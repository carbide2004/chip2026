#!/usr/bin/env python3
"""Train the first reproducible biomedical token boundary baseline on one CV fold.

This BIO model is deliberately an E3 baseline. It cannot represent every nested span;
the script reports spans dropped by label projection instead of hiding that limitation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import load_jsonl, save_jsonl

LABELS = {"O": 0, "B": 1, "I": 2}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def positive_spans(doc, passage):
    start = passage["offset"]
    end = start + len(passage["text"])
    spans = set()
    for entity in doc.get("entities", []):
        if str(entity.get("note") or "").upper() == "NO":
            continue
        left, right = entity["offset"], entity["offset"] + entity["length"]
        if start <= left and right <= end:
            spans.add((left - start, right - start))
    return sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0]))


class WindowDataset(Dataset):
    def __init__(self, docs, tokenizer, max_length=512, stride=128, include_metadata=False):
        self.items = []
        self.dropped_overlap_spans = 0
        self.gold_spans = defaultdict(set)
        for doc in docs:
            pmc_id = str(doc["pmc_id"])
            for passage_index, passage in enumerate(doc.get("full_text", [])):
                text = passage.get("text", "")
                if not text:
                    continue
                spans = positive_spans(doc, passage)
                for left, right in spans:
                    self.gold_spans[pmc_id].add((passage["offset"] + left, passage["offset"] + right))
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    stride=stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
                for input_ids, attention_mask, offsets in zip(
                    encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
                ):
                    labels = [-100 if tuple(offset) == (0, 0) else LABELS["O"] for offset in offsets]
                    content = [tuple(offset) for offset in offsets if tuple(offset) != (0, 0)]
                    if not content:
                        continue
                    window_start, window_end = min(x[0] for x in content), max(x[1] for x in content)
                    for span_start, span_end in spans:
                        if span_start < window_start or span_end > window_end:
                            continue
                        token_indices = [
                            index for index, (left, right) in enumerate(offsets)
                            if (left, right) != (0, 0) and left < span_end and right > span_start
                        ]
                        if not token_indices:
                            continue
                        if any(labels[index] != LABELS["O"] for index in token_indices):
                            self.dropped_overlap_spans += 1
                            continue
                        labels[token_indices[0]] = LABELS["B"]
                        for index in token_indices[1:]:
                            labels[index] = LABELS["I"]
                    item = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                    if include_metadata:
                        item["pmc_id"] = pmc_id
                        item["passage_offset"] = passage["offset"]
                        item["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
                    self.items.append(item)

    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def decode_batch(predictions, batch, output):
    for row, labels in enumerate(predictions.tolist()):
        pmc_id = batch["pmc_id"][row]
        passage_offset = int(batch["passage_offset"][row])
        offsets = batch["offset_mapping"][row].tolist()
        current = None
        for label, (left, right) in zip(labels, offsets):
            if (left, right) == (0, 0):
                continue
            if label == LABELS["B"] or (label == LABELS["I"] and current is None):
                if current is not None:
                    output[pmc_id].add(current)
                current = (passage_offset + left, passage_offset + right)
            elif label == LABELS["I"] and current is not None:
                current = (current[0], passage_offset + right)
            elif current is not None:
                output[pmc_id].add(current); current = None
        if current is not None:
            output[pmc_id].add(current)


def evaluate(model, loader, gold_spans, device):
    model.eval(); predicted = defaultdict(set)
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: batch[k].to(device) for k in ["input_ids", "attention_mask"]}
            labels = model(**inputs).logits.argmax(dim=-1).cpu()
            decode_batch(labels, batch, predicted)
    tp = fp = fn = 0
    for pmc_id in set(gold_spans) | set(predicted):
        gold, pred = gold_spans.get(pmc_id, set()), predicted.get(pmc_id, set())
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--locked-validation", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/raw/PatientPheX-V1-A")
    parser.add_argument("--e0-dir", type=Path, default=REPO_ROOT / "data/processed/e0")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed/e3/token_ner_fold0")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed); started = time.time()

    docs = load_jsonl(str(args.data_dir / "PatientPheX-train.jsonl")); by_id={str(d["pmc_id"]):d for d in docs}
    splits=json.loads((args.e0_dir/"splits.json").read_text())
    if args.locked_validation:
        train_ids=splits["development_pool"];validation_ids=splits["locked_validation"];scope="64-document development pool -> locked validation"
    else:
        fold=splits["cv_folds"][args.fold];train_ids=fold["train"];validation_ids=fold["validation"];scope=f"cv fold {args.fold}; locked validation untouched"
    train_docs=[by_id[x] for x in train_ids]; validation_docs=[by_id[x] for x in validation_ids]
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_set=WindowDataset(train_docs,tokenizer); validation_set=WindowDataset(validation_docs,tokenizer,include_metadata=True)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    validation_loader=DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=3,id2label={v:k for k,v in LABELS.items()},label2id=LABELS).to(device)
    revision=getattr(model.config,"_commit_hash",None); parameter_count=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
    total_steps=len(train_loader)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,total_steps//10),total_steps)
    counts=Counter(label for item in train_set.items for label in item["labels"].tolist() if label>=0)
    total=sum(counts.values()); weights=torch.tensor([math.sqrt(total/(3*counts[i])) for i in range(3)],device=device)

    history=[]; best_f1=-1; best_predictions=None; last_predictions=None
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids=batch["input_ids"].to(device); attention_mask=batch["attention_mask"].to(device); labels=batch["labels"].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                logits=model(input_ids=input_ids,attention_mask=attention_mask).logits
                loss=torch.nn.functional.cross_entropy(logits.view(-1,3),labels.view(-1),weight=weights,ignore_index=-100)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); losses.append(float(loss))
        metrics,predictions=evaluate(model,validation_loader,validation_set.gold_spans,device)
        last_predictions=predictions
        row={"epoch":epoch,"train_loss":sum(losses)/len(losses),**metrics};history.append(row);print(json.dumps(row),flush=True)
        if metrics["f1"]>best_f1:
            best_f1=metrics["f1"];best_predictions=predictions
            model.save_pretrained(args.output_dir/"model");tokenizer.save_pretrained(args.output_dir/"model")

    model.save_pretrained(args.output_dir/"model_last");tokenizer.save_pretrained(args.output_dir/"model_last")

    prediction_docs=[]
    for doc in validation_docs:
        entities=[]
        full_text=doc.get("full_text",[])
        for left,right in sorted(best_predictions.get(str(doc["pmc_id"]),set())):
            text=None
            for passage in full_text:
                local=left-passage["offset"]
                if 0<=local and right-passage["offset"]<=len(passage["text"]):text=passage["text"][local:right-passage["offset"]];break
            if text:
                entities.append({"identifier":"-1","type":"Phenotype","offset":left,"length":right-left,"text":text,"note":None})
        prediction_docs.append({"pmc_id":doc["pmc_id"],"pmid":doc.get("pmid"),"entities":entities,"association":[{"patient_id":p["patient_id"],"phenotype":[]} for p in doc.get("patient",[])]})
    save_jsonl(prediction_docs,str(args.output_dir/"span_predictions.jsonl"))
    report={"scope":scope,"model":{"name":args.model,"revision":revision,"parameter_count":parameter_count},"config":vars(args)|{"data_dir":str(args.data_dir),"e0_dir":str(args.e0_dir),"output_dir":str(args.output_dir)},"train_windows":len(train_set),"validation_windows":len(validation_set),"train_label_counts":dict(counts),"class_weights":weights.detach().cpu().tolist(),"dropped_overlap_span_projections":{"train":train_set.dropped_overlap_spans,"validation":validation_set.dropped_overlap_spans},"history":history,"best_span_f1":best_f1,"elapsed_seconds":time.time()-started}
    (args.output_dir/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
