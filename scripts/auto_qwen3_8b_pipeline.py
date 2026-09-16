"""End-to-end pipeline for official benchmark model Qwen3-8B:
1. Resilient ModelScope download with explicit progress reporting
2. Zero-shot Benchmark Evaluation
3. Fast 16-bit LoRA Supervised Fine-Tuning (SFT) on RTX 4090
4. Post-LoRA Benchmark Evaluation & A-test Submission Generation
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

MODEL_ID = "Qwen/Qwen3-8B"
LOG_FILE = repo_root / "qwen3_8b_pipeline.log"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def download_qwen3_8b(max_retries: int = 50) -> str:
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    log(f"Starting download for {MODEL_ID} from ModelScope...")
    api = HubApi()
    files = api.get_model_files(MODEL_ID)
    safetensors_files = [f for f in files if "safetensors" in f.get("Path", "")]
    total_expected_bytes = sum(f.get("Size", 0) for f in safetensors_files)
    total_expected_gb = total_expected_bytes / (1024**3)
    log(f"Model {MODEL_ID} contains {len(safetensors_files)} safetensors partitions, total {total_expected_gb:.2f} GB.")

    for attempt in range(1, max_retries + 1):
        try:
            log(f"Download attempt {attempt}/{max_retries}...")
            model_dir = snapshot_download(MODEL_ID)
            p = Path(model_dir)
            st_files = list(p.glob("*.safetensors"))
            total_down_bytes = sum(f.stat().st_size for f in st_files)
            total_down_gb = total_down_bytes / (1024**3)
            log(f"Local files in {model_dir}: {len(st_files)} files, {total_down_gb:.2f} GB downloaded.")

            if len(st_files) >= len(safetensors_files) and total_down_bytes >= (total_expected_bytes * 0.99):
                log("Download completely verified!")
                return model_dir
            else:
                log("Files incomplete, retrying download in 5s...")
                time.sleep(5)
        except Exception as e:
            log(f"Download warning on attempt {attempt}: {e}. Retrying in 10s...")
            time.sleep(10)

    raise RuntimeError("Download failed after reaching max retries.")


def run_zero_shot_eval(model_path: str):
    log("=========================================")
    log("=== Phase 2: Qwen3-8B Zero-Shot Eval  ===")
    log("=========================================")
    import subprocess
    cmd = [
        sys.executable,
        str(repo_root / "src" / "evaluate_qwen35.py"),
        "--num_docs", "20",
        "--model_id", model_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log(proc.stdout)
    with open(repo_root / "submissions" / "zeroshot_qwen3_eval.txt", "w", encoding="utf-8") as f:
        f.write(proc.stdout)


def train_lora(model_path: str) -> str:
    log("=========================================")
    log("=== Phase 3: Qwen3-8B 16-bit LoRA SFT ===")
    log("=========================================")
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    sft_data_file = repo_root / "data" / "processed" / "sft_association.json"
    with open(sft_data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    log(f"Loaded {len(raw_data)} training instruction pairs.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_data = []
    for item in raw_data:
        conv = [
            {"role": "system", "content": item.get("instruction", "")},
            {"role": "user", "content": item.get("input", "")},
            {"role": "assistant", "content": item.get("output", "")},
        ]
        text = tokenizer.apply_chat_template(conv, tokenize=False)
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1536,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    output_dir = repo_root / "saves" / "qwen3_8b_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading Qwen3-8B in bfloat16 onto RTX 4090...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3.0,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    log("Starting fast 16-bit LoRA training...")
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"LoRA Training successfully finished! Saved to: {output_dir}")
    return str(output_dir)


def run_post_eval_and_submission(model_path: str, lora_path: str):
    log("=========================================")
    log("=== Phase 4: Post-LoRA & Submissions  ===")
    log("=========================================")
    import subprocess
    sub_path = repo_root / "submissions" / "qwen3_8b_final_A.jsonl"
    cmd = [
        sys.executable,
        str(repo_root / "src" / "inference_8b.py"),
        "--base_model", model_path,
        "--lora_dir", lora_path,
        "--output", str(sub_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log(proc.stdout)
    log(f"A-test final submission generated at: {sub_path}")
    log("All pipeline steps for Qwen3-8B successfully completed!")


def main():
    log("==================================================")
    log("=== Official Benchmark Pipeline: Qwen3-8B (GPU) ===")
    log(f"=== GPU: NVIDIA GeForce RTX 4090 24GB           ===")
    log("==================================================")
    try:
        model_dir = download_qwen3_8b()
        run_zero_shot_eval(model_dir)
        lora_dir = train_lora(model_dir)
        run_post_eval_and_submission(model_dir, lora_dir)
    except Exception as e:
        log(f"Fatal error in pipeline: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
