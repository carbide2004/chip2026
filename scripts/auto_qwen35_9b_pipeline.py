"""End-to-end background pipeline for Qwen3.5-9B:
1. Resilient download via ModelScope
2. Zero-shot Evaluation on Benchmark
3. LoRA / QLoRA Fine-tuning on RTX 4090
4. Post-finetuning Evaluation & Benchmark Comparison
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Ensure utf-8 output and repo in path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

MODEL_ID = "Qwen/Qwen3.5-9B"
LOG_FILE = repo_root / "qwen35_9b_pipeline.log"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def download_model(max_retries: int = 50) -> str:
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    log(f"Starting resilient download for {MODEL_ID}...")
    for attempt in range(1, max_retries + 1):
        try:
            log(f"Download attempt {attempt}/{max_retries}...")
            model_dir = snapshot_download(MODEL_ID)
            
            # Verify that all 4 safetensors exist
            p = Path(model_dir)
            safetensors = list(p.glob("*.safetensors"))
            log(f"Downloaded files located at: {model_dir}")
            log(f"Found {len(safetensors)} safetensors partitions.")
            if len(safetensors) >= 4 or (p / "model.safetensors").exists():
                log("Download verification passed successfully!")
                return model_dir
            else:
                log(f"Incomplete partitions ({len(safetensors)}), retrying in 5s...")
                time.sleep(5)
        except Exception as e:
            log(f"Download error on attempt {attempt}: {e}. Retrying in 10s...")
            time.sleep(10)

    raise RuntimeError("Failed to complete download after maximum retries.")


def run_zero_shot_eval(model_path: str):
    log("=========================================")
    log("=== Running Phase 2: Zero-Shot Evaluation ===")
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
    with open(repo_root / "submissions" / "zeroshot_9b_eval.txt", "w", encoding="utf-8") as f:
        f.write(proc.stdout)


def train_lora_or_qlora(model_path: str) -> str:
    log("=========================================")
    log("=== Running Phase 3: LoRA / QLoRA Tuning ===")
    log("=========================================")
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    sft_data_file = repo_root / "data" / "processed" / "sft_association.json"
    with open(sft_data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    log(f"Loaded {len(raw_data)} training instruction pairs.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Format into chat messages
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
            max_length=2048,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    output_dir = repo_root / "saves" / "qwen35_9b_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try 16-bit LoRA first; if OOM occurs, fallback to 4-bit QLoRA
    mode = "LoRA (16-bit)"
    try:
        log("Attempting Standard 16-bit LoRA loading on RTX 4090...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
    except Exception as e:
        log(f"Standard LoRA failed ({e}). Switching to 4-bit QLoRA...")
        mode = "QLoRA (4-bit)"
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    log(f"Successfully loaded model using {mode}!")
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
        learning_rate=1e-4,
        num_train_epochs=3.0,
        logging_steps=10,
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

    log("Starting LoRA training...")
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"Training completed! Weights saved to: {output_dir}")
    return str(output_dir)


def run_final_eval(model_path: str, lora_path: str):
    log("=========================================")
    log("=== Running Phase 4: Post-LoRA Benchmark Evaluation ===")
    log("=========================================")
    import subprocess
    cmd = [
        sys.executable,
        str(repo_root / "src" / "inference_8b.py"),
        "--base_model", model_path,
        "--lora_dir", lora_path,
        "--output", str(repo_root / "submissions" / "qwen35_9b_final_A.jsonl"),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log(proc.stdout)
    log("All pipeline steps successfully finished!")


def main():
    log("=========================================")
    log("=== Qwen3.5-9B Automated Pipeline Started ===")
    log(f"=== GPU: NVIDIA GeForce RTX 4090 24GB ===")
    log("=========================================")
    try:
        model_dir = download_model()
        run_zero_shot_eval(model_dir)
        lora_dir = train_lora_or_qlora(model_dir)
        run_final_eval(model_dir, lora_dir)
    except Exception as e:
        log(f"Fatal error in pipeline: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
