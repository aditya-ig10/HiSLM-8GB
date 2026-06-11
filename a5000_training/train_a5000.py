#!/usr/bin/env python3
"""
Train TinyLlama-1.1B-Chat-v1.0 with LoRA on A5000 (24GB).

Uses bf16, higher rank, longer context, full 219k dataset.
"""
import json, os, gc, time, math
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch._dynamo
torch._dynamo.config.disable = True

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
SAVE_STEPS = 500
LOG_STEPS = 50

def format_example(example):
    user_msg = f"{example['instruction']}\n{example['input']}" if example['input'] else example['instruction']
    return {
        "text": f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{example['output']}</s>"
    }

def load_dataset(split="train"):
    fp = DATA_DIR / f"{split}.jsonl"
    if not fp.exists():
        raise FileNotFoundError(f"Dataset not found: {fp}")
    raw = []
    with open(fp) as f:
        for line in f:
            raw.append(json.loads(line))
    formatted = [format_example(r) for r in raw]
    from datasets import Dataset
    return Dataset.from_list(formatted)

def tokenize_fn(examples, tokenizer):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_tensors=None,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_dataset = load_dataset("train")
    print(f"  Train: {len(train_dataset)} samples")

    print("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["text"],
        desc="Tokenizing",
    )

    print("Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    print("Adding LoRA...")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    free, total = torch.cuda.mem_get_info()
    print(f"Memory after setup: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total\n")

    total_steps = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )

    print("Starting training...")
    start = time.time()
    trainer.train()
    elapsed = (time.time() - start) / 60
    print(f"\nTraining complete in {elapsed:.1f} min")

    print("Saving final LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR / "lora_adapter_final")
    tokenizer.save_pretrained(OUTPUT_DIR / "lora_adapter_final")
    print(f"Saved to {OUTPUT_DIR / 'lora_adapter_final'}")

if __name__ == "__main__":
    main()
