#!/usr/bin/env python3
"""
retrain_from_cache.py — QLoRA fine-tuning on cached AGX responses.

Reads the AGX response cache and fine-tunes TinyLlama-1.1B to learn
the queries that were previously forwarded to AGX.

Usage:
  python retrain_from_cache.py
  python retrain_from_cache.py --cache cache/agx_cache.json --lora-r 8 --samples 50
"""
import json
import os
import gc
import sys
import time
import signal
import faulthandler
from pathlib import Path

faulthandler.enable()
os.environ.update({
    "CUDA_MODULE_LOADING": "LAZY",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCHINDUCTOR_COMPILE_THREADS": "0",
    "PYTHONFAULTHANDLER": "1",
})
import torch
import torch._dynamo
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

BASE = Path(__file__).resolve().parent

# ── CLI ────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="Retrain from AGX cache")
parser.add_argument("--cache", default=str(BASE / "cache" / "agx_cache.json"),
                    help="Cache JSON path")
parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
parser.add_argument("--samples", type=int, default=0,
                    help="Max training samples (0 = all)")
parser.add_argument("--epochs", type=int, default=2,
                    help="Number of epochs")
args = parser.parse_args()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_R = args.lora_r
LORA_ALPHA = LORA_R * 2
TARGET_MODULES = ["q_proj", "v_proj"]
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = args.epochs
WARMUP_STEPS = 10
OUTPUT_DIR = BASE / "output" / f"retrain_cache_r{LORA_R}"
LOG_STEPS = 5
SAVE_STEPS = 999999  # only save at end

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output dir: {OUTPUT_DIR}")
print(f"Model: {MODEL_NAME}")
print(f"LoRA rank: {LORA_R}")

# ── Load cache ─────────────────────────────────────────────────────
cache_path = Path(args.cache)
if not cache_path.exists():
    print(f"Cache not found: {cache_path}")
    sys.exit(1)

with open(cache_path) as f:
    cache_data = json.load(f)

queries = cache_data.get("queries", {})
items = [
    {"instruction": v["query"], "output": v["response"]}
    for v in queries.values()
]

if args.samples > 0:
    items = items[:args.samples]

print(f"Loaded {len(items)} cached query-response pairs")

if len(items) < 5:
    print("Too few items to train. Need at least 5.")
    sys.exit(0)


# ── Format examples ────────────────────────────────────────────────
def format_example(example):
    user_msg = example["instruction"]
    return f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{example['output']}</s>"


def tokenize(tokenizer, texts):
    return tokenizer(
        texts, truncation=True, max_length=MAX_SEQ_LENGTH,
        padding=True, return_tensors="pt",
    )


# ── Phase 1: Tokenize ──────────────────────────────────────────────
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

texts = [format_example(ex) for ex in items]
enc = tokenize(tokenizer, texts)
del texts
gc.collect()

n = enc["input_ids"].shape[0]
labels = enc["input_ids"].clone()
ids_cpu, mask_cpu, lbl_cpu = enc["input_ids"], enc["attention_mask"], labels
del enc, labels
gc.collect()

print(f"Tokenized {n} sequences (max_len={MAX_SEQ_LENGTH})")

# ── Phase 2: Load model ────────────────────────────────────────────
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="cuda:0", torch_dtype=torch.float16, low_cpu_mem_usage=True,
)
gc.collect()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
gc.collect()

free, total = torch.cuda.mem_get_info()
print(f"Free before LoRA: {free / 1e9:.2f} GB")

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
gc.collect()

free, total = torch.cuda.mem_get_info()
print(f"Memory after setup: {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB")

# ── Phase 3: Train ─────────────────────────────────────────────────
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=WARMUP_STEPS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max((n * NUM_EPOCHS) // GRAD_ACCUM - WARMUP_STEPS, 1)
)

print(f"\nStarting: {n} samples, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, epochs={NUM_EPOCHS}")
global_step = 0
accum_loss = 0.0
start_time = time.time()
best_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    indices = torch.randperm(n)
    pbar = tqdm(range(0, n, BATCH_SIZE), desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()

    for step_offset in pbar:
        idx = indices[step_offset:step_offset + BATCH_SIZE]
        batch = {
            "input_ids": ids_cpu[idx].cuda(non_blocking=True),
            "attention_mask": mask_cpu[idx].cuda(non_blocking=True),
            "labels": lbl_cpu[idx].cuda(non_blocking=True),
        }
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        accum_loss += loss.item()
        loss.backward()
        del batch, outputs, loss

        if (step_offset // BATCH_SIZE + 1) % GRAD_ACCUM == 0 or (step_offset + BATCH_SIZE) >= n:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if global_step < WARMUP_STEPS:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            global_step += 1

            if global_step % LOG_STEPS == 0:
                elapsed = time.time() - start_time
                mem_free, _ = torch.cuda.mem_get_info()
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = accum_loss / LOG_STEPS
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "mem": f"{mem_free/1e9:.1f}GB",
                })
                with open(OUTPUT_DIR / "train_log.txt", "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "loss": round(avg_loss, 4),
                        "lr": lr, "mem_gb": round(mem_free / 1e9, 2),
                        "elapsed_s": round(elapsed),
                    }) + "\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                accum_loss = 0.0

            if global_step % SAVE_STEPS == 0 and global_step > 0:
                ckpt_dir = OUTPUT_DIR / f"lora_step_{global_step}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        gc.collect()
        torch.cuda.empty_cache()

print(f"\nBest loss: {best_loss:.4f}")
print("Saving final LoRA adapter...")
model.save_pretrained(OUTPUT_DIR / "lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR / "lora_adapter")
print(f"Saved to {OUTPUT_DIR / 'lora_adapter'}")
total_min = (time.time() - start_time) / 60
print(f"Complete in {total_min:.1f} min")
