#!/usr/bin/env python3
"""Minimal step-by-step test to find where training dies."""
import json, os, sys, torch, gc

os.environ.update({
    "CUDA_MODULE_LOADING": "LAZY",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCHINDUCTOR_COMPILE_THREADS": "0",
})
import torch._dynamo
torch._dynamo.config.disable = True

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def mem():
    free, total = torch.cuda.mem_get_info()
    print(f"  Memory: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
    return free

print("=== Step 1: Load tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"  Vocab size: {tokenizer.vocab_size}")
mem()

print("\n=== Step 2: Create dummy data ===")
texts = [
    "<|user|>What is the capital of France?</s>\n<|assistant|>Paris</s>"
    for _ in range(16)
]
enc = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
print(f"  Input shape: {enc['input_ids'].shape}")
mem()
ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["input_ids"].clone()})
print(f"  Dataset samples: {len(ds)}")

print("\n=== Step 3: Load model in 4-bit ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="cuda:0", torch_dtype=torch.float16, low_cpu_mem_usage=True,
)
mem()

print("\n=== Step 4: prepare_model_for_kbit_training (NO grad ckpt) ===")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
mem()

print("\n=== Step 5: Add LoRA ===")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
mem()

print("\n=== Step 6: Single forward pass ===")
def collate_fn(b):
    return {
        "input_ids": torch.stack([torch.as_tensor(x["input_ids"]) for x in b]).cuda(),
        "attention_mask": torch.stack([torch.as_tensor(x["attention_mask"]) for x in b]).cuda(),
        "labels": torch.stack([torch.as_tensor(x["labels"]) for x in b]).cuda(),
    }
train_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for step, batch in enumerate(train_loader):
    if step >= 8:
        break
    print(f"  Step {step}: ", end="")
    m0 = mem()
    outputs = model(**batch)
    print(f"    Loss: {outputs.loss.item():.4f}", end="")
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    m1 = mem()
    print(f", mem delta: {(m0-m1)/1e6:.1f}MB", end="")
    print()
    gc.collect(); torch.cuda.empty_cache()

print("\n=== SUCCESS ===")
