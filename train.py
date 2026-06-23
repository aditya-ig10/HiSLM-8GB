#!/usr/bin/env python3
"""Fine-tune TinyLlama with diagnostic signal trapping."""
import json, os, torch, gc, subprocess, time, signal, sys, faulthandler
from pathlib import Path

faulthandler.enable()
os.environ.update({
    "CUDA_MODULE_LOADING": "LAZY",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCHINDUCTOR_COMPILE_THREADS": "0",
    "PYTHONFAULTHANDLER": "1",
})
import torch._dynamo
torch._dynamo.config.disable = True

# ── Signal diagnostics ────────────────────────────────────────────
(Path(__file__).resolve().parent / "output").mkdir(exist_ok=True)
signal_log = open(Path(__file__).resolve().parent / "output" / "signal_death.log", "a")
signal_log.write(f"\n=== PID {os.getpid()} started at {time.strftime('%H:%M:%S')} ===\n")
signal_log.flush()

def signal_handler(signum, frame):
    msg = f"[SIGNAL] Caught signal {signum} ({signal.Signals(signum).name}) at {time.strftime('%H:%M:%S')}, elapsed={time.time()-global_start:.0f}s"
    print(msg, flush=True)
    signal_log.write(msg + "\n")
    signal_log.flush()
    # Print traceback of all threads
    import traceback
    for th_id, frame in sys._current_frames().items():
        signal_log.write(f"\n--- Thread {th_id} ---\n")
        traceback.print_stack(frame, file=signal_log)
    signal_log.flush()
    sys.exit(128 + signum)

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2, signal.SIGABRT, signal.SIGQUIT]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, OSError):
        pass

global_start = time.time()

subprocess.run(["pkill", "-9", "-f", "compile_worker"], capture_output=True)
subprocess.run(["pkill", "-9", "-f", "triton"], capture_output=True)
gc.set_threshold(100, 10, 10)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

BASE = Path(__file__).resolve().parent

# ── CLI overrides ─────────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
_parser.add_argument("--samples", type=int, default=500, help="Max training samples")
_parser.add_argument("--output", type=str, default=None, help="Output subdirectory")
_args, _ = _parser.parse_known_args()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_R = _args.lora_r
LORA_ALPHA = LORA_R * 2  # alpha = 2*r
TARGET_MODULES = ["q_proj", "v_proj"]
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_STEPS = 50
MAX_TRAIN_SAMPLES = _args.samples
OUTPUT_DIR = BASE / "output" / (_args.output or f"lora_r{LORA_R}")
LOG_STEPS = 5
SAVE_STEPS = max(200, MAX_TRAIN_SAMPLES)  # disable intermediate checkpoints for small runs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def format_example(example):
    user_msg = f"{example['instruction']}\n{example['input']}" if example['input'] else example['instruction']
    return f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{example['output']}</s>"

def load_texts():
    print("Loading raw data...")
    texts = []
    with open(BASE / "dataset" / "train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= MAX_TRAIN_SAMPLES:
                break
            texts.append(format_example(json.loads(line)))
    print(f"Samples: {len(texts)}")
    return texts

def train():
    # Phase 1: tokenize (CPU only)
    texts = load_texts()
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=True, return_tensors="pt")
    del texts
    gc.collect()
    n = enc["input_ids"].shape[0]
    labels = enc["input_ids"].clone()
    ids_cpu, mask_cpu, lbl_cpu = enc["input_ids"], enc["attention_mask"], labels
    del enc, labels
    gc.collect()

    # Phase 2: model (GPU)
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

    # Phase 3: training with pure PyTorch (no datasets library)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_STEPS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max((n * NUM_EPOCHS) // GRAD_ACCUM - WARMUP_STEPS, 1)
    )

    print(f"\nStarting: {n} samples, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    global_step = 0
    accum_loss = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        indices = torch.randperm(n)
        pbar = tqdm(range(0, n, BATCH_SIZE), desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for step_offset in pbar:
            idx = indices[step_offset:step_offset+BATCH_SIZE]
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
                    pbar.set_postfix({
                        "loss": f"{accum_loss / LOG_STEPS:.4f}",
                        "lr": f"{lr:.2e}",
                        "mem": f"{mem_free/1e9:.1f}GB",
                    })
                    with open(OUTPUT_DIR / "train_log.txt", "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "loss": round(accum_loss / LOG_STEPS, 4),
                            "lr": lr, "mem_gb": round(mem_free / 1e9, 2),
                            "elapsed_s": round(elapsed),
                        }) + "\n")
                    accum_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt_dir = OUTPUT_DIR / f"lora_step_{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\nCheckpoint saved: {ckpt_dir}")

            gc.collect()
            torch.cuda.empty_cache()

    print("\nSaving final LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR / "lora_adapter")
    tokenizer.save_pretrained(OUTPUT_DIR / "lora_adapter")
    print(f"Saved to {OUTPUT_DIR / 'lora_adapter'}")
    print(f"Complete in {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    train()
