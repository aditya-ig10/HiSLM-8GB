# HiSLM-8G Fine-tuning Project — Progress

## Goal
Fine-tune TinyLlama-1.1B-Chat-v1.0 on medical datasets (Med_QA, MT Samples, Pub_Med_QA) using QLoRA on Jetson Orin NX 8GB, then convert to GGUF and evaluate.

## Environment
- **Device:** NVIDIA Jetson Orin NX 8GB (aarch64, unified CPU/GPU memory)
- **JetPack:** R36.4.7 (JP 6.2), CUDA 12.6, Driver 540.4.0
- **PyTorch:** 2.5.0a0+nv24.08 (Jetson-specific)
- **bitsandbytes:** 0.50.0.dev0 (built from source for sm_87)
- **Unified memory bug:** `torch.cuda.memory_allocated()` always returns 0; use `mem_get_info()`

## Done
- [x] Environment setup: replaced incompatible torch, built bitsandbytes for sm_87, pinned numpy<2
- [x] Preprocessing (`preprocess.py`): 3 datasets → `train.jsonl` (219k) + `val.jsonl` (11.5k)
- [x] Dataset: 224,838 medical QA + 4,966 MT samples + 1,000 PubMed QA
- [x] Model: TinyLlama-1.1B-Chat-v1.0 in 4-bit QLoRA (nf4, double_quant)
- [x] LoRA config: rank=8, alpha=16, target q_proj+v_proj, 1.13M trainable params
- [x] Manual training loop (replaced Trainer — Trainer died at step 7)
- [x] Pure PyTorch training (no datasets library, no DataLoader for training)
- [x] Signal trapping + faulthandler diagnostics
- [x] **500-sample training completed successfully** — 20.6 min, loss 2.65→1.68, exit code 0
- [x] LoRA adapter saved: `output/lora_adapter/` (4.4MB .safetensors + tokenizer)

## Instability Solved
**The ~100s silent death** was fixed by:
1. **`CUDA_LAUNCH_BLOCKING=1`** — makes CUDA ops synchronous, preventing GPU watchdog timeouts
2. **Reduced samples (500→2000)** — more free memory headroom (1.0GB vs 0.2GB)
3. **Manual loop** — no Trainer internals causing memory fragmentation

Both 500 and 2000 sample configs now train stably with 0.9-1.0GB free memory.

## Current State
- `train.py` — manual training loop with diagnostics, working
- `train.sh` — launcher with exit code capture
- `test_step.py` — minimal proof-of-concept (still valid)
- `output/lora_adapter/` — saved LoRA adapter from 500-sample run
- `output/training.log` — latest training log
- `output/train_log.txt` — per-step loss/lr/mem log
- `output/signal_death.log` — signal diagnostics (empty if clean exit)

## Training Hyperparams (working)
| Param | Value |
|-------|-------|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Quant | 4-bit nf4, double_quant, fp16 compute |
| LoRA r/alpha | 8/16 |
| Target modules | q_proj, v_proj |
| Seq length | 128 |
| Batch size | 1 |
| Grad accum | 4 |
| Optimizer | AdamW (lr=2e-4) |
| Schedule | warmup 50 steps → cosine decay |
| Samples | 500 (tested) or 2000 (tested, stable) |
| Time/sample | ~2.5s (with CUDA_LAUNCH_BLOCKING=1) |
| Free memory | 0.9-1.0GB stable during training |

## To Do
- [ ] Run full training on more samples (2000+) — takes ~83 min for 2000
- [ ] Train with seq_len=256 (currently 128, needs memory testing)
- [ ] GGUF merge + conversion
- [ ] Evaluate model on medical QA benchmark
- [ ] Try full 219k dataset (estimate: ~150 hours)

## Commands
```bash
# Start training
cd /home/nvidia/llama/HiSLM-8G && bash train.sh 2>&1 | tee output/training.log
# Check progress
tail -f output/training.log
tail -f output/train_log.txt
```
