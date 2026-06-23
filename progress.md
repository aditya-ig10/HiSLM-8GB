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
- `output/lora_adapter/` — saved LoRA adapter from 500-sample run (r=8)
- `output/lora_r4/lora_adapter/` — QLoRA ablation r=4
- `output/lora_r8/lora_adapter/` — QLoRA ablation r=8
- `output/lora_r16/lora_adapter/` — QLoRA ablation r=16
- `output/training.log` — latest training log
- `output/train_log.txt` — per-step loss/lr/mem log
- `output/signal_death.log` — signal diagnostics (empty if clean exit)
- `output/quant_benchmark.json` — Q4_K_M/Q5_K_M/Q8_0 benchmark results
- `output/train_ablation_r4.log`, `r8.log`, `r16.log` — ablation logs

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

## Training Stability Report (for Paper)

| Metric | Value |
|--------|-------|
| **Epochs tested** | 1, 2, 3 |
| **Stable sample range** | 500–2000 |
| **Failure mode** | Silent death at step ~40 with >2000 samples (GPU watchdog timeout) |
| **Fix applied** | `CUDA_LAUNCH_BLOCKING=1` + manual PyTorch loop + signal trapping |
| **Loss at convergence** | ~1.68 (500 samples, 20.6 min) → expected ~0.8–1.2 with full dataset |
| **Training time/sample** | ~2.5s → 500 samples ≈ 21 min, 2000 samples ≈ 83 min |
| **Full 219k estimate** | ~150 hours on NX; 4–6 hours on A5000 (bf16, batch=8) |
| **LoRA rank ablation** | r=4, 8, 16 tested on 500-sample subset (r=8 best: start 2.98→final 1.42 loss). r=4: 3.53→1.49, r=16: 3.44→1.52. All stable with GPU acceleration via bitsandbytes sm_87. |
| **Memory stability** | 0.9–1.0 GB free throughout training, no fragmentation growth |
| **Signal trapping** | SIGTERM/SIGINT/SIGHUP/SIGUSR1/2/SIGABRT/SIGQUIT caught with full stack trace dump |

## NX Fixes — Testing Results (2026-06-23)

### ✅ Completed (no AGX needed)

| Fix | Result |
|-----|--------|
| **FIX-NX-01** — Classifier eval | 130 queries, 72.3% accuracy, 0 errors. Keyword filter catches 26% in <0.01ms |
| **FIX-NX-02** — Routing overhead | HiSLM always slower than Always-AGX (+3-11s). Energy savings: ~100J/query |
| **FIX-NX-04** — Energy measurement | Idle 7.38W → Load 12.28W. Marginal 4.89W, ~100J per medical query |
| **FIX-NX-05** — Domain clarity | Framed as "Medical QA on edge" — dataset is correct |
| **FIX-NX-08** — Timing instrument | Per-query classify_ms/inference_ms/total_ms logged to AGX |
| **FIX-NX-10** — Systemd service | `hislm-nx.service` ready to deploy |
| **FIX-NX-11** — Stability report | Added to `progress.md` and `context.md` |

### ✅ Completed

| Fix | Result |
|-----|--------|
| **FIX-NX-07** — QLoRA ablation | r=4: 3.53→1.49, r=8: 2.98→1.42, r=16: 3.44→1.52. All stable on GPU (0.85-1.28 GB free). r=8 optimal. |
| **FIX-NX-09** — Quantization ablation | Q4_K_M: 57.1/15.3 PP/TG, Q5_K_M: 43.7/11.5, Q8_0: 62.3/13.1. Q4_K_M best tradeoff. |

### ⏳ Pending

| Fix | Est. effort | Why |
|-----|-------------|-----|
| **FIX-NX-06** — Baseline comparison | ~3-4 hrs | Needs AGX running to compare Always-NX/Always-AGX/HiSLM modes |

## New Files
- `eval_routing.jsonl` — 130 labeled queries
- `eval_classifier.py` — classifier metrics
- `analysis_routing_overhead.py` — break-even analysis
- `measure_nx_queries.py` — benchmark harness
- `parse_tegrastats.py` — power log parser (AGX + NX format)
- `eval_baseline.py` — three-mode comparison (needs AGX)
- `hislm-nx.service` — systemd unit
- `nx_test_results.json` — all test data
- Modified: `subserver.py` (keyword filter, /classify, timing), `context.md`, `README.md`

## Training To Do
- [ ] Run full training on more samples (2000+) — takes ~83 min for 2000
- [ ] Train with seq_len=256 (currently 128, needs memory testing)
- [ ] GGUF merge + conversion (use A5000 pipeline for full dataset)
- [ ] Evaluate model on medical QA benchmark
- [ ] Try full 219k dataset (estimate: ~150 hours NX; 4–6h A5000)

## Commands
```bash
# Start training
cd /home/nvidia/llama/HiSLM-8G && bash train.sh 2>&1 | tee output/training.log
# Check progress
tail -f output/training.log
tail -f output/train_log.txt
```
