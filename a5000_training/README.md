# HiSLM — A5000 Training Package

Train TinyLlama-1.1B-Chat-v1.0 on the medical dataset using LoRA on an A5000 (24GB), then convert to GGUF for inference on Orin NX.

## Quick Start

```bash
# 1. Copy this folder + the dataset to the A5000
#    From NX:
scp -r a5000_training/ user@a5000:~/hislm_training/
scp dataset/train.jsonl dataset/val.jsonl user@a5000:~/hislm_training/data/

# 2. On the A5000, set up environment
cd ~/hislm_training
pip install -r requirements.txt
pip install --upgrade transformers peft accelerate bitsandbytes

# 3. Train
python train_a5000.py

# 4. Merge LoRA + convert to GGUF
python convert_to_gguf.py --output ./output/tinyllama-medical-q4_k_m.gguf

# 5. Copy the GGUF back
scp output/tinyllama-medical-q4_k_m.gguf user@nx:~/llama/HiSLM-8G/models/
```

## What's Different from the NX Training

| Setting | NX (8GB) | A5000 (24GB) |
|---------|----------|--------------|
| Precision | fp16 + 4-bit nf4 | bf16 (native) |
| LoRA rank | 8 | 16 |
| LoRA targets | q_proj, v_proj | q_proj, v_proj, k_proj, o_proj |
| Seq length | 128 | 512 |
| Batch size | 1 | 8 |
| Grad accum | 4 | 4 |
| Effective batch | 4 | 32 |
| Training time (219k) | ~150h | ~4-6h |
| Gradient checkpointing | No | No (24GB is enough) |
| Dataset | 2000 samples | Full 219k |

## Files

| File | Purpose |
|------|---------|
| `train_a5000.py` | Main training script |
| `merge_and_convert.py` | Merge LoRA → convert to GGUF |
| `requirements.txt` | Python dependencies |
| `run.sh` | One-shot launcher |
