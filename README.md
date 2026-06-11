# HiSLM-8G

Hierarchical Small Language Model inference system for edge devices. Fine-tune and serve 1-3B parameter LLMs on **Jetson Orin NX 8GB**, with a companion **AGX Orin server** and a desktop **A5000 training pipeline**.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AGX Orin (Server)                  │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Qwen 2.5-3B │  │  Flask/WS    │  │  ngrok      │  │
│  │ GGUF Model  │  │  Server      │  │  Tunnel     │  │
│  └────────────┘  └──────────────┘  └─────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ LAN / Tailscale
    ┌──────────────────┴──────────────────┐
    │          Orin NX (Client)           │
    │  client.py ── LAN mode              │
    │  client2.py ── Tailscale mode       │
    │  client_2.py ── Generic mode        │
    └─────────────────────────────────────┘
```

## Components

### Training (`train.py`)
Fine-tunes **TinyLlama-1.1B-Chat-v1.0** on medical datasets using QLoRA (4-bit). Runs on Jetson Orin NX 8GB with unified memory.

- **Datasets:** Med_QA, MT Samples, PubMed QA → 224k instruction pairs
- **Method:** QLoRA (rank=8, alpha=16), 4-bit nf4, double_quant
- **Hardware:** Jetson Orin NX 8GB (JetPack 6.2, CUDA 12.6)
- **Stable training:** 500–2000 samples, ~2.5s/sample, 0.9–1.0GB free memory

```bash
# Preprocess data
python preprocess.py

# Train (via launcher)
bash train.sh

# Or directly
python train.py
```

### Inference Server (`server_qwen.py`)
Flask + WebSocket server serving **Qwen 2.5-3B** (GGUF, Q4_K_M) via `llama-cli` subprocess. Supports SSE streaming, WebSocket chat, and ngrok tunneling.

```bash
bash run_qwen_web.sh
```

### Clients
| Client | Use Case |
|--------|----------|
| `client.py` | Connect to AGX Orin server over LAN |
| `client_2.py` | Connect to any HiSLM server (generic) |
| `client2.py` | Connect over Tailscale wireless |

### A5000 Training Package (`a5000_training/`)
Alternative training pipeline for desktop RTX A5000 (24GB) with full 219k dataset, bf16 precision, and higher LoRA rank.

```bash
cd a5000_training
pip install -r requirements.txt
python train_a5000.py
```

## Web UIs
- `static/index.html` — Sci-fi terminal chat UI (used by AGX server)
- `static/nx_index.html` — Wireless NX client UI
- `orin_index.html` — Standalone chat UI for Qwen server

## Project Structure

```
HiSLM-8G/
├── train.py                  # Main training script (QLoRA)
├── preprocess.py             # Dataset preprocessing
├── server_qwen.py            # Inference server (Flask + WS)
├── client.py                 # LAN client
├── client_2.py               # Generic client
├── client2.py                # Tailscale wireless client
├── test_step.py              # Training diagnostic script
├── run_qwen_web.sh           # Server deployment launcher
├── train.sh                  # Training launcher
├── static/
│   ├── index.html            # AGX chat UI
│   └── nx_index.html         # NX wireless UI
├── orin_index.html           # Standalone Qwen chat UI
├── a5000_training/           # Desktop GPU training pipeline
│   ├── train_a5000.py
│   ├── merge_and_convert.py
│   └── requirements.txt
├── dataset/                  # Medical datasets (gitignored)
├── models/                   # GGUF models (gitignored)
├── output/                   # Training outputs (gitignored)
├── CLIENT_TIMEOUT_TROUBLESHOOTING.md
└── CLIENT_UI.md
```

## Environment

- **Device:** NVIDIA Jetson Orin NX 8GB (aarch64)
- **JetPack:** R36.4.7, CUDA 12.6, Driver 540.4.0
- **PyTorch:** 2.5.0a0+nv24.08 (Jetson-specific)
- **Python:** 3.10+

| Hardware | Memory | Use Case |
|----------|--------|----------|
| Jetson Orin NX | 8GB unified | Training + Inference client |
| Jetson AGX Orin | 32GB+ | Inference server |
| Desktop RTX A5000 | 24GB | Full training pipeline |

## Docs
- [Client timeout troubleshooting](CLIENT_TIMEOUT_TROUBLESHOOTING.md)
- [Client/UI architecture](CLIENT_UI.md)
- [Training progress](progress.md)
