# HiSLM-8G

Hierarchical Small Language Model inference system for edge devices — Medical QA domain.
Fine-tune and serve 1-3B parameter LLMs on **Jetson Orin NX 8GB**, with a companion
**AGX Orin server** (Qwen2.5-3B) and a desktop **A5000 training pipeline**.

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │         AGX Orin (Server)            │
                     │  ┌────────────┐  ┌───────────────┐  │
                     │  │ Qwen 2.5-3B │  │  server2.py   │  │
                     │  │ llama.cpp   │  │  Flask + REST  │  │
                     │  └────────────┘  └───────┬───────┘  │
                     └──────────────────────────┼──────────┘
                                                │ LAN / Tailscale
     ┌──────────────────────────────────────────┴──────────────┐
     │               Orin NX (Edge Router)                      │
     │  ┌────────────────── subserver.py ────────────────────┐  │
     │  │  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  │  │
     │  │  │  Model Queue  │  │Classifier│→│   Router    │  │  │
     │  │  │  (thread-safe)│  │(3-stage) │  │(threshold)  │  │  │
     │  │  └──────────────┘  └──────────┘  └──────┬──────┘  │  │
     │  │                                     │            │  │
     │  │                               ┌─────┘     ┌─────┘  │  │
     │  │                               ▼           ▼        │  │
     │  │                        ┌────────────┐ ┌─────────┐  │  │
     │  │                        │ Qwen 1.5B  │ │ AGX     │  │  │
     │  │                        │ (local NX) │ │ (relay) │  │  │
     │  │                        └────────────┘ └─────────┘  │  │
     │  └────────────────────────────────────────────────────┘  │
     │                                                          │
     │  Clients: client.py / client2.py / Web UIs               │
     └──────────────────────────────────────────────────────────┘
```

### Hardware Tiers

| Tier | Device | Memory | Role |
|------|--------|--------|------|
| **Edge** | Jetson Orin NX 8GB | 8 GB unified | Hybrid router + local inference (Qwen2.5-1.5B) + QLoRA training |
| **Server** | Jetson AGX Orin 32GB+ | 32 GB+ | Inference server (Qwen2.5-3B, GPU accelerated) |
| **Desktop** | RTX A5000 | 24 GB | Full training pipeline (bf16, full 219k dataset) |

---

## Quick Start

### 1. Start NX Subserver (classify + route)

```bash
source venv/bin/activate
python subserver.py --agx-ip 172.16.6.21     # LAN AGX
python subserver.py --agx-ip 100.120.59.117  # Tailscale AGX
```

Optional flags:
```
--confidence 0.7    Min confidence to route to NX (default 0.7)
--port 8765         This server's port (default 8765)
--workers 1         Worker threads (keep at 1 for NX)
```

### 2. Start Standalone NX Server (no AGX needed)

```bash
python server_qwen.py                         # port 8765
python server_qwen.py --port 8080 --ngrok     # custom port + ngrok
```

### 3. Clients

```bash
# LAN GUI
python client.py --agx-ip 172.16.6.21
# Tailscale CLI
python client2.py --agx-ip 100.120.59.117 --cli --node-name nx-node
```

---

## Subserver (`subserver.py`) — How Routing Works

All model operations go through a **thread-safe serialised queue** (`queue.Queue` +
single daemon worker). This prevents concurrent `llama-cli` processes from
OOM'ing the 8GB NX.

### 3-Stage Classifier

**Stage 1 — Keyword pre-filter** (<0.01ms):
Checks query against 70+ medical/greeting keywords. Match → immediate
`is_medical=True, confidence=0.95`. Bypasses LLM entirely.

**Stage 2 — Multi-sample LLM scoring** (~11s):
Runs the classifier prompt at temps 0.0 (deterministic) and 0.5 (stochastic).
Each produces a score `0.0–1.0`. Treats each ≥0.5 as a medical vote →
empirical `P(medical)`. Confidence = `max(P(med), 1-P(med))`.

**Stage 3 — KL divergence + online k-means**:
- **KL divergence**: `KL(P_empirical || Uniform)` in bits — measures how far
  the model's opinion is from a coin-flip. High (≈1) = confident, low (≈0) = uncertain.
- **Online k-means**: 3-d feature vector `[confidence, kl_div, kw_ratio]`
  clustered into `confident-medical / uncertain / confident-non-medical`.
  Centroids update incrementally. Cold-start from first 9 samples.

### Routing Decision

```
Route to NX  ⇔  is_medical == True  AND  confidence >= 0.7
Route to AGX ⇔  otherwise
```

Uncertain or borderline queries always defer to the stronger AGX model.

### Response Format

Every response includes confidence metrics:
```json
{
  "content": "Paris is the capital of France.",
  "source": "AGX",
  "confidence": 0.95,
  "p_med": 0.0,
  "kl_div": 1.0,
  "kmeans_label": "confident-non-medical"
}
```

---

## API Reference

### POST /chat (JSON + SSE streaming)

```json
// Request
{"message": "What is hypertension?", "stream": false}

// Response (stream=false)
{"content": "...", "source": "NX", "confidence": 0.95, ...}

// SSE (stream=true)
data: {"token": "...", "source": "NX"}
data: {"done": true, "content": "...", "source": "NX", "confidence": 0.95, ...}
```

### POST /classify (confidence-only, no inference)

```json
// Request
{"text": "What are symptoms of diabetes?"}

// Response
{
  "is_medical": true,
  "confidence": 0.95,
  "p_med": 1.0,
  "kl_div": 1.0,
  "method": "keyword",
  "kmeans": {"cluster": 2, "label": "confident-medical", "dist_to_center": 0.12},
  "route": "NX",
  "classify_ms": 0.8
}
```

### GET /health
```json
{"status": "ok", "model": "/path/to/model.gguf"}
```

### WebSocket /ws

```json
// Client → Server
{"type": "ping"}
{"type": "message", "sender": "nx-node", "content": "hello"}

// Server → Client
{"type": "pong"}
{"type": "chunk", "content": "...", "source": "NX"}
{"type": "done", "content": "...", "source": "NX", "confidence": 0.95, ...}
```

---

## Tests & Eval

| Script | Purpose |
|--------|---------|
| `eval_classifier.py` | Run 130 eval queries against /classify, report accuracy + latency |
| `eval_baseline.py` | Compare Always-NX / Always-AGX / HiSLM on same eval set |
| `analysis_routing_overhead.py` | Break-even analysis for routing vs always-AGX |
| `measure_nx_queries.py` | Benchmark NX inference latency |
| `parse_tegrastats.py` | Parse tegrastats power logs |

### Classifier Results (130 queries, Jetson Orin NX)

| Metric | Value |
|--------|-------|
| Accuracy | 72.3% (94/130) |
| Precision (non-medical) | 100% (60/60) |
| Recall (medical) | 48.6% (34/70) |
| Avg classify latency | 3770 ms |
| Keyword catch rate | 26.2% (bypasses LLM classify) |

### NX Inference Performance (Qwen2.5-1.5B Q4_K_M, CPU-only)

| Scenario | Total Time |
|----------|-----------|
| Cold start | 4.5 s |
| Short QA (50 tok) | 4.6 s |
| Medical QA (128 tok) | 12.3 s |
| Long gen (256 tok) | 20.3 s |
| Medical classification | 3.8-11 s |

### Energy (tegrastats)

| State | Total Power | CPU+GPU |
|-------|-----------|---------|
| Idle | 7.38 W | 1.07 W |
| Inference | 12.28 W | 4.51 W |
| **Marginal** | **4.89 W** | **3.44 W** |

Energy per medical query: ~70J CPU+GPU marginal.

---

## Training (QLoRA)

Fine-tunes TinyLlama-1.1B-Chat-v1.0 on medical datasets using QLoRA on NX GPU.

```bash
python preprocess.py           # Prepare dataset (MedQA + PubMed + MT Samples)
python train.py                # Train (default: 500 samples, r=8)
python train.py --lora-r 16 --samples 2000   # Custom config
bash train.sh                  # Launcher with diagnostics
```

### Ablation Results (500 samples)

| Rank | Start Loss | Final Loss | Time | GPU Mem Free |
|------|-----------|-----------|------|-------------|
| r=4 | 3.53 | 1.49 | 5.7 min | 0.85 GB |
| **r=8** | **2.98** | **1.42** | **5.6 min** | **1.08 GB** |
| r=16 | 3.44 | 1.52 | 5.6 min | 1.28 GB |

### A5000 Training (full dataset)

```bash
cd a5000_training
pip install -r requirements.txt
python train_a5000.py           # bf16, full 219k, ~4-6h
python merge_and_convert.py     # PEFT → GGUF
```

---

## Quantization Benchmark (Qwen2.5-1.5B, NX CPU)

| Quant | Size | PP (t/s) | TG (t/s) |
|-------|------|---------|---------|
| Q4_K_M | 1.1 GB | 57.1 | **15.3** |
| Q5_K_M | 1.1 GB | 43.7 | 11.5 |
| Q8_0 | 1.6 GB | 62.3 | 13.1 |

Q4_K_M provides the best latency/quality tradeoff for edge deployment.

---

## Project Structure

```
HiSLM-8G/
├── subserver.py              # Hybrid NX/AGX server (queue + confidence routing)
├── server_qwen.py            # Standalone NX inference server
├── train.py                  # QLoRA fine-tuning (manual loop, TinyLlama-1.1B)
├── preprocess.py             # Dataset preprocessing (3 datasets → 224k pairs)
├── eval_classifier.py        # Classifier evaluation (130 queries)
├── eval_baseline.py          # Three-mode comparison script
├── analysis_routing_overhead.py  # Routing break-even analysis
├── measure_nx_queries.py     # NX inference benchmark harness
├── parse_tegrastats.py       # Power log parser
├── eval_routing.jsonl        # 130 labeled eval queries
├── hislm-nx.service          # Systemd unit for auto-restart
├── client.py                 # LAN client
├── client2.py                # Tailscale wireless client
├── client_2.py               # Generic client
├── orin_index.html           # Standalone chat UI (1573 lines)
├── static/
│   ├── index.html            # AGX server chat UI
│   └── nx_index.html         # NX wireless client UI
├── a5000_training/           # Desktop GPU training pipeline
│   ├── train_a5000.py
│   ├── merge_and_convert.py
│   └── requirements.txt
├── dataset/                  # Medical datasets (gitignored)
├── models/                   # GGUF models (gitignored)
├── output/                   # Training outputs (gitignored)
├── trained/                  # PEFT LoRA adapter (gitignored)
├── context.md                # Full project context
├── NX_FIXES.md               # NX paper-readiness checklist
├── progress.md               # Training progress log
├── BUG_ANALYSIS.md           # Resolved bugs
└── test_NX.md / test_AGX.md  # Hardware test reports
```

## Environment

| Component | NX (Edge) | AGX (Server) |
|-----------|----------|--------------|
| **Device** | Jetson Orin NX 8GB | Jetson AGX Orin 32GB |
| **JetPack** | R36.4.7 (CUDA 12.6) | R36.4.7 (CUDA 12.6) |
| **PyTorch** | 2.5.0a0+nv24.08 | — |
| **llama.cpp** | build b9453 (CPU-only) | build b9571 (CUDA) |
| **bitsandbytes** | 0.50.0.dev0 (sm_87) | — |
| **Model** | Qwen2.5-1.5B Q4_K_M | Qwen2.5-3B Q4_K_M |

## Docs

- [Full project context](context.md)
- [NX fix checklist](NX_FIXES.md)
- [Training progress](progress.md)
- [Bug analysis](BUG_ANALYSIS.md)
- [AGX test report](test_AGX.md)
- [NX test report](test_NX.md)
- [Client troubleshooting](CLIENT_TIMEOUT_TROUBLESHOOTING.md)
- [Client/UI architecture](CLIENT_UI.md)

## Key Findings

1. **Base model outperforms LoRA** for English medical QA when LoRA was
   trained on multilingual data — cross-lingual adapter transfer failure.
2. **Q4_K_M is optimal** for NX edge deployment — Q5_K_M is 25% slower at
   same size; Q8_0 has worse generation throughput.
3. **Conservative routing wins** — uncertain queries defer to AGX, preventing
   false positives from the lightweight classifier.
4. **CPU-only inference is viable** on NX for 1.5B models — 15-18 tok/s
   generation, sufficient for medical QA latency requirements.
