# HiSLM-8G — Comprehensive Project Context

**Hierarchical Small Language Model inference system for edge devices — Medical QA domain.**
Fine-tune and serve 1-3B parameter LLMs on **Jetson Orin NX 8GB**, with a companion **AGX Orin server** and a desktop **A5000 training pipeline**.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  AGX Orin (Server)                    │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │ Qwen 2.5-3B │  │  Flask/WS    │  │  ngrok      │   │
│  │ GGUF Model  │  │  Server      │  │  Tunnel     │   │
│  └────────────┘  └──────────────┘  └─────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       │ LAN / Tailscale
    ┌──────────────────┴──────────────────┐
    │          Orin NX (Server)            │
    │  ┌──────────────────────────────┐    │
    │  │  subserver.py                │    │
    │  │  ┌──────────┐  ┌─────────┐   │    │
    │  │  │Classifier│→ │ Router  │   │    │
    │  │  └──────────┘  └────┬────┘   │    │
    │  │               │     │        │    │
    │  │          ┌────┘     └──┐     │    │
    │  │          ▼             ▼     │    │
    │  │   ┌────────────┐ ┌────────┐  │    │
    │  │   │ Qwen 1.5B  │ │ AGX    │  │    │
    │  │   │ (local)    │ │ (relay)│  │    │
    │  │   └────────────┘ └────────┘  │    │
    │  └──────────────────────────────┘    │
    │                                      │
    │  client.py / client2.py              │
    │  (thin clients, connect to AGX)      │
    └──────────────────────────────────────┘
```

### Hardware Tiers

| Tier | Device | Memory | Role |
|------|--------|--------|------|
| **Desktop** | RTX A5000 | 24 GB | Full training pipeline (bf16, full 219k dataset) |
| **Server** | Jetson AGX Orin | 32 GB+ | Inference server (Qwen2.5-3B) |
| **Edge** | Jetson Orin NX | 8 GB | Hybrid router + local inference (Qwen2.5-1.5B) + training |

---

## 2. Directory Structure

```
HiSLM-8G/
├── train.py                        # QLoRA fine-tuning (manual loop, TinyLlama-1.1B)
├── preprocess.py                   # Dataset preprocessing (3 datasets → 224k pairs)
├── server_qwen.py                  # Flask + WebSocket inference server (llama-cli)
├── subserver.py                    # Hybrid NX/AGX server (classify + route)
├── client.py                       # LAN client (AGX over LAN)
├── client_2.py                     # Generic client (any server, env-configured)
├── client2.py                      # Tailscale wireless client
├── test_step.py                    # Training diagnostic script
├── run_qwen_web.sh                 # Server deployment launcher + ngrok
├── train.sh                        # Training launcher with diagnostics
├── test_NX.md                      # NX comprehensive test report (2026-06-17)
├── test_AGX.md                     # AGX comprehensive test report (2026-06-19)
├── Qwen2.5-3B-benchmark(1).md      # Qwen2.5-3B high-fidelity benchmark
├── BUG_ANALYSIS.md                 # 3 resolved bugs (LoRA, echo truncation, chunking)
├── progress.md                     # Training progress log
├── CLIENT_UI.md                    # Client/UI architecture reference
├── CLIENT_TIMEOUT_TROUBLESHOOTING.md  # Network timeout diagnostics
├── context.md                      # This file — full project context
├── .gitignore                      # Ignores dataset/, models/, output/, trained/, venv/
├── static/
│   ├── index.html                  # AGX server chat UI (sci-fi terminal, 1006 lines)
│   └── nx_index.html               # NX wireless client UI
├── orin_index.html                 # Standalone Qwen chat UI (1573 lines, sessions, themes)
├── a5000_training/                 # Desktop GPU training pipeline
│   ├── train_a5000.py             # bf16 LoRA training (HuggingFace Trainer)
│   ├── merge_and_convert.py       # PEFT → GGUF conversion
│   ├── requirements.txt           # Dependencies
│   └── README.md                  # A5000 training docs
├── dataset/                        # Medical datasets (gitignored)
├── models/                         # GGUF models (gitignored)
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf  # 985 MB (Qwen2.5-1.5B)
│   └── medical-lora-qwen2.5-1.5b.gguf     # ~1 MB (LoRA, deprecated)
├── output/                         # Training outputs (gitignored)
├── trained/                        # PEFT LoRA adapter (gitignored)
│   ├── adapter_config.json
│   └── adapter_model.safetensors   # 71 MB
└── venv/                           # Python virtual environment (gitignored)
```

---

## 3. Components — Detailed Reference

### 3.1 Inference Server (`server_qwen.py`, 221 lines)

**Purpose:** Flask + WebSocket server that spawns `llama-cli` as a subprocess to run Qwen2.5-1.5B-Instruct (GGUF, Q4_K_M).

**Key functions:**
- `build_prompt()` — wraps user message in Qwen2.5 chat template (`<|im_start|>user/assistant`)
- `_extract_response()` — strips banner, prompt echo, and stats from llama-cli stdout using last-header heuristic
- `stream_tokens()` — spawns `llama-cli`, calls `proc.communicate()`, buffers all output, yields characters

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check → `{"status":"ok","model":"..."}` |
| GET | `/` | Web chat UI (`orin_index.html`) |
| POST | `/chat` | Chat completion (JSON, SSE streaming) |
| WS | `/ws` | WebSocket chat (chunks + done) |

**llama-cli invocation:**
```
llama-cli -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "<prompt>" -n 512 --no-display-prompt --single-turn --simple-io -c 4096
```

**Performance (Orin NX, CPU-only):**
- Prompt processing: ~63 t/s
- Text generation: ~18 t/s
- Cold start (model load): ~2.5 s
- Short QA (50 tok): ~4.6 s
- Medical QA (128 tok): ~12.3 s
- Long gen (256 tok): ~20.3 s

**WebSocket protocol:**
- Client → Server: `{"type":"ping"}`, `{"type":"message","sender":"...","text":"..."}`
- Server → Client: `{"type":"chunk","content":"..."}`, `{"type":"done","content":"..."}`

**Known issues (resolved):**
- LoRA adapter removed — multilingual Chinese-trained adapter degraded English medical QA. **Finding:** Base instruction-tuned model outperforms cross-lingual LoRA transfer for English medical queries. See `BUG_ANALYSIS.md`.
- Echo truncation fix → buffered `_extract_response()` instead of line-by-line state machine
- Per-char chunking fix → batched 80-char chunks for SSE/WS

### 3.2 Hybrid Subserver (`subserver.py`, 844 lines)

**Purpose:** Runs on Orin NX. Classifies each query with probabilitic confidence
(logprob-style via multi-sample scoring + KL divergence + online k-means) and
routes accordingly.

**Architecture — Serialised Model Queue:**
All model operations (classify + infer) go through a single `queue.Queue` with
a daemon worker thread. This prevents resource contention from concurrent
llama-cli processes on the memory-constrained NX.

```
┌─────────────┐    ┌──────────────────────┐    ┌──────────────┐
│  HTTP/WS     │───▶│  queue.Queue          │───▶│  Worker      │
│  Handlers    │    │  (serialised access)   │    │  Thread      │
│              │    │  classify / infer ops   │    │  (1 at a time)│
└─────────────┘    └──────────────────────┘    └──────┬───────┘
                                                       │
                                               ┌───────▼───────┐
                                               │  llama-cli     │
                                               │  (Qwen1.5B)    │
                                               └───────────────┘
```

Each queued item carries a `threading.Event` for completion signalling.
Callers wait on the event with a timeout — the worker processes exactly
one operation at a time, in FIFO order.

**Confidence-based Routing — 3-Stage Classifier:**

Stage 1 — **Keyword pre-filter** (0ms):
- Checks query against `MEDICAL_KEYWORDS` set (70+ terms: symptoms,
  diagnosis, treatment, drugs, greetings, etc.)
- Match → immediate return `is_medical=True, confidence=0.95`
- Bypasses LLM entirely for obvious medical queries and greetings

Stage 2 — **Multi-sample LLM scoring** (~11s, 2× llama-cli calls):
- Runs the classifier prompt at temps 0.0 (deterministic) and 0.5 (stochastic)
- Each call outputs a score 0.0–1.0 via prompt:
  ```
  Rate the medical relevance from 0.0 to 1.0.
  0.0 = definitely NOT medical
  0.5 = uncertain
  1.0 = definitely medical
  Output only the number.
  ```
- Treats each score's ≥0.5 as a binary vote → empirical `P(medical)`
- Confidence = `max(P(medical), 1 - P(medical))` — how decisive the vote is

Stage 3 — **KL divergence + online k-means**:
- **KL divergence**: `KL(P_empirical || Uniform)` in bits.
  - P = [p_med, 1-p_med], Q = [0.5, 0.5]
  - KL = p·log₂(p/0.5) + (1-p)·log₂((1-p)/0.5)
  - High KL (≈1.0) → confident, Low KL (≈0.0) → uncertain
- **Online k-means**: 3-d feature vector `[confidence, kl_div, kw_ratio]`
  clustered into 3 groups via streaming k-means++:
  - Cluster 0: `confident-non-medical` (low confidence centroid)
  - Cluster 1: `uncertain` (mid confidence centroid)
  - Cluster 2: `confident-medical` (high confidence centroid)
  - Centroid labels auto-assigned by sorting the confidence dimension
  - Cold-start: first 9 samples initialise centers, then online update
  - 200-sample sliding window history for potential refit

**Routing Decision:**
```
Route to NX  ⇔  is_medical == True  AND  confidence >= CONFIDENCE_THRESHOLD (0.7)
Route to AGX ⇔  otherwise
```
This means uncertain or borderline queries always defer to the stronger AGX model.

**AGX relay** (REST polling, not WebSocket):
1. `POST /send` — submits query to AGX
2. `GET /messages?limit=20` — polls every 1s for server response
3. 120s timeout on polling

**Response format:**
- Every response includes `"source": "NX"` or `"source": "AGX"`
- SSE: `{"token":"...","source":"NX"}`, `{"done":true,"source":"AGX","content":"...","confidence":0.95,...}`
- WS: `{"type":"chunk","content":"...","source":"AGX"}`, `{"type":"done","content":"...","source":"NX"}`
- All responses now include: `confidence`, `p_med`, `kl_div`, `kmeans_label`

**Logging:** Every query, response, classification, routing decision, and
confidence metrics are POSTed to AGX at `/log` for centralised monitoring.

**CLI flags:**
- `--agx-ip` (required): AGX server IP
- `--agx-port` (default 8000): AGX port
- `--port` (default 8765): this server's port
- `--confidence` (default 0.7): min confidence threshold for NX routing
- `--workers` (default 1, keep at 1 for NX): worker threads

**Usage:**
```bash
python subserver.py --agx-ip 100.120.59.117                              # default
python subserver.py --agx-ip 100.120.59.117 --confidence 0.8            # stricter
python subserver.py --agx-ip 172.16.6.21 --port 9000                     # LAN
```

### 3.3 Training (`train.py`, 204 lines)

**Purpose:** Fine-tunes TinyLlama-1.1B-Chat-v1.0 on medical datasets using QLoRA.

**Method:** QLoRA (4-bit nf4, double_quant), rank=8, alpha=16, targets=`q_proj,v_proj`

**Hardware:** Jetson Orin NX 8GB (JetPack 6.2, CUDA 12.6, unified memory)

**Hyperparameters:**
| Param | Value |
|-------|-------|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Quant | 4-bit nf4, double_quant, fp16 compute |
| LoRA r/alpha | 8/16 |
| Target modules | q_proj, v_proj |
| Seq length | 128 |
| Batch size | 1 |
| Grad accum | 4 |
| Effective batch | 4 |
| Optimizer | AdamW (lr=2e-4) |
| Schedule | warmup 50 → cosine |
| Samples | 500-2000 (stable) |
| Time/sample | ~2.5s |
| Free memory | 0.9-1.0 GB |

**Training loop features:**
- Manual PyTorch loop (replaced HuggingFace Trainer — OOM at step 7)
- Pure CPU tokenization, GPU model with manual batch transfer
- Signal trapping (SIGTERM, SIGINT, SIGHUP, SIGUSR1/2, SIGABRT, SIGQUIT)
- `faulthandler` enabled, `CUDA_LAUNCH_BLOCKING=1` in launcher
- Per-step logging (loss, lr, memory, elapsed time)
- Checkpoint every 200 steps

**Instability resolved:** The ~100s silent death was fixed by `CUDA_LAUNCH_BLOCKING=1`, reduced samples (500-2000), and manual training loop (no Trainer internals).

**Usage:**
```bash
python preprocess.py               # Prepare dataset
python train.py                     # Direct training
bash train.sh                       # Launcher with diagnostics
```

### 3.4 Dataset Preprocessing (`preprocess.py`, 301 lines)

**Input datasets:**
1. **Med_QA** — US, Mainland (Chinese), Taiwan, Taiwan English translations, Textbooks, 4-options variant
2. **MT Samples** — Medical transcription CSV (specialty, description, sample, transcription)
3. **Pub_Med_QA** — PubMed QA parquet file

**Output:** `dataset/training_data.jsonl` → unified `{instruction, input, output, source}` format

**Pipeline:**
- Shuffle, deduplicate by (instruction, output)
- 95/5 train/val split
- **Total:** ~224,838 medical QA + ~4,966 MT samples + ~1,000 PubMed QA

### 3.5 A5000 Training Package (`a5000_training/`)

**Differences from NX training:**

| Setting | NX (8GB) | A5000 (24GB) |
|---------|----------|--------------|
| Precision | fp16 + 4-bit nf4 | bf16 (native) |
| LoRA rank | 8 | 16 |
| LoRA targets | q_proj, v_proj | q_proj, v_proj, k_proj, o_proj |
| Seq length | 128 | 512 |
| Batch size | 1 | 8 |
| Effective batch | 4 | 32 |
| Training time (219k) | ~150h | ~4-6h |
| Dataset | 2000 samples | Full 219k |

**`merge_and_convert.py`** — Merges PEFT LoRA adapter → HF model → GGUF (via llama.cpp `convert.py` + `quantize`)

### 3.6 Clients

| Client | File | Use Case | Libraries |
|--------|------|----------|-----------|
| LAN Client | `client.py` (394 lines) | Connect to AGX over LAN | `requests`, `websocket-client` |
| Tailscale Client | `client2.py` (301 lines) | Connect over Tailscale wireless | `httpx`, `websockets` |
| Generic Client | `client_2.py` (394 lines) | Any HiSLM server, env-configured | `requests`, `websocket-client` |

**Two modes per client:**
1. **GUI mode** (default) — opens browser pointing at AGX server URL with `?client_id=...`
2. **CLI mode** (`--cli`) — terminal REPL via WebSocket, with REST fallback

**Client features:**
- Health check at startup (exit with hints if unreachable)
- WebSocket with ping/pong keepalive (15s interval)
- REST fallback if WS unavailable (poll `/messages`, POST `/send`)
- `_rest_cli_loop()` for terminal mode when WS fails

### 3.7 Web UIs

| File | Lines | Purpose |
|------|-------|---------|
| `static/index.html` | 1006 | AGX server chat UI (sci-fi terminal, cyan/green theme) |
| `static/nx_index.html` | — | NX wireless client UI |
| `orin_index.html` | 1573 | Standalone Qwen chat UI (sessions, themes, streaming) |

**`static/index.html` features:**
- Sci-fi terminal look (`#080c10` bg, `--accent #00c8ff` NX cyan, `--accent2 #00ff9d` AGX green)
- Scanline overlay, corner brackets, glowing borders, blinking status dots
- CSS grid layout: `#app { grid-template-rows: 56px 1fr auto }`
- Message bubbles: `.from-agx` (left, green), `.from-nx` (right, cyan), `.from-system` (centred)
- Auto-connect via `?client_id=...` URL param
- WebSocket with 3s reconnect backoff + `POST /send` REST fallback
- Deduplication via `seenIds` Set
- HTML-escaping via `escHtml()`

**`orin_index.html` features:**
- Dark/light theme (persisted in localStorage)
- Session management (create, switch, delete sessions from sidebar)
- Multi-turn context (last 10 messages sent with each request)
- Live streaming (both WebSocket and SSE modes)
- Three modes: demo, WebSocket, and REST (SSE)
- Auto-migration of old system prompts

### 3.8 Diagnostic Script (`test_step.py`, 94 lines)

Step-by-step training test:
1. Load tokenizer (TinyLlama-1.1B, 32K vocab)
2. Create dummy data (16 samples)
3. Load model in 4-bit (blocks on bitsandbytes version mismatch)
4. prepare_model_for_kbit_training
5. Add LoRA (r=8, q_proj+v_proj)
6. Single forward/backward pass (8 steps)

**Known blocker:** requires `bitsandbytes>=0.50.0` for 4-bit QLoRA with `transformers>=4.41`.

---

## 4. Resolved Bugs (from BUG_ANALYSIS.md)

### Bug 1: LoRA Degrades English Medical QA
- Chinese-trained LoRA adapter overrode base model's English instruction following
- **Fix:** Removed `--lora` flag from llama-cli invocation

### Bug 2: Echo Truncation Causes 0-Char Responses
- `--single-turn` truncates prompt echo in llama-cli output
- Old line-by-line state machine never found expected number of headers
- **Fix:** Buffered `_extract_response()` using last-header + blank-line heuristic

### Bug 3: Per-Character Chunking Overwhelms WS/SSE
- 1000+ frames per response (one per character)
- **Fix:** Batch into 80-char chunks

---

## 5. Environment

### NX (Jetson Orin NX 8GB)
| Component | Value |
|-----------|-------|
| **Device** | NVIDIA Jetson Orin NX 8GB (aarch64) |
| **Board** | AVerMedia D115W |
| **JetPack** | R36.4.7 (JP 6.2), CUDA 12.6, Driver 540.4.0 |
| **PyTorch** | 2.5.0a0+nv24.08 (Jetson-specific) |
| **bitsandbytes** | 0.50.0.dev0 (built from source for sm_87) |
| **Python** | 3.10+ |
| **llama.cpp** | build b9453 (CPU-only, aarch64, no GPU layers) |
| **Kernel** | Linux aarch64 |

#### System Specs (NX)
| Component | Detail |
|-----------|--------|
| SoC | NVIDIA Orin NX 8GB |
| CPU | 6× Cortex-A78AE (2 clusters × 3 cores), 115–1984 MHz |
| L1/L2/L3 Cache | 384 KiB / 1.5 MiB / 4 MiB |
| RAM | 7.4 GiB (7803 MB) |
| Swap | 3.7 GiB (6× zram, ~635 MB each) |
| Storage | 456 GB NVMe (64 GB used, 370 GB free) |
| Power Mode | MAXN |
| Idle Power/Temp | ~6.8 W, CPU ~49°C, GPU ~47°C |

### AGX (Jetson AGX Orin 32GB)
| Component | Value |
|-----------|-------|
| **Device** | NVIDIA Jetson AGX Orin 32GB (aarch64) |
| **Board** | AVerMedia D315 |
| **JetPack** | R36.4.7 (CUDA 12.6, Driver 540.4.0) |
| **llama.cpp** | build b9571-e3471b3e7 (CUDA, aarch64, GNU 11.4.0) |
| **GPU Compute** | 8.7 (all 36 layers offloaded, `-ngl 99`) |
| **Kernel** | Linux 5.15.148-tegra |

#### System Specs (AGX)
| Component | Detail |
|-----------|--------|
| SoC | NVIDIA Jetson AGX Orin 32GB |
| CPU | 8× Cortex-A78AE (2 clusters × 4 cores), 115–2188 MHz |
| L1/L2/L3 Cache | 1 MiB / 2 MiB / 4 MiB |
| RAM | 29 GiB (30697 MB) |
| GPU VRAM | ~30 GiB (unified memory) |
| Swap | 14 GiB (8× zram, ~1.9 GB each) |
| Storage (root) | 59 GB eMMC (51 GB used, 3.4 GB free) |
| Storage (NVMe) | 932 GB NVMe (WD Blue SN5100, at /mnt/nvme) |
| Power Mode | MAXN |
| Idle Power | ~5.0 W (VIN_SYS_5V0) |
| Idle Temp | CPU ~45°C, GPU ~41°C, TJ ~45°C |

### Disk Usage (NX)
| Component | Size |
|-----------|------|
| `venv/` | 5.4 GB |
| `models/` | 3.1 GB |
| `dataset/` | 746 MB |
| `trained/` | 71 MB |
| `output/` | 6.6 MB |
| **Total** | ~9.4 GB |

---

## 6. Network Topology

| Path | Protocol | Status |
|------|----------|--------|
| NX ↔ AGX (LAN) | 172.16.6.x | ✅ Reachable |
| NX ↔ AGX (Tailscale) | 100.120.59.117:8000 | ✅ Reachable |
| AGX external | ngrok tunnel | ⚠️ Optional |
| DNS/Internet (NX) | via Tailscale | ⚠️ DNS unreachable |

AGX endpoints:
- `GET /health` → `{"status":"ok","node":"AGX-Orin-30GB","llama_ready":true,"llama_model":"Qwen2.5-3B-Q4_K_M.gguf"}`
- `POST /send` → submit query with `{sender, text}`
- `GET /messages?limit=N` → poll for responses
- No WebSocket on AGX (REST-only for subserver relay)

---

## 7. Commands Reference

### Inference
```bash
# Start NX server (local Qwen2.5-1.5B)
python server_qwen.py                          # port 8765
python server_qwen.py --port 8080              # custom port
python server_qwen.py --ngrok                  # expose via ngrok

# Start subserver (NX + AGX routing)
python subserver.py --agx-ip 172.16.6.21       # LAN AGX
python subserver.py --agx-ip 100.120.59.117    # Tailscale AGX

# Deploy with ngrok
bash run_qwen_web.sh
```

### Clients
```bash
# LAN client (GUI)
python client.py --agx-ip 172.16.6.21
# LAN client (CLI)
python client.py --agx-ip 172.16.6.21 --cli --name nx-node-1

# Tailscale client (GUI)
python client2.py --agx-ip 100.120.59.117
# Tailscale client (CLI)
python client2.py --agx-ip 100.120.59.117 --cli --node-name nx-node

# Generic client
python client_2.py --server-ip 192.168.1.10 --port 8001 --cli
```

### Training
```bash
# Prepare dataset
python preprocess.py

# Train on NX
python train.py
bash train.sh                                 # with diagnostics

# Train on A5000
cd a5000_training
python train_a5000.py

# Merge LoRA → GGUF (A5000)
python merge_and_convert.py --lora ./output/lora_adapter_final

# Diagnostic
python test_step.py
```

---

## 8. Inference Pipeline (Detailed)

```
User Query
  │
  ▼
build_prompt(user_msg, system, messages)
  → wraps in Qwen2.5 chat template:
    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{msg}<|im_end|>
    <|im_start|>assistant\n
  │
  ▼
stream_tokens(prompt)
  → subprocess.Popen(llama-cli, ...)
  → proc.communicate() — read ALL stdout
  → _extract_response(raw):
      1. Split by \n
      2. Find LAST line starting with <|im_start|>assistant
      3. Find first blank line after it (separator)
      4. Collect lines until [ Prompt: or Exiting
      5. Join + strip
  → yield character by character
  │
  ▼
SSE or WebSocket handler
  → "".join(stream_tokens) → batch 80-char chunks
  → yield/ws.send each chunk
  → final done frame with full content
```

---

## 9. Training Pipeline (Detailed)

```
preprocess.py
  → Med_QA (US/Mainland/Taiwan/Textbooks/4-options)
  → MT Samples (CSV transcriptions)
  → Pub_Med_QA (parquet)
  → deduplicate, shuffle, 95/5 split
  → dataset/train.jsonl (213k) + val.jsonl (11k)
  │
  ▼
train.py (QLoRA on Orin NX)
  Phase 1: CPU tokenization (tokenize all texts)
  Phase 2: GPU model (4-bit nf4, double_quant)
  Phase 3: LoRA (rank=8, alpha=16, q_proj+v_proj)
  Phase 4: Training loop
    - manual batch: ids_cpu[idx].cuda()
    - forward → loss.backward()
    - grad_accum=4 → optimizer.step()
    - warmup 50 steps → cosine decay
    - checkpoint every 200 steps
    - signal trapping for diagnostics
  → output/lora_adapter/ (PEFT safetensors)
  │
  ▼
merge_and_convert.py (on A5000)
  → merge LoRA into base model (bf16)
  → convert to GGUF FP16 via llama.cpp convert.py
  → quantize to Q4_K_M via llama.cpp quantize
  → models/tinyllama-medical-q4_k_m.gguf
```

---

## 10. AGX Orin (Server) Details

- **Node:** AGX-Orin-30GB (`tegra-ubuntu-1` via Tailscale)
- **Model:** Qwen2.5-3B-Instruct Q4_K_M (1.79 GiB GGUF, 3.09B params)
- **Engine:** llama-cli build b9571 (CUDA, aarch64, GNU 11.4.0)
- **CPU Threads:** 8 | **Context:** 4096 | **Batch:** 2048 | **UBatch:** 512
- **GPU Layers:** 99 (all 36 offloaded) | **KV Cache:** f16 | **Sampling:** greedy (temp=0)
- **Connectivity:** Reachable via Tailscale (`100.120.59.117:8000`)

### Endpoints
| Method | Path | Status | Purpose |
|--------|------|--------|---------|
| GET | `/health` | ✅ 200 (7ms) | Health + llama status |
| GET | `/` | ✅ 200 (33 KB) | Web UI (index.html) |
| GET | `/nx` | ✅ 200 | NX client web UI |
| POST | `/send` | ✅ 200 (5ms) | Submit query `{sender, text}` — queues inference |
| GET | `/messages` | ✅ 200 (3ms) | Poll message history |
| GET | `/ws` | ❌ 404 | No WebSocket on AGX (REST-only for subserver relay) |

### Health Payload
```json
{
  "status": "ok",
  "node": "AGX-Orin-30GB",
  "tailscale_ip": "100.120.59.117",
  "connected_clients": 0,
  "message_count": 0,
  "llama_ready": true,
  "llama_model": "/home/nvidia/HiSLM/llama.cpp/models/Qwen2.5-3B-Q4_K_M.gguf",
  "llama_runner": "/home/nvidia/HiSLM/llama.cpp/build/bin/llama-cli",
  "llama_errors": [],
  "timestamp": "2026-06-19T07:29:41.648804+00:00"
}
```

### GPU vs CPU Comparison (llama-bench)

| Metric | CPU Only | GPU (-ngl 99) | Speedup |
|--------|----------|---------------|---------|
| Prompt Processing (512 tok) | 514 tok/s | 494 tok/s | ~1x (bandwidth-bound) |
| Text Generation (128 tok) | 5.12 tok/s | 12.39 tok/s | **2.4x** |

For a 3B model on Orin, prompt processing is memory-bandwidth bound and runs similarly on CPU. Text generation benefits significantly from GPU offload (2.4x). Larger models (7B+) show even greater GPU advantage.

### High-Fidelity Benchmark (llama-bench, optimal conditions)
From `Qwen2.5-3B-benchmark(1).md` — run under ideal thermal conditions (max 42°C):

| Metric | CPU Only | GPU (-ngl 99) | Speedup |
|--------|----------|---------------|---------|
| Prompt Processing (512 tok) | 422 tok/s | **948 tok/s** | **2.2x** |
| Text Generation (128 tok) | 4.17 tok/s | **24.90 tok/s** | **6.0x** |

**Note:** The difference between the two benchmark sets is attributed to thermal throttling and power mode variations. The high-fidelity numbers represent peak capability under optimal cooling.

### Server Reliability (test_AGX.md)
| Metric | Value |
|--------|-------|
| Test duration | ~40 min |
| Server restarts needed | 0 |
| Crashes during testing | 0 |
| Concurrent request handling | ✅ 3 simultaneous OK |
| Inference timeout handling | ✅ 300s default config |

### AGX Observed Issues
| Issue | Details | Severity |
|-------|---------|----------|
| llama-cli banner output | Non-suppressible ASCII art / build info even with `--log-disable` | ✅ Low |
| Model reload on each request | POST `/send` spawns new llama-cli subprocess (no persistent model server, adds ~2.5s overhead) | ⚠️ Medium |
| Root disk space | Only 3.4 GB free on eMMC root partition | ⚠️ Medium (move models to /mnt/nvme) |

### Memory Under Load (AGX)
| State | Used | Available | Swap Used |
|-------|------|-----------|-----------|
| **Idle** | 4.3 GiB | 25 GiB | 0 GiB |
| **During inference (model loaded)** | 8.5-9.0 GiB | 20-21 GiB | 0 GiB |
| **Post-inference** | 8.5 GiB | 20 GiB | 0 GiB |

~4.2 GiB baseline includes desktop + background services. Model adds ~100 MiB to system RAM (most resides in GPU unified memory).

### Power & Thermal (AGX — tegrastats)

| State | CPU | GPU | SoC | TJ | VIN_SYS_5V0 | VDD_GPU_SOC | VDD_CPU_CV |
|-------|-----|-----|-----|----|-------------|-------------|------------|
| **Idle** | 45.1°C | 40.9°C | 42.2°C | 45.1°C | 5.2 W | 3.7 W | 1.4 W |
| **Inference (peak)** | 46.1°C | 40.8°C | 42.3°C | 46.1°C | 5.4 W | 3.9 W | 2.8 W |
| **Post-inference** | 45.3°C | 41.4°C | — | 45.3°C | 5.0 W | 3.7 W | 0.9 W |

Temperatures stayed well within safe limits. Max observed: CPU 46.1°C, GPU 41.4°C. GR3D_FREQ at 0-37% during load.

---

## 11. Performance Benchmarks

### NX Orin — Qwen2.5-1.5B Q4_K_M (CPU-only, llama-cli b9453)

**Model:** 1.12 GB GGUF | **Context:** 4096 | **Sampling:** greedy (temp=0)

| Scenario | Prompt t/s | Gen t/s | Total Time |
|----------|-----------|---------|------------|
| Cold start (first load) | 59.3 | 18.5 | 4.5 s |
| Short QA (50 tok) | 64.3 | 20.4 | 4.6 s |
| Medical QA (128 tok) | 63.7 | 16.7 | 12.3 s |
| Long gen (256 tok) | 62.6 | 16.2 | 20.3 s |
| Medical classification | 64.5 | 18.2 | 5.1 s |
| **Average** | **62.9 t/s** | **18.0 t/s** | — |

**Classification accuracy:** 2/2 direct tests passed (medical vs non-medical).

### AGX Orin — Qwen2.5-3B Q4_K_M (GPU offloaded, llama-cli b9571)

**Model:** 1.79 GiB GGUF | **3.09B params** | **36 layers offloaded** | **Context:** 4096

| Scenario | Total Latency | Notes |
|----------|--------------|-------|
| Cold start (first query) | ~3.4 s | Model load + prompt + 30 tok gen |
| Short QA (50 tok) | 5.10 s avg | ~75 chars response |
| Medical QA (128 tok) | 7.93 s avg | ~580 chars response |
| Long gen (256 tok) | 12.88 s avg | ~1300 chars response |

**GPU vs CPU comparison (llama-bench):**

| Metric | CPU Only | GPU (-ngl 99) | Speedup |
|--------|----------|---------------|---------|
| Prompt Processing (512 tok) | 514 tok/s | 494 tok/s | ~1x |
| Text Generation (128 tok) | 5.12 tok/s | 12.39 tok/s | **2.4x** |

Under optimal thermal conditions (max 42°C), peak benchmarks reach **948 tok/s** prompt and **24.90 tok/s** generation with GPU offload (6.0x vs CPU).

### NX vs AGX Summary

| Scenario | NX (1.5B, CPU) | AGX (3B, GPU) | AGX Advantage |
|----------|----------------|----------------|---------------|
| Cold start | 4.5 s | 3.4 s | 1.3x faster |
| Short QA (50 tok) | 4.6 s | 5.1 s | 0.9x (3B model is larger) |
| Medical QA (128 tok) | 12.3 s | 7.9 s | **1.6x faster** |
| Long gen (256 tok) | 20.3 s | 12.9 s | **1.6x faster** |
| Model size | 1.12 GB / 1.5B | 1.79 GB / 3.09B | 2x parameters |
| Power (idle) | ~6.8 W | ~5.2 W | AGX more efficient |

### Server Endpoint Tests (NX — server_qwen.py)

| Scenario | Streaming | Latency | Response |
|----------|-----------|---------|----------|
| Health check | — | <10 ms | `{"status":"ok"}` |
| "What is the capital of France?" | No | 4.96 s | 35 chars |
| "What are symptoms of type 2 diabetes?" | No | 20.8 s | 537 chars |
| "Explain ML in 2 sentences" | Yes (SSE) | 6.6 s | 4 chunks |
| Multi-turn (remembered name) | No | ~5 s | Correct |
| **3 concurrent requests** | No | All ~7.5 s | All correct |

### Server Endpoint Tests (AGX — server2.py)

| Scenario | Latency | Response |
|----------|---------|----------|
| Health check | 7 ms | Full JSON with llama status |
| POST /send (queues inference) | 5 ms | `{"ok": true, ...}` |
| Qwen2.5-3B "capital of France" | ~11 s | "Paris 🐛" |
| Qwen2.5-3B "2+2" | ~5 s | "4" |
| GET /messages (poll) | 3 ms | Full message history |
| **3 concurrent POST /send** | 5-10 ms each | All 200 OK, all processed |

### WebSocket Tests (/ws)

| Feature | NX (server_qwen.py) | AGX (server2.py) |
|---------|---------------------|-------------------|
| Connect with client_id | ✅ | ✅ |
| Ping/Pong | ✅ <50ms | ✅ 20s interval |
| Message send | ✅ ~5s | ✅ |
| Chunk streaming | ✅ 80-char | ✅ |
| Done response | ✅ | ✅ |
| Multi-turn context | ✅ Correct | ✅ |
| History on connect | — | ✅ Last 200 messages |
| Qwen auto-reply | — | ✅ |

---

## 12. Wire Protocols

### WebSocket (server_qwen.py / subserver.py)

**Client → Server:**
```json
{"type": "ping"}
{"type": "message", "sender": "nx-node", "text": "hello"}
```

**Server → Client:**
```json
{"type": "connected", "client_id": "nx-node", "node": "agx"}
{"type": "history", "payload": [{/*message*/}, ...]}
{"type": "message", "payload": {"id":"...", "sender":"...", "role":"server|user|system", "text":"...", "timestamp":"ISO8601"}}
{"type": "ack", "id": "..."}
{"type": "pong"}
{"type": "error", "detail": "..."}
```

### SSE (server_qwen.py / subserver.py `/chat` endpoint)
```json
data: {"token": "...", "source": "NX"}
data: {"done": true, "source": "NX", "content": "..."}
```

### REST Surface
| Verb | Path | Body / Query |
|------|------|-------------|
| GET | `/health` | — |
| GET | `/` | serves `orin_index.html` |
| POST | `/chat` | `{"message":"...","system":"...","stream":true}` |
| GET | `/messages` | `?limit=N` → `{"messages":[...]}` |
| POST | `/send` | `{"sender":"...","text":"..."}` |
| POST | `/log` | `{"sender":"...","type":"subserver_log","payload":{...}}` |

---

## 13. .gitignore Rules

Ignores: `__pycache__/`, `*.pyc`, `venv/`, `*.log`, `dataset/`, `models/`, `output/`, `trained/`, `cache/`, `.vscode/`, `.idea/`, `*.swp`, `*.whl`

---

## 14. Verified Features (from test reports)

### NX Orin (server_qwen.py) — test_NX.md
- [x] Local inference (llama-cli subprocess, CPU-only)
- [x] REST API (`/chat`, `/health`)
- [x] SSE streaming (`/chat?stream=true`)
- [x] WebSocket (`/ws` with ping/pong, chunks, done)
- [x] Concurrent request handling (3 simultaneous)
- [x] Multi-turn conversation context
- [x] Web UI (orin_index.html — dark/light theme, session management)
- [x] Medical/non-medical query classification
- [x] AGX routing (REST send/poll protocol)
- [x] Tailscale connectivity (NX ↔ AGX)
- [x] AGX auto-inference (3B model)
- [x] QLoRA training pipeline (base deps OK, Step 3 blocked by bnb version)

### AGX Orin (server2.py) — test_AGX.md
- [x] Local inference (llama-cli subprocess, CUDA + CPU)
- [x] Qwen2.5-3B Q4_K_M model (1.79 GiB, 36 layers offloaded)
- [x] REST API (`/health`, `/send`, `/messages`)
- [x] Web UI (`GET /` → index.html, `GET /nx` → nx_index.html)
- [x] WebSocket (`/ws` with ping/pong, history on connect)
- [x] Concurrent request handling (3 simultaneous)
- [x] Qwen auto-inference on POST /send and WS message
- [x] Tailscale IP auto-detection
- [x] GPU acceleration (2.4x generation speedup vs CPU)
- [x] Efficient thermal performance (max 46°C under load)

---

## 15. Key Design Decisions

1. **CPU-only inference** — llama-cli runs without GPU layers on NX (avoids CUDA OOM)
2. **Q4_K_M quantization** — reduces Qwen2.5-1.5B from ~3.1 GB to 985 MB
3. **REST polling for AGX relay** — more reliable than WebSocket over Tailscale/wireless
4. **Manual training loop** — replaces HuggingFace Trainer (crashed at step 7 due to memory fragmentation)
5. **Signal trapping** — all critical signals caught with stack trace dumps for debugging silent deaths
6. **Base model over LoRA** — Qwen2.5-1.5B-Instruct outperforms Chinese-medical LoRA for English medical QA
7. **Buffered response extraction** — reads all llama-cli output then parses, instead of fragile line-by-line state machine
8. **80-char chunking** — prevents overwhelming WebSocket/SSE with per-character frames
9. **Serialised model queue** — single worker thread for all llama-cli ops prevents resource contention on NX; classify + infer requests are FIFO-ordered with timeout-based signalling
10. **Probabilitic confidence via multi-sampling** — 2 LLM calls at different temperatures approximate logprobs; empirical P(medical) replaces hard binary classify
11. **KL divergence as uncertainty signal** — `KL(P || Uniform)` quantifies how far the model's opinion is from a coin-flip; KL≈1 = confident, KL≈0 = guessing
12. **Online k-means clustering** — streamed 3-cluster model groups queries into confident-medical / uncertain / confident-non-medical; centroids update incrementally with 1/count learning rate; cold-start initialises from first 9 samples
13. **Conservative routing** — AGX is the default for any uncertain or borderline query (confidence < 0.7), even if the classifier says medical; "when in doubt, defer to the bigger model"

---

## 16. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| llama-cli prints `> ` forever | Interactive mode without prompt | Add `--single-turn` flag |
| Response includes ASCII banner | No output filtering | Use `_extract_response()` to strip non-content |
| OOM on model load | FP16 model too large (3.1 GB) | Switch to Q4_K_M quantized (985 MB) |
| Empty response | Wrong marker in extract | Search for `<|im_start|>assistant\n\n` (not `> <|im_start|>...`) |
| Client timeout (90s+ 0 bytes) | Server handler blocking or inference too slow | Check AGX logs, increase client timeout, add timing logs |
| Training silent death (~100s) | GPU watchdog timeout | `CUDA_LAUNCH_BLOCKING=1`, reduce samples, use manual loop |
| Training Step 3 fails (4-bit load) | bitsandbytes version mismatch | `bitsandbytes>=0.50.0` required for `transformers>=4.41` |
| WebSocket fails | Network issue | Falls back to REST polling automatically |
| Subserver process killed | Shell timeout (nohup dies on session end) | Use `tmux`, `screen`, or systemd service |
| AGX llama-cli banner output | Non-suppressible ASCII art even with `--log-disable` | ✅ Low severity, does not affect functionality |
| AGX model reload per request | POST `/send` spawns new llama-cli each time (~2.5s overhead) | ⚠️ Medium — implement persistent model server |
| AGX root disk full | Only 3.4 GB free on eMMC root partition | Move models and data to `/mnt/nvme` |
| AGX inference intermittent | Requires active WebSocket client connection | ⚠️ Medium — check WS heartbeat and reconnect logic |
| NX server process killed by shell | Background process dies when bash session terminates | Use `tmux`, `screen`, or `nohup` with proper process management |
