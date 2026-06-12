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
    │          Orin NX (Server)           │
    │  ┌──────────────────────────────┐   │
    │  │  subserver.py                │   │
    │  │  ┌──────────┐  ┌─────────┐  │   │
    │  │  │Classifier│→ │ Router  │  │   │
    │  │  └──────────┘  └────┬────┘  │   │
    │  │               │     │       │   │
    │  │          ┌────┘     └──┐    │   │
    │  │          ▼             ▼    │   │
    │  │   ┌────────────┐ ┌────────┐│   │
    │  │   │ Qwen 1.5B  │ │ AGX    ││   │
    │  │   │ (local)    │ │ (relay)││   │
    │  │   └────────────┘ └────────┘│   │
    │  └──────────────────────────────┘   │
    │                                     │
    │  client.py / client2.py             │
    │  (thin clients, connect to AGX)     │
    └─────────────────────────────────────┘
```

## Components

### Inference Server (`server_qwen.py`)

Flask + WebSocket server that spawns `llama-cli` as a subprocess. Serves the **Qwen2.5-1.5B-Instruct** model (GGUF, Q4_K_M).

> **Note:** The medical LoRA adapter was removed because it degraded English medical QA. The base Qwen2.5-1.5B-Instruct outperforms it for conversational English medical queries. See `BUG_ANALYSIS.md` for details.

#### Quick start

```bash
# Activate environment
source venv/bin/activate

# Start server (default port 8765)
python server_qwen.py

# Custom port
python server_qwen.py --port 8080

# Expose via ngrok
python server_qwen.py --ngrok
```

#### API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/` | Web chat UI (`orin_index.html`) |
| POST | `/chat` | Chat completion (JSON, SSE streaming) |
| WS | `/ws` | WebSocket chat |

**POST /chat** accepts:
```json
{
  "message": "What is hypertension?",
  "system": "Optional system prompt",
  "stream": false
}
```

- `stream: true` (default) — returns SSE stream of character tokens
- `stream: false` — returns JSON `{"content": "..."}`

#### How it works

**1. Prompt construction** (`build_prompt` in `server_qwen.py:38`):

Wraps the user message in Qwen2.5 chat template tokens:
```
<|im_start|>user
What is hypertension?<|im_end|>
<|im_start|>assistant
```

**2. Subprocess invocation** (`stream_tokens` in `server_qwen.py:60`):

Spawns llama-cli (CPU-only, aarch64 build b9453):

```
llama-cli \
  -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant" \
  -n 512 \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  -c 4096
```

Key flags:
- `--single-turn` — prevents llama-cli from entering interactive mode (without this it prints `> ` prompts in an infinite loop)
- `--no-display-prompt` — prevents the prompt from being echoed in the output

**3. Response extraction** (`_extract_response` in `server_qwen.py:47`):

The raw stdout from llama-cli contains the loading banner, ASCII logo, build info, command menu, and performance stats. `_extract_response` strips all of that by:
- Finding the last occurrence of `<|im_start|>assistant\n\n` — everything before it is discarded
- Truncating at `\n[ Prompt:` — the stats footer is discarded

**4. Streaming** — characters are yielded one at a time for SSE or WebSocket delivery.

#### Performance

On Jetson Orin NX 8GB (CPU-only llama-cli, Q4_K_M):

- Prompt processing: ~50 t/s
- Text generation: ~11-17 t/s (varies with context length)

Total time per request (512 tokens): ~30-50 seconds depending on response length.

#### Response extraction details

The raw llama-cli output looks like:

```
\nLoading model... \n\n[ASCII logo]\n\nbuild: ...\nmodel: ...\n
available commands:\n  ...\n\n
> <|im_start|>user\nWhat is hypertension?<|im_end|\n
<|im_start|>assistant\n\n
[GENERATED RESPONSE TEXT]
\n[ Prompt: 50 t/s | Generation: 11 t/s ]\n\nExiting...\n
```

`_extract_response` removes everything before and including `<|im_start|>assistant\n\n`, then removes `[ Prompt:` and everything after, returning only the generated text.

### Hybrid Subserver (`subserver.py`)

Runs on **Orin NX**. Classifies each user query and routes it to the appropriate backend:
- **Medical queries & simple greetings** → answered locally by Qwen2.5-1.5B
- **Non-medical / out-of-domain queries** → forwarded to AGX Orin for the more powerful Qwen2.5-3B

Every response includes a `"source"` field (`"NX"` or `"AGX"`) so the UI can label where the answer came from.

#### Classification

Uses the local Qwen2.5-1.5B with a minimal prompt:
```
Is this a medical/health topic or a simple greeting?
Answer with exactly one number: 1 for yes, 0 for no.
```
The model outputs a single digit (`0` or `1`), extracted via regex. If classification fails, it defaults to True (local) as a safe fallback.

#### AGX relay

Uses REST polling instead of WebSocket for reliability:
1. `POST /send` — submits the query to AGX
2. `GET /messages?limit=20` — polls every 1s until a server response appears
3. Response forwarded back to the client with `source: "AGX"`

#### Quick start

```bash
python subserver.py --agx-ip 172.16.6.21          # default port 8765
python subserver.py --agx-ip 172.16.6.21 --port 9000
```

#### API

Same surface as `server_qwen.py`, plus `source` in every response:

**SSE chunk:** `{"token": "...", "source": "NX"}`
**SSE done:**   `{"done": true, "source": "AGX", "content": "..."}`
**WS chunk:**   `{"type": "chunk", "content": "...", "source": "AGX"}`
**WS done:**    `{"type": "done", "content": "...", "source": "NX"}`

#### Logging

Every query, response, classification, and routing decision is POSTed to AGX at `/log` for centralised monitoring.



### Training (`train.py`)

Fine-tunes **TinyLlama-1.1B-Chat-v1.0** on medical datasets using QLoRA (4-bit). Runs on Jetson Orin NX 8GB with unified memory.

- **Datasets:** Med_QA, MT Samples, PubMed QA -> 224k instruction pairs
- **Method:** QLoRA (rank=8, alpha=16), 4-bit nf4, double_quant
- **Hardware:** Jetson Orin NX 8GB (JetPack 6.2, CUDA 12.6)
- **Stable training:** 500-2000 samples, ~2.5s/sample, 0.9-1.0GB free memory

```bash
# Preprocess data
python preprocess.py

# Train (via launcher)
bash train.sh

# Or directly
python train.py
```

### Clients

| Client | Use Case |
|--------|----------|
| `client.py` | Connect to AGX Orin server over LAN |
| `client_2.py` | Connect to any HiSLM server (generic) |
| `client2.py` | Connect over Tailscale wireless |

> Running `subserver.py` replaces the need for clients — they connect directly to the NX server, which handles local inference or relays to AGX automatically.

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
- `orin_index.html` — Standalone chat UI for the Qwen server with:
  - **Session management** — create, switch, and delete chat sessions from the sidebar
  - **Multi-turn context** — last 10 messages sent with each request for coherent conversation
  - **Live streaming** — tokens appear character-by-character as the model generates (both WebSocket and REST modes)
  - **Dark/light theme** — persisted across sessions
  - **Three modes** — demo, WebSocket, and REST (SSE streaming)

## LoRA Conversion (PEFT -> GGUF)

The medical LoRA was originally trained as a PEFT adapter (`trained/`), then converted to GGUF format for use with llama-cli:

```bash
python ~/llama/llama.cpp/convert_lora_to_gguf.py \
  --base models/qwen2.5-1.5b-instruct-f16.gguf \
  --lora trained/ \
  --output models/medical-lora-qwen2.5-1.5b.gguf
```

Conversion reduces the adapter from 71 MB (safetensors) to ~1 MB (GGUF) and makes it loadable with llama-cli's `--lora` flag.

## Troubleshooting notes

| Issue | Cause | Fix |
|-------|-------|-----|
| llama-cli prints `> ` forever | Interactive mode without prompt | Add `--single-turn` flag |
| Response includes ASCII banner/logo | No output filtering | Use `_extract_response` to strip non-content |
| OOM on model load | FP16 model too large (3.1 GB) | Switch to Q4_K_M quantized (985 MB) |
| Server returns empty content | Wrong marker in extract | Search for `<|im_start|>assistant\n\n` (not `> <|im_start|>...`) |

## Project Structure

```
HiSLM-8G/
├── train.py                        # Main training script (QLoRA)
├── preprocess.py                   # Dataset preprocessing
├── server_qwen.py                  # Inference server (Flask + WS)
├── subserver.py                    # Hybrid NX/AGX server (classify + route)
├── client.py                       # LAN client
├── client_2.py                     # Generic client
├── client2.py                      # Tailscale wireless client
├── test_step.py                    # Training diagnostic script
├── run_qwen_web.sh                 # Server deployment launcher
├── train.sh                        # Training launcher
├── static/
│   ├── index.html                  # AGX chat UI
│   └── nx_index.html               # NX wireless UI
├── orin_index.html                 # Standalone Qwen chat UI
├── a5000_training/                 # Desktop GPU training pipeline
│   ├── train_a5000.py
│   ├── merge_and_convert.py
│   └── requirements.txt
├── dataset/                        # Medical datasets (gitignored)
├── models/                         # GGUF models (gitignored)
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf   # 985 MB
│   └── medical-lora-qwen2.5-1.5b.gguf      # ~1 MB
├── output/                         # Training outputs (gitignored)
├── trained/                        # PEFT LoRA adapter (gitignored)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors   # 71 MB
│   └── README.md
├── CLIENT_TIMEOUT_TROUBLESHOOTING.md
├── CLIENT_UI.md
└── README.md
```

## Environment

- **Device:** NVIDIA Jetson Orin NX 8GB (aarch64)
- **JetPack:** R36.4.7, CUDA 12.6, Driver 540.4.0
- **PyTorch:** 2.5.0a0+nv24.08 (Jetson-specific)
- **Python:** 3.10+
- **llama.cpp:** build b9453 (CPU-only, aarch64, no GPU layers)

| Hardware | Memory | Use Case |
|----------|--------|----------|
| Jetson Orin NX | 8 GB unified | Training + Inference client |
| Jetson AGX Orin | 32 GB+ | Inference server |
| Desktop RTX A5000 | 24 GB | Full training pipeline |

## Docs

- [Client timeout troubleshooting](CLIENT_TIMEOUT_TROUBLESHOOTING.md)
- [Client/UI architecture](CLIENT_UI.md)
- [Training progress](progress.md)
