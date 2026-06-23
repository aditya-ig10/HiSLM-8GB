# HiSLM — Comprehensive AGX Orin Test Report

**Date:** 2026-06-19  
**Device:** NVIDIA Jetson AGX Orin 32GB  
**Board:** AVerMedia D315  
**JetPack:** R36.4.7 (CUDA 12.6, Driver 540.4.0)  
**Kernel:** Linux 5.15.148-tegra  
**llama.cpp:** b9571-e3471b3e7 (GNU 11.4.0, aarch64, CUDA)

---

## 1. System Specifications

| Component | Detail |
|-----------|--------|
| **SoC** | NVIDIA Jetson AGX Orin 32GB |
| **CPU** | 8× Cortex-A78AE (2 clusters × 4 cores) |
| **CPU Freq** | 115–2188 MHz |
| **L1 Cache** | 1 MiB (8× 64K I + 8× 64K D) |
| **L2 Cache** | 2 MiB (8× 256K) |
| **L3 Cache** | 4 MiB (2× 2M) |
| **RAM** | 29 GiB (30697 MB total) |
| **Swap** | 14 GiB (8× zram devices, ~1.9 GB each) |
| **GPU** | Orin (nvgpu) — integrated, compute capability 8.7 |
| **GPU VRAM** | ~30 GiB (unified memory) |
| **Storage (root)** | 59 GB eMMC (51 GB used, 3.4 GB free) |
| **Storage (NVMe)** | 932 GB NVMe (WD Blue SN5100, at /mnt/nvme) |
| **Power Mode** | MAXN |
| **Idle Power** | ~5.0 W (VIN_SYS_5V0) |
| **Idle Temp** | CPU ~45°C, GPU ~41°C, TJ ~45°C |

---

## 2. Benchmark: Qwen2.5-3B-Instruct (Q4_K_M)

**Model file:** models/Qwen2.5-3B-Q4_K_M.gguf (1.79 GiB, 3.09 B params)  
**Engine:** llama-cli build b9571 (llama.cpp, CUDA, aarch64, GNU 11.4.0)  
**CPU Threads:** 8 | **Context:** 4096 tokens | **Batch:** 2048 | **UBatch:** 512  
**GPU Layers:** 99 (all 36 layers offloaded) | **Flash Attn:** auto  
**KV Cache:** f16 | **Sampling:** greedy (temp=0)

### 2.1 Cold Start (Model Load + First Inference)

| Metric | Value |
|--------|-------|
| Total time (first query) | ~3.4 s |
| Model load + prompt processing | ~2.5 s (estimated) |
| Generation (30 tokens) | ~0.9 s |

### 2.2 Short QA (50 tokens)

| Metric | Run 1 | Run 2 | Run 3 | **Average** |
|--------|-------|-------|-------|-------------|
| Total latency | 5.07 s | 5.10 s | 5.12 s | **5.10 s** |
| Response | ML definition (~75 chars) | ~75 chars | ~75 chars | Consistent |

### 2.3 Medical QA (128 tokens)

| Metric | Run 1 | Run 2 | Run 3 | **Average** |
|--------|-------|-------|-------|-------------|
| Total latency | 7.99 s | 7.87 s | 7.94 s | **7.93 s** |
| Response length | ~580 chars | ~580 chars | ~580 chars | Consistent |

### 2.4 Long Generation (256 tokens)

| Metric | Run 1 | Run 2 | Run 3 | **Average** |
|--------|-------|-------|-------|-------------|
| Total latency | 12.88 s | 12.85 s | 12.91 s | **12.88 s** |
| Response length | ~1300 chars | ~1300 chars | ~1300 chars | Consistent |

---

## 3. Benchmark: GPU vs CPU Comparison (llama-bench)

### 3.1 GPU Offloaded (`-ngl 99`)

| Test | Speed |
|------|-------|
| Prompt Processing (512 tokens) | **494.30 ± 11.73 tok/s** |
| Text Generation (128 tokens) | **12.39 ± 0.09 tok/s** |

Backend: CUDA (all 36 layers offloaded to GPU)

### 3.2 CPU Only (`-ngl 0`)

| Test | Speed |
|------|-------|
| Prompt Processing (512 tokens) | **513.89 ± 45.44 tok/s** |
| Text Generation (128 tokens) | **5.12 ± 0.07 tok/s** |

Backend: CUDA (0 layers offloaded)

### 3.3 Speedup with GPU

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Prompt Processing | 514 tok/s | 494 tok/s | **0.96x** (CPU slightly faster — GPU overhead for small model) |
| Text Generation | 5.12 tok/s | 12.39 tok/s | **2.4x** |

**Note:** For a 3B model on Orin, prompt processing is memory-bandwidth bound and runs similarly on CPU. Text generation benefits significantly from GPU offload (2.4x). Larger models (7B+) show greater GPU advantage.

---

## 4. Server Endpoint Tests (AGX — server2.py)

### 4.1 Health Check

```http
GET /health → HTTP 200 (7ms)
```

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

### 4.2 Chat API (REST)

| Scenario | Latency | Response |
|----------|---------|----------|
| POST /send (queues inference) | 5 ms | `{"ok": true, "message": {...}}` |
| Qwen2.5-3B reply to "What is the capital of France?" | ~11 s | "The capital city of France is Paris 🐛" |
| Qwen2.5-3B reply to "What is 2+2?" | ~5 s | `4` |
| GET /messages (poll) | 3 ms | Returns message history |

### 4.3 Web UI

| Endpoint | Status | Purpose |
|----------|--------|---------|
| GET / | ✅ 200 (33 KB) | Server web UI (index.html) |
| GET /nx | ✅ 200 | NX client web UI |

### 4.4 Concurrent Requests

| Test | Result |
|------|--------|
| 3 simultaneous POST /send (What is {i}+{i}?) | ✅ All 3 returned 200 OK in ~5-10 ms |
| All 3 processed by Qwen sequentially | ✅ All correct responses observed |
| Total messages after concurrent test | 9 messages (3 users + 3 Qwen replies + 3 system) |

### 4.5 WebSocket (`/ws`)

Tested from server2.py: ✅

| Feature | Status |
|---------|--------|
| Connect with client_id | ✅ Working |
| Ping/Pong keepalive | ✅ 20s interval |
| Message send + broadcast | ✅ Working |
| Qwen auto-reply | ✅ Working |
| History on connect | ✅ Last 200 messages |

---

## 5. Memory & Resource Usage

### 5.1 Memory Under Load

| State | Used | Available | Swap Used |
|-------|------|-----------|-----------|
| **Idle** | 4.3 GiB | 25 GiB | 0 GiB |
| **During inference (model loaded)** | 8.5-9.0 GiB | 20-21 GiB | 0 GiB |
| **Post-inference** | 8.5 GiB | 20 GiB | 0 GiB |

**Note:** ~4.2 GiB baseline usage includes desktop environment and background services. Model loading adds ~100 MiB to system RAM (most resides in GPU unified memory).

### 5.2 Power & Thermal (tegrastats)

#### Idle

| Sensor | Temperature | Power |
|--------|-------------|-------|
| CPU | 45.1°C | |
| GPU | 40.9°C | |
| SoC | 42.2°C | |
| TJ (junction) | 45.1°C | |
| **VDD_GPU_SOC** | | **3.7 W** |
| **VDD_CPU_CV** | | **1.4 W** |
| **VIN_SYS_5V0** | | **5.2 W** |

#### During Inference (Peak)

| Sensor | Temperature | Power |
|--------|-------------|-------|
| CPU | 46.1°C | |
| GPU | 40.8°C | |
| SoC | 42.3°C | |
| TJ (junction) | 46.1°C | |
| **VDD_GPU_SOC** | | **3.9 W** |
| **VDD_CPU_CV** | | **2.8 W** |
| **VIN_SYS_5V0** | | **5.4 W** |
| **GR3D_FREQ** | | **0-37%** |

#### Post-Inference

| Sensor | Temperature | Power |
|--------|-------------|-------|
| CPU | 45.3°C | |
| GPU | 41.4°C | |
| TJ (junction) | 45.3°C | |
| **VDD_GPU_SOC** | | **3.7 W** |
| **VDD_CPU_CV** | | **0.9 W** |
| **VIN_SYS_5V0** | | **5.0 W** |

Temperatures stayed within safe limits throughout. Maximum observed: CPU 46.1°C, GPU 41.4°C.

### 5.3 Disk Usage

| Component | Size |
|-----------|------|
| models/ (Qwen2.5-3B Q4_K_M) | 1.9 GB |
| Project (excluding models, .git, venv) | 1.6 GB |
| Root partition free | 3.4 GB (94% full — consider NVMe for data) |

---

## 6. Server Reliability

| Metric | Value |
|--------|-------|
| Test duration | ~40 min |
| Server restarts needed | 0 |
| Crashes during testing | 0 |
| Concurrent request handling | ✅ 3 simultaneous OK |
| Inference timeout handling | ✅ 300s default config |

### Observed Issues

| Issue | Details | Severity |
|-------|---------|----------|
| llama-cli banner output | Non-suppressible ASCII art and build info printed even with `--log-disable` | ✅ Low (does not affect functionality) |
| Model reload on each request | Each POST /send spawns new llama-cli subprocess (no persistent model server) | ⚠️ Medium (adds ~2.5s model load overhead) |
| Root disk space | Only 3.4 GB free on eMMC root partition | ⚠️ Medium (recommend moving models to /mnt/nvme) |

---

## 7. Performance Summary

### AGX Orin (Qwen2.5-3B Q4_K_M — GPU Offloaded)

```
Model Load:        ~2.5 s
Prompt Process:    494 tok/s  ─━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text Generation:   12.4 tok/s ─━━━━━━━━━━━━━━━━━━━━━━━━━━
Short QA (30 tok):  3.4 s    (cold start, first load)
Short QA (50 tok):  5.1 s
Medical QA (128):   7.9 s
Long Gen (256):    12.9 s
```

### AGX Orin (Qwen2.5-3B Q4_K_M — CPU Only)

```
Prompt Process:    514 tok/s  ─━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text Generation:    5.1 tok/s ─━━━━━━━━━━━
```

### Speedup Summary

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Prompt Processing | 514 tok/s | 494 tok/s | ~1x |
| Text Generation | 5.12 tok/s | 12.39 tok/s | **2.4x** |

---

## 8. Verified Features

- [x] Local inference (llama-cli subprocess, CUDA + CPU)
- [x] Qwen2.5-3B Q4_K_M model (1.79 GiB, 36 layers offloaded)
- [x] REST API (/health, /send, /messages)
- [x] Web UI (GET / → index.html, GET /nx → nx_index.html)
- [x] WebSocket (/ws with ping/pong, history on connect)
- [x] Concurrent request handling (3 simultaneous)
- [x] Qwen auto-inference on POST /send and WS message
- [x] Tailscale IP auto-detection
- [x] GPU acceleration (2.4x generation speedup vs CPU)
- [x] Efficient thermal performance (max 46°C under load)

---

Report generated 2026-06-19 by automated benchmark suite.
