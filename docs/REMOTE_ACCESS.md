# HiSLM Remote Control — MacBook Setup

Single script to fully control the Jetson Orin NX (and AGX Orin) from your MacBook over the internet via Tailscale + SSH.

---

## Quick Start (30 seconds)

```bash
# 1. Download the control script from NX to your MacBook
scp nvidia@100.85.30.17:~/llama/HiSLM-8G/hislm-remote.sh .
chmod +x hislm-remote.sh

# 2. One-time setup (installs Tailscale, copies SSH key, tests connection)
./hislm-remote.sh setup

# 3. Full remote chat — starts server, opens tunnel, launches browser
./hislm-remote.sh chat
```

---

## All Commands

### One-Time
| Command | What it does |
|---------|-------------|
| `setup` | Install Tailscale (if missing), generate SSH key, copy to NX + AGX, test connection |

### Control
| Command | What it does |
|---------|-------------|
| `ssh` | Open interactive SSH shell to NX |
| `status` | Show running servers, health, memory, uptime, disk |
| `info` | Full system info (NX + AGX + MacBook) |
| `health` | Curl the NX `/health` endpoint |
| `watch` | Live monitoring — refreshes status every 5s |

### Servers
| Command | What it does |
|---------|-------------|
| `start-server` | Start Qwen2.5-1.5B inference server on NX |
| `start-sub` | Start hybrid subserver (classifies + routes to AGX) |
| `stop` | Stop all servers |
| `restart` | Restart servers |

### Access
| Command | What it does |
|---------|-------------|
| `chat` | Start server + open SSH tunnel + open browser — all in one |
| `tunnel` | SSH port forward `localhost:8765` → NX (for browser access) |
| `tunnel-agx` | SSH port forward `localhost:8000` → AGX |

### Logs
| Command | What it does |
|---------|-------------|
| `logs server` | Follow NX server log (tail -f) |
| `logs subserver` | Follow subserver log |
| `logs train` | Follow training log |
| `log 30` | Show last 30 lines of server log |

### Training
| Command | What it does |
|---------|-------------|
| `train` | Start QLoRA training on NX |
| `train-log` | Follow training progress (loss/lr/mem per step) |
| `test` | Run training diagnostic (test_step.py) |

### Files
| Command | What it does |
|---------|-------------|
| `upload my_script.py` | Copy file from MacBook → NX project folder |
| `download ~/path/file` | Copy file from NX → MacBook current directory |

### Shell
| Command | What it does |
|---------|-------------|
| `shell "free -h"` | Run any command on NX and see output |
| `shell "ls -la ~/llama/HiSLM-8G/"` | List project files |
| `shell "nohup python3 train.py &"` | Start anything in background |

---

## Examples

```bash
# Check if servers are running
./hislm-remote.sh status

# Start the subserver (routes medical→NX, other→AGX)
./hislm-remote.sh start-sub

# In another terminal, open the tunnel + browser
./hislm-remote.sh tunnel

# Run a command on NX
./hislm-remote.sh shell "tail -20 ~/llama/HiSLM-8G/output/server.log"

# Download training results
./hislm-remote.sh download ~/llama/HiSLM-8G/output/train_log.txt

# Upload a custom script
./hislm-remote.sh upload my_medical_test.py
```

---

## Architecture

```
MacBook (your desk)
  │
  ├─ Tailscale (encrypted tunnel over internet)
  │
  ├─ Jetson Orin NX  (100.85.30.17)  ← Qwen2.5-1.5B + training
  │   └─ hislm-remote.sh controls via SSH
  │
  └─ Jetson AGX Orin (100.120.59.117) ← Qwen2.5-3B
      └─ accessible via SSH, or NX relays requests
```

---

## Manual SSH (if you prefer)

```bash
# Connect to NX
ssh nvidia@100.85.30.17

# Connect to AGX
ssh xadityaxsr@100.120.59.117

# Tunnel for web UI
ssh -L 8765:localhost:8765 nvidia@100.85.30.17
# Then open http://localhost:8765
```
