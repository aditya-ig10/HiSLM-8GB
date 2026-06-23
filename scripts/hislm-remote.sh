#!/bin/bash
# HiSLM Remote Control — Run this on your MacBook
# ==================================================
# Full remote control of Jetson Orin NX (+ AGX Orin) via SSH+Tailscale.
#
# Quick start:
#   chmod +x hislm-remote.sh
#   ./hislm-remote.sh setup       # one-time: install deps + copy SSH key
#   ./hislm-remote.sh status      # check everything
#   ./hislm-remote.sh chat        # start server + tunnel + open browser
#
# All commands:
#   setup          One-time: install Tailscale, copy SSH key, test connection
#   ssh            Open interactive SSH shell to NX
#   status         Show running servers, health, uptime, disk/memory
#   start-server   Start Qwen2.5-1.5B inference server on NX
#   start-sub      Start hybrid subserver (routes to AGX)
#   stop           Stop all servers on NX
#   restart        Restart servers
#   logs           Follow NX server logs (tail -f)
#   log            Show last N lines of server log
#   chat           Full flow: start-server + tunnel + open browser
#   tunnel         SSH port forward localhost:8765 → NX:8765
#   tunnel-agx     SSH port forward localhost:8000 → AGX:8000
#   health         Curl the NX /health endpoint
#   train          Start training on NX
#   train-log      Follow training log
#   test           Run training diagnostic (test_step.py)
#   upload <file>  Copy file from MacBook → NX project dir
#   download <path> Copy file from NX → MacBook current dir
#   shell <cmd>    Run any command on NX, e.g. ./hislm-remote.sh shell "ls -la"
#   scp <args>     Pass raw SCP args through to NX
#   info           Print system info (NX + AGX)
#   watch          Montior mode: refresh status every 5s

NX_HOST="100.85.30.17"
NX_USER="nvidia"
AGX_HOST="100.120.59.117"
AGX_USER="xadityaxsr"
PROJECT="~/llama/HiSLM-8G"
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new"

sshexec() { ssh $SSH_OPTS "${NX_USER}@${NX_HOST}" "$@"; }
sshexec_agx() { ssh $SSH_OPTS "${AGX_USER}@${AGX_HOST}" "$@"; }

echo_bold() { printf "\033[1m%s\033[0m\n" "$1"; }
echo_green() { printf "\033[32m%s\033[0m\n" "$1"; }
echo_red() { printf "\033[31m%s\033[0m\n" "$1"; }
echo_dim() { printf "\033[2m%s\033[0m\n" "$1"; }
echo_arrow() { printf "\033[36m ▶\033[0m %s\n" "$1"; }

check_nx() {
  if ! ssh $SSH_OPTS -q "${NX_USER}@${NX_HOST}" exit 2>/dev/null; then
    echo_red "✗ Cannot reach NX at ${NX_HOST}"
    echo "  Make sure you're on Tailscale: sudo tailscale up"
    echo "  Then try: ./hislm-remote.sh setup"
    exit 1
  fi
}

check_agx() {
  ssh $SSH_OPTS -q "${AGX_USER}@${AGX_HOST}" exit 2>/dev/null
}

cmd_setup() {
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║     HiSLM — One-Time Remote Setup        ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""

  # 1. Check Tailscale
  echo_arrow "Checking Tailscale..."
  if command -v tailscale &>/dev/null && tailscale status &>/dev/null; then
    echo_green "  ✓ Tailscale is running"
  else
    echo "  Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "  Run: sudo tailscale up"
    echo "  Then re-run this setup."
    exit 1
  fi

  # 2. Generate SSH key if missing
  if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo_arrow "Generating SSH key..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q
    echo_green "  ✓ Key generated"
  else
    echo_green "  ✓ SSH key exists"
  fi

  # 3. Copy to NX
  echo_arrow "Copying SSH key to NX ($NX_HOST)..."
  if ssh-copy-id $SSH_OPTS "${NX_USER}@${NX_HOST}" 2>/dev/null; then
    echo_green "  ✓ Key copied to NX"
  else
    echo_red "  ✗ Failed. Try manually: ssh-copy-id ${NX_USER}@${NX_HOST}"
  fi

  # 4. Copy to AGX
  if check_agx; then
    echo_arrow "Copying SSH key to AGX ($AGX_HOST)..."
    if ssh-copy-id $SSH_OPTS "${AGX_USER}@${AGX_HOST}" 2>/dev/null; then
      echo_green "  ✓ Key copied to AGX"
    else
      echo_red "  ✗ Failed. Try manually: ssh-copy-id ${AGX_USER}@${AGX_HOST}"
    fi
  else
    echo_dim "  - AGX not reachable (skip key copy, run later)"
  fi

  # 5. Test connection
  echo_arrow "Testing SSH connection..."
  if ssh $SSH_OPTS -q "${NX_USER}@${NX_HOST}" "echo 'Connected to:' && hostname && uname -m"; then
    echo_green "  ✓ SSH to NX works!"
  else
    echo_red "  ✗ SSH failed. Check Tailscale and try again."
    exit 1
  fi

  # 6. Download remote script to MacBook
  echo_arrow "Downloading this script to MacBook..."
  echo_dim "  (already done — you're running it)"

  # 7. Print summary
  echo ""
  echo_green "╔══════════════════════════════════════════╗"
  echo_green "║  Setup complete!                         ║"
  echo_green "╠══════════════════════════════════════════╣"
  echo_green "║  NX:  ssh ${NX_USER}@${NX_HOST}"
  echo_green "║  AGX: ssh ${AGX_USER}@${AGX_HOST}"
  echo_green "║                                          ║"
  echo_green "║  Try:  ./hislm-remote.sh chat            ║"
  echo_green "╚══════════════════════════════════════════╝"
}

cmd_ssh() {
  check_nx
  echo "🔗 Opening SSH shell to NX... (type exit to return)"
  ssh $SSH_OPTS "${NX_USER}@${NX_HOST}"
}

cmd_status() {
  check_nx
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║           HiSLM — System Status          ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""

  echo_arrow "NX Orin ($NX_HOST)"
  echo "────────────────────────────────────────"
  sshexec "
    echo '  Hostname  : \$(hostname)'
    echo '  Uptime    :\$(uptime | sed 's/.*up //' | sed 's/,.*//')'
    echo '  CPU Load  : \$(uptime | grep -oP 'load average:.*' | cut -d: -f2)'
    echo '  Memory    : \$(free -h | awk '/Mem:/{printf \$3\"/\"\$2}')'
    echo '  Disk      : \$(df -h ~ | tail -1 | awk '{print \$3\"/\"\$2\" (\"\$5\")\"}')'
    echo '  Temp      : \$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{printf \"%.1f°C\", \$1/1000}')'
  "
  echo ""

  echo_arrow "Servers"
  echo "────────────────────────────────────────"
  local server_pid=$(sshexec "pgrep -f server_qwen.py 2>/dev/null")
  local sub_pid=$(sshexec "pgrep -f subserver.py 2>/dev/null")
  if [ -n "$server_pid" ]; then
    local server_uptime=$(sshexec "ps -p $server_pid -o etime= 2>/dev/null | xargs")
    sshexec "curl -s http://localhost:8765/health 2>/dev/null" | python3 -m json.tool 2>/dev/null || echo "  Server    : running (PID $server_pid, up $server_uptime)"
  else
    echo_dim "  Server    : stopped"
  fi
  if [ -n "$sub_pid" ]; then
    echo "  Subserver : running (PID $sub_pid)"
  else
    echo_dim "  Subserver : stopped"
  fi
  echo ""

  echo_arrow "AGX Orin ($AGX_HOST)"
  echo "────────────────────────────────────────"
  if check_agx; then
    sshexec_agx "
      echo '  Hostname  : \$(hostname)'
      echo '  Uptime    :\$(uptime | sed 's/.*up //' | sed 's/,.*//')'
      echo '  Memory    : \$(free -h | awk '/Mem:/{printf \$3\"/\"\$2}')'
      echo '  Server    : \$(pgrep -f server2.py && echo \"running\" || echo \"stopped\")'
    "
  else
    echo_dim "  Unreachable (not on Tailscale or AGX is off)"
  fi
  echo ""

  echo_arrow "Tunnel Status (MacBook)"
  echo "────────────────────────────────────────"
  if lsof -i :8765 &>/dev/null; then
    echo_green "  Port 8765 : tunnel ACTIVE → http://localhost:8765"
  else
    echo_dim "  Port 8765 : no tunnel (run: ./hislm-remote.sh tunnel)"
  fi
  echo ""
}

cmd_start_server() {
  check_nx
  sshexec "pgrep -f server_qwen.py" >/dev/null && {
    echo_red "Server already running"
    sshexec "pgrep -f server_qwen.py"
    exit 1
  }
  echo_arrow "Starting Qwen2.5-1.5B inference server on NX..."
  sshexec "cd $PROJECT && nohup python3 server_qwen.py --port 8765 > output/server.log 2>&1 &"
  sleep 3
  if sshexec "curl -s http://localhost:8765/health" | grep -q ok; then
    echo_green "  ✓ Server is online!"
    sshexec "curl -s http://localhost:8765/health"
    echo ""
    echo "  Tunnel:  ./hislm-remote.sh tunnel"
    echo "  Browser: http://localhost:8765"
  else
    echo_red "  ✗ Server failed to start. Check logs:"
    sshexec "tail -5 $PROJECT/output/server.log 2>/dev/null || echo 'No log'"
  fi
}

cmd_start_sub() {
  check_nx
  sshexec "pgrep -f subserver.py" >/dev/null && {
    echo_red "Subserver already running"
    exit 1
  }
  echo_arrow "Starting hybrid subserver (NX + AGX routing)..."
  sshexec "cd $PROJECT && nohup python3 subserver.py --agx-ip $AGX_HOST --port 8765 > output/subserver.log 2>&1 &"
  sleep 3
  if sshexec "curl -s http://localhost:8765/health" | grep -q ok; then
    echo_green "  ✓ Subserver is online!"
    echo "  Queries will be routed: medical→NX local, other→AGX"
    echo ""
    echo "  Tunnel:  ./hislm-remote.sh tunnel"
    echo "  Browser: http://localhost:8765"
  else
    echo_red "  ✗ Subserver failed to start"
    sshexec "tail -5 $PROJECT/output/subserver.log 2>/dev/null || echo 'No log'"
  fi
}

cmd_stop() {
  check_nx
  echo_arrow "Stopping servers..."
  sshexec "
    pkill -f server_qwen.py 2>/dev/null && echo '  ✓ Server stopped' || echo '  - Server was not running'
    pkill -f subserver.py 2>/dev/null && echo '  ✓ Subserver stopped' || echo '  - Subserver was not running'
  "
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start_server
}

cmd_logs() {
  check_nx
  local which="${1:-server}"
  case "$which" in
    server|subserver|train)
      echo "📜 Following $which log on NX... (Ctrl+C to stop)"
      sshexec "tail -f $PROJECT/output/${which}.log 2>/dev/null || echo 'No ${which}.log found'"
      ;;
    *)
      echo "Usage: $0 logs [server|subserver|train]"
      ;;
  esac
}

cmd_log() {
  check_nx
  local which="${2:-server}"
  local lines="${3:-30}"
  sshexec "tail -$lines $PROJECT/output/${which}.log 2>/dev/null || echo 'No ${which}.log found'"
}

cmd_chat() {
  cmd_start_server
  echo ""
  echo_arrow "Opening SSH tunnel..."
  echo "  Open http://localhost:8765 in your browser"
  echo "  Press Ctrl+C to stop everything"
  echo ""
  ssh $SSH_OPTS -L 8765:localhost:8765 -N "${NX_USER}@${NX_HOST}"
  echo ""
  cmd_stop
}

cmd_tunnel() {
  check_nx
  echo "🔌 SSH Tunnel: localhost:8765 → $NX_HOST:8765"
  echo "  Open http://localhost:8765 in your browser"
  echo "  Press Ctrl+C to close tunnel"
  echo ""
  ssh $SSH_OPTS -L 8765:localhost:8765 -N "${NX_USER}@${NX_HOST}"
}

cmd_tunnel_agx() {
  if ! check_agx; then
    echo_red "AGX unreachable. Make sure Tailscale is connected."
    exit 1
  fi
  echo "🔌 SSH Tunnel: localhost:8000 → $AGX_HOST:8000"
  echo "  Open http://localhost:8000 in your browser"
  echo "  Press Ctrl+C to close tunnel"
  echo ""
  ssh $SSH_OPTS -L 8000:localhost:8000 -N "${AGX_USER}@${AGX_HOST}"
}

cmd_health() {
  check_nx
  if sshexec "curl -s http://localhost:8765/health" | python3 -m json.tool 2>/dev/null; then
    :
  else
    sshexec "curl -s http://localhost:8765/health" || echo_red "Server not running"
  fi
}

cmd_train() {
  check_nx
  echo_arrow "Starting training on NX..."
  sshexec "cd $PROJECT && bash train.sh > output/train.log 2>&1 &"
  echo "  Training started. Monitor with:"
  echo "  ./hislm-remote.sh logs train"
  echo "  ./hislm-remote.sh train-log"
}

cmd_train_log() {
  check_nx
  echo "📜 Following training log... (Ctrl+C to stop)"
  sshexec "tail -f $PROJECT/output/train_log.txt 2>/dev/null || echo 'No train_log.txt yet'"
}

cmd_test() {
  check_nx
  echo_arrow "Running training diagnostic on NX..."
  sshexec "cd $PROJECT && python3 test_step.py 2>&1"
}

cmd_upload() {
  if [ -z "$2" ]; then
    echo "Usage: $0 upload <local-file> [remote-path]"
    echo "  e.g.: $0 upload my_script.py"
    echo "  e.g.: $0 upload my_script.py ~/llama/HiSLM-8G/tools/"
    exit 1
  fi
  local src="$2"
  local dst="${3:-$PROJECT/}"
  if [ ! -f "$src" ]; then
    echo_red "File not found: $src"
    exit 1
  fi
  check_nx
  echo_arrow "Uploading $src → NX:$dst"
  scp $SSH_OPTS "$src" "${NX_USER}@${NX_HOST}:$dst"
  echo_green "  ✓ Done"
}

cmd_download() {
  if [ -z "$2" ]; then
    echo "Usage: $0 download <remote-path>"
    echo "  e.g.: $0 download ~/llama/HiSLM-8G/output/train_log.txt"
    echo "  e.g.: $0 download ~/llama/HiSLM-8G/output/server.log"
    exit 1
  fi
  check_nx
  local fname=$(basename "$2")
  echo_arrow "Downloading $2 → ./$fname"
  scp $SSH_OPTS "${NX_USER}@${NX_HOST}:$2" "./$fname"
  echo_green "  ✓ Saved to ./$fname"
}

cmd_shell() {
  check_nx
  shift
  if [ $# -eq 0 ]; then
    echo "Usage: $0 shell <command>"
    echo "  e.g.: $0 shell 'ls -la ~/llama/HiSLM-8G/'"
    echo "  e.g.: $0 shell 'free -h && nvidia-smi'"
    exit 1
  fi
  echo_arrow "Running on NX: $*"
  sshexec "$@"
}

cmd_scp() {
  check_nx
  shift
  scp $SSH_OPTS "$@"
}

cmd_info() {
  check_nx
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║          HiSLM — System Info             ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""

  echo_green "── NX Orin (8GB) ──"
  sshexec "
    echo '  Hostname : \$(hostname)'
    echo '  CPU      : \$(lscpu | grep 'Model name' | head -1 | sed 's/.*://' | xargs)'
    echo '  Cores    : \$(nproc)'
    echo '  Memory   : \$(free -h | awk '/Mem:/{print \$2}')'
    echo '  Disk     : \$(df -h / | tail -1 | awk '{print \$2\" total, \"\$4\" free\"}')'
    echo '  Kernel   : \$(uname -r)'
    echo '  Arch     : \$(uname -m)'
    echo '  Project  : \$(du -sh $PROJECT --exclude=venv --exclude=dataset --exclude=models 2>/dev/null | cut -f1)'
  "
  echo ""
  echo_green "── Models on NX ──"
  sshexec "ls -lh $PROJECT/models/ 2>/dev/null || echo '  (no models directory)'"
  echo ""
  echo_green "── AGX Orin ──"
  if check_agx; then
    sshexec_agx "
      echo '  Hostname : \$(hostname)'
      echo '  Memory   : \$(free -h | awk '/Mem:/{print \$2}')'
    "
  else
    echo_dim "  Unreachable"
  fi
  echo ""
  echo_green "── MacBook ──"
  echo "  OS       : $(sw_vers -productName 2>/dev/null || echo "macOS")"
  echo "  Tailscale: $(tailscale ip -4 2>/dev/null || echo 'not connected')"
}

cmd_watch() {
  while true; do
    clear
    cmd_status
    sleep 5
  done
}

cmd_help() {
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║    HiSLM Remote Control — MacBook        ║"
  echo_bold "╠══════════════════════════════════════════╣"
  echo_bold "║  NX:  ${NX_USER}@${NX_HOST}"
  echo_bold "║  AGX: ${AGX_USER}@${AGX_HOST}"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""
  echo "Usage: ./hislm-remote.sh <command> [args]"
  echo ""
  echo_bold "── One-time ──"
  echo "  setup          Install deps, copy SSH keys, test connection"
  echo ""
  echo_bold "── Control ──"
  echo "  ssh            Open interactive SSH shell to NX"
  echo "  status         Show servers, health, memory, uptime"
  echo "  info           Full system information (both machines)"
  echo "  health         Curl NX /health endpoint"
  echo "  watch          Live monitoring (refreshes every 5s)"
  echo ""
  echo_bold "── Servers ──"
  echo "  start-server   Start inference server (Qwen2.5-1.5B)"
  echo "  start-sub      Start subserver (NX + AGX routing)"
  echo "  stop           Stop all servers"
  echo "  restart        Restart servers"
  echo ""
  echo_bold "── Access ──"
  echo "  chat           Start server + tunnel + open browser in one go"
  echo "  tunnel         SSH tunnel localhost:8765 → NX"
  echo "  tunnel-agx     SSH tunnel localhost:8000 → AGX"
  echo ""
  echo_bold "── Logs ──"
  echo "  logs [name]    Follow logs (server|subserver|train)"
  echo "  log [n] [name] Show last N log lines"
  echo ""
  echo_bold "── Training ──"
  echo "  train          Start QLoRA training on NX"
  echo "  train-log      Follow training progress"
  echo "  test           Run training diagnostic"
  echo ""
  echo_bold "── Files ──"
  echo "  upload <f>     Copy file from MacBook to NX"
  echo "  download <p>   Copy file from NX to MacBook"
  echo "  scp <args>     Raw SCP passthrough"
  echo ""
  echo_bold "── Shell ──"
  echo "  shell <cmd>    Run any command on NX"
  echo "                 e.g. ./hislm-remote.sh shell 'free -h'"
  echo "                 e.g. ./hislm-remote.sh shell 'ls -la ~/llama/HiSLM-8G/'"
}

# ── Main Dispatch ─────────────────────────────────────
case "${1:-help}" in
  setup)        cmd_setup ;;
  ssh)          cmd_ssh ;;
  status)       cmd_status ;;
  info)         cmd_info ;;
  health)       cmd_health ;;
  watch)        cmd_watch ;;
  start-server) cmd_start_server ;;
  start-sub)    cmd_start_sub ;;
  stop)         cmd_stop ;;
  restart)      cmd_restart ;;
  chat)         cmd_chat ;;
  tunnel)       cmd_tunnel ;;
  tunnel-agx)   cmd_tunnel_agx ;;
  logs)         cmd_logs "$2" ;;
  log)          cmd_log "$@" ;;
  train)        cmd_train ;;
  train-log)    cmd_train_log ;;
  test)         cmd_test ;;
  upload)       cmd_upload "$@" ;;
  download)     cmd_download "$@" ;;
  shell)        cmd_shell "$@" ;;
  scp)          cmd_scp "$@" ;;
  help)         cmd_help ;;
  *)
    echo_red "Unknown command: $1"
    echo "Usage: ./hislm-remote.sh help"
    exit 1
    ;;
esac
