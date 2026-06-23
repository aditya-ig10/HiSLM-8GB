#!/bin/bash
# HiSLM Remote Desktop for MacBook — one-click GUI access to Jetson Orin NX
# ==========================================================================
# Connects to NX via SSH tunnel + Microsoft Remote Desktop.
# Two modes:
#   1. Tailscale (default) — RDP over SSH tunnel via Tailscale IP
#   2. ngrok — get a public URL (no Tailscale needed on MacBook)
#
# Usage:
#   ./remote-desktop.sh                # Open RDP via Tailscale tunnel
#   ./remote-desktop.sh ngrok          # Use ngrok public URL instead
#   ./remote-desktop.sh tunnel-only    # Just create the SSH tunnel
#   ./remote-desktop.sh install        # Install dependencies (brew)
#   ./remote-desktop.sh copy-key       # Copy PEM key to ~/.ssh
#   ./remote-desktop.sh help           # Show this message

NX_HOST="100.85.30.17"
NX_USER="nvidia"
PEM_KEY="$(dirname "$0")/../hislm-remote.pem"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

echo_bold() { printf "\033[1m%s\033[0m\n" "$1"; }
echo_green() { printf "\033[32m%s\033[0m\n" "$1"; }
echo_red() { printf "\033[31m%s\033[0m\n" "$1"; }
echo_arrow() { printf "\033[36m ▶\033[0m %s\n" "$1"; }

check_pem() {
  if [ ! -f "$PEM_KEY" ]; then
    echo_red "✗ PEM key not found: $PEM_KEY"
    echo "  Download it from NX:"
    echo "    scp ${NX_USER}@${NX_HOST}:~/llama/HiSLM-8G/hislm-remote.pem ."
    echo "    chmod 600 hislm-remote.pem"
    exit 1
  fi
  chmod 600 "$PEM_KEY" 2>/dev/null
}

check_nx() {
  if ! ssh -i "$PEM_KEY" $SSH_OPTS -q "${NX_USER}@${NX_HOST}" exit 2>/dev/null; then
    echo_red "✗ Cannot reach NX at ${NX_HOST}"
    echo "  Make sure you're on Tailscale: sudo tailscale up"
    echo "  Or use: $0 ngrok"
    exit 1
  fi
}

cmd_install() {
  echo_arrow "Checking dependencies..."
  
  if ! command -v brew &>/dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  
  if ! command -v mstsc &>/dev/null && ! ls /Applications/Microsoft\ Remote\ Desktop.app 2>/dev/null; then
    echo "  Installing Microsoft Remote Desktop..."
    brew install --cask microsoft-remote-desktop
  fi
  
  if ! command -v tailscale &>/dev/null; then
    echo "  Installing Tailscale..."
    brew install --cask tailscale
    echo "  Open Tailscale app and sign in, then re-run this script."
  fi
  
  echo_green "  ✓ All dependencies ready"
}

cmd_copy_key() {
  mkdir -p ~/.ssh
  cp "$PEM_KEY" ~/.ssh/hislm-remote.pem
  chmod 600 ~/.ssh/hislm-remote.pem
  echo_green "  ✓ PEM key copied to ~/.ssh/hislm-remote.pem"
  
  # Add to SSH config for convenience
  if ! grep -q "hislm-nx" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config << EOF

Host hislm-nx
    HostName $NX_HOST
    User $NX_USER
    IdentityFile ~/.ssh/hislm-remote.pem
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 3

Host hislm-nx-rdp
    HostName $NX_HOST
    User $NX_USER
    IdentityFile ~/.ssh/hislm-remote.pem
    StrictHostKeyChecking accept-new
    LocalForward 33389 localhost:3389
    ServerAliveInterval 30
    ServerAliveCountMax 3
EOF
    echo_green "  ✓ SSH config entries added (use: ssh hislm-nx)"
  fi
}

cmd_ngrok_tunnel() {
  check_pem
  
  echo_arrow "Starting ngrok TCP tunnel on NX (port 3389)..."
  ssh -i "$PEM_KEY" $SSH_OPTS "${NX_USER}@${NX_HOST}" "
    # Kill any old ngrok
    pkill -f 'ngrok.*3389' 2>/dev/null
    sleep 1
    # Start new ngrok TCP tunnel
    nohup ngrok tcp 3389 --log=stdout > ~/llama/HiSLM-8G/output/ngrok.log 2>&1 &
    sleep 3
  "
  
  # Get the ngrok URL
  sleep 2
  local ngrok_url=$(ssh -i "$PEM_KEY" $SSH_OPTS "${NX_USER}@${NX_HOST}" \
    "curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); t=d['tunnels'][0]; print(t['public_url'])\" 2>/dev/null")
  
  if [ -z "$ngrok_url" ]; then
    echo_red "  ✗ Failed to get ngrok URL"
    echo "  Check ngrok auth: ssh -i $PEM_KEY ${NX_USER}@${NX_HOST} 'ngrok config add-authtoken YOUR_TOKEN'"
    echo "  Get token from: https://dashboard.ngrok.com"
    exit 1
  fi
  
  echo_green "  ✓ ngrok tunnel active!"
  echo ""
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║     RDP is LIVE on the internet!         ║"
  echo_bold "╠══════════════════════════════════════════╣"
  echo_bold "║                                          ║"
  echo_bold "║  $ngrok_url"
  echo_bold "║                                          ║"
  echo_bold "║  Open Microsoft Remote Desktop           ║"
  echo_bold "║  → Add PC → enter the URL above          ║"
  echo_bold "║  → Username: $NX_USER                    ║"
  echo_bold "║  → Password: (your NX password)          ║"
  echo_bold "║                                          ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""
  echo "  Press Ctrl+C to stop the tunnel"
  echo ""
  
  # Keep tunnel alive
  ssh -i "$PEM_KEY" $SSH_OPTS -L 33389:localhost:3389 -N "${NX_USER}@${NX_HOST}"
}

cmd_tunnel_only() {
  check_pem
  check_nx
  
  echo_arrow "Creating SSH tunnel: localhost:33389 → $NX_HOST:3389 (RDP)"
  echo "  Open Microsoft Remote Desktop → Add PC → localhost:33389"
  echo "  Username: $NX_USER"
  echo "  Password: (your NX password)"
  echo ""
  echo "  Press Ctrl+C to close tunnel"
  echo ""
  
  ssh -i "$PEM_KEY" $SSH_OPTS -L 33389:localhost:3389 -N "${NX_USER}@${NX_HOST}"
}

cmd_default() {
  check_pem
  check_nx
  
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║   HiSLM — Remote Desktop (RDP)           ║"
  echo_bold "╠══════════════════════════════════════════╣"
  echo_bold "║  Connecting to NX Orin via SSH tunnel... ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""
  
  # Check xRDP is running on NX
  echo_arrow "Checking xRDP on NX..."
  if ssh -i "$PEM_KEY" $SSH_OPTS "${NX_USER}@${NX_HOST}" "ss -tlnp | grep -q :3389"; then
    echo_green "  ✓ xRDP is running"
  else
    echo_red "  ✗ xRDP is not running on NX"
    echo "  Run this on NX to install:"
    echo "    cd ~/llama/HiSLM-8G && chmod +x scripts/setup-remote-desktop.sh && ./scripts/setup-remote-desktop.sh"
    exit 1
  fi
  
  # Start tunnel in background
  echo_arrow "Creating SSH tunnel (localhost:33389 → NX:3389)..."
  ssh -i "$PEM_KEY" $SSH_OPTS -L 33389:localhost:3389 -N "${NX_USER}@${NX_HOST}" &
  TUNNEL_PID=$!
  sleep 2
  
  if kill -0 $TUNNEL_PID 2>/dev/null; then
    echo_green "  ✓ Tunnel active (PID $TUNNEL_PID)"
  else
    echo_red "  ✗ Tunnel failed"
    exit 1
  fi
  
  # Open Microsoft Remote Desktop
  echo_arrow "Opening Microsoft Remote Desktop..."
  if [ -d "/Applications/Microsoft Remote Desktop.app" ]; then
    open "/Applications/Microsoft Remote Desktop.app"
    echo_green "  ✓ Microsoft Remote Desktop launched"
  elif command -v open &>/dev/null; then
    # Try to create and open an RDP file
    local rdp_file="/tmp/hislm-nx.rdp"
    cat > "$rdp_file" << RDP
full address:s:localhost:33389
username:s:$NX_USER
screen mode id:i:2
session bpp:i:24
connection type:i:6
disable wallpaper:i:1
allow font smoothing:i:1
allow desktop composition:i:0
disable full window drag:i:1
disable menu anims:i:1
disable themes:i:0
RDP
    open "$rdp_file"
    echo_green "  ✓ RDP file created and opened"
  fi
  
  echo ""
  echo_bold "╔══════════════════════════════════════════╗"
  echo_bold "║  RDP ready!                              ║"
  echo_bold "╠══════════════════════════════════════════╣"
  echo_bold "║  Connect to: localhost:33389             ║"
  echo_bold "║  Username:   $NX_USER                    ║"
  echo_bold "║  Password:   (your NX login password)    ║"
  echo_bold "╠══════════════════════════════════════════╣"
  echo_bold "║  Press Ctrl+C to disconnect              ║"
  echo_bold "╚══════════════════════════════════════════╝"
  echo ""
  
  # Wait
  wait $TUNNEL_PID 2>/dev/null
}

cmd_help() {
  echo_bold "HiSLM Remote Desktop — MacBook"
  echo ""
  echo "Usage: ./remote-desktop.sh [command]"
  echo ""
  echo "Commands:"
  echo "  (no args)     Open RDP via SSH tunnel (Tailscale)"
  echo "  ngrok         Use ngrok public URL (no Tailscale needed)"
  echo "  tunnel-only   Just create SSH tunnel, don't open client"
  echo "  install       Install dependencies via Homebrew"
  echo "  copy-key      Copy PEM key to ~/.ssh + add SSH config"
  echo "  help          Show this message"
  echo ""
  echo_bold "One-time setup:"
  echo "  1. ./remote-desktop.sh install       # Install deps"
  echo "  2. ./remote-desktop.sh copy-key      # Install PEM key"
  echo "  3. On NX: ./setup-remote-desktop.sh  # Install xRDP"
  echo "  4. ./remote-desktop.sh               # Connect!"
}

case "${1:-default}" in
  default)      cmd_default ;;
  ngrok)        cmd_ngrok_tunnel ;;
  tunnel-only)  cmd_tunnel_only ;;
  install)      cmd_install ;;
  copy-key)     cmd_copy_key ;;
  help)         cmd_help ;;
  *)
    echo_red "Unknown: $1"
    cmd_help
    exit 1
    ;;
esac
