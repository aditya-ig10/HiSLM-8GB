#!/bin/bash
# Start Qwen web server + ngrok tunnel

PORT=8765
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Qwen server on port $PORT..."
python3 "$DIR/server_qwen.py" --port "$PORT" &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 10); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Starting ngrok tunnel..."
/home/nvidia/.local/bin/ngrok-bin http "$PORT" --log=stdout &
NGROK_PID=$!

sleep 3

# Fetch the ngrok public URL
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null)

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║          Qwen Web — Deployed             ║"
echo "  ╠══════════════════════════════════════════╣"
echo "  ║  Local:   http://localhost:$PORT"
echo "  ║  Tunnel:  $NGROK_URL"
echo "  ║                                          ║"
echo "  ║  Open the tunnel URL in any browser!     ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

cleanup() {
    echo "Shutting down..."
    kill "$SERVER_PID" "$NGROK_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

wait
