"""
client2.py — Run on Orin NX (Tailscale wireless client)
=========================================================
Connects to AGX over Tailscale for SLM inference.

Two modes:
  1. GUI mode (default):  Serves nx_index.html on localhost, opens browser
  2. CLI mode (--cli):    Terminal chat interface via WebSocket

Usage:
  python client2.py --agx-ip 100.x.y.z
  python client2.py --agx-ip 100.x.y.z --cli --node-name nx-node
  python client2.py --agx-ip 100.x.y.z --port 8000 --gui-port 8501
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("client2.log", mode="a"),
    ],
)
log = logging.getLogger("NX-CLIENT2")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

def parse_args():
    p = argparse.ArgumentParser(description="HiSLM NX Wireless Client")
    p.add_argument("--agx-ip", required=True, help="AGX Tailscale IP (e.g. 100.x.y.z)")
    p.add_argument("--port", type=int, default=int(os.getenv("AGX_PORT", "8000")), help="AGX server port (default: 8000)")
    p.add_argument("--node-name", default=os.getenv("NX_NODE_NAME", "nx-node"), help="This node's name (default: nx-node)")
    p.add_argument("--gui-port", type=int, default=int(os.getenv("NX_GUI_PORT", "8501")), help="Local GUI port (default: 8501)")
    p.add_argument("--cli", action="store_true", help="CLI mode instead of GUI")
    return p.parse_args()

args = parse_args()
AGX_HOST = args.agx_ip
AGX_PORT = args.port
NODE_NAME = args.node_name
GUI_PORT = args.gui_port
CLI_MODE = args.cli

BASE_URL = f"http://{AGX_HOST}:{AGX_PORT}"
WS_URL = f"ws://{AGX_HOST}:{AGX_PORT}/ws?client_id={NODE_NAME}"

_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0))
    return _http_client

async def check_agx_health() -> dict:
    client = get_http_client()
    try:
        r = await asyncio.wait_for(client.get(f"{BASE_URL}/health"), timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}

async def rest_send(text: str) -> dict:
    client = get_http_client()
    r = await client.post(f"{BASE_URL}/send", json={"sender": NODE_NAME, "text": text})
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# GUI mode — lightweight HTTP server
# ─────────────────────────────────────────────

AGX_CONFIG_DATA = {
    "agx_host": AGX_HOST,
    "agx_port": AGX_PORT,
    "node_name": NODE_NAME,
    "ws_url": WS_URL,
    "base_url": BASE_URL,
}

class NXGUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.debug(f"[HTTP] {self.client_address[0]} - {format % args}")

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _send_404(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def _read_file(self, path: Path) -> Optional[str]:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def do_GET(self):
        if self.path == "/config.json":
            self._send_json(AGX_CONFIG_DATA)
        elif self.path == "/" or self.path == "":
            html = self._read_file(STATIC_DIR / "nx_index.html")
            if html:
                self._send_html(html)
            else:
                self._send_404()
        elif self.path.startswith("/static/"):
            fname = self.path[8:]
            content = self._read_file(STATIC_DIR / fname)
            if content:
                ext = fname.split(".")[-1]
                ctype = {"html": "text/html", "css": "text/css", "js": "application/javascript", "png": "image/png", "svg": "image/svg+xml"}.get(ext, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.end_headers()
                self.wfile.write(content.encode("utf-8") if isinstance(content, str) else content)
            else:
                self._send_404()
        else:
            html = self._read_file(STATIC_DIR / "nx_index.html")
            if html:
                self._send_html(html)
            else:
                self._send_404()

def start_gui():
    server = HTTPServer(("127.0.0.1", GUI_PORT), NXGUIHandler)
    gui_url = f"http://localhost:{GUI_PORT}/"
    print(f"\n  NX GUI → {gui_url}")
    print(f"  Server → {BASE_URL}")
    print(f"  Node   → {NODE_NAME}\n")
    log.info(f"[GUI] Starting HTTP server on port {GUI_PORT}")
    log.info(f"[GUI] AGX server at {BASE_URL}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.5)
    try:
        webbrowser.open(gui_url)
        log.info(f"[GUI] Browser opened to {gui_url}")
    except Exception as exc:
        log.warning(f"[GUI] Could not open browser: {exc}")
    print(f"  Open {gui_url} in your browser to start chatting.")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        log.info("[GUI] Server stopped")


# ─────────────────────────────────────────────
# CLI mode — async WebSocket chat
# ─────────────────────────────────────────────

async def cli_chat():
    health = await check_agx_health()
    if "error" in health:
        print(f"\n  ✗ AGX not reachable at {BASE_URL}")
        print(f"  Error: {health['error']}")
        print(f"  Check: tailscale status and AGX_URL\n")
        return

    print(f"\n  ✓ AGX reachable at {BASE_URL}")
    print(f"  Node: {NODE_NAME}")
    print(f"  Model: {health.get('llama_model', 'unknown')}")
    print(f"  LLAMA ready: {health.get('llama_ready', False)}")
    print(f"\n  Type your message and press Enter. Ctrl+C to quit.\n")

    try:
        import websockets
    except ImportError:
        log.error("[CLI] websockets library not installed. Install with: pip install websockets")
        print("Error: websockets library not installed.")
        print("Install: pip install websockets")
        return

    ws = None
    reader_task = None

    async def read_loop():
        nonlocal ws
        while ws and ws.open:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                frame = json.loads(raw)
                ftype = frame.get("type", "")
                if ftype == "message":
                    msg = frame["payload"]
                    if msg["role"] == "server" and msg["sender"] != "system":
                        sender = msg["sender"].replace("-qwen2.5-3b", "").replace("agx-", "AGX ").upper() or "AGX"
                        print(f"\n  [{sender}] {msg['text']}\n")
                        print("  ── ── ── ── ── ── ── ── ── ──")
                        print(f"  [You] ", end="", flush=True)
                    elif msg["role"] == "system":
                        print(f"\n  ⚡ {msg['text']}")
                        print(f"  [You] ", end="", flush=True)
                elif ftype == "history":
                    for m in frame.get("payload", []):
                        if m["role"] == "server" and m["sender"] != "system":
                            sender = m["sender"].replace("-qwen2.5-3b", "").replace("agx-", "AGX ").upper() or "AGX"
                            print(f"\n  [{sender}] {m['text']}")
                    print(f"  ── ── ── ── ── ── ── ── ── ──")
                    print(f"  [You] ", end="", flush=True)
                elif ftype == "connected":
                    print(f"  ✓ Connected to {frame.get('node', 'AGX')}\n")
                    print(f"  ── ── ── ── ── ── ── ── ── ──")
                    print(f"  [You] ", end="", flush=True)
                elif ftype == "ack":
                    pass
            except asyncio.TimeoutError:
                if ws and ws.open:
                    await ws.send(json.dumps({"type": "ping"}))
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as exc:
                log.debug(f"[CLI] read error: {exc}")
                break

    try:
        ws = await websockets.connect(WS_URL, ping_interval=20, ping_timeout=10)
        reader_task = asyncio.create_task(read_loop())
        await asyncio.sleep(0.3)

        print(f"  [You] ", end="", flush=True)
        loop = asyncio.get_running_loop()
        while True:
            text = await loop.run_in_executor(None, lambda: sys.stdin.readline())
            text = text.strip()
            if not text:
                print(f"  [You] ", end="", flush=True)
                continue
            await ws.send(json.dumps({"type": "message", "sender": NODE_NAME, "text": text}))
            print(f"  (waiting for AGX...) ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Goodbye!")
    except websockets.exceptions.WebSocketException as exc:
        log.error(f"[CLI] WebSocket error: {exc}")
        print(f"\n  ✗ Connection failed: {exc}")
    finally:
        if reader_task:
            reader_task.cancel()
        if ws:
            await ws.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     HiSLM NX Wireless Client         ║")
    print(f"  ║     → {AGX_HOST}:{AGX_PORT}{' ' * (24 - len(str(AGX_PORT)) - len(AGX_HOST))}║")
    print(f"  ║     Node: {NODE_NAME}{' ' * (33 - len(NODE_NAME))}║")
    mode = "CLI" if CLI_MODE else "GUI (Web)"
    print(f"  ║     Mode: {mode}{' ' * (33 - len(mode))}║")
    print("  ╚══════════════════════════════════════╝")
    print()

    if CLI_MODE:
        asyncio.run(cli_chat())
    else:
        start_gui()


if __name__ == "__main__":
    main()
