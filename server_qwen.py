#!/usr/bin/env python3
"""Qwen server — Flask + WebSocket, streams from llama-cli."""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file
from flask_sock import Sock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger("qwen-server")

MODEL = os.path.expanduser(
    "~/llama/HiSLM-8G/models/qwen2.5-3b-instruct-q4_k_m.gguf"
)
LLAMA_CLI = os.path.expanduser(
    "~/llama/llama.cpp/build-x64-linux-gcc-release/bin/llama-cli"
)
ORIN_HTML = os.path.join(os.path.dirname(__file__), "orin_index.html")

app = Flask(__name__)
sock = Sock(app)


# ── Helpers ──────────────────────────────────────────────────────────

def build_prompt(user_msg: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def stream_tokens(prompt: str, max_tokens: int = 512):
    """Yield token strings from llama-cli one at a time."""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "-p", prompt,
        "-n", str(max_tokens),
        "--no-display-prompt",
        "--simple-io",
        "-c", "4096",
    ]
    log.info(f"Spawning: {' '.join(cmd[-8:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    full = []
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        for ch in line:
            full.append(ch)
            yield ch
    proc.stdout.close()
    proc.wait()
    log.info(f"Generation done, {len(full)} chars, exit={proc.returncode}")


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(ORIN_HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": str(MODEL)})


# SSE streaming endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", data.get("content", ""))
    system = data.get("system", "")
    stream = data.get("stream", True)

    prompt = build_prompt(user_msg, system)

    if not stream:
        full = "".join(stream_tokens(prompt))
        return jsonify({"content": full.strip()})

    def generate():
        for token in stream_tokens(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# WebSocket streaming endpoint
@sock.route("/ws")
def ws_chat(ws):
    log.info("WebSocket connected")
    while True:
        raw = ws.receive()
        if raw is None:
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            continue

        msg_type = data.get("type", "")

        if msg_type == "ping":
            ws.send(json.dumps({"type": "pong"}))
            continue

        if msg_type != "message":
            continue

        user_msg = data.get("content", "")
        system = data.get("system", "")

        prompt = build_prompt(user_msg, system)
        buf = []

        for token in stream_tokens(prompt):
            buf.append(token)
            ws.send(json.dumps({"type": "chunk", "content": token}))

        full = "".join(buf).strip()
        ws.send(json.dumps({"type": "done", "content": full}))
        log.info(f"WebSocket reply done ({len(full)} chars)")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ngrok", action="store_true", help="Expose via ngrok")
    args = parser.parse_args()

    port = args.port

    if args.ngrok:
        from pyngrok import ngrok
        tunnel = ngrok.connect(port, proto="http")
        log.info(f"ngrok tunnel: {tunnel.public_url}")
        print(f"\n  ngrok URL: {tunnel.public_url}\n")

    print(f"\n  Qwen server: http://{args.host}:{port}")
    print(f"  WebSocket:   ws://{args.host}:{port}/ws")
    print(f"  Chat API:    http://{args.host}:{port}/chat (POST)\n")

    app.run(host=args.host, port=port, threaded=True)
