#!/usr/bin/env python3
"""
subserver.py — Hybrid NX/AGX inference server

Runs on Orin NX. For each user query:
  1. Classifies it by medical-domain confidence (0 or 1).
  2. If medical or simple greeting → answer locally via Qwen2.5-1.5B.
  3. If non-medical (out of domain) → forward to AGX, relay response.
  4. Logs every query + response + routing decision to AGX.
  5. Each response includes a `source` field ("NX" or "AGX").

Usage:
  python subserver.py --agx-ip 100.x.y.z             # default port 8765
  python subserver.py --agx-ip 100.x.y.z --port 9000
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file
from flask_sock import Sock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger("subserver")

MODEL = os.path.expanduser(
    "~/llama/HiSLM-8G/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
LLAMA_CLI = os.path.expanduser(
    "~/llama/llama.cpp/build-x64-linux-gcc-release/bin/llama-cli"
)
ORIN_HTML = os.path.join(os.path.dirname(__file__), "orin_index.html")

app = Flask(__name__)
sock = Sock(app)

agx_host: str = ""
agx_port: int = 8000
node_name: str = "nx-subserver"


def agx_base_url() -> str:
    return f"http://{agx_host}:{agx_port}"


# ── Classification ──────────────────────────────────────────────────

CLASSIFY_PROMPT = (
    'Is the following query about a medical/health topic '
    'or a simple greeting like "hi" / "hello"?\n'
    "Answer with exactly one number: 1 for yes, 0 for no.\n"
    "Do not output anything else.\n\n"
    'Query: {query}\n'
    "Answer: "
)


def is_medical_query(query: str) -> bool:
    """Use local Qwen to decide if query is medical or a greeting.

    Returns True → handle on NX locally.
    Returns False → route to AGX.
    Always returns True (local) on any error — safe default.
    """
    prompt = CLASSIFY_PROMPT.replace("{query}", query)
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "-p", prompt,
        "-n", "8",
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "-c", "512",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        out, _ = proc.communicate(timeout=30)
        proc.wait()
    except Exception as exc:
        log.warning(f"Classification error: {exc}")
        return True

    # Extract 0 or 1 from output
    match = re.search(r"[01]", out)
    if match:
        val = int(match.group())
        log.info(f"Classification: {query[:40]!r} → {'medical' if val else 'non-medical'}")
        return bool(val)

    log.warning(f"Could not parse classifier output: {out[:80]!r}")
    return True


# ── AGX Communication (REST-based) ─────────────────────────────────

def fetch_from_agx(query: str) -> str:
    """Send query to AGX via REST, poll for the response.

    Uses POST /send to submit, then polls GET /messages until the
    server response appears.
    """
    base = agx_base_url()
    log.info(f"Forwarding query to AGX via REST: {base}/send")

    try:
        r = requests.post(
            f"{base}/send",
            json={"sender": node_name, "text": query},
            timeout=15,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        log.error(f"AGX REST send failed: {exc}")
        return f"[AGX unreachable]"

    # Poll for response — look for the first server message after now
    deadline = time.time() + 120
    poll_interval = 1.0
    last_seen = time.time()

    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/messages?limit=20", timeout=10)
            r.raise_for_status()
            msgs = r.json().get("messages", [])
        except requests.RequestException as exc:
            log.warning(f"AGX poll failed: {exc}")
            time.sleep(poll_interval)
            continue

        for msg in reversed(msgs):
            ts = msg.get("timestamp", "")
            role = msg.get("role", "")
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            if role == "server" and sender != "system":
                # Found a server response — does it relate to our query?
                msg_time = _parse_timestamp(ts)
                if msg_time and msg_time > last_seen:
                    log.info(f"AGX response received ({len(text)} chars)")
                    return text

        time.sleep(poll_interval)

    log.error("AGX response timed out")
    return "[AGX timeout]"


def _parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to unix float, return 0 on failure."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def log_to_agx(entry: dict):
    try:
        requests.post(
            f"{agx_base_url()}/log",
            json={"sender": node_name, "type": "subserver_log", "payload": entry},
            timeout=5,
        )
    except requests.RequestException:
        pass


# ── Local inference ─────────────────────────────────────────────────

def build_prompt(
    user_msg: str,
    system: str = "",
    messages: list[dict] | None = None,
) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    if messages:
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _extract_response(raw: str) -> str:
    lines = raw.split("\n")
    last_asst = -1
    for i, line in enumerate(lines):
        if line.startswith("<|im_start|>assistant"):
            last_asst = i
    if last_asst < 0:
        return ""
    sep = -1
    for i in range(last_asst + 1, len(lines)):
        if not lines[i].strip():
            sep = i
            break
    if sep < 0:
        return ""
    gen_lines = []
    for j in range(sep + 1, len(lines)):
        if lines[j].startswith("[ Prompt:") or lines[j].startswith("Exiting"):
            break
        gen_lines.append(lines[j])
    return "\n".join(gen_lines).strip()


def stream_tokens(prompt: str, max_tokens: int = 512):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "-p", prompt,
        "-n", str(max_tokens),
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "-c", "4096",
    ]
    log.info(f"Local inference: {' '.join(cmd[-6:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    out, _ = proc.communicate()
    proc.wait()
    response = _extract_response(out)
    for ch in response:
        yield ch


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(ORIN_HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": str(MODEL)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", data.get("content", ""))
    system = data.get("system", "")
    stream = data.get("stream", True)
    messages = data.get("messages")

    medical = is_medical_query(user_msg)
    route_agx = not medical

    log.info(
        f"Query: {user_msg[:60]!r}  medical={medical}  "
        f"route={'AGX' if route_agx else 'NX'}"
    )

    if route_agx:
        full = fetch_from_agx(user_msg)
    else:
        prompt = build_prompt(user_msg, system, messages)
        full = "".join(stream_tokens(prompt))

    log_to_agx({
        "query": user_msg,
        "response": full,
        "medical": medical,
        "routed_to": "AGX" if route_agx else "NX",
    })

    source = "AGX" if route_agx else "NX"

    if not stream:
        return jsonify({"content": full.strip(), "source": source})

    def generate():
        for i in range(0, len(full), 80):
            yield f"data: {json.dumps({'token': full[i:i+80], 'source': source})}\n\n"
        yield f"data: {json.dumps({'done': True, 'source': source, 'content': full.strip()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


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
        messages = data.get("messages")

        medical = is_medical_query(user_msg)
        route_agx = not medical

        log.info(
            f"Query: {user_msg[:60]!r}  medical={medical}  "
            f"route={'AGX' if route_agx else 'NX'}"
        )

        if route_agx:
            full = fetch_from_agx(user_msg)
        else:
            prompt = build_prompt(user_msg, system, messages)
            full = "".join(stream_tokens(prompt))

        log_to_agx({
            "query": user_msg,
            "response": full,
            "medical": medical,
            "routed_to": "AGX" if route_agx else "NX",
        })

        source = "AGX" if route_agx else "NX"

        for i in range(0, len(full), 80):
            ws.send(json.dumps({"type": "chunk", "content": full[i:i+80], "source": source}))
        ws.send(json.dumps({
            "type": "done", "content": full.strip(), "source": source
        }))


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid NX/AGX subserver")
    parser.add_argument("--agx-ip", required=True, help="AGX Tailscale IP")
    parser.add_argument("--agx-port", type=int, default=8000, help="AGX server port")
    parser.add_argument("--port", type=int, default=8765, help="This server's port")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--node-name", default="nx-subserver",
                        help="Client ID for AGX connection")
    args = parser.parse_args()

    agx_host = args.agx_ip
    agx_port = args.agx_port
    node_name = args.node_name

    print(f"\n  Subserver: http://{args.host}:{args.port}")
    print(f"  AGX:       http://{agx_host}:{agx_port}")
    print(f"  Node:      {node_name}")
    print(f"  WS:        ws://{args.host}:{args.port}/ws")
    print(f"  Chat API:  http://{args.host}:{args.port}/chat (POST)")
    print(f"  Routing:   medical/greeting → NX local,  other → AGX\n")

    app.run(host=args.host, port=args.port, threaded=True)
