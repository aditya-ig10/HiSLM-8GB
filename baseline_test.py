#!/usr/bin/env python3
"""
baseline_test.py — Compare Always-NX, Always-AGX, and HiSLM routing
on a focused set of medical + non-medical queries.

Each mode runs sequentially with proper server management:
  - always_nx:  start server_qwen standalone → run → stop
  - always_agx: query AGX directly (must be running)
  - hislm:      subserver running → run queries → stop

Usage:
  python baseline_test.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import requests

AGX_CHAT = "http://100.120.59.117:8000/chat"
SUBSERVER_PORT = 8765
STANDALONE_PORT = 8768
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "bin", "python")

QUERIES = [
    ("m1", "What are the symptoms of diabetes?", 1),
    ("m2", "How is hypertension treated?", 1),
    ("m3", "What causes pneumonia?", 1),
    ("g1", "What is the capital of France?", 0),
    ("g2", "Write a Python function to sort a list.", 0),
    ("g3", "Explain how a diesel engine works.", 0),
]


def agx_chat(endpoint, message, timeout=120):
    """Send query to AGX via /send + poll /messages."""
    base = endpoint.rstrip("/")
    t0 = time.time()
    try:
        r = requests.post(
            f"{base}/send",
            json={"sender": "baseline-test", "text": message},
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        if "reply" in body:
            reply = body["reply"]
            text = reply.get("text", "")
            elapsed = time.time() - t0
            return {
                "ok": True,
                "status": 200,
                "latency_s": round(elapsed, 2),
                "content": text,
                "content_len": len(text),
                "source": "AGX",
                "error": None,
            }
        deadline = time.time() + 120
        while time.time() < deadline:
            r = requests.get(f"{base}/messages?limit=20", timeout=10)
            msgs = r.json().get("messages", [])
            for msg in reversed(msgs):
                role = msg.get("role", "")
                sender = msg.get("sender", "")
                text = msg.get("text", "")
                if role == "server" and sender != "system" and text:
                    elapsed = time.time() - t0
                    return {
                        "ok": True,
                        "status": 200,
                        "latency_s": round(elapsed, 2),
                        "content": text,
                        "content_len": len(text),
                        "source": "AGX",
                        "error": None,
                    }
            time.sleep(0.5)
        return {
            "ok": False, "status": 0, "latency_s": round(time.time()-t0, 2),
            "content": "", "content_len": 0, "source": "error",
            "error": "AGX poll timeout",
        }
    except Exception as e:
        return {
            "ok": False, "status": 0, "latency_s": round(time.time()-t0, 2),
            "content": "", "content_len": 0, "source": "error",
            "error": str(e),
        }


def chat(endpoint, message, timeout=120):
    """Generic /chat endpoint caller."""
    t0 = time.time()
    try:
        r = requests.post(
            endpoint,
            json={"message": message, "stream": False},
            timeout=timeout,
        )
        elapsed = time.time() - t0
        body = r.json() if r.ok else {}
        return {
            "ok": r.ok,
            "status": r.status_code,
            "latency_s": round(elapsed, 2),
            "content": body.get("content", body.get("response", "")),
            "content_len": len(body.get("content", body.get("response", ""))),
            "source": body.get("source", "?"),
            "confidence": body.get("confidence"),
            "p_med": body.get("p_med"),
            "kl_div": body.get("kl_div"),
            "classify_ms": body.get("timing_ms", {}).get("classify_ms") if isinstance(body.get("timing_ms"), dict) else None,
            "inference_ms": body.get("timing_ms", {}).get("inference_ms") if isinstance(body.get("timing_ms"), dict) else None,
            "total_ms": body.get("timing_ms", {}).get("total_ms") if isinstance(body.get("timing_ms"), dict) else None,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": 0,
            "latency_s": round(time.time() - t0, 2),
            "content": "",
            "content_len": 0,
            "source": "error",
            "confidence": None,
            "p_med": None,
            "kl_div": None,
            "classify_ms": None,
            "inference_ms": None,
            "total_ms": None,
            "error": str(e),
        }


def wait_for_server(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def start_server(script, port):
    proc = subprocess.Popen(
        [VENV_PYTHON, os.path.join(BASE_DIR, script), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if wait_for_server(f"http://localhost:{port}/health", timeout=45):
        return proc
    proc.kill()
    return None


def stop_server(proc):
    if proc:
        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()


def run_mode(mode, queries):
    print(f"\n{'='*65}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*65}")

    proc = None
    if mode == "always_nx":
        print("  Starting standalone NX server...")
        proc = start_server("server_qwen.py", STANDALONE_PORT)
        if not proc:
            print("  FAILED to start NX server")
            return []
        chat_fn = lambda m: chat(f"http://localhost:{STANDALONE_PORT}/chat", m)
    elif mode == "always_agx":
        proc = None
        chat_fn = lambda m: agx_chat("http://100.120.59.117:8000", m)
    else:  # hislm
        proc = None
        chat_fn = lambda m: chat(f"http://localhost:{SUBSERVER_PORT}/chat", m)

    results = []
    for qid, qtext, label in queries:
        res = chat_fn(qtext)
        res["qid"] = qid
        res["query"] = qtext[:50]
        res["label"] = label
        results.append(res)

        label_str = "MED" if label == 1 else "GEN"
        ok_str = "OK" if res["ok"] else f"ERR:{str(res.get('error','?'))[:25]}"
        route_str = res.get("source", "?")
        c_str = str(res.get("confidence", "?"))
        print(f"  [{qid:>3s} {label_str}] {ok_str:30s} {res['latency_s']:6.2f}s  "
              f"route={route_str:>4s}  len={res['content_len']:4d}  c={c_str:>5}")

    if proc:
        stop_server(proc)
        print("  Stopped NX server")

    return results


def summarize(modes_results):
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Mode':<15s} {'Avg Lat':>8s} {'Med Lat':>8s} {'Gen Lat':>8s} "
          f"{'Avg Resp':>9s} {'Err':>5s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*5}")
    for mode, results in modes_results.items():
        lats = [r["latency_s"] for r in results if r["ok"]]
        med_lats = [r["latency_s"] for r in results if r["ok"] and r["label"] == 1]
        gen_lats = [r["latency_s"] for r in results if r["ok"] and r["label"] == 0]
        lens = [r["content_len"] for r in results if r["ok"]]
        errs = sum(1 for r in results if not r["ok"])
        avg_lat = sum(lats) / len(lats) if lats else 0
        avg_med = sum(med_lats) / len(med_lats) if med_lats else 0
        avg_gen = sum(gen_lats) / len(gen_lats) if gen_lats else 0
        avg_len = sum(lens) / len(lens) if lens else 0
        print(f"  {mode:<15s} {avg_lat:>7.1f}s {avg_med:>7.1f}s {avg_gen:>7.1f}s "
              f"{avg_len:>8.0f}c {errs:>4d}")

    hislm = modes_results.get("hislm", [])
    print(f"\n  HiSLM Routing Breakdown:")
    routed_nx = [r for r in hislm if r.get("source") == "NX"]
    routed_agx = [r for r in hislm if r.get("source") == "AGX"]
    print(f"    Routed to NX:  {len(routed_nx)}/{len(hislm)}  "
          f"({round(len(routed_nx)/len(hislm)*100)}%)")
    print(f"    Routed to AGX: {len(routed_agx)}/{len(hislm)}  "
          f"({round(len(routed_agx)/len(hislm)*100)}%)")
    for r in hislm:
        label_s = "MED" if r["label"] == 1 else "GEN"
        c_str = f"{r.get('confidence','?'):.2f}" if isinstance(r.get('confidence'), (int, float)) else str(r.get('confidence','?'))
        print(f"    {r['qid']:>4s} {label_s}  → {r.get('source','?'):>4s}  "
              f"c={c_str}  '{r['query'][:35]}'")

    # Compute latency savings
    if "always_agx" in modes_results and "hislm" in modes_results:
        agx_lats = [r["latency_s"] for r in modes_results["always_agx"] if r["ok"]]
        hislm_lats = [r["latency_s"] for r in modes_results["hislm"] if r["ok"]]
        if agx_lats and hislm_lats:
            agx_total = sum(agx_lats)
            hislm_total = sum(hislm_lats)
            pct = (hislm_total - agx_total) / agx_total * 100
            print(f"\n  HiSLM vs Always-AGX:  {pct:+.1f}% total latency")
            print(f"    Always-AGX total: {agx_total:.1f}s")
            print(f"    HiSLM total:      {hislm_total:.1f}s")


def main():
    print(f"Baseline Test — {datetime.now().isoformat()}")
    print(f"  Queries: {len(QUERIES)} ({sum(1 for q in QUERIES if q[2]==1)} medical, "
          f"{sum(1 for q in QUERIES if q[2]==0)} general)")

    # Check AGX is reachable
    try:
        r = requests.get("http://100.120.59.117:8000/health", timeout=5)
        print(f"  AGX: {r.json().get('status','?')}")
    except Exception as e:
        print(f"  AGX: DOWN — {e}")
        sys.exit(1)

    modes_results = {}
    for mode in ["always_nx", "always_agx", "hislm"]:
        results = run_mode(mode, QUERIES)
        modes_results[mode] = results

    summarize(modes_results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "agx_chat": AGX_CHAT,
            "standalone_port": STANDALONE_PORT,
            "queries": [{"id": q[0], "text": q[1], "label": q[2]} for q in QUERIES],
        },
        "modes": modes_results,
    }
    with open("results/baseline_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw results saved to baseline_test_results.json")


if __name__ == "__main__":
    main()
