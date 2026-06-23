#!/usr/bin/env python3
"""Test general queries on AGX, then medical on NX."""

import json, requests, time

AGX = "http://100.120.59.117:8000"
NX = "http://localhost:8765"

GENERAL = [
    ("g1", "What is the capital of France?"),
    ("g2", "Write a Python function to sort a list."),
    ("g3", "Explain how a diesel engine works."),
    ("g4", "What is 2+2?"),
    ("g5", "Who wrote Romeo and Juliet?"),
]

MEDICAL = [
    ("m1", "What are the symptoms of diabetes?"),
    ("m2", "How is hypertension treated?"),
    ("m3", "What causes pneumonia?"),
    ("m4", "Describe the symptoms of anaphylaxis."),
    ("m5", "How does insulin work in the body?"),
]


def agx_chat(text, timeout=120):
    t0 = time.time()
    r = requests.post(f"{AGX}/send", json={"sender": "test", "text": text}, timeout=timeout)
    body = r.json()
    if "reply" in body:
        text = body["reply"].get("text", "")
        return {"s": round(time.time()-t0, 2), "content": text, "len": len(text)}
    deadline = time.time() + 120
    while time.time() < deadline:
        r = requests.get(f"{AGX}/messages?limit=20", timeout=10)
        for msg in reversed(r.json().get("messages", [])):
            if msg.get("role") == "server" and msg.get("sender") != "system" and msg.get("text"):
                return {"s": round(time.time()-t0, 2), "content": msg["text"], "len": len(msg["text"])}
        time.sleep(0.5)
    return {"s": round(time.time()-t0, 2), "content": "", "len": 0}


def nx_chat(text, timeout=120):
    t0 = time.time()
    r = requests.post(f"{NX}/chat", json={"message": text, "stream": False}, timeout=timeout)
    body = r.json() if r.ok else {}
    return {
        "s": round(time.time()-t0, 2),
        "content": body.get("content", ""),
        "len": len(body.get("content", "")),
        "source": body.get("source", "?"),
        "confidence": body.get("confidence"),
    }


print("=" * 60)
print("  PART 1: GENERAL queries → AGX directly")
print("=" * 60)
for qid, qtext in GENERAL:
    r = agx_chat(qtext)
    c = r["content"][:80]
    print(f"  [{qid}] {r['s']:6.2f}s  {r['len']:4d}c  \"{c}\"")

print()
print("=" * 60)
print("  PART 2: MEDICAL queries → Subserver (keyword→NX inference)")
print("=" * 60)
for qid, qtext in MEDICAL:
    r = nx_chat(qtext)
    c = r["content"][:80]
    print(f"  [{qid}] {r['s']:6.2f}s  {r['len']:4d}c  src={r['source']}  c={r['confidence']}  \"{c}\"")

print("\nDone.")
