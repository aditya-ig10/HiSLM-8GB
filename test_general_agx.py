#!/usr/bin/env python3
"""Send all 60 general queries to AGX directly."""

import json, requests, time, sys

AGX = "http://100.120.59.117:8000"

with open("eval_routing.jsonl") as f:
    queries = [json.loads(l) for l in f if l.strip()]

general = [q for q in queries if q["label"] == 0]
print(f"Total general queries: {len(general)}\n")

def agx_query(text, timeout=120):
    t0 = time.time()
    r = requests.post(f"{AGX}/send", json={"sender": "test", "text": text}, timeout=timeout)
    body = r.json()
    if "reply" in body and body["reply"].get("text"):
        return {"s": round(time.time()-t0, 2), "text": body["reply"]["text"]}
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{AGX}/messages?limit=20", timeout=10)
        for msg in reversed(r.json().get("messages", [])):
            if msg.get("role") == "server" and msg.get("sender") != "system" and msg.get("text"):
                return {"s": round(time.time()-t0, 2), "text": msg["text"]}
        time.sleep(0.5)
    return {"s": round(time.time()-t0, 2), "text": ""}

results = []
for i, q in enumerate(general):
    r = agx_query(q["text"])
    r["id"] = q["id"]
    r["query"] = q["text"][:55]
    results.append(r)
    status = f"[{i+1}/{len(general)}] {r['s']:6.2f}s  {len(r['text']):4d}c  \"{r['text'][:60]}\""
    print(status)

# Summary
lats = [r["s"] for r in results]
lens = [len(r["text"]) for r in results]
print(f"\n{'='*60}")
print(f"  AGX GENERAL QUERIES — SUMMARY")
print(f"{'='*60}")
print(f"  Total:    {len(results)}")
print(f"  Avg lat:  {sum(lats)/len(lats):.2f}s")
print(f"  P50 lat:  {sorted(lats)[len(lats)//2]:.2f}s")
print(f"  P95 lat:  {sorted(lats)[int(0.95*len(lats))]:.2f}s")
print(f"  Avg resp: {sum(lens)/len(lens):.0f} chars")
print(f"  Min lat:  {min(lats):.2f}s")
print(f"  Max lat:  {max(lats):.2f}s")

with open("agx_general_results.json", "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nSaved to agx_general_results.json")
