#!/usr/bin/env python3
"""Run all 60 medical queries through the subserver (keyword→NX inference)."""

import json, requests, time, sys

NX = "http://localhost:8765"

with open("data/eval_routing.jsonl") as f:
    queries = [json.loads(l) for l in f if l.strip()]

medical = [q for q in queries if q["label"] == 1 and q["category"] == "medical"]
print(f"Medical queries: {len(medical)}\n")

results = []
for i, q in enumerate(medical):
    t0 = time.time()
    try:
        r = requests.post(f"{NX}/chat", json={"message": q["text"], "stream": False}, timeout=300)
        body = r.json() if r.ok else {}
        elapsed = time.time() - t0
        res = {
            "id": q["id"],
            "query": q["text"][:55],
            "ok": r.ok,
            "latency_s": round(elapsed, 2),
            "response_len": len(body.get("content", "")),
            "source": body.get("source", "?"),
            "confidence": body.get("confidence"),
            "classify_ms": body.get("timing_ms", {}).get("classify_ms") if isinstance(body.get("timing_ms"), dict) else None,
            "inference_ms": body.get("timing_ms", {}).get("inference_ms") if isinstance(body.get("timing_ms"), dict) else None,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        res = {"id": q["id"], "query": q["text"][:55], "ok": False, "latency_s": round(elapsed, 2), "error": str(e)[:40], "response_len": 0, "source": "?", "confidence": None, "classify_ms": None, "inference_ms": None}
    results.append(res)

    ok_s = "OK" if res["ok"] else f"ERR:{res['error']}"
    print(f"  [{i+1:>2d}/{len(medical)}] {q['id']:>4s} {ok_s:20s} {res['latency_s']:7.2f}s  "
          f"src={res.get('source','?'):>4s}  c={str(res.get('confidence','?')):>5}  "
          f"len={res['response_len']:4d}")

# Summary
oks = [r for r in results if r["ok"]]
lats = [r["latency_s"] for r in oks]
lens = [r["response_len"] for r in oks]
srcs = {}
for r in oks:
    s = r.get("source", "?")
    srcs[s] = srcs.get(s, 0) + 1

print(f"\n{'='*65}")
print(f"  NX MEDICAL QUERIES — SUMMARY")
print(f"{'='*65}")
print(f"  Total:    {len(results)}")
print(f"  OK:       {len(oks)}")
print(f"  Errors:   {len(results) - len(oks)}")
print(f"  Avg lat:  {sum(lats)/len(lats):.2f}s" if lats else "  Avg lat:  N/A")
print(f"  P50 lat:  {sorted(lats)[len(lats)//2]:.2f}s" if lats else "")
print(f"  P95 lat:  {sorted(lats)[int(0.95*len(lats))]:.2f}s" if lats else "")
print(f"  Avg resp: {sum(lens)/len(lens):.0f} chars" if lens else "")
print(f"  Min lat:  {min(lats):.2f}s" if lats else "")
print(f"  Max lat:  {max(lats):.2f}s" if lats else "")
print(f"  Sources:  {srcs}")

# Latency distribution
if lats:
    print(f"\n  Latency Distribution:")
    for bucket in [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,120)]:
        c = sum(1 for l in lats if bucket[0] <= l < bucket[1])
        if c:
            print(f"    {bucket[0]:>3d}-{bucket[1]:>3d}s: {c:>3d} ({c/len(lats)*100:.0f}%)")

with open("results/nx_medical_results.json", "w") as f:
    json.dump({"summary": {
        "total": len(results), "ok": len(oks), "errors": len(results)-len(oks),
        "avg_latency_s": round(sum(lats)/len(lats),2) if lats else 0,
        "p50_latency_s": round(sorted(lats)[len(lats)//2],2) if lats else 0,
        "p95_latency_s": round(sorted(lats)[int(0.95*len(lats))],2) if lats else 0,
        "avg_response_len": round(sum(lens)/len(lens),1) if lens else 0,
        "min_latency_s": round(min(lats),2) if lats else 0,
        "max_latency_s": round(max(lats),2) if lats else 0,
        "source_breakdown": srcs,
    }, "results": results}, f, indent=2)
print(f"\n  Saved to nx_medical_results.json")
