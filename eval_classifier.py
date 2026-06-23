#!/usr/bin/env python3
"""
eval_classifier.py — Evaluate routing classifier on labeled eval set.

Usage:
    python eval_classifier.py \
    --queries data/eval_routing.jsonl \
    --server http://localhost:8765 \
    --output results/classifier_eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Evaluate routing classifier")
    parser.add_argument("--queries", default="data/eval_routing.jsonl",
                        help="Path to labeled queries JSONL")
    parser.add_argument("--server", default="http://localhost:8765",
                        help="Subserver base URL")
    parser.add_argument("--output", default="results/classifier_eval_results.json",
                        help="Output results file")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(queries)} labeled queries from {args.queries}")
    print(f"Server: {args.server}/classify\n")

    results = []
    errors = 0
    for i, q in enumerate(queries):
        try:
            t0 = time.time()
            r = requests.post(
                f"{args.server}/classify",
                json={"text": q["text"]},
                timeout=60,
            )
            latency_ms = (time.time() - t0) * 1000
            r.raise_for_status()
            body = r.json()
            pred = body["is_medical"]
            method = body.get("method", "unknown")
            results.append({
                "id": q["id"],
                "text": q["text"],
                "label": q["label"],
                "category": q.get("category", ""),
                "pred": pred,
                "correct": pred == q["label"],
                "latency_ms": round(latency_ms, 1),
                "method": method,
                "classify_ms": body.get("classify_ms", 0),
            })
        except Exception as exc:
            errors += 1
            print(f"  [{i+1}/{len(queries)}] ERROR {q['id']}: {exc}")
            results.append({
                "id": q["id"],
                "text": q["text"],
                "label": q["label"],
                "pred": -1,
                "correct": False,
                "latency_ms": 0,
                "method": "error",
                "error": str(exc),
            })

        if (i + 1) % 10 == 0 or i == len(queries) - 1:
            done = sum(1 for r in results if r["pred"] != -1)
            print(f"  Progress: {i+1}/{len(queries)} ({done} done, {errors} errors)")

    # Compute metrics
    y_true = [r["label"] for r in results if r["pred"] != -1]
    y_pred = [r["pred"] for r in results if r["pred"] != -1]
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    classify_times = [r["classify_ms"] for r in results if r.get("classify_ms", 0) > 0]

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / total if total else 0

    # Per-class metrics
    classes = {"medical": (1,), "non-medical": (0,), "greeting": (1,)}
    for cname, clabels in classes.items():
        c_true = [t for t, p in zip(y_true, y_pred) if t in clabels]
        c_pred = [p for t, p in zip(y_true, y_pred) if t in clabels]
        c_correct = sum(1 for t, p in zip(c_true, c_pred) if t == p)
        c_total = len(c_true)
        print(f"  {cname}: {c_correct}/{c_total} = {c_correct/c_total*100:.1f}%")

    # Method breakdown
    method_counts = {}
    for r in results:
        m = r.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Classifier Evaluation Report")
    print(f"{'='*60}")
    print(f"  Total queries:    {len(queries)}")
    print(f"  Errors:           {errors}")
    print(f"  Accuracy:         {accuracy*100:.1f}% ({correct}/{total})")
    print(f"  Avg latency:      {sum(latencies)/len(latencies):.0f}ms" if latencies else "  Avg latency:      N/A")
    print(f"  P50 latency:      {sorted(latencies)[int(0.50*len(latencies))]:.0f}ms" if latencies else "")
    print(f"  P95 latency:      {sorted(latencies)[int(0.95*len(latencies))]:.0f}ms" if latencies else "")
    print(f"  P99 latency:      {sorted(latencies)[int(0.99*len(latencies))]:.0f}ms" if latencies else "")
    if classify_times:
        print(f"  Avg classify_ms:  {sum(classify_times)/len(classify_times):.0f}ms")
    print(f"  Method breakdown: {json.dumps(method_counts)}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}\n")

    # Also compute sklearn metrics if available
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        target_names = ["non-medical", "medical"]
        print("Scikit-learn classification report:")
        print(classification_report(y_true, y_pred, target_names=target_names))
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion matrix:\n{cm}")
    except ImportError:
        print("Install scikit-learn for detailed metrics: pip install scikit-learn")

    with open(args.output, "w") as f:
        json.dump({
            "summary": {
                "total": len(queries),
                "errors": errors,
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "avg_latency_ms": round(sum(latencies)/len(latencies), 1) if latencies else 0,
                "p50_latency_ms": round(sorted(latencies)[int(0.50*len(latencies))], 1) if latencies else 0,
                "p95_latency_ms": round(sorted(latencies)[int(0.95*len(latencies))], 1) if latencies else 0,
                "p99_latency_ms": round(sorted(latencies)[int(0.99*len(latencies))], 1) if latencies else 0,
                "method_breakdown": method_counts,
            },
            "results": results,
        }, f, indent=2)

    sys.exit(0 if accuracy >= 0.5 else 1)


if __name__ == "__main__":
    main()
