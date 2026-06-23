#!/usr/bin/env python3
"""
measure_nx_queries.py — Send N queries to subserver and record latency.

Used in conjunction with tegrastats for energy measurement.

Usage:
  # Start tegrastats in another terminal:
  tegrastats --interval 200 --logfile /tmp/nx_inference.log

  # Run benchmark:
  python measure_nx_queries.py --server http://localhost:8765 --count 20

  # After done, stop tegrastats (Ctrl+C) and parse:
  python parse_tegrastats.py /tmp/nx_inference.log
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


QUERIES = [
    # Short medical (5 tokens)
    "What is hypertension?",
    "Explain diabetes treatment.",
    "What causes asthma?",
    "Describe heart disease symptoms.",
    "How is pneumonia treated?",
    # Medium medical (25-50 tokens)
    "What are the common symptoms of type 2 diabetes and how is it diagnosed?",
    "Explain the treatment options for high blood pressure including lifestyle changes.",
    "What causes chronic lower back pain and when should surgery be considered?",
    "Describe the difference between bacterial and viral pneumonia symptoms.",
    "How does insulin resistance develop and what are the early warning signs?",
    # Short general
    "What is the capital of France?",
    "Write a Python function.",
    "Explain gravity simply.",
    "How does a car engine work?",
    "What is machine learning?",
    # Greetings
    "Hello!",
    "Hi there, how are you?",
    "Good morning!",
    # Mixed length
    "What is the difference between SQL and NoSQL databases and when would you use each?",
    "Explain how climate change affects global weather patterns and agricultural production.",
    "What are the side effects of common blood pressure medications like ACE inhibitors?",
    "Describe the process of photosynthesis and why it is important for life on Earth.",
    "How do vaccines work at the cellular level to provide immunity against diseases?",
]


def main():
    parser = argparse.ArgumentParser(description="NX query benchmark")
    parser.add_argument("--server", default="http://localhost:8765",
                        help="Subserver base URL")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of queries to send")
    parser.add_argument("--output", default="nx_benchmark_results.json",
                        help="Output results file")
    args = parser.parse_args()

    queries = QUERIES[:args.count]
    print(f"Sending {len(queries)} queries to {args.server}/chat ...\n")

    results = []
    for i, q in enumerate(queries):
        t0 = time.time()
        try:
            r = requests.post(
                f"{args.server}/chat",
                json={"message": q, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            body = r.json()
            content_len = len(body.get("content", ""))
            timing = body.get("timing_ms", {})
            source = body.get("source", "?")
            results.append({
                "query_idx": i,
                "query": q[:60],
                "latency_s": round(elapsed, 2),
                "response_len": content_len,
                "source": source,
                "timing": timing,
            })
            print(f"  [{i+1}/{len(queries)}] {elapsed:6.2f}s  {source:3s}  "
                  f"{content_len:4d}chars  {q[:50]}")
        except Exception as exc:
            print(f"  [{i+1}/{len(queries)}] ERROR: {exc}")
            results.append({
                "query_idx": i,
                "query": q[:60],
                "latency_s": 0,
                "error": str(exc),
            })

    # Summary
    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    if latencies:
        print(f"\n{'='*50}")
        print(f"  Summary ({len(latencies)} successful queries)")
        print(f"  Avg latency:    {sum(latencies)/len(latencies):.2f}s")
        print(f"  Min latency:    {min(latencies):.2f}s")
        print(f"  Max latency:    {max(latencies):.2f}s")
        print(f"  P50 latency:    {sorted(latencies)[int(0.50*len(latencies))]:.2f}s")
        print(f"  P95 latency:    {sorted(latencies)[int(0.95*len(latencies))]:.2f}s")
        print(f"  Total time:     {sum(latencies):.2f}s")

        # Count by source
        nx_count = sum(1 for r in results if r.get("source") == "NX")
        agx_count = sum(1 for r in results if r.get("source") == "AGX")
        print(f"  NX responses:   {nx_count}")
        print(f"  AGX responses:  {agx_count}")

    with open(args.output, "w") as f:
        json.dump({
            "config": {"server": args.server, "count": args.count},
            "summary": {
                "total": len(results),
                "successful": len(latencies),
                "avg_latency_s": round(sum(latencies)/len(latencies), 2) if latencies else 0,
                "p50_latency_s": round(sorted(latencies)[int(0.50*len(latencies))], 2) if latencies else 0,
                "p95_latency_s": round(sorted(latencies)[int(0.95*len(latencies))], 2) if latencies else 0,
                "total_time_s": round(sum(latencies), 2) if latencies else 0,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
