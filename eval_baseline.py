#!/usr/bin/env python3
"""
eval_baseline.py — Compare Always-NX, Always-AGX, and HiSLM routing.

Runs the same eval set through all three modes and compares latency,
response quality, and energy estimates.

Usage:
  # AGX server must be running and reachable
  python eval_baseline.py --queries eval_routing.jsonl --output baseline_comparison.json
  python eval_baseline.py --queries eval_routing.jsonl --mode hislm --output hislm_only.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


NX_ENDPOINT = "http://localhost:8765"
AGX_ENDPOINT = "http://100.120.59.117:8000"

# Models used by each server
NX_MODEL = "Qwen2.5-1.5B-Instruct Q4_K_M"
AGX_MODEL = "Qwen2.5-3B-Instruct Q4_K_M"


def run_single_query(endpoint: str, query: str, timeout: int = 120) -> dict:
    """Send a chat query and return timing + response."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{endpoint}/chat",
            json={"message": query, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        elapsed_s = time.time() - t0
        body = r.json()
        content = body.get("content", "")
        timing = body.get("timing_ms", {})
        source = body.get("source", endpoint)
        return {
            "latency_s": round(elapsed_s, 2),
            "response": content.strip(),
            "response_len": len(content.strip()),
            "source": source,
            "timing_ms": timing,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "latency_s": time.time() - t0,
            "response": "",
            "response_len": 0,
            "source": "error",
            "timing_ms": {},
            "error": str(exc),
        }


def run_baseline(queries: list[dict], mode: str, agx_endpoint: str) -> list[dict]:
    """Run all queries through one mode.

    Modes:
      - always_nx:  force all queries to NX
      - always_agx: force all queries to AGX
      - hislm:      use subserver routing (POST /chat on NX)
    """
    results = []
    for i, q in enumerate(queries):
        if mode == "always_nx":
            # Direct to NX server's /chat endpoint
            res = run_single_query(NX_ENDPOINT, q["text"])
            res["model"] = NX_MODEL
        elif mode == "always_agx":
            # Direct to AGX server's /chat endpoint
            res = run_single_query(agx_endpoint, q["text"])
            res["model"] = AGX_MODEL
        elif mode == "hislm":
            # Hit subserver /chat — it will route automatically
            res = run_single_query(NX_ENDPOINT, q["text"])
            res["model"] = f"HiSLM (→{res['source']})"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        res["query_id"] = q["id"]
        res["query"] = q["text"][:60]
        res["label"] = q["label"]
        results.append(res)

        # Progress
        status = "OK" if res["error"] is None else f"ERR:{res['error'][:30]}"
        print(f"  [{i+1}/{len(queries)}] {status:30s}  {res['latency_s']:6.2f}s  "
              f"{res.get('model','?'):20s}  {res.get('response_len',0):4d}chars")

    return results


def compute_quality_metrics(all_results: dict[str, list[dict]]) -> dict:
    """Compute ROUGE-L-like quality proxy using response length ratio.

    For a proper eval, use ROUGE-L or BLEU vs reference answers.
    This is a simple proxy: compare response lengths and non-empty rate.
    """
    metrics = {}
    for mode, results in all_results.items():
        latencies = [r["latency_s"] for r in results if r["error"] is None]
        response_lens = [r["response_len"] for r in results if r["error"] is None]
        errors = sum(1 for r in results if r["error"] is not None)

        metrics[mode] = {
            "total": len(results),
            "successful": len(latencies),
            "errors": errors,
            "avg_latency_s": round(sum(latencies)/len(latencies), 2) if latencies else 0,
            "p50_latency_s": round(sorted(latencies)[int(0.50*len(latencies))], 2) if latencies else 0,
            "p95_latency_s": round(sorted(latencies)[int(0.95*len(latencies))], 2) if latencies else 0,
            "min_latency_s": round(min(latencies), 2) if latencies else 0,
            "max_latency_s": round(max(latencies), 2) if latencies else 0,
            "avg_response_len": round(sum(response_lens)/len(response_lens), 1) if response_lens else 0,
            "total_time_s": round(sum(latencies), 2) if latencies else 0,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument("--queries", default="eval_routing.jsonl",
                        help="Labeled queries JSONL")
    parser.add_argument("--mode", default="all",
                        choices=["all", "always_nx", "always_agx", "hislm"],
                        help="Which mode to run (default: all)")
    parser.add_argument("--agx-endpoint", default=AGX_ENDPOINT,
                        help="AGX server URL")
    parser.add_argument("--output", default="baseline_comparison_results.json",
                        help="Output results file")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"\n  Loaded {len(queries)} queries from {args.queries}")
    print(f"  NX endpoint:  {NX_ENDPOINT}/chat")
    print(f"  AGX endpoint: {args.agx_endpoint}/chat")
    print()

    modes = [args.mode] if args.mode != "all" else ["always_nx", "always_agx", "hislm"]
    all_results = {}

    for mode in modes:
        print(f"{'='*55}")
        print(f"  Mode: {mode}")
        print(f"{'='*55}")
        results = run_baseline(queries, mode, args.agx_endpoint)
        all_results[mode] = results
        print()

    metrics = compute_quality_metrics(all_results)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Mode':<20s} {'Avg Lat':>8s} {'P50':>8s} {'P95':>8s} "
          f"{'Avg Resp':>10s} {'Err':>5s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*5} {'-'*8}")
    for mode in modes:
        m = metrics[mode]
        print(f"  {mode:<20s} {m['avg_latency_s']:>7.1f}s {m['p50_latency_s']:>7.1f}s "
              f"{m['p95_latency_s']:>7.1f}s {m['avg_response_len']:>9.1f}c "
              f"{m['errors']:>4d}  {m['total_time_s']:>6.1f}s")

    # Compute relative to baselines
    if "always_nx" in metrics and "hislm" in metrics:
        h = metrics["hislm"]
        n = metrics["always_nx"]
        a = metrics.get("always_agx", {})
        print(f"\n  HiSLM vs Always-NX:  "
              f"{((h['avg_latency_s'] - n['avg_latency_s'])/n['avg_latency_s']*100):+.1f}% latency")
    if "always_agx" in metrics and "hislm" in metrics:
        h = metrics["hislm"]
        a = metrics["always_agx"]
        print(f"  HiSLM vs Always-AGX: "
              f"{((h['avg_latency_s'] - a['avg_latency_s'])/a['avg_latency_s']*100):+.1f}% latency")

    # Save
    output = {
        "config": {
            "nx_endpoint": NX_ENDPOINT,
            "agx_endpoint": args.agx_endpoint,
            "queries_file": args.queries,
            "modes_run": modes,
        },
        "metrics": metrics,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")
    print()


if __name__ == "__main__":
    main()
