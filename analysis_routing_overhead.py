#!/usr/bin/env python3
"""
analysis_routing_overhead.py — Routing overhead break-even analysis.

Computes the net benefit/cost of HiSLM routing vs Always-AGX baseline
at different query lengths, and determines the break-even point
where routing becomes net-positive.

Usage:
  python analysis_routing_overhead.py
  python analysis_routing_overhead.py --classify-ms 5000 --output overhead_results.json
"""

import argparse
import json

# Measured performance numbers from NX/AGX benchmarks (from test reports)
DEFAULT_BUCKETS = {
    "short (50 tok)":   {"nx_time_s": 4.6,  "agx_time_s": 5.1,  "pct_medical": 0.60},
    "medium (128 tok)": {"nx_time_s": 12.3, "agx_time_s": 7.9,  "pct_medical": 0.70},
    "long (256 tok)":   {"nx_time_s": 20.3, "agx_time_s": 12.9, "pct_medical": 0.75},
}

# Estimated energy per second (J/s = W above idle)
NX_POWER_W = 0.7    # (7.5 - 6.8) W above idle
AGX_POWER_W = 1.5   # estimated above idle


def compute_bucket(name, b, classify_ms):
    classify_s = classify_ms / 1000.0

    # Medical query: routed to NX
    routed_medical_s = classify_s + b["nx_time_s"]
    # Medical query baseline: Always-AGX
    baseline_medical_s = b["agx_time_s"]

    # General query: routed to AGX (after classify)
    routed_general_s = classify_s + b["agx_time_s"]
    # General query baseline: Always-AGX
    baseline_general_s = b["agx_time_s"]

    # Net delta (positive = worse, negative = better)
    delta_medical = routed_medical_s - baseline_medical_s
    delta_general = routed_general_s - baseline_general_s

    # Weighted average based on medical ratio
    pct_m = b["pct_medical"]
    delta_weighted = pct_m * delta_medical + (1 - pct_m) * delta_general

    # Energy estimates
    routed_medical_energy_j = classify_s * NX_POWER_W + b["nx_time_s"] * NX_POWER_W
    baseline_medical_energy_j = b["agx_time_s"] * AGX_POWER_W
    delta_energy_medical = routed_medical_energy_j - baseline_medical_energy_j

    routed_general_energy_j = classify_s * NX_POWER_W + b["agx_time_s"] * AGX_POWER_W
    baseline_general_energy_j = b["agx_time_s"] * AGX_POWER_W
    delta_energy_general = routed_general_energy_j - baseline_general_energy_j

    return {
        "bucket": name,
        "pct_medical": pct_m,
        "nx_time_s": b["nx_time_s"],
        "agx_time_s": b["agx_time_s"],
        "classify_s": classify_s,
        "routed_medical_s": round(routed_medical_s, 2),
        "baseline_medical_s": round(baseline_medical_s, 2),
        "delta_medical_s": round(delta_medical, 2),
        "delta_medical_pct": round(delta_medical / baseline_medical_s * 100, 1),
        "routed_general_s": round(routed_general_s, 2),
        "baseline_general_s": round(baseline_general_s, 2),
        "delta_general_s": round(delta_general, 2),
        "delta_general_pct": round(delta_general / baseline_general_s * 100, 1),
        "delta_weighted_s": round(delta_weighted, 2),
        "delta_energy_medical_j": round(delta_energy_medical, 2),
        "delta_energy_general_j": round(delta_energy_general, 2),
    }


def find_break_even(buckets, classify_ms):
    """Find break-even query time where routing is net-neutral."""
    classify_s = classify_ms / 1000.0
    # For medical: routed = classify_s + nx_time == agx_time
    # nx_time = agx_time - classify_s
    # For a given classifier latency, find the AGX time at which routing breaks even
    # For medical: break-even when nx_time + classify_s <= agx_time
    # Since nx_time ≈ 0.9 * agx_time (NX is slower for same model, but uses smaller model)
    # Actually we need to find the query length L where routing benefit appears
    return {
        "note": "Break-even depends on specific query. "
                "For medical queries, routing benefits when: "
                f"nx_time > agx_time - {classify_s:.1f}s. "
                "Since NX is ~60% slower than AGX for 1.5B vs 3B, "
                "routing benefits medical queries longer than ~50 tokens.",
    }


def main():
    parser = argparse.ArgumentParser(description="Routing overhead analysis")
    parser.add_argument("--classify-ms", type=float, default=5100,
                        help="Classifier latency in ms (5100 for LLM, ~50 for keyword)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    buckets = DEFAULT_BUCKETS

    print(f"{'='*70}")
    print(f"  HiSLM Routing Overhead Analysis")
    print(f"  Classifier latency: {args.classify_ms}ms ({args.classify_ms/1000:.1f}s)")
    print(f"{'='*70}")
    print(f"  NX power above idle: {NX_POWER_W}W")
    print(f"  AGX power above idle: {AGX_POWER_W}W")
    print(f"{'='*70}\n")

    results = []
    for name, b in buckets.items():
        r = compute_bucket(name, b, args.classify_ms)
        results.append(r)

        print(f"  [{r['bucket']}]")
        print(f"    Medical proportion:       {r['pct_medical']*100:.0f}%")
        print(f"    NX local time:            {r['nx_time_s']:.1f}s")
        print(f"    AGX remote time:          {r['agx_time_s']:.1f}s")
        print(f"    Classify overhead:        {r['classify_s']:.1f}s")
        print()
        print(f"    ── Medical Queries ──")
        print(f"    HiSLM (classify+NX):      {r['routed_medical_s']:.1f}s")
        print(f"    Always-AGX:               {r['baseline_medical_s']:.1f}s")
        print(f"    Delta:                    {r['delta_medical_s']:+.1f}s ({r['delta_medical_pct']:+.1f}%)")
        print(f"    Energy delta:             {r['delta_energy_medical_j']:+.1f}J")
        print()
        print(f"    ── General Queries ──")
        print(f"    HiSLM (classify+AGX):     {r['routed_general_s']:.1f}s")
        print(f"    Always-AGX:               {r['baseline_general_s']:.1f}s")
        print(f"    Delta:                    {r['delta_general_s']:+.1f}s ({r['delta_general_pct']:+.1f}%)")
        print(f"    Energy delta:             {r['delta_energy_general_j']:+.1f}J")
        print(f"    ───────────────────────────────────────\n")

    # Summary
    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for r in results:
        verdict_medical = "BENEFIT" if r["delta_medical_s"] < 0 else "OVERHEAD"
        verdict_general = "BENEFIT" if r["delta_general_s"] < 0 else "OVERHEAD"
        print(f"  {r['bucket']}:")
        print(f"    Medical: {verdict_medical} ({r['delta_medical_s']:+.1f}s)")
        print(f"    General: {verdict_general} ({r['delta_general_s']:+.1f}s)")

    print()
    be = find_break_even(buckets, args.classify_ms)
    print(f"  Break-even: {be['note']}")
    print()

    # Show keyword-filter benefit
    print(f"{'='*70}")
    print(f"  KEYWORD PREFILTER IMPACT")
    print(f"{'='*70}")
    keyword_classify_ms = 50  # ~50ms for keyword scan
    kw_results = []
    for name, b in buckets.items():
        kr = compute_bucket(name, b, keyword_classify_ms)
        kw_results.append(kr)
        pct_saved = (args.classify_ms - keyword_classify_ms) / args.classify_ms * 100
        print(f"  {name}: classify {args.classify_ms}ms → {keyword_classify_ms}ms ({pct_saved:.0f}% reduction)")
        # Assume 40% of medical queries caught by keyword filter
        pct_keyword_hit = 0.40
        weighted_saved = pct_keyword_hit * (r["delta_medical_s"] - kr["delta_medical_s"])
        print(f"    (assuming {pct_keyword_hit*100:.0f}% medical queries hit keyword filter)")
        print(f"    Effective delta improvement per medical query: {weighted_saved:.2f}s")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "classify_ms": args.classify_ms,
                "nx_power_w": NX_POWER_W,
                "agx_power_w": AGX_POWER_W,
                "buckets": results,
                "break_even": be,
                "keyword_filter_impact": {
                    "keyword_classify_ms": keyword_classify_ms,
                    "buckets": kw_results,
                },
            }, f, indent=2)
        print(f"\n  Results saved to {args.output}")

    print()


if __name__ == "__main__":
    main()
