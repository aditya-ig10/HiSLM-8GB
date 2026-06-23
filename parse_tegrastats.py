#!/usr/bin/env python3
"""
parse_tegrastats.py — Parse tegrastats log and compute energy metrics.

Usage:
  # Record idle baseline (30s):
  tegrastats --interval 500 --logfile /tmp/nx_idle.log &
  sleep 30; kill %1

  # Record inference load (concurrent with measure_nx_queries.py):
  tegrastats --interval 200 --logfile /tmp/nx_inference.log &
  python measure_nx_queries.py --server http://localhost:8765 --count 20
  kill %1

  # Parse results:
  python parse_tegrastats.py /tmp/nx_idle.log --label idle
  python parse_tegrastats.py /tmp/nx_inference.log --label inference \
    --benchmark nx_benchmark_results.json
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def parse_tegrastats_line(line: str) -> dict | None:
    """Extract power and temperature from a tegrastats line.

    Example line:
      RAM 3986/7803MB (lfb 125x4MB) SWAP 1527/3755MB (cached 1263MB)
      CPU [33%@1190,29%@1190,33%@1190,28%@1190,25%@1190,23%@1190]
      EMC_FREQ 7%@2133 GR3D_FREQ 0%@130
      VDD_GPU_SOC 3776/3843 VDD_CPU_CV 1411/1448 VIN_SYS_5V0 6650/6675
      VDD_GPU_SOC 3756/3824 VDD_CPU_CV 1405/1442 VIN_SYS_5V0 6656/6681
      TEMP GPU 47.5C CPU 55.5C Tboard 41.0C Tdiode 48.0C

    Returns dict with extracted values.
    """
    data = {}

    # RAM
    ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if ram_match:
        data["ram_used_mb"] = int(ram_match.group(1))
        data["ram_total_mb"] = int(ram_match.group(2))

    # CPU frequency
    cpu_match = re.search(r'CPU\s+\[([^\]]+)\]', line)
    if cpu_match:
        freqs = re.findall(r'(\d+)%', cpu_match.group(1))
        if freqs:
            data["cpu_pct"] = [int(f) for f in freqs]
            data["cpu_pct_avg"] = statistics.mean(int(f) for f in freqs)

    # GPU freq
    gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if gpu_match:
        data["gpu_pct"] = int(gpu_match.group(1))

    # Power — Orin AGX format (VIN_SYS_5V0, VDD_GPU_SOC, VDD_CPU_CV)
    vin_match = re.search(r'VIN_SYS_5V0\s+(\d+)/', line)
    if vin_match:
        data["vin_sys_5v0_mw"] = int(vin_match.group(1))

    vdd_gpu_match = re.search(r'VDD_GPU_SOC\s+(\d+)/', line)
    if vdd_gpu_match:
        data["vdd_gpu_soc_mw"] = int(vdd_gpu_match.group(1))

    vdd_cpu_match = re.search(r'VDD_CPU_CV\s+(\d+)/', line)
    if vdd_cpu_match:
        data["vdd_cpu_cv_mw"] = int(vdd_cpu_match.group(1))

    # Power — Orin NX format (VDD_IN 7660mW/7660mW, VDD_CPU_GPU_CV, VDD_SOC)
    vdd_in_match = re.search(r'VDD_IN\s+(\d+)mW/', line)
    if vdd_in_match:
        data["vdd_in_mw"] = int(vdd_in_match.group(1))

    vdd_cpu_gpu_match = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW/', line)
    if vdd_cpu_gpu_match:
        data["vdd_cpu_gpu_mw"] = int(vdd_cpu_gpu_match.group(1))

    vdd_soc_match = re.search(r'VDD_SOC\s+(\d+)mW/', line)
    if vdd_soc_match:
        data["vdd_soc_mw"] = int(vdd_soc_match.group(1))

    # Temperatures — AGX format: GPU 47.5C CPU 55.5C
    temp_gpu = re.search(r'GPU\s+([\d.]+)C', line)
    if temp_gpu:
        data["temp_gpu_c"] = float(temp_gpu.group(1))

    temp_cpu = re.search(r'CPU\s+([\d.]+)C', line)
    if temp_cpu:
        data["temp_cpu_c"] = float(temp_cpu.group(1))

    # Temperatures — NX format: gpu@47.625C cpu@49.312C
    temp_gpu_nx = re.search(r'gpu@([\d.]+)C', line)
    if temp_gpu_nx and "temp_gpu_c" not in data:
        data["temp_gpu_c"] = float(temp_gpu_nx.group(1))

    temp_cpu_nx = re.search(r'cpu@([\d.]+)C', line)
    if temp_cpu_nx and "temp_cpu_c" not in data:
        data["temp_cpu_c"] = float(temp_cpu_nx.group(1))

    return data if data else None


def main():
    parser = argparse.ArgumentParser(description="Parse tegrastats log")
    parser.add_argument("logfile", help="Path to tegrastats log file")
    parser.add_argument("--label", default="measurement",
                        help="Label for this measurement")
    parser.add_argument("--benchmark", default=None,
                        help="Path to benchmark results JSON for query count")
    args = parser.parse_args()

    logfile = Path(args.logfile)
    if not logfile.exists():
        print(f"Error: {logfile} not found")
        return

    # Parse all lines
    samples = []
    with open(logfile) as f:
        for line in f:
            parsed = parse_tegrastats_line(line.strip())
            if parsed and ("vin_sys_5v0_mw" in parsed or "vdd_in_mw" in parsed):
                samples.append(parsed)

    if not samples:
        print("Error: No valid tegrastats samples found")
        return

    # Detect format: NX uses vdd_in_mw, AGX uses vin_sys_5v0_mw
    use_nx_format = "vdd_in_mw" in samples[0]
    use_agx_format = "vin_sys_5v0_mw" in samples[0]

    if use_nx_format:
        avg_total_mw = statistics.mean(s["vdd_in_mw"] for s in samples)
        avg_cpu_gpu_mw = statistics.mean(s.get("vdd_cpu_gpu_mw", 0) for s in samples)
        avg_soc_mw = statistics.mean(s.get("vdd_soc_mw", 0) for s in samples)
    elif use_agx_format:
        avg_total_mw = statistics.mean(s["vin_sys_5v0_mw"] for s in samples)
    else:
        avg_total_mw = 0

    avg_temp_gpu = statistics.mean(s.get("temp_gpu_c", 0) for s in samples)
    avg_temp_cpu = statistics.mean(s.get("temp_cpu_c", 0) for s in samples)
    avg_ram_mb = statistics.mean(s.get("ram_used_mb", 0) for s in samples)
    avg_gpu_pct = statistics.mean(s.get("gpu_pct", 0) for s in samples)

    # Load benchmark data if provided
    query_count = 0
    total_time_s = 0
    if args.benchmark:
        with open(args.benchmark) as f:
            bm = json.load(f)
        query_count = bm.get("summary", {}).get("successful", 0)
        total_time_s = bm.get("summary", {}).get("total_time_s", 0)

    # Power in Watts
    avg_power_w = avg_total_mw / 1000.0

    print(f"\n{'='*55}")
    print(f"  Tegrastats Power Analysis — {args.label}")
    print(f"{'='*55}")
    print(f"  Samples:            {len(samples)}")
    interval_s = 0.5 if len(samples) < 50 else 0.2  # guess based on sample count
    print(f"  Duration:           ~{len(samples) * interval_s:.0f}s")
    if use_nx_format:
        print(f"  Avg VDD_IN (total): {avg_total_mw:.0f} mW ({avg_power_w:.2f} W)")
        print(f"  Avg VDD_CPU_GPU:    {avg_cpu_gpu_mw:.0f} mW")
        print(f"  Avg VDD_SOC:        {avg_soc_mw:.0f} mW")
    elif use_agx_format:
        print(f"  Avg VIN_SYS_5V0:    {avg_power_w:.2f} W")
    print(f"  Avg GPU util:       {avg_gpu_pct:.1f}%")
    print(f"  Avg RAM used:       {avg_ram_mb:.0f} MB")
    print(f"  Avg GPU temp:       {avg_temp_gpu:.1f}°C")
    print(f"  Avg CPU temp:       {avg_temp_cpu:.1f}°C")

    if query_count > 0 and total_time_s > 0:
        # Energy = avg_power * total_time
        total_energy_j = avg_power_w * total_time_s
        energy_per_query_j = total_energy_j / query_count
        print(f"\n  ── Energy Metrics ──")
        print(f"  Queries:            {query_count}")
        print(f"  Total time:         {total_time_s:.1f}s")
        print(f"  Total energy:       {total_energy_j:.1f} J")
        print(f"  Energy per query:   {energy_per_query_j:.2f} J")

    # Also compute idle subtractions if comparing
    print(f"\n  Note: For marginal energy, subtract idle baseline power.")
    print(f"  Marginal power = P_inference - P_idle")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
