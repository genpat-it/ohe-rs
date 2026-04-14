#!/usr/bin/env python3
"""Thread scaling benchmark for ohe-rs.

Tests how performance scales with 1, 2, 4, 8, 16, 32, 64, 80 threads.
NOTE: rayon's global thread pool can only be set once per process,
so we spawn a subprocess for each thread count.
"""

import subprocess
import sys
import os
import platform
import numpy as np

REPEATS = 7
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WORKER_SCRIPT = '''
import gc, time, sys, numpy as np
from ohe_rs import encode_sparse, set_threads

REPEATS = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
THREADS = int(sys.argv[4])
MODE = sys.argv[5]  # "e2e" or "transform"
SEED = 42

set_threads(THREADS)
rng = np.random.default_rng(SEED)
data = rng.integers(0, K, size=N, dtype=np.int64)

if MODE == "transform":
    fn = lambda: encode_sparse(data, num_classes=K)
else:
    fn = lambda: encode_sparse(data)

# warm-up
fn()

gc.disable()
times = []
for _ in range(REPEATS):
    gc.collect()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    times.append(t1 - t0)
gc.enable()

median = sorted(times)[len(times) // 2]
print(f"{median:.6f}")
'''

def run_bench(n, k, threads, mode):
    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, str(REPEATS), str(n), str(k), str(threads), mode],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return None
    return float(result.stdout.strip())

scenarios = [
    ("K=10",      10_000_000,      10),
    ("K=1,000",   10_000_000,   1_000),
    ("K=100,000", 10_000_000, 100_000),
]

thread_counts = [1, 2, 4, 8, 16, 32, 64, 80]

print("=" * 95)
print("  THREAD SCALING BENCHMARK — ohe-rs sparse encoding")
print(f"  N=10,000,000 rows, {REPEATS} repeats (median), warm-up excluded")
print()

# System info
try:
    import subprocess as sp
    cpu = sp.check_output(["lscpu"], text=True)
    for line in cpu.splitlines():
        if line.startswith("Model name:"):
            print(f"  CPU: {line.split(':',1)[1].strip()}")
        if line.startswith("CPU(s):"):
            print(f"  Total cores: {line.split(':',1)[1].strip()}")
except Exception:
    pass

print("=" * 95)

for mode, mode_label in [("e2e", "END-TO-END (discovery + encode)"), ("transform", "TRANSFORM-ONLY (num_classes=K)")]:
    print(f"\n  {mode_label}")
    print(f"  {'─'*85}")

    # Header
    header = f"  {'Threads':>8s}"
    for name, _, _ in scenarios:
        header += f"  {name:>18s}"
    header += f"  {'Speedup (1→80)':>16s}"
    print(header)
    print(f"  {'─'*8}" + f"  {'─'*18}" * len(scenarios) + f"  {'─'*16}")

    # Collect results for speedup calculation
    results = {name: {} for name, _, _ in scenarios}

    for t in thread_counts:
        row = f"  {t:>8d}"
        for name, n, k in scenarios:
            ms = run_bench(n, k, t, mode)
            if ms is not None:
                results[name][t] = ms
                rate = n / ms / 1e6
                row += f"  {ms*1000:>10.1f} ms {rate:>4.0f}M/s"
            else:
                row += f"  {'ERROR':>18s}"

        # Speedup column (vs single thread, for this row)
        if t == 80:
            speedups = []
            for name, _, _ in scenarios:
                if 1 in results[name] and 80 in results[name]:
                    sp = results[name][1] / results[name][80]
                    speedups.append(f"{sp:.1f}x")
                else:
                    speedups.append("?")
            row += f"  {'/'.join(speedups):>16s}"

        print(row)

    print()

# Summary: scaling efficiency
print(f"\n  SCALING EFFICIENCY (time_1thread / time_Nthreads / N)")
print(f"  {'─'*85}")
for mode, mode_label in [("e2e", "E2E"), ("transform", "Transform")]:
    print(f"\n  {mode_label}:")
    for name, n, k in scenarios:
        line = f"    {name:>10s}:"
        base = None
        for t in thread_counts:
            ms = run_bench(n, k, t, mode)
            if ms is not None:
                if t == 1:
                    base = ms
                    line += f"  1T={ms*1000:.0f}ms"
                elif base:
                    speedup = base / ms
                    efficiency = speedup / t * 100
                    line += f"  {t}T={efficiency:.0f}%"
        print(line)

print(f"\n{'='*95}\n")
