#!/usr/bin/env python3
"""Fair benchmark: ohe-rs vs state-of-the-art one-hot encoding.

Fairness protocol:
  - All implementations use uint8 output dtype
  - GC disabled during timing
  - Warm-up run excluded from measurements
  - fit() cost included for ALL methods (category discovery)
  - fit() and transform() timed separately for transparency
  - Same input data, same random seed
  - 7 repeats, report median (robust to outliers)
  - Correctness verified: every row sums to 1, nnz == N
"""

import gc
import time
import numpy as np
from scipy.sparse import csr_matrix

REPEATS = 7
SEED = 42

def bench(name, fn, repeats=REPEATS):
    """Warm-up + timed repeats with GC disabled. Returns median time and last result."""
    # Warm-up (excluded)
    result = fn()

    # Timed runs
    gc.disable()
    times = []
    for _ in range(repeats):
        gc.collect()  # clean state before each run
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    gc.enable()

    median = sorted(times)[len(times) // 2]
    rate = N / median / 1e6
    print(f"  {name:45s}  {median*1000:8.1f} ms  ({rate:7.1f} M rows/s)")
    return median, result

def verify(matrix, label):
    """Verify one-hot correctness."""
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    ok = True
    if matrix.nnz != matrix.shape[0]:
        print(f"    FAIL {label}: nnz={matrix.nnz} != N={matrix.shape[0]}")
        ok = False
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    if not np.all(row_sums == 1):
        bad = np.sum(row_sums != 1)
        print(f"    FAIL {label}: {bad} rows don't sum to 1")
        ok = False
    return ok

# ── Config ────────────────────────────────────────────────────────────

scenarios = [
    ("Low cardinality",   10_000_000,      10),
    ("Med cardinality",   10_000_000,   1_000),
    ("High cardinality",  10_000_000, 100_000),
]

print("=" * 78)
print("  FAIR BENCHMARK: one-hot encoding")
print(f"  Repeats: {REPEATS}, metric: median, warm-up: 1 run excluded")
print(f"  Output dtype: uint8, GC disabled during timing")
print("=" * 78)

for scenario_name, N, K in scenarios:
    rng = np.random.default_rng(SEED)
    data_int = rng.integers(0, K, size=N, dtype=np.int64)
    data_2d = data_int.reshape(-1, 1)

    print(f"\n{'─'*78}")
    print(f"  {scenario_name}: N={N:,}, K={K:,}")
    print(f"{'─'*78}")

    all_ok = True

    # ── ohe-rs: fit+transform (end-to-end, includes category discovery) ──
    print(f"\n  --- ohe-rs (Rust, parallel) ---")
    try:
        from ohe_rs import encode_sparse, encode_dense

        t, (v, idx, ptr, nc) = bench(
            "sparse (fit+transform)",
            lambda: encode_sparse(data_int)
        )
        all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs sparse")

        if K <= 100:
            t, m = bench(
                "dense (fit+transform)",
                lambda: encode_dense(data_int)
            )
            all_ok &= verify(m, "ohe-rs dense")
    except ImportError:
        print("  ohe-rs: not installed (run: maturin develop --release)")

    # ── ohe-rs GPU ───────────────────────────────────────────────────────
    try:
        from ohe_rs import gpu_available
        if gpu_available():
            from ohe_rs import gpu_encode_sparse, gpu_encode_dense
            print(f"\n  --- ohe-rs GPU (CUDA) ---")

            t, (v, idx, ptr, nc) = bench(
                "GPU sparse (fit+transform)",
                lambda: gpu_encode_sparse(data_int)
            )
            all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs GPU sparse")

            if K <= 100:
                t, m = bench(
                    "GPU dense (fit+transform)",
                    lambda: gpu_encode_dense(data_int)
                )
                all_ok &= verify(m, "ohe-rs GPU dense")
    except ImportError:
        pass

    # ── sklearn: fit+transform together (fair comparison) ────────────────
    print(f"\n  --- scikit-learn ---")
    try:
        from sklearn.preprocessing import OneHotEncoder

        # Fair: fit+transform together (category discovery included)
        def sklearn_fit_transform():
            enc = OneHotEncoder(sparse_output=True, dtype=np.uint8)
            return enc.fit_transform(data_2d)

        t, m = bench("sparse fit_transform", sklearn_fit_transform)
        all_ok &= verify(m, "sklearn fit_transform")

        # Also show transform-only for reference (categories pre-known)
        enc_prefitted = OneHotEncoder(sparse_output=True, dtype=np.uint8)
        enc_prefitted.fit(data_2d)
        t, m = bench("sparse transform-only (prefitted)", lambda: enc_prefitted.transform(data_2d))
        all_ok &= verify(m, "sklearn transform-only")

    except ImportError:
        print("  scikit-learn: not installed")

    # ── numpy eye indexing ───────────────────────────────────────────────
    if K <= 100:
        print(f"\n  --- numpy ---")
        # Fair: numpy eye requires knowing K upfront, so category discovery
        # is "free" (you must know K). We include the eye() matrix creation.
        t, m = bench(
            "eye indexing (K pre-known)",
            lambda: np.eye(K, dtype=np.uint8)[data_int]
        )
        all_ok &= verify(m, "numpy eye")

    # ── polars ───────────────────────────────────────────────────────────
    try:
        import polars as pl
        if K <= 100:
            print(f"\n  --- polars ---")
            df = pl.DataFrame({"x": data_int})
            t, _ = bench("to_dummies (fit+transform)", lambda: df.to_dummies())
    except ImportError:
        pass

    # ── Correctness ──────────────────────────────────────────────────────
    status = "ALL PASS" if all_ok else "FAILURES DETECTED"
    print(f"\n  Correctness: {status}")

print(f"\n{'='*78}")
print(f"  Benchmark complete.")
print(f"{'='*78}\n")
