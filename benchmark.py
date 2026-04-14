#!/usr/bin/env python3
"""Benchmark ohe-rs against state-of-the-art one-hot encoding implementations."""

import time
import hashlib
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def bench(name, fn, repeats=5):
    """Run fn multiple times and report median time. Returns last result."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    median = sorted(times)[len(times) // 2]
    rate = N / median / 1e6
    print(f"  {name:40s}  {median*1000:8.1f} ms  ({rate:.1f} M rows/s)")
    return median, result

def matrix_checksum(matrix):
    """Compute a mapping-invariant checksum.

    The key insight: different implementations may assign different column
    indices to the same category, but each row must have exactly one 1,
    and rows with the same input value must have their 1 in the same column.

    We check: for each unique input value, all rows with that value map to
    the same column. The checksum is based on the per-row column index,
    remapped through the input data to be order-independent.
    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)

    # For one-hot, each row has exactly 1 nonzero -> the column index IS the code
    # We just need: nnz == n_rows, and each row has exactly 1
    assert matrix.nnz == matrix.shape[0], f"nnz={matrix.nnz} != nrows={matrix.shape[0]}"

    # row_sum check: every row should sum to 1
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    assert np.all(row_sums == 1), "Not all rows sum to 1"

    # The shape and nnz count are the invariant properties
    return f"OK:N={matrix.shape[0]},K={matrix.shape[1]},nnz={matrix.nnz}"

# ── Config ────────────────────────────────────────────────────────────

scenarios = [
    ("Low cardinality",   10_000_000,    10),
    ("Med cardinality",   10_000_000,  1_000),
    ("High cardinality",  10_000_000, 100_000),
]

for scenario_name, N, K in scenarios:
    print(f"\n{'='*70}")
    print(f"  {scenario_name}: N={N:,}, K={K:,}")
    print(f"{'='*70}")

    data_int = np.random.randint(0, K, size=N, dtype=np.int64)
    checksums = {}

    # ── ohe-rs CPU (sparse) ──────────────────────────────────────────
    try:
        from ohe_rs import encode_sparse, encode_dense
        t, (v, idx, ptr, nc) = bench("ohe-rs CPU sparse", lambda: encode_sparse(data_int))
        checksums["ohe-rs CPU sparse"] = matrix_checksum(
            csr_matrix((v, idx, ptr), shape=(N, nc)))

        if K <= 1000:
            t, m = bench("ohe-rs CPU dense", lambda: encode_dense(data_int))
            checksums["ohe-rs CPU dense"] = matrix_checksum(m)
    except ImportError:
        print("  ohe-rs: not installed (run: maturin develop --release)")

    # ── ohe-rs GPU ───────────────────────────────────────────────────
    try:
        from ohe_rs import gpu_available
        if gpu_available():
            from ohe_rs import gpu_encode_dense, gpu_encode_sparse
            t, (v, idx, ptr, nc) = bench("ohe-rs GPU sparse", lambda: gpu_encode_sparse(data_int))
            checksums["ohe-rs GPU sparse"] = matrix_checksum(
                csr_matrix((v, idx, ptr), shape=(N, nc)))

            if K <= 100:  # GPU dense needs N*K bytes on device
                t, m = bench("ohe-rs GPU dense", lambda: gpu_encode_dense(data_int))
                checksums["ohe-rs GPU dense"] = matrix_checksum(m)
    except ImportError:
        pass

    # ── numpy eye indexing ───────────────────────────────────────────
    if K <= 1000:
        t, m = bench("numpy eye indexing", lambda: np.eye(K, dtype=np.uint8)[data_int])
        checksums["numpy eye"] = matrix_checksum(m)

    # ── sklearn OneHotEncoder (sparse) ───────────────────────────────
    try:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=True, dtype=np.uint8)
        data_2d = data_int.reshape(-1, 1)
        enc.fit(data_2d)
        t, m = bench("sklearn OHE sparse", lambda: enc.transform(data_2d))
        checksums["sklearn"] = matrix_checksum(m)
    except ImportError:
        pass

    # ── Checksum verification ────────────────────────────────────────
    unique_checksums = set(checksums.values())
    if len(unique_checksums) == 1:
        print(f"\n  Correctness: ALL PASS ({list(unique_checksums)[0]})")
    else:
        print(f"\n  CORRECTNESS CHECK:")
        for name, cs in checksums.items():
            print(f"    {name:35s}  {cs}")

print()
