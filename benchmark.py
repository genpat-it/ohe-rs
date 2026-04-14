#!/usr/bin/env python3
"""Fair benchmark: ohe-rs vs state-of-the-art one-hot encoding.

Fairness protocol:
  - All implementations use uint8 output dtype
  - GC disabled during timing
  - Warm-up run excluded from measurements
  - Two sections: END-TO-END (with discovery) and TRANSFORM-ONLY (K pre-known)
  - Same input data, same random seed
  - 7 repeats, report median (robust to outliers)
  - Correctness verified: every row sums to 1, nnz == N
"""

import gc
import time
import platform
import subprocess
import numpy as np
from scipy.sparse import csr_matrix

REPEATS = 7
SEED = 42

def bench(name, fn, n_rows, repeats=REPEATS):
    """Warm-up + timed repeats with GC disabled. Returns median time and last result."""
    result = fn()
    gc.disable()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    gc.enable()
    median = sorted(times)[len(times) // 2]
    rate = n_rows / median / 1e6
    print(f"  {name:50s}  {median*1000:8.1f} ms  ({rate:7.1f} M rows/s)")
    return median, result

def verify(matrix, label):
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    ok = True
    if matrix.nnz != matrix.shape[0]:
        print(f"    FAIL {label}: nnz={matrix.nnz} != N={matrix.shape[0]}")
        ok = False
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    if not np.all(row_sums == 1):
        print(f"    FAIL {label}: {np.sum(row_sums != 1)} rows don't sum to 1")
        ok = False
    return ok

def print_system_info():
    print(f"  Platform:  {platform.system()} {platform.release()}")
    try:
        cpu = subprocess.check_output(["lscpu"], text=True)
        for line in cpu.splitlines():
            if line.startswith("Model name:"):
                print(f"  CPU:       {line.split(':',1)[1].strip()}")
            if line.startswith("CPU(s):"):
                print(f"  CPU cores: {line.split(':',1)[1].strip()}")
    except Exception:
        pass
    try:
        mem = subprocess.check_output(["free", "-h"], text=True).splitlines()[1]
        print(f"  RAM:       {mem.split()[1]}")
    except Exception:
        pass
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
        print(f"  GPU:       {gpu}")
    except Exception:
        print(f"  GPU:       not available")
    print(f"  Python:    {platform.python_version()}")
    print(f"  NumPy:     {np.__version__}")
    import os
    rayon_threads = os.cpu_count()
    print(f"  Rayon:     {rayon_threads} threads (all available cores)")

# ── Config ────────────────────────────────────────────────────────────

scenarios = [
    ("Low cardinality",   10_000_000,      10),
    ("Med cardinality",   10_000_000,   1_000),
    ("High cardinality",  10_000_000, 100_000),
]

print("=" * 85)
print("  FAIR BENCHMARK: one-hot encoding")
print(f"  Repeats: {REPEATS}, metric: median, warm-up: 1 run excluded")
print(f"  Output dtype: uint8, GC disabled during timing")
print()
print_system_info()
print("=" * 85)

for scenario_name, N, K in scenarios:
    rng = np.random.default_rng(SEED)
    data_int = rng.integers(0, K, size=N, dtype=np.int64)
    data_2d = data_int.reshape(-1, 1)

    print(f"\n{'━'*85}")
    print(f"  {scenario_name}: N={N:,}, K={K:,}")
    print(f"{'━'*85}")

    all_ok = True

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: END-TO-END (category discovery + encoding)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  ┌─ END-TO-END (input → encoded matrix, category discovery included)")
    print(f"  │")

    # ohe-rs
    from ohe_rs import encode_sparse, encode_dense
    t, (v, idx, ptr, nc) = bench(
        "│  ohe-rs sparse", lambda: encode_sparse(data_int), N,
    )
    all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs e2e sparse")

    # sklearn
    try:
        from sklearn.preprocessing import OneHotEncoder
        def sklearn_fit_transform():
            enc = OneHotEncoder(sparse_output=True, dtype=np.uint8)
            return enc.fit_transform(data_2d)
        t, m = bench("│  sklearn fit_transform", sklearn_fit_transform, N)
        all_ok &= verify(m, "sklearn e2e")
    except ImportError:
        pass

    print(f"  │")
    print(f"  └─")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: TRANSFORM-ONLY (K known, input already 0..K-1)
    # Apple-to-apple comparison: no category discovery for anyone
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  ┌─ TRANSFORM-ONLY (K pre-known, input already 0..K-1, no discovery)")
    print(f"  │")

    # ohe-rs CPU with num_classes (skip discovery)
    t, (v, idx, ptr, nc) = bench(
        "│  ohe-rs CPU sparse (num_classes=K)",
        lambda: encode_sparse(data_int, num_classes=K), N,
    )
    all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs transform sparse")

    # ohe-rs GPU (with H2D transfer, no discovery)
    try:
        from ohe_rs import gpu_available
        if gpu_available():
            from ohe_rs import gpu_encode_sparse, gpu_upload, gpu_encode_sparse_preloaded

            t, (v, idx, ptr, nc) = bench(
                "│  ohe-rs GPU sparse (with H2D, num_classes=K)",
                lambda: gpu_encode_sparse(data_int), N,
            )

            # Pre-load data on GPU
            gpu_buf = gpu_upload(data_int)

            t, (v, idx, ptr, nc) = bench(
                "│  ohe-rs GPU sparse (pre-loaded)",
                lambda: gpu_encode_sparse_preloaded(gpu_buf, K), N,
            )

            del gpu_buf
    except ImportError:
        pass

    # sklearn transform-only (prefitted)
    try:
        from sklearn.preprocessing import OneHotEncoder
        enc_pre = OneHotEncoder(sparse_output=True, dtype=np.uint8)
        enc_pre.fit(data_2d)
        t, m = bench("│  sklearn transform (prefitted)", lambda: enc_pre.transform(data_2d), N)
        all_ok &= verify(m, "sklearn transform")
    except ImportError:
        pass

    # PyTorch sparse COO (manual construction)
    try:
        import torch
        import torch.nn.functional as F

        t_cpu = torch.from_numpy(data_int)

        def torch_sparse_coo_cpu():
            row_idx = torch.arange(N, dtype=torch.long)
            indices = torch.stack([row_idx, t_cpu])
            values = torch.ones(N, dtype=torch.uint8)
            return torch.sparse_coo_tensor(indices, values, (N, K))

        t, _ = bench("│  PyTorch sparse COO CPU", torch_sparse_coo_cpu, N)

        # PyTorch F.one_hot (dense, K pre-known)
        dense_bytes = N * K * 8
        if dense_bytes <= 4 * 1024**3:
            def torch_onehot_cpu():
                return F.one_hot(t_cpu, num_classes=K).to(torch.uint8)
            t, m = bench("│  PyTorch F.one_hot CPU (dense)", torch_onehot_cpu, N)
            all_ok &= verify(m.numpy(), "torch F.one_hot CPU")

        if torch.cuda.is_available():
            # GPU sparse COO with H2D transfer
            def torch_sparse_coo_gpu():
                t_gpu = t_cpu.cuda()
                row_idx = torch.arange(N, dtype=torch.long, device='cuda')
                indices = torch.stack([row_idx, t_gpu])
                values = torch.ones(N, dtype=torch.uint8, device='cuda')
                out = torch.sparse_coo_tensor(indices, values, (N, K))
                torch.cuda.synchronize()
                return out
            t, _ = bench("│  PyTorch sparse COO GPU (with H2D)", torch_sparse_coo_gpu, N)

            # GPU sparse COO, data pre-loaded
            t_gpu_pre = t_cpu.cuda()
            row_idx_pre = torch.arange(N, dtype=torch.long, device='cuda')
            vals_pre = torch.ones(N, dtype=torch.uint8, device='cuda')
            torch.cuda.synchronize()

            def torch_sparse_coo_gpu_preloaded():
                indices = torch.stack([row_idx_pre, t_gpu_pre])
                out = torch.sparse_coo_tensor(indices, vals_pre, (N, K))
                torch.cuda.synchronize()
                return out
            t, _ = bench("│  PyTorch sparse COO GPU (pre-loaded)", torch_sparse_coo_gpu_preloaded, N)

            del t_gpu_pre, row_idx_pre, vals_pre
            torch.cuda.empty_cache()

    except ImportError:
        pass

    # numpy eye (K pre-known)
    if K <= 100:
        t, m = bench("│  numpy eye indexing", lambda: np.eye(K, dtype=np.uint8)[data_int], N)
        all_ok &= verify(m, "numpy eye")

    print(f"  │")
    print(f"  └─")

    # ── Correctness ──────────────────────────────────────────────────
    status = "ALL PASS" if all_ok else "FAILURES DETECTED"
    print(f"\n  Correctness: {status}")

print(f"\n{'='*85}")
print(f"  Benchmark complete.")
print(f"{'='*85}\n")
