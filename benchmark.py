#!/usr/bin/env python3
"""Fair benchmark: ohe-rs vs state-of-the-art one-hot encoding.

Cartesian product of all dimensions:
  - Category discovery: YES / NO (num_classes pre-known)
  - H2D transfer: YES / NO (data pre-loaded on GPU)
  - D2H transfer: YES / NO (output stays on GPU)
  - Device: CPU / GPU

Fair protocol: warm-up excluded, GC disabled, 7 repeats (median), uint8.
"""

import gc
import time
import platform
import subprocess
import os
import numpy as np
from scipy.sparse import csr_matrix

REPEATS = 7
SEED = 42

def bench(name, fn, n_rows, repeats=REPEATS):
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
    print(f"    {name:55s} {median*1000:8.1f} ms  ({rate:7.1f} M rows/s)")
    return median, result

def verify(matrix, label):
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    ok = True
    if matrix.nnz != matrix.shape[0]:
        print(f"      FAIL {label}: nnz={matrix.nnz} != N={matrix.shape[0]}")
        ok = False
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    if not np.all(row_sums == 1):
        print(f"      FAIL {label}: {np.sum(row_sums != 1)} rows don't sum to 1")
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
                print(f"  Cores:     {line.split(':',1)[1].strip()}")
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
    print(f"  Python:    {platform.python_version()}, NumPy: {np.__version__}")
    print(f"  Threads:   {os.cpu_count()} (rayon uses all by default)")

scenarios = [
    ("Low cardinality",   10_000_000,      10),
    ("Med cardinality",   10_000_000,   1_000),
    ("High cardinality",  10_000_000, 100_000),
]

print("=" * 90)
print("  FAIR BENCHMARK — one-hot encoding (sparse CSR/COO)")
print(f"  Protocol: {REPEATS} repeats, median, warm-up excluded, GC off, uint8")
print()
print_system_info()
print("=" * 90)

for scenario_name, N, K in scenarios:
    rng = np.random.default_rng(SEED)
    data_int = rng.integers(0, K, size=N, dtype=np.int64)
    data_2d = data_int.reshape(-1, 1)

    print(f"\n{'━'*90}")
    print(f"  {scenario_name}: N={N:,}, K={K:,}")
    print(f"{'━'*90}")

    all_ok = True

    # ==================================================================
    # A) END-TO-END: raw data → usable result (discovery + encoding)
    # ==================================================================
    print(f"\n  A) END-TO-END (discovery=YES)")
    print(f"  ─────────────────────────────")

    from ohe_rs import encode_sparse

    t, (v, idx, ptr, nc) = bench(
        "ohe-rs CPU  [discovery + encode]",
        lambda: encode_sparse(data_int), N)
    all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs e2e")

    try:
        from sklearn.preprocessing import OneHotEncoder
        def sklearn_e2e():
            enc = OneHotEncoder(sparse_output=True, dtype=np.uint8)
            return enc.fit_transform(data_2d)
        t, m = bench("sklearn      [fit_transform]", sklearn_e2e, N)
        all_ok &= verify(m, "sklearn e2e")
    except ImportError:
        pass

    # ==================================================================
    # B) TRANSFORM-ONLY: K known, input=0..K-1, no discovery
    # ==================================================================
    print(f"\n  B) TRANSFORM-ONLY (discovery=NO, K={K:,})")
    print(f"  ──────────────────────────────────────────")

    # --- CPU ---
    print(f"    ── CPU ──")

    t, (v, idx, ptr, nc) = bench(
        "ohe-rs CPU  [encode only]",
        lambda: encode_sparse(data_int, num_classes=K), N)
    all_ok &= verify(csr_matrix((v, idx, ptr), shape=(N, nc)), "ohe-rs cpu transform")

    try:
        from sklearn.preprocessing import OneHotEncoder
        enc_pre = OneHotEncoder(sparse_output=True, dtype=np.uint8)
        enc_pre.fit(data_2d)
        t, m = bench("sklearn      [transform, prefitted]", lambda: enc_pre.transform(data_2d), N)
        all_ok &= verify(m, "sklearn transform")
    except ImportError:
        pass

    try:
        import torch
        t_cpu = torch.from_numpy(data_int)

        def torch_sparse_coo_cpu():
            row_idx = torch.arange(N, dtype=torch.long)
            indices = torch.stack([row_idx, t_cpu])
            values = torch.ones(N, dtype=torch.uint8)
            return torch.sparse_coo_tensor(indices, values, (N, K))
        t, _ = bench("PyTorch      [sparse COO construct]", torch_sparse_coo_cpu, N)
    except ImportError:
        torch = None

    # --- GPU: with H2D, with D2H ---
    print(f"    ── GPU (H2D=YES, D2H=YES) ──")

    try:
        from ohe_rs import gpu_available
        if gpu_available():
            from ohe_rs import gpu_encode_sparse
            t, (v, idx, ptr, nc) = bench(
                "ohe-rs GPU   [H2D + kernel + D2H]",
                lambda: gpu_encode_sparse(data_int), N)
    except ImportError:
        pass

    if torch and torch.cuda.is_available():
        def torch_gpu_h2d_d2h():
            t_gpu = t_cpu.cuda()
            row_idx = torch.arange(N, dtype=torch.long, device='cuda')
            idx_t = torch.stack([row_idx, t_gpu])
            vals = torch.ones(N, dtype=torch.uint8, device='cuda')
            out = torch.sparse_coo_tensor(idx_t, vals, (N, K))
            # D2H: coalesce + move to CPU to materialize result
            out_cpu = out.coalesce().to('cpu')
            torch.cuda.synchronize()
            return out_cpu
        t, _ = bench("PyTorch GPU  [H2D + construct + D2H]", torch_gpu_h2d_d2h, N)

    # --- GPU: pre-loaded input, with D2H ---
    print(f"    ── GPU (H2D=NO, D2H=YES) — input pre-loaded ──")

    try:
        if gpu_available():
            from ohe_rs import gpu_upload, gpu_encode_sparse_preloaded
            gpu_buf = gpu_upload(data_int)
            t, (v, idx, ptr, nc) = bench(
                "ohe-rs GPU   [kernel + D2H]",
                lambda: gpu_encode_sparse_preloaded(gpu_buf, K), N)
    except ImportError:
        pass

    if torch and torch.cuda.is_available():
        t_gpu_pre = t_cpu.cuda()
        row_pre = torch.arange(N, dtype=torch.long, device='cuda')
        vals_pre = torch.ones(N, dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()

        def torch_gpu_preloaded_d2h():
            idx_t = torch.stack([row_pre, t_gpu_pre])
            out = torch.sparse_coo_tensor(idx_t, vals_pre, (N, K))
            out_cpu = out.coalesce().to('cpu')
            torch.cuda.synchronize()
            return out_cpu
        t, _ = bench("PyTorch GPU  [construct + D2H]", torch_gpu_preloaded_d2h, N)

    # --- GPU: pre-loaded input, output stays on GPU ---
    print(f"    ── GPU (H2D=NO, D2H=NO) — all on device ──")

    try:
        if gpu_available():
            from ohe_rs.ohe_rs import gpu_encode_sparse_kernel_only_py
            t, _ = bench(
                "ohe-rs GPU   [kernel only, output on GPU]",
                lambda: gpu_encode_sparse_kernel_only_py(gpu_buf), N)
            del gpu_buf
    except ImportError:
        pass

    if torch and torch.cuda.is_available():
        def torch_gpu_all_on_device():
            idx_t = torch.stack([row_pre, t_gpu_pre])
            out = torch.sparse_coo_tensor(idx_t, vals_pre, (N, K))
            torch.cuda.synchronize()
            return out
        t, _ = bench("PyTorch GPU  [construct only, on device]", torch_gpu_all_on_device, N)

        del t_gpu_pre, row_pre, vals_pre
        torch.cuda.empty_cache()

    # ── Correctness ──
    status = "ALL PASS" if all_ok else "FAILURES DETECTED"
    print(f"\n  Correctness: {status}")

print(f"\n{'='*90}")
print(f"  Benchmark complete.")
print(f"{'='*90}\n")
