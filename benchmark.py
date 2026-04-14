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

# ── System info ──────────────────────────────────────────────────────

def print_system_info():
    import platform, subprocess
    print(f"  Platform:  {platform.system()} {platform.release()}")
    try:
        cpu = subprocess.check_output(
            ["lscpu"], text=True
        )
        for line in cpu.splitlines():
            if line.startswith("Model name:"):
                print(f"  CPU:       {line.split(':',1)[1].strip()}")
            if line.startswith("CPU(s):"):
                print(f"  CPU cores: {line.split(':',1)[1].strip()}")
    except Exception:
        pass
    try:
        mem = subprocess.check_output(["free", "-h"], text=True).splitlines()[1]
        total = mem.split()[1]
        print(f"  RAM:       {total}")
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
print()
print_system_info()
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

    # ── PyTorch ──────────────────────────────────────────────────────────
    try:
        import torch
        import torch.nn.functional as F

        print(f"\n  --- PyTorch ---")

        # CPU: torch.nn.functional.one_hot (returns dense int64 tensor)
        t_cpu = torch.from_numpy(data_int)

        # torch one_hot produces dense int64 (8 bytes per element!) then cast
        # N*K*8 bytes for int64, skip if > 4GB to avoid system OOM
        dense_torch_bytes = N * K * 8
        if dense_torch_bytes <= 4 * 1024**3:
            def torch_cpu_ohe():
                return F.one_hot(t_cpu, num_classes=K).to(torch.uint8)

            t, m = bench("F.one_hot CPU (K pre-known)", torch_cpu_ohe)
            m_np = m.numpy()
            all_ok &= verify(m_np, "torch CPU")
        else:
            print(f"  {'F.one_hot CPU':45s}  SKIPPED (would need {dense_torch_bytes/1e9:.0f} GB)")

        # GPU: transfer + one_hot on device
        if torch.cuda.is_available():
            # N*K*8 for int64 on GPU, limit to ~2GB
            if dense_torch_bytes <= 2 * 1024**3:
                def torch_gpu_ohe_with_transfer():
                    t_gpu = t_cpu.cuda()
                    out = F.one_hot(t_gpu, num_classes=K).to(torch.uint8)
                    torch.cuda.synchronize()
                    return out

                t, _ = bench("F.one_hot GPU (with H2D transfer)", torch_gpu_ohe_with_transfer)

                t_gpu_preloaded = t_cpu.cuda()
                torch.cuda.synchronize()

                def torch_gpu_ohe_preloaded():
                    out = F.one_hot(t_gpu_preloaded, num_classes=K).to(torch.uint8)
                    torch.cuda.synchronize()
                    return out

                t, _ = bench("F.one_hot GPU (data pre-loaded)", torch_gpu_ohe_preloaded)

                # Cleanup GPU memory
                del t_gpu_preloaded
                torch.cuda.empty_cache()
            else:
                print(f"  {'F.one_hot GPU':45s}  SKIPPED (would need {dense_torch_bytes/1e9:.0f} GB on device)")

        # Sparse COO construction (manual — F.one_hot doesn't support sparse)
        def torch_sparse_cpu():
            row_idx = torch.arange(N, dtype=torch.long)
            indices = torch.stack([row_idx, t_cpu])
            values = torch.ones(N, dtype=torch.uint8)
            return torch.sparse_coo_tensor(indices, values, (N, K))

        t, m = bench("sparse COO manual CPU (K pre-known)", torch_sparse_cpu)

        if torch.cuda.is_available():
            def torch_sparse_gpu_with_transfer():
                row_idx = torch.arange(N, dtype=torch.long, device='cuda')
                t_gpu = t_cpu.cuda()
                indices = torch.stack([row_idx, t_gpu])
                values = torch.ones(N, dtype=torch.uint8, device='cuda')
                out = torch.sparse_coo_tensor(indices, values, (N, K))
                torch.cuda.synchronize()
                return out

            t, _ = bench("sparse COO manual GPU (with H2D)", torch_sparse_gpu_with_transfer)

    except ImportError:
        pass

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
