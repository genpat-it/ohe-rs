# ohe-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Ultra-fast one-hot encoding powered by Rust + CUDA, with Python bindings.

## Why ohe-rs?

One-hot encoding is a fundamental operation in machine learning pipelines, yet existing implementations in Python (scikit-learn, pandas, numpy) are surprisingly slow on large datasets. They suffer from Python overhead, single-threaded execution, and suboptimal memory access patterns.

**ohe-rs** solves this by implementing one-hot encoding in Rust with:

- **Parallel category discovery** using rayon + FxHashMap (lock-free, per-thread local maps with global merge)
- **Zero-copy Python integration** via PyO3 + numpy array protocol
- **Sparse CSR output** that uses ~13 bytes/row regardless of cardinality (vs N*K for dense)
- **Optional CUDA acceleration** for GPU-resident data pipelines
- **Memory-safe operation** with upfront estimation and chunked processing for large datasets

## Benchmark Results

**Machine:** 2x Intel Xeon Gold 6542Y (80 cores), 504 GB RAM, NVIDIA L4 (24 GB), Linux.
**Protocol:** 10M rows, warm-up excluded, GC disabled, 7 repeats (median), uint8 output, 80 rayon threads.

### End-to-End (category discovery + encoding)

| Cardinality (K) | ohe-rs CPU | scikit-learn | Speedup |
|---|---|---|---|
| K = 10 | **26 ms** (387 M rows/s) | 381 ms | **15x** |
| K = 1,000 | **21 ms** (468 M rows/s) | 740 ms | **35x** |
| K = 100,000 | **56 ms** (179 M rows/s) | 1,310 ms | **23x** |

### Transform-Only (K pre-known, no discovery) — full cartesian product

Every combination of H2D (host-to-device) and D2H (device-to-host) transfer benchmarked for both ohe-rs and PyTorch.

**CPU:**

| Method | K=10 | K=1,000 | K=100,000 |
|---|---|---|---|
| **ohe-rs CPU** | **18 ms** | **18 ms** | **21 ms** |
| PyTorch sparse COO CPU | 29 ms | 29 ms | 28 ms |
| sklearn (prefitted) | 400 ms | 698 ms | 1,337 ms |

**GPU with H2D + D2H (data on host, result on host):**

| Method | K=10 | K=1,000 | K=100,000 |
|---|---|---|---|
| **ohe-rs GPU** | **48 ms** | **44 ms** | 75 ms |
| PyTorch GPU | 75 ms | 73 ms | **73 ms** |

**GPU pre-loaded input, D2H output (kernel + D2H):**

| Method | K=10 | K=1,000 | K=100,000 |
|---|---|---|---|
| **ohe-rs GPU** | **25 ms** | **25 ms** | **25 ms** |
| PyTorch GPU | 66 ms | 65 ms | 64 ms |

**GPU all on device — kernel only (no transfer):**

| Method | K=10 | K=1,000 | K=100,000 |
|---|---|---|---|
| **ohe-rs GPU** | **1.3 ms** | **1.4 ms** | **1.4 ms** |
| PyTorch GPU | 1.5 ms | 1.5 ms | 1.5 ms |

> **ohe-rs wins in nearly every scenario.** At K=100K with full H2D+D2H, PyTorch edges ahead (73ms vs 75ms) due to lower transfer overhead for COO metadata vs CSR arrays.

> **PyTorch `F.one_hot` limitation:** allocates a dense **int64** tensor (8 bytes/element) before casting. At K=1,000 with 10M rows this requires **80 GB of RAM**. ohe-rs sparse uses ~13 bytes/row regardless of K.

### Thread Scaling

One-hot encoding is **memory-bandwidth bound**, not compute-bound. More threads help only up to the point where RAM bandwidth saturates. On our 80-core machine, the sweet spot is **8-16 threads**:

| Threads | E2E K=10 | E2E K=100K | Transform K=10 |
|---|---|---|---|
| 1 | 58 ms | 273 ms | 24 ms |
| 2 | 38 ms | 148 ms | 20 ms |
| 4 | 30 ms | 101 ms | 22 ms |
| 8 | **20 ms** | 70 ms | **16 ms** |
| 16 | 20 ms | **62 ms** | 16 ms |
| 32 | 20 ms | 55 ms | 20 ms |
| 80 | 28 ms | 64 ms | 29 ms |

Beyond 16 threads, performance **degrades** due to cache contention. On typical workstations (4-8 cores), all cores are useful. Use `set_threads()` to tune:

```python
from ohe_rs import set_threads
set_threads(8)  # recommended for machines with >16 cores
```

## Installation

### From source (recommended)

```bash
# Clone
git clone https://github.com/genpat-it/ohe-rs.git
cd ohe-rs

# CPU-only build
pip install maturin
maturin develop --release

# With CUDA support (requires CUDA toolkit)
CUDA_ROOT=/usr/local/cuda maturin develop --release
```

### Docker (ghcr.io)

```bash
# CPU-only
docker pull ghcr.io/genpat-it/ohe-rs:latest
docker run --rm ghcr.io/genpat-it/ohe-rs -c "from ohe_rs import encode_sparse; print('OK')"

# With CUDA support
docker pull ghcr.io/genpat-it/ohe-rs:latest-cuda
docker run --rm --gpus all ghcr.io/genpat-it/ohe-rs:latest-cuda -c "from ohe_rs import gpu_available; print('GPU:', gpu_available())"
```

Images are automatically built and pushed on each release.

### Bioconda

```bash
# Coming soon
conda install -c bioconda ohe-rs
```

### Build requirements

- Rust 1.70+
- Python 3.9+
- numpy >= 1.20
- scipy >= 1.7
- CUDA toolkit (optional, for GPU support)

## Usage

### Sparse encoding (recommended)

```python
import numpy as np
from scipy.sparse import csr_matrix
from ohe_rs import encode_sparse

data = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64)
values, indices, indptr, n_categories = encode_sparse(data)

# Build scipy sparse matrix
matrix = csr_matrix((values, indices, indptr), shape=(len(data), n_categories))
print(matrix.toarray())
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]]
```

### Dense encoding

```python
from ohe_rs import encode_dense

data = np.array([0, 1, 2, 0], dtype=np.int64)
matrix = encode_dense(data)  # np.ndarray, shape (4, 3), dtype uint8
```

### String input

```python
from ohe_rs import encode_strings_sparse

strings = ["cat", "dog", "cat", "bird", "dog"]
values, indices, indptr, categories, n_cats = encode_strings_sparse(strings)
print(categories)  # ['cat', 'dog', 'bird']
```

### Multi-column encoding (cgMLST / allele profiles)

For datasets with many categorical columns (e.g. cgMLST allele profiles), `encode_multi_sparse` encodes all columns in a single Rust call, avoiding Python loop overhead.

```python
import numpy as np
from scipy.sparse import csr_matrix
from ohe_rs import encode_multi_sparse

# cgMLST-like matrix: 10K samples x 8K loci, each cell is an allele ID
profiles = np.random.randint(0, 300, size=(10_000, 8_000), dtype=np.int64)

# Single call — encodes all columns in parallel
values, indices, indptr, total_cols, per_col_sizes = encode_multi_sparse(profiles)

# Build scipy sparse matrix (rows=samples, cols=concatenated one-hot of all loci)
matrix = csr_matrix((values, indices, indptr), shape=(10_000, total_cols))
# matrix.shape = (10000, ~2.4M)  — each row has exactly 8000 non-zeros
```

**Performance (10K samples x 8K loci, ~50-500 alleles per locus):**

| Method | Time | Speedup |
|---|---|---|
| **ohe-rs encode_multi_sparse** | **724 ms** | **12x** |
| ohe-rs per-column Python loop | 2,491 ms | 3.5x |
| sklearn per-column | 8,618 ms | baseline |

### Memory estimation

```python
from ohe_rs import estimate_memory

data = np.random.randint(0, 100_000, size=10_000_000, dtype=np.int64)
dense_bytes, sparse_bytes = estimate_memory(data)
print(f"Dense: {dense_bytes / 1e9:.1f} GB")   # Dense: 1000.0 GB
print(f"Sparse: {sparse_bytes / 1e6:.1f} MB")  # Sparse: 130.0 MB
```

### Memory-safe dense encoding

```python
# Automatically processes in chunks if needed
matrix = encode_dense(data, max_memory_mb=512)
```

### GPU acceleration

```python
from ohe_rs import gpu_available

if gpu_available():
    from ohe_rs import gpu_encode_sparse, gpu_encode_dense

    values, indices, indptr, n_cats = gpu_encode_sparse(data)
    dense_matrix = gpu_encode_dense(data)  # for small K
```

### Thread control

```python
from ohe_rs import set_threads
set_threads(4)  # Limit to 4 threads
```

## Architecture

```
Input (Python numpy array)
         |
         v
+----------------------------+
|  Rust Core (PyO3 bindings) |
|                            |
|  1. Category Discovery     |
|     rayon parallel chunks  |
|     FxHashMap per-thread   |
|     + sequential merge     |
|                            |
|  2. Encoding               |
|     CPU: parallel write    |
|     GPU: CUDA kernel       |
|                            |
|  3. Output                 |
|     Sparse CSR (zero-copy) |
|     Dense ndarray          |
+----------------------------+
         |
         v
scipy.sparse.csr_matrix / np.ndarray
```

### Why CPU beats GPU here

One-hot encoding is **memory-bound**, not compute-bound. Each element requires:
- 1 hash lookup (category mapping)
- 1 memory write (set the bit)

The GPU kernel itself runs in microseconds, but the host-to-device transfer of N int64 values (~80 MB for 10M rows) dominates the total time. GPU wins when:
- Data is **already on the GPU** (e.g., in a cuML/PyTorch pipeline)
- You combine OHE with other GPU operations, amortizing the transfer cost

## Development

```bash
# Build
cargo build --release

# Tests
cargo test

# Build with CUDA
CUDA_ROOT=/usr/local/cuda cargo build --release --features cuda

# Python development install
maturin develop --release

# Run benchmarks
python benchmark.py
```

## License

MIT

