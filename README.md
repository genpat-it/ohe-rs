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

All benchmarks on 10,000,000 rows, single machine (Linux, AMD EPYC, NVIDIA L4).
Fair protocol: warm-up run excluded, GC disabled, 7 repeats (median), fit+transform end-to-end for all methods, uint8 output.

### Sparse Encoding (fit+transform, category discovery included)

| Cardinality (K) | ohe-rs CPU | ohe-rs GPU | scikit-learn | Speedup vs sklearn |
|---|---|---|---|---|
| K = 10 | **31 ms** (326 M rows/s) | 54 ms | 475 ms | **15x** |
| K = 1,000 | **28 ms** (357 M rows/s) | 65 ms | 825 ms | **29x** |
| K = 100,000 | **58 ms** (172 M rows/s) | 84 ms | 1,338 ms | **23x** |

### Dense Encoding (K=10)

| Method | Time | Notes |
|---|---|---|
| PyTorch GPU (data pre-loaded) | 11 ms | Requires data already on GPU |
| **ohe-rs CPU dense** | **19 ms** | **No GPU required** |
| PyTorch GPU (with H2D transfer) | 21 ms | Includes host-to-device copy |
| PyTorch CPU | 82 ms | |
| numpy eye indexing | 97 ms | |

> **PyTorch limitation:** `F.one_hot` allocates a dense **int64** tensor (8 bytes/element) before casting to uint8. At K=1,000 with 10M rows this requires **80 GB of RAM**, making it unusable for any real cardinality. ohe-rs sparse uses ~13 bytes/row regardless of K.

> **GPU note:** GPU sparse is ~2x slower than CPU due to host-device transfer overhead. GPU shines when data is already on the device or in end-to-end GPU pipelines.

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

