"""ohe-rs: Ultra-fast one-hot encoding powered by Rust + CUDA.

Usage:
    import numpy as np
    from scipy.sparse import csr_matrix
    from ohe_rs import encode_sparse, encode_dense, encode_strings_sparse

    # Integer input -> sparse CSR
    data = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    values, indices, indptr, n_cats = encode_sparse(data)
    matrix = csr_matrix((values, indices, indptr), shape=(len(data), n_cats))

    # Integer input -> dense numpy array
    dense = encode_dense(data)  # shape (5, 3), dtype uint8

    # Memory estimation before encoding
    dense_bytes, sparse_bytes = estimate_memory(data)

    # GPU-accelerated encoding (if CUDA available)
    if gpu_available():
        dense_gpu = gpu_encode_dense(data)
        values, indices, indptr, n_cats = gpu_encode_sparse(data)
"""

from .ohe_rs import (
    encode_sparse_py as encode_sparse,
    encode_dense_py as encode_dense,
    encode_strings_sparse_py as encode_strings_sparse,
    estimate_memory_py as estimate_memory,
    set_threads,
    gpu_available,
)

__all__ = [
    "encode_sparse",
    "encode_dense",
    "encode_strings_sparse",
    "estimate_memory",
    "set_threads",
    "gpu_available",
]

if gpu_available():
    from .ohe_rs import (
        gpu_encode_dense_py as gpu_encode_dense,
        gpu_encode_sparse_py as gpu_encode_sparse,
    )
    __all__ += ["gpu_encode_dense", "gpu_encode_sparse"]

__version__ = "0.1.0"
