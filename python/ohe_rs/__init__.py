"""ohe-rs: Ultra-fast one-hot encoding powered by Rust + CUDA."""

from .ohe_rs import (
    encode_sparse_py as encode_sparse,
    encode_dense_py as encode_dense,
    encode_strings_sparse_py as encode_strings_sparse,
    encode_multi_sparse_py as encode_multi_sparse,
    estimate_memory_py as estimate_memory,
    set_threads,
    gpu_available,
)

__all__ = [
    "encode_sparse",
    "encode_dense",
    "encode_strings_sparse",
    "encode_multi_sparse",
    "estimate_memory",
    "set_threads",
    "gpu_available",
]

if gpu_available():
    from .ohe_rs import (
        gpu_encode_dense_py as gpu_encode_dense,
        gpu_encode_sparse_py as gpu_encode_sparse,
        gpu_upload,
        gpu_encode_sparse_preloaded_py as gpu_encode_sparse_preloaded,
        GpuBufferPy as GpuBuffer,
    )
    __all__ += [
        "gpu_encode_dense",
        "gpu_encode_sparse",
        "gpu_upload",
        "gpu_encode_sparse_preloaded",
        "GpuBuffer",
    ]

__version__ = "0.1.0"
