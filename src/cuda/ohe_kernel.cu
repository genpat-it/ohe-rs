// Ultra-fast one-hot encoding CUDA kernels
// Each thread handles one row: zero-init + set one bit

extern "C" __global__ void ohe_dense_u8(
    const long long* __restrict__ input,   // [N] category indices (pre-mapped to 0..K-1)
    unsigned char* __restrict__ output,     // [N x K] output matrix, pre-zeroed
    const int K,                            // number of categories
    const long long N                       // number of rows
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < N) {
        int cat = (int)input[idx];
        if (cat >= 0 && cat < K) {
            output[idx * K + cat] = 1;
        }
    }
}

extern "C" __global__ void ohe_dense_f32(
    const long long* __restrict__ input,
    float* __restrict__ output,
    const int K,
    const long long N
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < N) {
        int cat = (int)input[idx];
        if (cat >= 0 && cat < K) {
            output[idx * K + cat] = 1.0f;
        }
    }
}

// Sparse CSR indices kernel — just writes the category index for each row
extern "C" __global__ void ohe_sparse_indices(
    const long long* __restrict__ input,   // [N] category indices
    int* __restrict__ indices,              // [N] output CSR indices
    const long long N
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < N) {
        indices[idx] = (int)input[idx];
    }
}

// Sparse CSR indptr kernel — trivial: indptr[i] = i
extern "C" __global__ void ohe_sparse_indptr(
    long long* __restrict__ indptr,
    const long long N
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx <= N) {
        indptr[idx] = idx;
    }
}
