use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray as _, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[cfg(feature = "cuda")]
mod gpu;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
static GPU_ENCODER: OnceLock<Option<gpu::GpuEncoder>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn get_gpu_encoder() -> Result<&'static gpu::GpuEncoder, String> {
    let opt = GPU_ENCODER.get_or_init(|| gpu::GpuEncoder::new(0).ok());
    opt.as_ref().ok_or_else(|| "Failed to initialize CUDA".to_string())
}

// ── Core encoder ──────────────────────────────────────────────────────

/// Maps arbitrary category labels to dense integer codes.
struct CategoryMap {
    map: FxHashMap<i64, u32>,
    n_categories: u32,
}

impl CategoryMap {
    fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            n_categories: 0,
        }
    }

    fn insert(&mut self, key: i64) -> u32 {
        let n = &mut self.n_categories;
        *self.map.entry(key).or_insert_with(|| {
            let code = *n;
            *n += 1;
            code
        })
    }

    fn get(&self, key: &i64) -> Option<u32> {
        self.map.get(key).copied()
    }

    fn len(&self) -> usize {
        self.n_categories as usize
    }
}

// ── Parallel category discovery ──────────────────────────────────────

/// Discover all unique categories in parallel, returning a global map.
fn discover_categories_parallel(data: &[i64]) -> CategoryMap {
    const CHUNK_SIZE: usize = 1 << 16; // 64K elements per chunk

    // Phase 1: per-chunk local sets (parallel)
    let local_maps: Vec<FxHashMap<i64, ()>> = data
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut local = FxHashMap::default();
            for &val in chunk {
                local.entry(val).or_insert(());
            }
            local
        })
        .collect();

    // Phase 2: merge into global map (sequential — maps are small)
    let mut global = CategoryMap::new();
    for local in &local_maps {
        for &key in local.keys() {
            global.insert(key);
        }
    }
    global
}

// ── Sparse CSR output ────────────────────────────────────────────────

/// One-hot encode to sparse CSR format.
///
/// For one-hot, each row has exactly one non-zero (1), so:
/// - data    = [1, 1, ..., 1]  (length N)
/// - indices = [code_0, code_1, ..., code_{N-1}]  (length N)
/// - indptr  = [0, 1, 2, ..., N]  (length N+1)
fn encode_sparse(data: &[i64], cat_map: &CategoryMap) -> (Vec<u8>, Vec<i32>, Vec<i64>) {
    let n = data.len();

    let indptr: Vec<i64> = (0..=(n as i64)).collect();
    let values = vec![1u8; n];

    let indices: Vec<i32> = data
        .par_iter()
        .map(|&val| cat_map.get(&val).unwrap() as i32)
        .collect();

    (values, indices, indptr)
}

// ── Dense output ─────────────────────────────────────────────────────

/// One-hot encode to dense matrix (N × K) of u8.
fn encode_dense(data: &[i64], cat_map: &CategoryMap) -> Array2<u8> {
    let n = data.len();
    let k = cat_map.len();

    let codes: Vec<u32> = data
        .par_iter()
        .map(|&val| cat_map.get(&val).unwrap())
        .collect();

    let mut matrix = Array2::<u8>::zeros((n, k));
    let raw = matrix.as_slice_mut().unwrap();

    codes.par_iter().enumerate().for_each(|(i, &code)| {
        let offset = i * k + code as usize;
        // SAFETY: each thread writes to a unique row, no data race
        unsafe {
            let ptr = raw.as_ptr() as *mut u8;
            *ptr.add(offset) = 1;
        }
    });

    matrix
}

// ── String encoding ──────────────────────────────────────────────────

/// Fast string interning + one-hot encoding to sparse CSR.
fn intern_and_encode_sparse(
    strings: &[&str],
) -> (Vec<u8>, Vec<i32>, Vec<i64>, Vec<String>, usize) {
    let n = strings.len();

    let mut intern_map: FxHashMap<&str, u32> = FxHashMap::default();
    let mut categories: Vec<String> = Vec::new();
    let mut codes = Vec::with_capacity(n);

    for &s in strings {
        let code = if let Some(&c) = intern_map.get(s) {
            c
        } else {
            let c = categories.len() as u32;
            intern_map.insert(s, c);
            categories.push(s.to_owned());
            c
        };
        codes.push(code);
    }

    let k = categories.len();
    let indptr: Vec<i64> = (0..=(n as i64)).collect();
    let values = vec![1u8; n];
    let indices: Vec<i32> = codes.into_iter().map(|c| c as i32).collect();

    (values, indices, indptr, categories, k)
}

// ── Public API for benchmarks ────────────────────────────────────────

pub fn discover_categories_parallel_pub(data: &[i64]) -> usize {
    discover_categories_parallel(data).len()
}

pub fn encode_sparse_pub(data: &[i64]) -> (Vec<u8>, Vec<i32>, Vec<i64>, usize) {
    let cat_map = discover_categories_parallel(data);
    let n = cat_map.len();
    let (v, i, p) = encode_sparse(data, &cat_map);
    (v, i, p, n)
}

pub fn encode_dense_pub(data: &[i64]) -> Array2<u8> {
    let cat_map = discover_categories_parallel(data);
    encode_dense(data, &cat_map)
}

// ── Python bindings ──────────────────────────────────────────────────

/// One-hot encode an integer array to sparse CSR.
/// Returns (data, indices, indptr, n_categories).
#[pyfunction]
fn encode_sparse_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    let data = input.as_slice()?;
    let cat_map = discover_categories_parallel(data);
    let n_cats = cat_map.len();
    let (values, indices, indptr) = encode_sparse(data, &cat_map);

    Ok((
        Array1::from_vec(values).into_pyarray_bound(py),
        Array1::from_vec(indices).into_pyarray_bound(py),
        Array1::from_vec(indptr).into_pyarray_bound(py),
        n_cats,
    ))
}

/// One-hot encode an integer array to dense matrix.
#[pyfunction]
fn encode_dense_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let data = input.as_slice()?;
    let cat_map = discover_categories_parallel(data);
    let matrix = encode_dense(data, &cat_map);
    Ok(matrix.into_pyarray_bound(py))
}

/// One-hot encode a list of strings to sparse CSR.
/// Returns (data, indices, indptr, categories, n_categories).
#[pyfunction]
fn encode_strings_sparse_py<'py>(
    py: Python<'py>,
    input: Vec<String>,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Vec<String>,
    usize,
)> {
    let refs: Vec<&str> = input.iter().map(|s| s.as_str()).collect();
    let (values, indices, indptr, categories, n_cats) = intern_and_encode_sparse(&refs);

    Ok((
        Array1::from_vec(values).into_pyarray_bound(py),
        Array1::from_vec(indices).into_pyarray_bound(py),
        Array1::from_vec(indptr).into_pyarray_bound(py),
        categories,
        n_cats,
    ))
}

/// Set the number of threads for parallel operations.
#[pyfunction]
fn set_threads(n: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .ok();
}

// ── GPU Python bindings ──────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[pyfunction]
fn gpu_encode_dense_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let data = input.as_slice()?;

    // Category discovery on CPU (hash map — faster than GPU sort+unique)
    let cat_map = discover_categories_parallel(data);
    let k = cat_map.len();

    // Map to dense codes
    let codes: Vec<i64> = data
        .par_iter()
        .map(|&val| cat_map.get(&val).unwrap() as i64)
        .collect();

    // GPU kernel (cached encoder)
    let encoder = get_gpu_encoder()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let output = encoder
        .encode_dense_u8(&codes, k)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel: {e}")))?;

    let n = data.len();
    let matrix = Array2::from_shape_vec((n, k), output)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}")))?;
    Ok(matrix.into_pyarray_bound(py))
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn gpu_encode_sparse_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    let data = input.as_slice()?;
    let cat_map = discover_categories_parallel(data);
    let n_cats = cat_map.len();

    let codes: Vec<i64> = data
        .par_iter()
        .map(|&val| cat_map.get(&val).unwrap() as i64)
        .collect();

    let encoder = get_gpu_encoder()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let (indices, indptr) = encoder
        .encode_sparse_indices(&codes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA kernel: {e}")))?;

    let values = vec![1u8; data.len()];

    Ok((
        Array1::from_vec(values).into_pyarray_bound(py),
        Array1::from_vec(indices).into_pyarray_bound(py),
        Array1::from_vec(indptr).into_pyarray_bound(py),
        n_cats,
    ))
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn gpu_available() -> bool {
    true
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn gpu_available() -> bool {
    false
}

#[pymodule]
fn ohe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_strings_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(gpu_encode_dense_py, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_encode_sparse_py, m)?)?;
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_map() {
        let mut cm = CategoryMap::new();
        assert_eq!(cm.insert(10), 0);
        assert_eq!(cm.insert(20), 1);
        assert_eq!(cm.insert(10), 0);
        assert_eq!(cm.insert(30), 2);
        assert_eq!(cm.len(), 3);
    }

    #[test]
    fn test_sparse_basic() {
        let data = vec![0i64, 1, 2, 0, 1];
        let cat_map = discover_categories_parallel(&data);
        let (values, indices, indptr) = encode_sparse(&data, &cat_map);

        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|&v| v == 1));
        assert_eq!(indptr, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(indices[0], indices[3]);
        assert_eq!(indices[1], indices[4]);
        assert_eq!(cat_map.len(), 3);
    }

    #[test]
    fn test_dense_basic() {
        let data = vec![0i64, 1, 2, 0];
        let cat_map = discover_categories_parallel(&data);
        let matrix = encode_dense(&data, &cat_map);

        assert_eq!(matrix.shape(), &[4, 3]);
        for row in matrix.rows() {
            assert_eq!(row.iter().filter(|&&v| v == 1).count(), 1);
            assert_eq!(row.iter().filter(|&&v| v == 0).count(), 2);
        }
        assert_eq!(matrix.row(0), matrix.row(3));
    }

    #[test]
    fn test_string_encoding() {
        let strings = vec!["cat", "dog", "cat", "bird", "dog"];
        let (values, indices, indptr, categories, n_cats) =
            intern_and_encode_sparse(&strings);

        assert_eq!(n_cats, 3);
        assert_eq!(categories.len(), 3);
        assert_eq!(values.len(), 5);
        assert_eq!(indptr.len(), 6);
        assert_eq!(indices[0], indices[2]);
        assert_eq!(indices[1], indices[4]);
    }

    #[test]
    fn test_single_category() {
        let data = vec![42i64; 1000];
        let cat_map = discover_categories_parallel(&data);
        assert_eq!(cat_map.len(), 1);

        let (_, indices, _) = encode_sparse(&data, &cat_map);
        assert!(indices.iter().all(|&i| i == 0));
    }

    #[test]
    fn test_high_cardinality() {
        let data: Vec<i64> = (0..10_000).collect();
        let cat_map = discover_categories_parallel(&data);
        assert_eq!(cat_map.len(), 10_000);

        let (values, indices, indptr) = encode_sparse(&data, &cat_map);
        assert_eq!(values.len(), 10_000);
        assert_eq!(indptr.len(), 10_001);

        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 10_000);
    }
}
