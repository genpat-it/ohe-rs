use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ohe_kernel.ptx"));

pub struct GpuEncoder {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

impl GpuEncoder {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        let module = ctx.load_module(Ptx::from_src(PTX_SRC))?;
        Ok(Self { ctx, module })
    }

    /// One-hot encode to dense u8 matrix on GPU.
    /// `codes` must be pre-mapped integer codes in 0..K-1.
    pub fn encode_dense_u8(&self, codes: &[i64], k: usize) -> Result<Vec<u8>, DriverError> {
        let n = codes.len() as i64;
        let total = codes.len() * k;
        let stream = self.ctx.default_stream();

        // Upload input
        let d_input = stream.clone_htod(codes)?;

        // Allocate zeroed output
        let d_output = stream.alloc_zeros::<u8>(total)?;

        let func = self.module.load_function("ohe_dense_u8").unwrap();
        let cfg = LaunchConfig::for_num_elems(codes.len() as u32);

        let k_i32 = k as i32;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&d_input);
        launch.arg(&d_output);
        launch.arg(&k_i32);
        launch.arg(&n);
        unsafe { launch.launch(cfg) }?;

        let output = stream.clone_dtoh(&d_output)?;
        Ok(output)
    }

    /// One-hot encode to dense f32 matrix on GPU.
    pub fn encode_dense_f32(&self, codes: &[i64], k: usize) -> Result<Vec<f32>, DriverError> {
        let n = codes.len() as i64;
        let total = codes.len() * k;
        let stream = self.ctx.default_stream();

        let d_input = stream.clone_htod(codes)?;
        let d_output = stream.alloc_zeros::<f32>(total)?;

        let func = self.module.load_function("ohe_dense_f32").unwrap();
        let cfg = LaunchConfig::for_num_elems(codes.len() as u32);

        let k_i32 = k as i32;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&d_input);
        launch.arg(&d_output);
        launch.arg(&k_i32);
        launch.arg(&n);
        unsafe { launch.launch(cfg) }?;

        let output = stream.clone_dtoh(&d_output)?;
        Ok(output)
    }

    /// Build sparse CSR indices on GPU.
    /// Returns (indices_vec, indptr_vec).
    pub fn encode_sparse_indices(
        &self,
        codes: &[i64],
    ) -> Result<(Vec<i32>, Vec<i64>), DriverError> {
        let n = codes.len() as i64;
        let stream = self.ctx.default_stream();

        let d_input = stream.clone_htod(codes)?;
        let d_indices = stream.alloc_zeros::<i32>(codes.len())?;
        let d_indptr = stream.alloc_zeros::<i64>(codes.len() + 1)?;

        let f_indices = self.module.load_function("ohe_sparse_indices").unwrap();
        let f_indptr = self.module.load_function("ohe_sparse_indptr").unwrap();

        let cfg_n = LaunchConfig::for_num_elems(codes.len() as u32);
        let cfg_np1 = LaunchConfig::for_num_elems((codes.len() + 1) as u32);

        let mut launch_idx = stream.launch_builder(&f_indices);
        launch_idx.arg(&d_input);
        launch_idx.arg(&d_indices);
        launch_idx.arg(&n);
        unsafe { launch_idx.launch(cfg_n) }?;

        let mut launch_ptr = stream.launch_builder(&f_indptr);
        launch_ptr.arg(&d_indptr);
        launch_ptr.arg(&n);
        unsafe { launch_ptr.launch(cfg_np1) }?;

        let indices = stream.clone_dtoh(&d_indices)?;
        let indptr = stream.clone_dtoh(&d_indptr)?;
        Ok((indices, indptr))
    }
}
