use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA kernel if the cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = "src/cuda/ohe_kernel.cu";

    println!("cargo:rerun-if-changed={}", kernel_src);

    let ptx_path = out_dir.join("ohe_kernel.ptx");

    let status = Command::new("nvcc")
        .args([
            "--ptx",
            "-O3",
            "-o",
            ptx_path.to_str().unwrap(),
            kernel_src,
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");

    if !status.success() {
        panic!("nvcc failed to compile CUDA kernel");
    }
}
