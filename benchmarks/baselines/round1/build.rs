use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("cannot run {command:?}: {e}"));
    assert!(status.success(), "command failed ({status}): {command:?}");
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "CUDA feature currently targets Linux/WSL2"
    );
    let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()));
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "120".into());
    assert!(
        !arch.is_empty() && arch.chars().all(|c| c.is_ascii_digit()),
        "CUDA_ARCH must be a numeric SM architecture, e.g. 120"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("kernels.o");
    run(Command::new(cuda.join("bin/nvcc"))
        .args(["-c", "kernels/kernels.cu", "-o"])
        .arg(&object)
        .args([
            "-O3",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "--expt-relaxed-constexpr",
        ])
        .arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}")));
    run(Command::new("ar")
        .arg("crs")
        .arg(out.join("libnano_cuda.a"))
        .arg(object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=static=nano_cuda");
    for library in ["cudart", "cublas", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        cuda.join("lib64").display()
    );
}
