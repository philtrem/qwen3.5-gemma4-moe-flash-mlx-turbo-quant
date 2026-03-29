fn main() {
    // Find mlx-sys build output directory (contains MLX C++ headers)
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".to_string());
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());

    // Glob for the mlx-sys build output
    let pattern = format!("{}/{}/build/mlx-sys-*/out/build/include", target_dir, profile);
    let mlx_include = glob::glob(&pattern)
        .expect("glob pattern error")
        .filter_map(|p| p.ok())
        .next()
        .expect("could not find mlx-sys build output — run cargo build first");

    // The MLX source headers (for backend/metal) are in _deps
    let mlx_src_pattern = format!("{}/{}/build/mlx-sys-*/out/build/_deps/mlx-src", target_dir, profile);
    let mlx_src = glob::glob(&mlx_src_pattern)
        .expect("glob pattern error")
        .filter_map(|p| p.ok())
        .next()
        .expect("could not find mlx source directory");

    // Private C API headers (mlx/c/private/array.h) are in the mlx-sys crate source
    let home = std::env::var("HOME").unwrap();
    let mlxc_private_pattern = format!(
        "{}/.cargo/registry/src/*/mlx-sys-0.2.*/src/mlx-c",
        home
    );
    let mlxc_private = glob::glob(&mlxc_private_pattern)
        .expect("glob pattern error")
        .filter_map(|p| p.ok())
        .next()
        .expect("could not find mlx-c private headers in cargo registry");

    // metal-cpp headers are bundled with MLX source
    let metal_cpp = mlx_src.join("mlx/backend/metal/kernels/metal_3_1");
    let metal_cpp_fallback = mlx_src.join("mlx/backend/metal");

    println!("cargo:rerun-if-changed=src/ffi_zerocopy.cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("src/ffi_zerocopy.cpp")
        .include(&mlx_include)
        .include(&mlx_src)
        .include(&mlxc_private);

    // metal-cpp headers (Metal/Metal.hpp etc.)
    if metal_cpp.exists() {
        build.include(&metal_cpp);
    }
    if metal_cpp_fallback.exists() {
        build.include(&metal_cpp_fallback);
    }

    // metal-cpp headers (Metal/Metal.hpp) — fetched by MLX's cmake
    let metal_cpp_pattern = format!("{}/{}/build/mlx-sys-*/out/build/_deps/metal_cpp-src", target_dir, profile);
    if let Some(p) = glob::glob(&metal_cpp_pattern).ok().and_then(|mut g| g.next()).and_then(|p| p.ok()) {
        build.include(&p);
    }
    // Also check the installed include path (may have metal_cpp/ subfolder)
    let metal_cpp_installed = mlx_include.join("metal_cpp");
    if metal_cpp_installed.exists() {
        build.include(&metal_cpp_installed);
    }

    build.compile("ffi_zerocopy");
}
