use std::env;
use std::{fs, path::PathBuf};

fn main() {
    let cuda_path = env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string());

    let cuda_lib_path = format!("{}/lib64", cuda_path);
    let cuda_header_path = format!("{}/include", cuda_path);
    let cuda_runtime_header_path = format!("{}/include/cuda_runtime.h", cuda_path);
    let cuda_driver_header_path = format!("{}/include/cuda.h", cuda_path);
    let cufile_header_path = format!("{}/include/cufile.h", cuda_path);

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    // CUDA headers have represented CUmemLocation::id both as a direct field
    // and as an anonymous union.  Bindgen preserves that layout distinction,
    // so expose it to the Rust code based on the bindings actually generated.
    println!("cargo:rustc-check-cfg=cfg(cu_mem_location_anonymous_union)");

    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=dylib=cudart"); // Link dynamically.
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cufile");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(cuda_runtime_header_path)
        .header(cuda_driver_header_path)
        .header(cufile_header_path)
        .clang_arg(format!("-I{}", cuda_header_path))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");

    let mut bindings = fs::read_to_string(&out_path).expect("Couldn't read generated bindings!");
    let cu_mem_location_body = bindings
        .split("pub struct CUmemLocation_st {")
        .nth(1)
        .and_then(|rest| rest.split('}').next())
        .expect("bindgen did not generate CUmemLocation_st");
    if cu_mem_location_body.contains("__bindgen_anon_") {
        println!("cargo:rustc-cfg=cu_mem_location_anonymous_union");
    }
    let has_function = |name: &str| {
        bindings.contains(&format!("pub fn {name}("))
            || bindings.contains(&format!("pub fn {name} ("))
    };
    let has_device_properties = has_function("cudaGetDeviceProperties");
    let has_device_properties_v2 = has_function("cudaGetDeviceProperties_v2");

    if !has_device_properties && has_device_properties_v2 {
        bindings.push_str(
            "\n#[allow(unused_imports)]\npub use self::cudaGetDeviceProperties_v2 as cudaGetDeviceProperties;\n",
        );
        fs::write(out_path, bindings).expect("Couldn't update generated bindings!");
    }
}
