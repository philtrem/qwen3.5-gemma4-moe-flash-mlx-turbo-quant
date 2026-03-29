//! FFI wrapper for mlx-c functions not yet exposed by mlx-rs.

use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

extern "C" {
    fn mlx_array_from_mmap(
        mmap_ptr: *const std::ffi::c_void,
        byte_offset: usize,
        byte_length: usize,
        shape: *const i32,
        dim: i32,
        dtype: mlx_sys::mlx_dtype,
    ) -> mlx_sys::mlx_array;
}

/// Create a zero-copy MLX array backed by mmap'd memory via Metal's newBufferWithBytesNoCopy.
/// The mmap must outlive the returned array.
pub unsafe fn array_from_mmap(
    mmap_ptr: *const u8,
    byte_offset: usize,
    byte_length: usize,
    shape: &[i32],
    dtype: Dtype,
) -> Array {
    let result = mlx_array_from_mmap(
        mmap_ptr as *const std::ffi::c_void,
        byte_offset,
        byte_length,
        shape.as_ptr(),
        shape.len() as i32,
        dtype.into(),
    );
    Array::from_ptr(result)
}

/// Safe wrapper around mlx_gather_qmm from mlx-c.
///
/// Uses the same Guarded pattern as mlx-rs internal ops.
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    rhs_indices: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array, Exception> {
    // Use a dummy quantized_matmul call to validate, then call gather_qmm
    // via the Array::try_from_op pattern exposed by as_ptr/from_ptr.
    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let null_lhs = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let status = mlx_sys::mlx_gather_qmm(
            &mut result as *mut _,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            null_lhs,
            rhs_indices.as_ptr(),
            transpose,
            group_size,
            bits,
            sorted_indices,
            stream,
        );
        mlx_sys::mlx_stream_free(stream);
        if status == 0 {
            Ok(Array::from_ptr(result))
        } else {
            mlx_sys::mlx_array_free(result);
            // Can't construct Exception directly (fields are private).
            // Panic with the error — this matches the behavior of other
            // mlx-rs ops that use internal Guarded::try_from_op.
            panic!("mlx_gather_qmm failed (status {})", status);
        }
    }
}
