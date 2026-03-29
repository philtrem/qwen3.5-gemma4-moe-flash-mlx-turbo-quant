// Zero-copy MLX array creation from mmap'd memory via Metal's newBufferWithBytesNoCopy.
//
// This bypasses mlx_array_new_data (which always copies) by using the internal
// array(allocator::Buffer, shape, dtype, deleter) constructor that accepts an
// existing Metal buffer.

#include "mlx/array.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"

extern "C" mlx_array mlx_array_from_mmap(
    void* mmap_ptr,
    size_t byte_offset,
    size_t byte_length,
    const int* shape,
    int dim,
    mlx_dtype dtype
) {
    void* data_ptr = static_cast<char*>(mmap_ptr) + byte_offset;

    // Get MLX's Metal device singleton and create a no-copy Metal buffer
    auto& dev = mlx::core::metal::device(mlx::core::Device::gpu);
    auto* mtl_buf = dev.mtl_device()->newBuffer(
        data_ptr,
        byte_length,
        MTL::ResourceStorageModeShared,
        nullptr);  // no deallocator — mmap owns memory

    if (!mtl_buf) {
        // Fall back to empty array on failure
        return mlx_array_new_();
    }

    // Wrap as allocator::Buffer (just a void* holding MTL::Buffer*)
    auto buf = mlx::core::allocator::Buffer(mtl_buf);

    // No-op deleter: mmap lifetime exceeds array lifetime
    auto noop = [](mlx::core::allocator::Buffer) {};

    std::vector<int> cpp_shape(shape, shape + dim);
    auto cpp_dtype = mlx_dtype_to_cpp(dtype);

    return mlx_array_new_(mlx::core::array(buf, cpp_shape, cpp_dtype, noop));
}
