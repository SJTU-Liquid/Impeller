use halo2curves::bn256;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping;
use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::ScalarArray;

// Define the field type used in tests.
type TestField = bn256::Fr;

// Helper: create a CPU `ScalarArray` using `PinnedMemoryPool`.
fn create_cpu_array<const N: usize>(
    values: [TestField; N],
) -> (ScalarArray<TestField>, CpuMemoryPool) {
    let mut pool = CpuMemoryPool::new(10, size_of::<TestField>()); // Pool with sufficient capacity.
    let ptr = pool.allocate(N);
    unsafe {
        copy_nonoverlapping(values.as_ptr(), ptr, N);
    }
    let array = ScalarArray::new(N, ptr, DeviceType::CPU);
    (array, pool)
}

// Helper: create a GPU `ScalarArray`.
fn create_gpu_array<const N: usize>(
    values: [TestField; N],
    stream: &CudaStream,
) -> ScalarArray<TestField> {
    let (cpu_array, _pool) = create_cpu_array(values);
    let ptr = stream.allocate(N);
    let mut gpu_array = ScalarArray::new(
        N,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    cpu_array.cpu2gpu(&mut gpu_array, stream);
    gpu_array
}

#[test]
fn test_rotate_slice_basic() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];
    let (mut array, _pool) = create_cpu_array(values);

    // Rotate by 2; the array becomes [6, 7, 0, 1, 2, 3, 4, 5].
    array.rotate(2);

    // Test a regular slice (no wraparound).
    let slice = array.slice(3, 6);
    assert_eq!(slice.len, 3);
    // Check that the slice contents are correct (expected [1, 2, 3]).
    for i in 0..3 {
        assert_eq!(slice[i], values[(i + 1) % 8]);
    }
}

#[test]
fn test_rotate_slice_cross_boundary() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];
    let (mut array, _pool) = create_cpu_array(values);

    // Rotate by 1; logically, the array becomes [7, 0, 1, 2, 3, 4, 5, 6].
    array.rotate(1);

    // Test a slice that wraps around the boundary ([7, 0, 1], a non-contiguous memory region).
    let slice = array.slice(0, 3);
    assert_eq!(slice.len, 3);
    // Verify the slice contents.
    assert_eq!(slice[0], values[7]);
    assert_eq!(slice[1], values[0]);
    assert_eq!(slice[2], values[1]);
}

#[test]
fn test_rotate_slice_to_array_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // Create and rotate a GPU array.
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(3); // Rotate by 3; logically, it becomes [5, 6, 7, 0, 1, 2, 3, 4].

    // Create a slice that includes a region crossing the boundary.
    let slice = gpu_array.slice(1, 4); // Expected contents: [6, 7, 0].

    // Create the destination array and transfer data.
    let ptr = stream.allocate(3);
    let mut dst_array = ScalarArray::new(
        3,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );

    // Transfer from the rotated slice to a regular array.
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&slice, &mut dst_array, &stream);

    // Verify the transfer result.
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(3, verify_pool.allocate(3), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // Verify the contents.
    assert_eq!(verify_array[0], values[6]); // 6
    assert_eq!(verify_array[1], values[7]); // 7
    assert_eq!(verify_array[2], values[0]); // 0

    // Free GPU memory.
    stream.free(gpu_array.values);
    stream.free(dst_array.values);
}

#[test]
fn test_rotate_slice_to_rotate_array_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // Create and rotate a GPU array.
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(3); // Rotate by 3; logically, it becomes [5, 6, 7, 0, 1, 2, 3, 4].

    // Create a slice that includes a region crossing the boundary.
    let slice = gpu_array.slice(1, 4); // Expected contents: [6, 7, 0].

    // Create the destination array and transfer data.
    let ptr = stream.allocate(3);
    let mut dst_array = ScalarArray::new(
        3,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    dst_array.rotate(2);

    // Transfer from the rotated slice to a regular array.
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&slice, &mut dst_array, &stream);

    // Verify the transfer result.
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(3, verify_pool.allocate(3), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // Verify the contents.
    assert_eq!(verify_array[0], values[6]); // 6
    assert_eq!(verify_array[1], values[7]); // 7
    assert_eq!(verify_array[2], values[0]); // 0

    // Free GPU memory.
    stream.free(gpu_array.values);
    stream.free(dst_array.values);
}

#[test]
fn test_rotate_array_to_rotate_slice_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // Create and rotate the destination GPU array.
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(2); // Rotate by 2; logically, it becomes [6, 7, 0, 1, 2, 3, 4, 5].

    // Create the source array.
    let src_values = [
        TestField::from(10),
        TestField::from(11),
        TestField::from(12),
    ];
    let mut src_array = create_gpu_array(src_values, &stream);
    src_array.rotate(2); // Rotate by 2; logically, it becomes [11, 12, 10].

    // Use a slice of the rotated array as the destination.
    let mut dst_slice = gpu_array.slice(1, 4); // Region crossing the boundary: [7, 0, 1].

    // Transfer from a regular array to the rotated slice.
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&src_array, &mut dst_slice, &stream);

    // Verify the transfer result.
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(8, verify_pool.allocate(8), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&gpu_array, &mut verify_array, &stream);
    stream.sync();

    // Verify the entire array and ensure only the slice was modified.
    assert_eq!(verify_array[0], values[6]);
    assert_eq!(verify_array[1], src_values[1]); // Modified element.
    assert_eq!(verify_array[2], src_values[2]); // Modified element.
    assert_eq!(verify_array[3], src_values[0]); // Modified element.
    assert_eq!(verify_array[4], values[2]); // Unmodified element.
    assert_eq!(verify_array[5], values[3]);
    assert_eq!(verify_array[6], values[4]);
    assert_eq!(verify_array[7], values[5]);

    // Free GPU memory.
    stream.free(gpu_array.values);
    stream.free(src_array.values);
}

#[test]
fn test_rotate_slice_to_slice_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // Create and rotate two GPU arrays.
    let mut src_array = create_gpu_array(values, &stream);
    let mut dst_array = create_gpu_array(values, &stream);

    src_array.rotate(2); // Rotate the source array by 2: [6, 7, 0, 1, 2, 3, 4, 5].
    dst_array.rotate(3); // Rotate the destination array by 3: [5, 6, 7, 0, 1, 2, 3, 4].

    // Create source and destination slices that both wrap around the boundary.
    let src_slice = src_array.slice(1, 4); // [7,0,1]
    let mut dst_slice = dst_array.slice(1, 4); // [6,7,0]

    // Transfer from one rotated slice to another.
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&src_slice, &mut dst_slice, &stream);

    // Verify the transfer result.
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(8, verify_pool.allocate(8), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // Verify the contents.
    assert_eq!(verify_array[0], values[5]);
    assert_eq!(verify_array[1], values[7]); // The first element was changed to 7.
    assert_eq!(verify_array[2], values[0]); // The second element was changed to 0.
    assert_eq!(verify_array[3], values[1]); // The third element was changed to 1.
    assert_eq!(verify_array[4], values[1]); // The unmodified element remains unchanged.

    // Free GPU memory.
    stream.free(src_array.values);
    stream.free(dst_array.values);
}
