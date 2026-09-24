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
fn test_basic_slice_operations() {
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
    let (array, _pool) = create_cpu_array(values);

    // Test basic slicing.
    let slice = array.slice(2, 6);
    assert_eq!(slice.len, 4);
    for i in 0..4 {
        assert_eq!(slice[i], values[i + 2]);
    }

    // Test slice range validation.
    let slice = array.slice(0, 8);
    assert_eq!(slice.len, 8);
    for i in 0..8 {
        assert_eq!(slice[i], values[i]);
    }
}

#[test]
#[should_panic(expected = "slice can't be sliced again")]
fn test_nested_slice_panic() {
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
    let (array, _pool) = create_cpu_array(values);
    let slice = array.slice(2, 6);
    let _ = slice.slice(1, 2); // This should panic.
}

#[test]
fn test_rotate_with_slice() {
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

    // Rotate the original array.
    array.rotate(2);
    let slice = array.slice(1, 5);

    // Verify the slice contents.
    for i in 0..4 {
        assert_eq!(slice[i], array[(i + 1) % 8]);
    }
}

#[test]
fn test_cpu_to_cpu_transfer() {
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
    let (src, mut pool) = create_cpu_array(values);

    // Test contiguous memory transfer.
    let ptr = pool.allocate(8);
    let mut dst = ScalarArray::new(8, ptr, DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::cpu2cpu(&src, &mut dst);
    for i in 0..8 {
        assert_eq!(dst[i], values[i]);
    }

    // Test transfer with a slice.
    let slice = src.slice(2, 6);
    let ptr = pool.allocate(4);
    let mut dst = ScalarArray::new(4, ptr, DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::cpu2cpu(&slice, &mut dst);
    for i in 0..4 {
        assert_eq!(dst[i], values[i + 2]);
    }
}

#[test]
fn test_gpu_transfers() {
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

    // Test CPU-to-GPU transfer.
    let (cpu_array, _pool) = create_cpu_array(values);
    let gpu_array = create_gpu_array(values, &stream);

    // Test transfer with a slice.
    let slice = cpu_array.slice(2, 6);
    let ptr = stream.allocate(4);
    let mut gpu_slice = ScalarArray::new(
        4,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    <ScalarArray<TestField> as Transfer>::cpu2gpu(&slice, &mut gpu_slice, &stream);

    // Verify slice transfer.
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(4, verify_pool.allocate(4), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&gpu_slice, &mut verify_array, &stream);
    stream.sync();

    for i in 0..4 {
        assert_eq!(verify_array[i], values[i + 2]);
    }

    // Free GPU memory.
    stream.free(gpu_array.values);
    stream.free(gpu_slice.values);
}

#[test]
fn test_boundary_conditions() {
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
    let (array, _pool) = create_cpu_array(values);

    // Test an empty slice.
    let slice = array.slice(4, 4);
    assert_eq!(slice.len, 0);

    // Test a slice covering the entire array.
    let slice = array.slice(0, 8);
    assert_eq!(slice.len, 8);
    for i in 0..8 {
        assert_eq!(slice[i], values[i]);
    }
}

#[test]
#[should_panic(expected = "assertion failed")]
fn test_invalid_slice_indices() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
    ];
    let (array, _pool) = create_cpu_array(values);
    let _ = array.slice(3, 2); // The start index is greater than the end index.
}

#[test]
#[should_panic(expected = "assertion failed")]
fn test_out_of_bounds_slice() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
    ];
    let (array, _pool) = create_cpu_array(values);
    let _ = array.slice(0, 5); // The end index is outside the array bounds.
}

#[test]
fn test_gpu_slice_memory_safety() {
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

    let gpu_array = create_gpu_array(values, &stream);
    let slice = gpu_array.slice(2, 6);

    // Verify that slicing does not modify the original data.
    let ptr = stream.allocate(4);
    let mut gpu_slice = ScalarArray::new(
        4,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(4, verify_pool.allocate(4), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&slice, &mut gpu_slice, &stream);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&gpu_slice, &mut verify_array, &stream);
    stream.sync();

    for i in 0..4 {
        assert_eq!(verify_array[i], values[i + 2]);
    }

    // Free GPU memory.
    stream.free(gpu_array.values);
    stream.free(gpu_slice.values);
}
