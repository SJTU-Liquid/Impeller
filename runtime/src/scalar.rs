use core::panic;
use std::{
    ops::{Index, IndexMut},
    ptr::copy_nonoverlapping,
};

use group::ff::Field;
use rand_core::RngCore;
use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::{
    bindings::cudaDeviceSynchronize,
    cuda_check,
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};
use zkpoly_memory_pool::{
    buddy_disk_pool::{
        cpu_read_from_disk, cpu_write_to_disk, gpu_read_chunk_from_disk, gpu_read_from_disk,
        gpu_write_chunk_to_disk, gpu_write_to_disk, DiskAllocInfo,
    },
    BuddyDiskPool, CpuMemoryPool,
};

use crate::runtime::transfer::Transfer;

#[derive(Clone)]
pub struct Scalar<F: Field> {
    pub value: *mut F,
    pub device: DeviceType,
}

impl<F: Field> std::fmt::Debug for Scalar<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let scalar_val = if self.device == DeviceType::CPU {
            unsafe { self.value.as_ref().unwrap().clone() }
        } else if let DeviceType::GPU { device_id } = self.device {
            unsafe { cuda_check!(cudaDeviceSynchronize()) };
            let mut cpu_scalar = Scalar::new_cpu();
            let stream = CudaStream::new(device_id);
            self.gpu2cpu(&mut cpu_scalar, &stream);
            stream.sync();
            let ret = unsafe { cpu_scalar.value.as_ref().unwrap().clone() };
            cpu_scalar.deallocate();
            ret
        } else {
            panic!("scalar on disk can't be printed");
        };
        f.debug_struct("Scalar")
            .field("value_ptr", &self.value)
            .field("device", &self.device)
            .field("value", &scalar_val)
            .finish()
    }
}

#[test]
fn test_scalar_debug() {
    // test scalar print
    use halo2curves::bn256::Fr as F;
    let mut s = Scalar::<F>::new_cpu();
    *s.as_mut() = F::from(42);
    println!("{:?}", s); // should print 0x2a
    let stream = CudaStream::new(0);
    let mut s_gpu = Scalar::new_gpu(stream.allocate(std::mem::size_of::<F>()), 0);
    s.cpu2gpu(&mut s_gpu, &stream);
    println!("{:?}", s_gpu); // should print 0x2a
}

unsafe impl<F: Field> Send for Scalar<F> {}
unsafe impl<F: Field> Sync for Scalar<F> {}

impl<F: Field> Scalar<F> {
    pub fn new_cpu() -> Self {
        Self {
            value: unsafe { alloc_pinned(1) },
            device: DeviceType::CPU,
        }
    }

    pub fn deallocate(&mut self) {
        if self.device == DeviceType::CPU {
            unsafe {
                free_pinned(self.value, 1);
            }
        } else {
            panic!("deallocate is only supported for CPU");
        }
    }

    pub fn new_gpu(value: *mut F, device_id: i32) -> Self {
        Self {
            value,
            device: DeviceType::GPU { device_id },
        }
    }

    pub fn from_ff(value: &F) -> Self {
        let mut r = Self::new_cpu();
        *r.as_mut() = value.clone();
        r
    }

    pub fn to_ff(&self) -> F {
        unsafe {
            let mut r = F::ZERO;
            self.value.copy_to(&mut r, 1);
            r
        }
    }

    pub fn as_ref(&self) -> &F {
        assert!(self.device == DeviceType::CPU);
        unsafe { &*self.value }
    }

    pub fn as_mut(&mut self) -> &mut F {
        assert!(self.device == DeviceType::CPU);
        unsafe { &mut *self.value }
    }
}

impl<F: Field> Transfer for Scalar<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(self.value, target.value, 1);
        }
    }

    fn cpu2disk(&self, _: &mut Self) {
        unreachable!("scalar doesn't need to be transferred to disk");
    }

    fn disk2cpu(&self, _: &mut Self) {
        unreachable!("scalar doesn't need to be transferred from disk");
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_h2d(target.value, self.value, 1);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_d2h(target.value, self.value, 1);
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        stream.memcpy_d2d(target.value, self.value, 1);
    }
}

#[derive(Clone)]
pub struct ScalarArray<F: Field> {
    // when this is a slice, the pointer is pointed to the base array's start,
    // you have to visit the slice with slice_offset or by index
    pub values: *mut F,
    pub len: usize,
    pub(crate) rotate: i64, // the i64 is just to support neg during add, when getting rotate, we can safely assume it is positive
    pub device: DeviceType,
    pub slice_info: Option<ScalarSlice>,
    pub disk_pos: Vec<DiskAllocInfo>, // if the poly is stored on disk, this is the offset of each part in each file, (fd, offset)
}

impl<F: Field> std::fmt::Debug for ScalarArray<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cpu_array = match self.device {
            DeviceType::CPU => None,
            DeviceType::GPU { device_id } => {
                unsafe { cuda_check!(cudaDeviceSynchronize()) };
                let mut cpu_array = ScalarArray::alloc_std(self.len());
                let stream = CudaStream::new(device_id);
                self.gpu2cpu(&mut cpu_array, &stream);
                stream.sync();
                Some(cpu_array)
            }
            DeviceType::Disk => unimplemented!(),
        };

        let array_ref = if let Some(cpu_array) = cpu_array.as_ref() {
            &cpu_array
        } else {
            self
        };

        struct Array<F: Field>(*mut F, usize);
        impl<F: Field> std::fmt::Debug for Array<F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                for i in 0..self.1 {
                    write!(f, "{:?},", unsafe { self.0.add(i).as_ref().unwrap() })?;
                }
                Ok(())
            }
        }
        let res = f
            .debug_struct("ScalarArray")
            .field("values_ptr", &self.values)
            .field("len", &self.len)
            .field("rotate", &self.rotate)
            .field("device", &self.device)
            .field("slice_info", &self.slice_info)
            .field("values", &Array(array_ref.values, array_ref.len))
            .finish();
        // free cpu_array if allocated
        if let Some(cpu_array) = cpu_array {
            unsafe { cpu_array.dealloc_std() };
        }
        res
    }
}

#[test]
fn test_scalar_array_debug() {
    use halo2curves::bn256::Fr as F;
    let stream = CudaStream::new(0);
    let mut cpu_array = ScalarArray::alloc_std(4);
    for i in 0..4 {
        cpu_array[i] = F::from(i as u64);
    }
    println!("{:?}", cpu_array); // should print [0,1,2,3]
    let mut gpu_array = ScalarArray::new(
        4,
        stream.allocate(4 * std::mem::size_of::<F>()),
        DeviceType::GPU { device_id: 0 },
    );
    cpu_array.cpu2gpu(&mut gpu_array, &stream);
    println!("{:?}", gpu_array); // should print [0,1,2,3]

    let slice = gpu_array.slice(1, 3);
    println!("{:?}", slice); // should print [1,2]
}
#[derive(Debug, Clone)]
pub struct ScalarSlice {
    pub offset: usize,
    pub whole_len: usize,
}

unsafe impl<F: Field> Send for ScalarArray<F> {}
unsafe impl<F: Field> Sync for ScalarArray<F> {}

impl<F: Field> PartialEq for ScalarArray<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

const DISK_ALIGNMENT: usize = 4096;

impl<F: Field> ScalarArray<F> {
    pub fn new(len: usize, ptr: *mut F, device: DeviceType) -> Self {
        Self {
            values: ptr,
            len,
            rotate: 0,
            device,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn alloc_std(len: usize) -> Self {
        let ptr = unsafe { std::alloc::alloc(std::alloc::Layout::array::<F>(len).unwrap()) };

        Self::new(len, ptr as *mut F, DeviceType::CPU)
    }

    pub unsafe fn dealloc_std(self) {
        std::alloc::dealloc(
            self.values as *mut u8,
            std::alloc::Layout::array::<F>(self.len).unwrap(),
        );
    }

    pub fn alloc_cpu(len: usize, allocator: &mut CpuMemoryPool) -> Self {
        let ptr = allocator.allocate(len);
        Self {
            values: ptr,
            len,
            rotate: 0,
            device: DeviceType::CPU,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn alloc_disk(len: usize, allocator: &mut Vec<BuddyDiskPool>) -> Self {
        // assert!(len % allocator.len() == 0);
        let part_len = len / allocator.len();
        let disk_pose = allocator
            .iter_mut()
            .map(|pool| {
                let pos = pool.allocate(part_len * size_of::<F>()).unwrap();
                DiskAllocInfo::new(pos, pool)
            })
            .collect::<Vec<_>>();
        Self {
            values: std::ptr::null_mut(),
            len,
            rotate: 0,
            device: DeviceType::Disk,
            slice_info: None,
            disk_pos: disk_pose,
        }
    }

    pub fn free_disk(&mut self, allocator: &mut Vec<BuddyDiskPool>) {
        allocator
            .iter_mut()
            .zip(self.disk_pos.iter())
            .for_each(|(disk_pool, dai)| {
                disk_pool
                    .deallocate(dai.offset)
                    .expect("deallocation failed");
            });
    }

    pub fn from_vec(v: &[F], allocator: &mut CpuMemoryPool) -> Self {
        let r = Self::alloc_cpu(v.len(), allocator);
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), r.values, v.len());
        }
        r
    }

    pub fn borrow_vec(v: &[F]) -> Self {
        Self {
            values: v.as_ptr() as *mut F,
            len: v.len(),
            rotate: 0,
            device: DeviceType::CPU,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn from_iter(
        v: impl Iterator<Item = F>,
        len: usize,
        allocator: &mut CpuMemoryPool,
    ) -> Self {
        let r = Self::alloc_cpu(len, allocator);
        for (i, x) in v.take(len).enumerate() {
            unsafe {
                *r.values.add(i) = x;
            }
        }
        r
    }

    pub fn blind(&mut self, start: usize, end: usize, rng: impl RngCore + Clone) {
        for i in start..end {
            self[i] = F::random(rng.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get_rotation(&self) -> usize {
        self.rotate as usize
    }

    pub fn set_slice_raw(&self, mut offset: usize, len: usize) -> Option<Self> {
        let whole_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        offset = offset % whole_len;
        if len > whole_len || offset >= whole_len {
            eprintln!(
                "get invalid slice: offset {}, len {}, whole_len {}",
                offset, len, whole_len
            );
            return None;
        }

        let slice_info = if len == whole_len && offset == 0 {
            None
        } else {
            Some(ScalarSlice {
                offset,
                whole_len: if self.slice_info.is_none() {
                    self.len
                } else {
                    self.slice_info.as_ref().unwrap().whole_len
                },
            })
        };

        Some(Self {
            values: self.values.clone(),
            len,
            rotate: 0,
            device: self.device.clone(),
            slice_info,
            disk_pos: self.disk_pos.clone(),
        })
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= self.len);
        if let Some(si) = &self.slice_info {
            panic!("this is already a slice {:?}, can't be sliced again", si)
        }
        let rotate = self.get_rotation();
        let actual_pos = (self.len + start - rotate) % self.len;
        Self {
            values: self.values,
            len: end - start,
            rotate: 0, // slice can't rotate
            device: self.device.clone(),
            slice_info: Some(ScalarSlice {
                offset: actual_pos,
                whole_len: self.len,
            }),
            disk_pos: self.disk_pos.clone(),
        }
    }

    pub fn as_ref(&self) -> &[F] {
        assert!(self.slice_info.is_none());
        assert!(self.rotate == 0);
        unsafe { std::slice::from_raw_parts(self.values, self.len) }
    }

    pub fn as_mut(&mut self) -> &mut [F] {
        assert!(self.slice_info.is_none());
        assert!(self.rotate == 0);
        unsafe { std::slice::from_raw_parts_mut(self.values, self.len) }
    }

    pub fn rotate(&mut self, shift: i64) {
        assert!(self.slice_info.is_none());
        self.rotate = (self.rotate + shift) % self.len as i64;
        if self.rotate < 0 {
            self.rotate += self.len as i64
        }
    }

    pub fn iter(&self) -> ScalarArrayIter<'_, F> {
        let mod_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        ScalarArrayIter {
            array: self,
            pos: 0,
            mod_len,
        }
    }

    pub fn iter_mut(&mut self) -> ScalarArrayIterMut<'_, F> {
        let mod_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        ScalarArrayIterMut {
            array: self,
            pos: 0,
            mod_len,
        }
    }

    // helper function for transfer check
    pub fn check_target_len(&self, target: &Self) {
        if self.len != target.len {
            if self.len >= target.len {
                panic!(
                    "source array length {} is larger than target array length {}",
                    self.len, target.len
                );
            }
            assert!(self.len < target.len);
            assert!(target.rotate == 0);
            assert!(target.slice_info.is_none());
        }
    }

    /// Split the first `logical_len` values into at most two contiguous physical ranges.
    fn physical_segments(&self, logical_len: usize) -> ([(usize, usize); 2], usize) {
        assert!(logical_len <= self.len);
        if logical_len == 0 {
            return ([(0, 0); 2], 0);
        }

        let (offset, whole_len) = match &self.slice_info {
            Some(slice) => (slice.offset, slice.whole_len),
            None => (0, self.len),
        };
        assert!(whole_len > 0);
        assert!(logical_len <= whole_len);

        let rotation = self.rotate.rem_euclid(whole_len as i64) as usize;
        let start = (offset + whole_len - rotation) % whole_len;
        let first_len = logical_len.min(whole_len - start);
        let second_len = logical_len - first_len;
        (
            [(start, first_len), (0, second_len)],
            if second_len == 0 { 1 } else { 2 },
        )
    }

    /// Match source and destination ranges into `(src_offset, dst_offset, len)` copies.
    fn transfer_chunks(&self, target: &Self) -> ([(usize, usize, usize); 3], usize) {
        self.check_target_len(target);
        let (src_segments, src_count) = self.physical_segments(self.len);
        let (dst_segments, dst_count) = target.physical_segments(self.len);
        let mut chunks = [(0, 0, 0); 3];
        let (mut src_idx, mut dst_idx, mut src_pos, mut dst_pos, mut chunk_count) = (0, 0, 0, 0, 0);

        while src_idx < src_count && dst_idx < dst_count {
            let src_remaining = src_segments[src_idx].1 - src_pos;
            let dst_remaining = dst_segments[dst_idx].1 - dst_pos;
            let chunk_len = src_remaining.min(dst_remaining);
            debug_assert!(chunk_count < chunks.len());
            chunks[chunk_count] = (
                src_segments[src_idx].0 + src_pos,
                dst_segments[dst_idx].0 + dst_pos,
                chunk_len,
            );
            chunk_count += 1;
            src_pos += chunk_len;
            dst_pos += chunk_len;

            if src_pos == src_segments[src_idx].1 {
                src_idx += 1;
                src_pos = 0;
            }
            if dst_pos == dst_segments[dst_idx].1 {
                dst_idx += 1;
                dst_pos = 0;
            }
        }

        debug_assert_eq!(
            chunks[..chunk_count]
                .iter()
                .map(|(_, _, len)| len)
                .sum::<usize>(),
            self.len
        );
        (chunks, chunk_count)
    }

    pub fn get_ptr(&self, index: usize) -> *mut F {
        assert!(index < self.len);
        let (offset, mod_len) = if self.slice_info.is_none() {
            (0, self.len)
        } else {
            let slice_info = self.slice_info.as_ref().unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        let rotate = self.rotate as usize;
        let actual_pos = (index + mod_len + offset - rotate) % mod_len;
        unsafe { self.values.add(actual_pos) }
    }
}

pub struct ScalarArrayIter<'a, F: Field> {
    array: &'a ScalarArray<F>,
    pos: usize,
    mod_len: usize,
}

pub struct ScalarArrayIterMut<'a, F: Field> {
    array: &'a mut ScalarArray<F>,
    pos: usize,
    mod_len: usize,
}

impl<'a, F: Field> Iterator for ScalarArrayIter<'a, F> {
    type Item = &'a F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len {
            None
        } else {
            let rotate = self.array.rotate as usize;
            let offset = if self.array.slice_info.is_none() {
                0
            } else {
                self.array.slice_info.as_ref().unwrap().offset
            };
            let actual_pos = (self.pos + self.mod_len + offset - rotate) % self.mod_len;
            self.pos += 1;
            Some(unsafe { &*self.array.values.add(actual_pos) })
        }
    }
}

impl<'a, F: Field> Iterator for ScalarArrayIterMut<'a, F> {
    type Item = &'a mut F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len {
            None
        } else {
            let rotate = self.array.rotate as usize;
            let offset = if self.array.slice_info.is_none() {
                0
            } else {
                self.array.slice_info.as_ref().unwrap().offset
            };
            let actual_pos = (self.pos + self.mod_len + offset - rotate) % self.mod_len;
            self.pos += 1;
            Some(unsafe { &mut *self.array.values.add(actual_pos) })
        }
    }
}

impl<F: Field> Index<usize> for ScalarArray<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.get_ptr(index) }
    }
}

impl<F: Field> IndexMut<usize> for ScalarArray<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.get_ptr(index) }
    }
}

impl<F: Field> Transfer for ScalarArray<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);
        let (chunks, count) = self.transfer_chunks(target);
        for &(src_offset, dst_offset, len) in &chunks[..count] {
            unsafe {
                copy_nonoverlapping(
                    self.values.add(src_offset),
                    target.values.add(dst_offset),
                    len,
                );
            }
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                },
            "target device {:?} is not GPU with stream device {}",
            target.device,
            stream.get_device()
        );
        let (chunks, count) = self.transfer_chunks(target);
        for &(src_offset, dst_offset, len) in &chunks[..count] {
            unsafe {
                stream.memcpy_h2d(
                    target.values.add(dst_offset),
                    self.values.add(src_offset),
                    len,
                );
            }
        }
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        let (chunks, count) = self.transfer_chunks(target);
        for &(src_offset, dst_offset, len) in &chunks[..count] {
            unsafe {
                stream.memcpy_d2h(
                    target.values.add(dst_offset),
                    self.values.add(src_offset),
                    len,
                );
            }
        }
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        let (chunks, count) = self.transfer_chunks(target);
        for &(src_offset, dst_offset, len) in &chunks[..count] {
            unsafe {
                stream.memcpy_d2d(
                    target.values.add(dst_offset),
                    self.values.add(src_offset),
                    len,
                );
            }
        }
    }

    fn cpu2disk(&self, target: &mut Self) {
        assert_eq!(self.device, DeviceType::CPU);
        assert_eq!(target.device, DeviceType::Disk);
        assert_eq!(self.rotate, target.rotate);
        assert!(self.slice_info.is_none());
        assert!(target.slice_info.is_none());
        let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
        cpu_write_to_disk(self.values as *const u8, &target.disk_pos, alligned_size);
    }

    fn disk2cpu(&self, target: &mut Self) {
        assert_eq!(self.device, DeviceType::Disk);
        assert_eq!(target.device, DeviceType::CPU);
        assert_eq!(self.rotate, target.rotate);
        assert!(self.slice_info.is_none());
        assert!(target.slice_info.is_none());
        let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
        cpu_read_from_disk(target.values as *mut u8, &self.disk_pos, alligned_size);
    }

    fn gpu2disk(&self, target: &mut Self) {
        if let DeviceType::GPU { device_id } = self.device {
            assert_eq!(target.device, DeviceType::Disk);
            assert_eq!(self.rotate, target.rotate);
            assert!(self.slice_info.is_none());
            assert!(!target.disk_pos.is_empty());
            if self.len == 0 {
                return;
            }
            let elem_size = size_of::<F>();
            if target.slice_info.is_none() {
                let alligned_size = (self.len * elem_size).next_multiple_of(4096);
                gpu_write_to_disk(
                    self.values as *const u8,
                    &target.disk_pos,
                    alligned_size,
                    device_id,
                );
            } else {
                assert_eq!(self.len, target.len);
                let slice_info = target.slice_info.as_ref().unwrap();
                assert!(slice_info.whole_len > 0);
                assert!(target.len <= slice_info.whole_len);
                let disk_count = target.disk_pos.len();
                assert_eq!(slice_info.whole_len % disk_count, 0);
                let per_disk_elems = slice_info.whole_len / disk_count;
                let per_disk_bytes = per_disk_elems * elem_size;
                let mut slice_offset = slice_info.offset % slice_info.whole_len;
                let mut remaining = self.len;
                let mut gpu_offset = 0usize;
                let src_ptr = self.values as *const u8;
                while remaining > 0 {
                    let available = slice_info.whole_len - slice_offset;
                    let chunk_elems = remaining.min(available);
                    let chunk_bytes_total = chunk_elems * elem_size;
                    assert_eq!((slice_offset * elem_size) % DISK_ALIGNMENT, 0);
                    assert_eq!(chunk_bytes_total % DISK_ALIGNMENT, 0);
                    let mut segment_start = slice_offset * elem_size;
                    let mut bytes_left = chunk_bytes_total;
                    while bytes_left > 0 {
                        let disk_idx = segment_start / per_disk_bytes;
                        assert!(disk_idx < disk_count);
                        let disk_offset = segment_start % per_disk_bytes;
                        let chunk_bytes = bytes_left.min(per_disk_bytes - disk_offset);
                        let alloc = &target.disk_pos[disk_idx];
                        let file_offset = alloc.offset + disk_offset;
                        assert_eq!(file_offset % DISK_ALIGNMENT, 0);
                        assert_eq!(chunk_bytes % DISK_ALIGNMENT, 0);
                        gpu_write_chunk_to_disk(
                            src_ptr,
                            alloc,
                            chunk_bytes,
                            file_offset,
                            gpu_offset,
                            device_id,
                        );
                        segment_start += chunk_bytes;
                        bytes_left -= chunk_bytes;
                        gpu_offset += chunk_bytes;
                    }
                    remaining -= chunk_elems;
                    slice_offset = 0;
                }
                debug_assert_eq!(gpu_offset, self.len * elem_size);
            }
        } else {
            panic!("gpu2disk is only supported for GPU arrays");
        }
    }

    fn disk2gpu(&self, target: &mut Self) {
        if let DeviceType::GPU { device_id } = target.device {
            assert_eq!(self.device, DeviceType::Disk);
            assert_eq!(self.rotate, target.rotate);
            assert!(target.slice_info.is_none());
            assert!(!self.disk_pos.is_empty());
            assert!(target.len >= self.len);
            if self.len == 0 {
                return;
            }
            let elem_size = size_of::<F>();
            if self.slice_info.is_none() {
                let alligned_size = (self.len * elem_size).next_multiple_of(4096);
                gpu_read_from_disk(
                    target.values as *mut u8,
                    &self.disk_pos,
                    alligned_size,
                    device_id,
                );
            } else {
                let slice_info = self.slice_info.as_ref().unwrap();
                assert!(slice_info.whole_len > 0);
                assert!(self.len <= slice_info.whole_len);
                let disk_count = self.disk_pos.len();
                assert_eq!(slice_info.whole_len % disk_count, 0);
                let per_disk_elems = slice_info.whole_len / disk_count;
                let per_disk_bytes = per_disk_elems * elem_size;
                let mut slice_offset = slice_info.offset % slice_info.whole_len;
                let mut remaining = self.len;
                let mut gpu_offset = 0usize;
                let target_ptr = target.values as *mut u8;
                while remaining > 0 {
                    let available = slice_info.whole_len - slice_offset;
                    let chunk_elems = remaining.min(available);
                    let chunk_bytes_total = chunk_elems * elem_size;
                    assert_eq!((slice_offset * elem_size) % DISK_ALIGNMENT, 0);
                    assert_eq!(chunk_bytes_total % DISK_ALIGNMENT, 0);
                    let mut segment_start = slice_offset * elem_size;
                    let mut bytes_left = chunk_bytes_total;
                    while bytes_left > 0 {
                        let disk_idx = segment_start / per_disk_bytes;
                        assert!(disk_idx < disk_count);
                        let disk_offset = segment_start % per_disk_bytes;
                        let chunk_bytes = bytes_left.min(per_disk_bytes - disk_offset);
                        let alloc = &self.disk_pos[disk_idx];
                        let file_offset = alloc.offset + disk_offset;
                        assert_eq!(file_offset % DISK_ALIGNMENT, 0);
                        assert_eq!(chunk_bytes % DISK_ALIGNMENT, 0);
                        gpu_read_chunk_from_disk(
                            target_ptr,
                            alloc,
                            chunk_bytes,
                            file_offset,
                            gpu_offset,
                            device_id,
                        );
                        segment_start += chunk_bytes;
                        bytes_left -= chunk_bytes;
                        gpu_offset += chunk_bytes;
                    }
                    remaining -= chunk_elems;
                    slice_offset = 0;
                }
                debug_assert_eq!(gpu_offset, self.len * elem_size);
            }
        } else {
            panic!("disk2gpu is only supported for GPU arrays");
        }
    }
}

#[test]
fn test_transfer_rotated_array_to_wrapping_cpu_slice_exhaustive() {
    use halo2curves::bn256::Fr as F;

    const SRC_LEN: usize = 8;
    const DST_LEN: usize = 13;

    for src_rotation in 0..SRC_LEN {
        for dst_offset in 0..DST_LEN {
            let mut src = ScalarArray::<F>::alloc_std(SRC_LEN);
            for i in 0..SRC_LEN {
                src[i] = F::from(i as u64);
            }
            src.rotate(src_rotation as i64);
            let expected = src.iter().copied().collect::<Vec<_>>();

            let mut dst = ScalarArray::<F>::alloc_std(DST_LEN);
            for i in 0..DST_LEN {
                dst[i] = F::from(1000 + i as u64);
            }
            let mut dst_slice = dst.set_slice_raw(dst_offset, SRC_LEN).unwrap();

            src.cpu2cpu(&mut dst_slice);

            assert_eq!(
                dst_slice.iter().copied().collect::<Vec<_>>(),
                expected,
                "source rotation {src_rotation}, destination offset {dst_offset}"
            );

            unsafe {
                src.dealloc_std();
                dst.dealloc_std();
            }
        }
    }

    for src_offset in 0..DST_LEN {
        for dst_rotation in 0..SRC_LEN {
            let mut src = ScalarArray::<F>::alloc_std(DST_LEN);
            for i in 0..DST_LEN {
                src[i] = F::from(i as u64);
            }
            let src_slice = src.set_slice_raw(src_offset, SRC_LEN).unwrap();
            let expected = src_slice.iter().copied().collect::<Vec<_>>();

            let mut dst = ScalarArray::<F>::alloc_std(SRC_LEN);
            dst.rotate(dst_rotation as i64);
            src_slice.cpu2cpu(&mut dst);

            assert_eq!(
                dst.iter().copied().collect::<Vec<_>>(),
                expected,
                "source offset {src_offset}, destination rotation {dst_rotation}"
            );

            unsafe {
                src.dealloc_std();
                dst.dealloc_std();
            }
        }
    }

    const SECOND_DST_LEN: usize = 17;
    for src_offset in 0..DST_LEN {
        for dst_offset in 0..SECOND_DST_LEN {
            let mut src = ScalarArray::<F>::alloc_std(DST_LEN);
            for i in 0..DST_LEN {
                src[i] = F::from(i as u64);
            }
            let src_slice = src.set_slice_raw(src_offset, SRC_LEN).unwrap();
            let expected = src_slice.iter().copied().collect::<Vec<_>>();

            let dst = ScalarArray::<F>::alloc_std(SECOND_DST_LEN);
            let mut dst_slice = dst.set_slice_raw(dst_offset, SRC_LEN).unwrap();
            src_slice.cpu2cpu(&mut dst_slice);

            assert_eq!(
                dst_slice.iter().copied().collect::<Vec<_>>(),
                expected,
                "source offset {src_offset}, destination offset {dst_offset}"
            );

            unsafe {
                src.dealloc_std();
                dst.dealloc_std();
            }
        }
    }
}

#[test]
fn test_transfer_rotated_gpu_array_to_wrapping_gpu_slice_exhaustive() {
    use halo2curves::bn256::Fr as F;

    const SRC_LEN: usize = 8;
    const DST_LEN: usize = 13;

    let stream = CudaStream::new(0);
    for src_rotation in 0..SRC_LEN {
        for dst_offset in 0..DST_LEN {
            let mut src_cpu = ScalarArray::<F>::alloc_std(SRC_LEN);
            for i in 0..SRC_LEN {
                src_cpu[i] = F::from(i as u64);
            }

            let mut src_gpu = ScalarArray::new(
                SRC_LEN,
                stream.allocate(SRC_LEN),
                DeviceType::GPU {
                    device_id: stream.get_device(),
                },
            );
            src_cpu.cpu2gpu(&mut src_gpu, &stream);
            src_gpu.rotate(src_rotation as i64);

            let mut dst_cpu = ScalarArray::<F>::alloc_std(DST_LEN);
            for i in 0..DST_LEN {
                dst_cpu[i] = F::from(1000 + i as u64);
            }
            let mut dst_gpu = ScalarArray::new(
                DST_LEN,
                stream.allocate(DST_LEN),
                DeviceType::GPU {
                    device_id: stream.get_device(),
                },
            );
            dst_cpu.cpu2gpu(&mut dst_gpu, &stream);
            let mut dst_slice = dst_gpu.set_slice_raw(dst_offset, SRC_LEN).unwrap();
            src_gpu.gpu2gpu(&mut dst_slice, &stream);

            dst_gpu.gpu2cpu(&mut dst_cpu, &stream);
            stream.sync();

            let mut touched = [false; DST_LEN];
            for i in 0..SRC_LEN {
                let src_idx = (i + SRC_LEN - src_rotation) % SRC_LEN;
                let dst_idx = (dst_offset + i) % DST_LEN;
                touched[dst_idx] = true;
                assert_eq!(
                    dst_cpu[dst_idx], src_cpu[src_idx],
                    "source rotation {src_rotation}, destination offset {dst_offset}, \
                     logical index {i}"
                );
            }
            for i in 0..DST_LEN {
                if !touched[i] {
                    assert_eq!(
                        dst_cpu[i],
                        F::from(1000 + i as u64),
                        "source rotation {src_rotation}, destination offset {dst_offset}, \
                         untouched index {i}"
                    );
                }
            }

            stream.free(src_gpu.values);
            stream.free(dst_gpu.values);
            unsafe {
                src_cpu.dealloc_std();
                dst_cpu.dealloc_std();
            }
        }
    }
}

#[test]
fn test_tranfer_cpu_disk() {
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(10, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(28), None).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some(std::env::temp_dir())).unwrap(),
    ];
    let mut array1 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array1.iter_mut().for_each(|v| {
        *v = F::random(rand_core::OsRng);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array1.cpu2disk(&mut array2);
    let mut array3 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array2.disk2cpu(&mut array3);
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }
}

#[test]
fn test_tranfer_cpu_disk_large() {
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(26, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), None).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some(std::env::temp_dir())).unwrap(),
    ];
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(26), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(26), &mut disk_pool);
    array1.cpu2disk(&mut array2);
    let mut array3 = ScalarArray::<F>::alloc_cpu(2usize.pow(26), &mut cpu_pool);
    array2.disk2cpu(&mut array3);
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }
}

#[test]
fn test_transfer_bandwidth_2gb() {
    use halo2curves::bn256::Fr as F;
    use std::mem::size_of;
    use std::time::Instant;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    // 2GB = 2 * 1024 * 1024 * 1024 bytes
    let bytes_2gb = 2usize * 1024 * 1024 * 1024;
    let n_elem = bytes_2gb / size_of::<F>();

    // Allocate CPU and disk pools.
    let temp_dir = std::env::temp_dir();
    let secondary_temp_dir = std::env::var_os("IMPELLER_SECONDARY_TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let mut cpu_pool = CpuMemoryPool::new((bytes_2gb as f64).log2().ceil() as u32, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(bytes_2gb, Some(temp_dir)).unwrap(),
        BuddyDiskPool::new(bytes_2gb, Some(secondary_temp_dir)).unwrap(),
    ];

    // Allocate and initialize the data.
    let mut array1 = ScalarArray::<F>::alloc_cpu(n_elem, &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });

    // Write to disk and measure the elapsed time.
    let mut array2 = ScalarArray::<F>::alloc_disk(n_elem, &mut disk_pool);
    let start_write = Instant::now();
    array1.cpu2disk(&mut array2);
    let elapsed_write = start_write.elapsed().as_secs_f64();
    let write_bw = (bytes_2gb as f64) / elapsed_write / (1024.0 * 1024.0 * 1024.0);

    // Read back from disk and measure the elapsed time.
    let mut array3 = ScalarArray::<F>::alloc_cpu(n_elem, &mut cpu_pool);
    let start_read = Instant::now();
    array2.disk2cpu(&mut array3);
    let elapsed_read = start_read.elapsed().as_secs_f64();
    let read_bw = (bytes_2gb as f64) / elapsed_read / (1024.0 * 1024.0 * 1024.0);

    // Verify the data by comparing the first 1024 elements.
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }

    println!(
        "Write bandwidth: {:.2} GB/s, Read bandwidth: {:.2} GB/s",
        write_bw, read_bw
    );
}

#[test]
fn test_transfer_gpu_disk() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(10, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(28), None).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some(std::env::temp_dir())).unwrap(),
    ];
    let mut gpu_pool =
        PageAllocator::new(DeviceType::GPU { device_id: 0 }, 1024 * 1024 * 2, 2, true);
    let mut array1 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array1.iter_mut().for_each(|v| {
        *v = F::random(rand_core::OsRng);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 2, vec![0], 0);
    let mut array3 = ScalarArray::<F>::new(1024, ptr_gpu, DeviceType::GPU { device_id: 0 });
    array2.disk2gpu(&mut array3);

    let mut array4 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..1024 {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let mut array5 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array4.cpu2disk(&mut array5);
    let mut array6 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..1024 {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}

#[test]
fn test_transfer_gpu_disk_slice_roundtrip() {
    use halo2curves::bn256::Fr as F;
    use std::mem::size_of;
    use zkpoly_cuda_api::stream::CudaStream;
    use zkpoly_memory_pool::{BuddyDiskPool, CpuMemoryPool};

    let mut cpu_pool = CpuMemoryPool::new(10, size_of::<F>());
    let mut disk_pool = vec![BuddyDiskPool::new(2usize.pow(20), None).unwrap()];
    let stream = CudaStream::new(0);

    let len = 512;
    let slice_offset = 384;
    let slice_len = 256;
    let mut cpu_array = ScalarArray::<F>::alloc_cpu(len, &mut cpu_pool);
    for i in 0..len {
        cpu_array[i] = F::from(i as u64);
    }
    let mut disk_array = ScalarArray::<F>::alloc_disk(len, &mut disk_pool);
    cpu_array.cpu2disk(&mut disk_array);

    let mut disk_slice = disk_array.set_slice_raw(slice_offset, slice_len).unwrap();
    let expected_idx: Vec<_> = (0..slice_len).map(|i| (slice_offset + i) % len).collect();
    let mut gpu_slice = ScalarArray::<F>::new(
        expected_idx.len(),
        stream.allocate(expected_idx.len()),
        DeviceType::GPU { device_id: 0 },
    );

    disk_slice.disk2gpu(&mut gpu_slice);
    let mut cpu_verify = ScalarArray::<F>::alloc_cpu(expected_idx.len(), &mut cpu_pool);
    gpu_slice.gpu2cpu(&mut cpu_verify, &stream);
    stream.sync();
    for (i, idx) in expected_idx.iter().enumerate() {
        assert_eq!(cpu_verify[i], cpu_array[*idx]);
    }

    let mut cpu_update = ScalarArray::<F>::alloc_cpu(expected_idx.len(), &mut cpu_pool);
    for i in 0..expected_idx.len() {
        cpu_update[i] = F::from((100 + i) as u64);
    }
    cpu_update.cpu2gpu(&mut gpu_slice, &stream);
    stream.sync();
    gpu_slice.gpu2disk(&mut disk_slice);

    let mut cpu_after = ScalarArray::<F>::alloc_cpu(len, &mut cpu_pool);
    disk_array.disk2cpu(&mut cpu_after);
    let mut touched = vec![false; len];
    for &idx in &expected_idx {
        touched[idx] = true;
    }
    for (i, idx) in expected_idx.iter().enumerate() {
        assert_eq!(cpu_after[*idx], cpu_update[i]);
    }
    for idx in 0..len {
        if !touched[idx] {
            assert_eq!(cpu_after[idx], cpu_array[idx]);
        }
    }

    stream.free(gpu_slice.values);
}

#[test]
fn test_transfer_gpu_disk_large() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(28, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), None).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some(std::env::temp_dir())).unwrap(),
    ];
    let mut gpu_pool = PageAllocator::new(
        DeviceType::GPU { device_id: 0 },
        1024 * 1024 * 1024 * 8,
        1,
        true,
    );
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 1024 * 16, vec![0], 0);
    let mut array3 =
        ScalarArray::<F>::new(2usize.pow(28), ptr_gpu, DeviceType::GPU { device_id: 0 });
    array2.disk2gpu(&mut array3);

    let mut array4 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..2usize.pow(28) {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let mut array5 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array4.cpu2disk(&mut array5);
    let mut array6 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..2usize.pow(28) {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}

#[test]
fn test_transfer_gpu_disk_large_no_direct() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(28, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), None).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some(std::env::temp_dir())).unwrap(),
    ];
    let mut gpu_pool = PageAllocator::new(
        DeviceType::GPU { device_id: 0 },
        1024 * 1024 * 1024 * 8,
        1,
        true,
    );
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 1024 * 16, vec![0], 0);
    let array3 = ScalarArray::<F>::new(2usize.pow(28), ptr_gpu, DeviceType::GPU { device_id: 0 });

    let temp_size: usize = 1024 * 1024 * 2; // 2MB temporary buffer size
    let temp_buffer = unsafe { alloc_pinned::<u8>(temp_size) }; // 2MB temporary buffer
    let temp_buffers = vec![temp_buffer];

    // array2.disk2gpu(&mut array3);
    zkpoly_memory_pool::buddy_disk_pool::gpu_read_from_disk_no_direct(
        array3.values.cast(),
        &array2.disk_pos,
        2usize.pow(28) * size_of::<F>(),
        0,
        &temp_buffers,
        temp_size,
    );
    // gpu_write_to_disk_no_direct(array3.values.cast(), &array2.disk_pos, 2usize.pow(28) * size_of::<F>(), 0, &temp_buffers, temp_size);

    let mut array4 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..2usize.pow(28) {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let array5 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);

    // array3.gpu2disk(&mut array5);
    zkpoly_memory_pool::buddy_disk_pool::gpu_write_to_disk_no_direct(
        array3.values.cast(),
        &array5.disk_pos,
        2usize.pow(28) * size_of::<F>(),
        0,
        &temp_buffers,
        temp_size,
    );
    let mut array6 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..2usize.pow(28) {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}
