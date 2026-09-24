use std::collections::BTreeMap;

use zkpoly_cuda_api::mem::{alloc_pinned, free_pinned};

struct SanityChecker {
    /// A mapping from start to end of allocated ranges.
    ranges: BTreeMap<usize, usize>,
}

impl SanityChecker {
    pub fn new() -> Self {
        Self {
            ranges: BTreeMap::new(),
        }
    }

    pub fn allocate(&mut self, left: usize, right: usize) {
        let gap = self.ranges.lower_bound(std::ops::Bound::Included(&left));
        if let Some((pred_start, pred_end)) = gap.peek_prev() {
            if !(pred_end <= &left) {
                panic!(
                    "range collision detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                    left, right, pred_start, pred_end
                )
            }
        }
        if let Some((next_start, next_end)) = gap.peek_next() {
            if !(next_start >= &right) {
                panic!(
                    "range collision detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                    left, right, next_start, next_end
                )
            }
        }
        self.ranges.insert(left, right);
    }

    pub fn free(&mut self, left: usize) {
        if !self.ranges.contains_key(&left) {
            panic!("freeing unallocated range [{},?)", left);
        }
        self.ranges.remove(&left);
    }
}

pub struct CpuStaticAllocator {
    capacity: usize,
    checker: Option<SanityChecker>,
    base_ptr: *mut u8,
    primary: bool,
}

unsafe impl Send for CpuStaticAllocator {}

impl CpuStaticAllocator {
    pub fn new(capacity: usize, check: bool) -> Self {
        Self {
            capacity,
            checker: if check {
                Some(SanityChecker::new())
            } else {
                None
            },
            base_ptr: unsafe { alloc_pinned(capacity) },
            primary: true,
        }
    }

    /// Allocate CUDA pinned host memory from a thread constrained to `cpus`.
    ///
    /// CUDA's host allocator follows the allocating thread's Linux memory policy on
    /// the machines we target.  Keeping the affinity in this short-lived thread
    /// makes the placement policy explicit without changing the affinity of the
    /// scheduler or its executor threads.
    pub fn new_on_cpu_set(capacity: usize, check: bool, cpus: Vec<usize>) -> Self {
        assert!(!cpus.is_empty(), "CPU affinity set for pinned allocation is empty");

        std::thread::spawn(move || {
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            for cpu in &cpus {
                unsafe {
                    libc::CPU_SET(*cpu, &mut set);
                }
            }
            let result = unsafe {
                libc::sched_setaffinity(
                    0,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &set,
                )
            };
            assert_eq!(
                result,
                0,
                "failed to set affinity for NUMA-local pinned allocation: {}",
                std::io::Error::last_os_error()
            );

            let allocator = Self::new(capacity, check);
            println!(
                "allocated {} GiB of pinned CPU memory on CPUs {:?} at {:p}",
                capacity as f64 / 2f64.powi(30),
                cpus,
                allocator.base_ptr
            );
            allocator
        })
        .join()
        .expect("NUMA-local pinned allocation thread panicked")
    }

    pub fn slice(&self, offset: usize, size: usize) -> Self {
        let base_ptr = unsafe { self.base_ptr.byte_add(offset) };
        Self {
            capacity: size,
            checker: self.checker.as_ref().map(|_| SanityChecker::new()),
            base_ptr,
            primary: false,
        }
    }

    pub fn allocate(&mut self, offset: usize, size: usize) -> *mut u8 {
        let left = offset;
        let right = offset + size;
        if let Some(checker) = &mut self.checker {
            checker.allocate(left, right);
        }
        if right > self.capacity {
            panic!(
                "trying to allocate [{}, {}), but capacity is {}",
                left, right, self.capacity
            );
        }

        unsafe { self.base_ptr.add(left) }
    }

    pub fn deallocate(&mut self, ptr: *mut u8) {
        if let Some(checker) = &mut self.checker {
            let offset = unsafe { ptr.offset_from(self.base_ptr) };
            checker.free(offset.try_into().expect("negative offset"));
        }
    }
}

impl Drop for CpuStaticAllocator {
    fn drop(&mut self) {
        if self.primary {
            unsafe {
                free_pinned(self.base_ptr, self.capacity);
            }
        }
    }
}
