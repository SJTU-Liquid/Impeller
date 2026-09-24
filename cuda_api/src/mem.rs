use std::ffi::c_void;

use crate::{
    bindings::{cudaFreeHost, cudaMallocHost},
    cuda_check,
};

pub mod page_allocator;
pub mod static_allocator;

pub use page_allocator::PageAllocator;
pub use static_allocator::StaticAllocator;

pub struct CudaAllocator {
    pub statik: static_allocator::StaticAllocator,
    pub page: page_allocator::PageAllocator,
}

pub unsafe fn alloc_pinned<T: Sized>(len: usize) -> *mut T {
    let mut ptr: *mut T = std::ptr::null_mut();
    unsafe {
        cuda_check!(cudaMallocHost(
            &mut ptr as *mut *mut T as *mut *mut c_void,
            len * std::mem::size_of::<T>()
        ));
    }
    ptr

    // unsafe {
    //     std::alloc::alloc(
    //         std::alloc::Layout::from_size_align(len * std::mem::size_of::<T>(), 4096).unwrap(),
    //     ) as *mut T
    // }
}

pub unsafe fn free_pinned<T: Sized>(ptr: *mut T, _len: usize) {
    unsafe {
        cuda_check!(cudaFreeHost(ptr as *mut c_void));
    }
    // unsafe {
    //     std::alloc::dealloc(
    //         ptr as *mut u8,
    //         std::alloc::Layout::from_size_align(len * std::mem::size_of::<T>(), 4096).unwrap(),
    //     )
    // }
}
