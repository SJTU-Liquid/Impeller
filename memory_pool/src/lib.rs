#![feature(btree_cursors)]
pub mod buddy_disk_pool;
pub mod cpu_pool;
pub mod memory_pool;
pub use buddy_disk_pool::{BuddyDiskPool, DiskIoMode};
pub use cpu_pool::CpuMemoryPool;
pub mod static_allocator;
