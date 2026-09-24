use std::{collections::HashMap, path::PathBuf};
use zkpoly_cuda_api::mem::CudaAllocator;
use zkpoly_memory_pool::{
    buddy_disk_pool::{DiskIoMode, DiskMemoryPool},
    static_allocator::CpuStaticAllocator,
    BuddyDiskPool,
};

/// Configure a device's memory.
#[derive(Debug, PartialOrd, Ord, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MemoryInfo {
    pub(crate) memory_limit: u64,
}

impl MemoryInfo {
    /// Configure the device to use `memory_limit` bytes of memory,
    /// `smithereen_space` of which are dedicated to small memory objects.
    pub fn new(memory_limit: u64) -> Self {
        Self { memory_limit }
    }

    /// Get memory limit in bytes of the configuration.
    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }

    /// Get memory limit in giga bytes of the configuration.
    pub fn memory_limit_gigabytes(&self) -> f64 {
        self.memory_limit as f64 / 2.0_f64.powi(30)
    }
}

/// Configure a disk device to provide memory for computation.
#[derive(Debug, Clone)]
pub struct DiskMemoryInfo {
    disk_path: Option<PathBuf>,
    io_mode: DiskIoMode,
}

impl DiskMemoryInfo {
    /// Configure to use the disk device holding `path` in file system.
    /// If [`None`] is provided, use system temporary directory.
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            disk_path: path,
            io_mode: DiskIoMode::CuFile,
        }
    }

    /// Selects the disk I/O implementation used by pools created from this info.
    pub fn with_io_mode(mut self, io_mode: DiskIoMode) -> Self {
        self.io_mode = io_mode;
        self
    }

    /// Selects POSIX pread/pwrite with host-memory staging for GPU transfers.
    pub fn compat_mode(self) -> Self {
        self.with_io_mode(DiskIoMode::Compat)
    }

    pub fn io_mode(&self) -> DiskIoMode {
        self.io_mode
    }

    /// Get the path configured in `new`.
    pub fn disk_path(&self) -> Option<&PathBuf> {
        self.disk_path.as_ref()
    }

    pub fn has_path(&self) -> bool {
        self.disk_path.is_none()
    }
}

/// Configure various devices in system for computation.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    gpus: Vec<MemoryInfo>,
    cpu: MemoryInfo,
    disk: Vec<DiskMemoryInfo>,
}

impl HardwareInfo {
    /// Get configured number of GPU's in the system.
    pub fn n_gpus(&self) -> usize {
        self.gpus.len()
    }

    /// Iterate through each GPU's configuration.
    pub fn gpus(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.gpus.iter()
    }

    /// Iterate through each disk device's configuration.
    pub fn disks(&self) -> impl Iterator<Item = &DiskMemoryInfo> {
        self.disk.iter()
    }

    /// Get CPU memory's configuration.
    pub fn cpu(&self) -> &MemoryInfo {
        &self.cpu
    }

    /// Create a configuation with only one CPU, no GPU nor disk.
    pub fn new(cpu: MemoryInfo) -> Self {
        Self {
            gpus: Vec::new(),
            cpu,
            disk: Vec::new(),
        }
    }

    /// Configure CPU.
    pub fn with_cpu(self, cpu: MemoryInfo) -> Self {
        HardwareInfo { cpu, ..self }
    }

    /// Add a disk configuation.
    pub fn with_disk(mut self, info: DiskMemoryInfo) -> Self {
        self.disk.push(info);
        self
    }

    /// Add a GPU configuation.
    pub fn with_gpu(mut self, gpu: MemoryInfo) -> Self {
        self.gpus.push(gpu);
        self
    }

    /// Get number of disks in the system.
    pub fn disks_available(&self) -> usize {
        self.disk.len()
    }

    /// Build the CPU memory pool as configured.
    pub fn cpu_allocator(&self, memory_check: bool) -> CpuStaticAllocator {
        CpuStaticAllocator::new(self.cpu().memory_limit() as usize, memory_check)
    }

    /// Build the GPU memory pools as configured, indexed by CUDA device ID's.
    pub fn gpu_allocators(
        &self,
        memory_check: bool,
        memory_config: &super::MemoryPlanningConfig,
    ) -> HashMap<i32, CudaAllocator> {
        self.gpu_allocators_for(memory_check, 0..(self.gpus.len() as i32), memory_config)
    }

    /// Build the GPU memory pools as configured, for a subset of GPU's identified by
    /// CUDA device ID's.
    pub fn gpu_allocators_for(
        &self,
        memory_check: bool,
        device_ids: impl Iterator<Item = i32>,
        memory_config: &super::MemoryPlanningConfig,
    ) -> HashMap<i32, CudaAllocator> {
        use zkpoly_cuda_api::mem;
        device_ids
            .map(|id| {
                let gpu = &self.gpus[id as usize];
                (
                    id as i32,
                    match memory_config.gpu_allocator() {
                        super::GpuAllocatorChoice::Slab => CudaAllocator {
                            statik: mem::StaticAllocator::new(
                                id,
                                gpu.memory_limit() as usize,
                                memory_check,
                            ),
                            page: mem::PageAllocator::new(
                                zkpoly_common::devices::DeviceType::GPU { device_id: id },
                                2u64.pow(24) as usize,
                                0,
                                memory_check,
                            ),
                        },
                        super::GpuAllocatorChoice::Page(page_size) => CudaAllocator {
                            statik: mem::StaticAllocator::new(
                                id,
                                memory_config.gpu_smithereen_space() as usize,
                                memory_check,
                            ),
                            page: mem::PageAllocator::new(
                                zkpoly_common::devices::DeviceType::GPU { device_id: id },
                                page_size as usize,
                                memory_config.gpu_page_number(gpu) as usize,
                                memory_check,
                            ),
                        },
                    },
                )
            })
            .collect()
    }

    /// Build the Disk memory pools as configured.
    pub fn disk_allocator(&self, max_block: usize) -> DiskMemoryPool {
        self.disks()
            .map(|disk_info| {
                let disk_path = disk_info.disk_path().cloned();
                BuddyDiskPool::new_with_mode(max_block, disk_path, disk_info.io_mode())
                    .expect("cannot create disk pool")
            })
            .collect()
    }

    pub(crate) fn smaller_cpus<'a>(
        &'a self,
        divisors: impl Iterator<Item = u32> + 'a,
    ) -> impl Iterator<Item = MemoryInfo> + 'a {
        divisors.map(|divisor| {
            assert!(divisor > 0, "CPU-memory divisor must be positive");
            MemoryInfo::new(self.cpu.memory_limit / divisor as u64)
        })
    }
}
