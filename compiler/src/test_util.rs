pub mod prelude {
    pub use crate::{
        ast::{self},
        driver::{self, DiskIoMode, DiskMemoryInfo, MemoryInfo},
        transit::type2::{self, Typ},
    };
    pub use zkpoly_common::heap::Heap;
    pub use zkpoly_memory_pool::{buddy_disk_pool::DiskMemoryPool, BuddyDiskPool, CpuMemoryPool};
    pub use zkpoly_runtime::{
        args::{EntryTable, RuntimeType, Variable},
        async_rng::AsyncRng,
        runtime::{self, RuntimeDebug},
        scalar::ScalarArray,
    };
}

use prelude::*;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;

pub struct Tester {
    constant_pool: ast::ConstantPool,
    disk_pool: Arc<Mutex<DiskMemoryPool>>,
    hd_info: driver::HardwareInfo,
    cpu_alloc: CpuMemoryPool,
}

pub fn assert_eq<Rt: RuntimeType>(left: &Variable<Rt>, right: &Variable<Rt>) {
    if !runtime::assert_eq::assert_eq(left, right) {
        panic!("assert_eq failed\nLeft = {:?}\nRight = {:?}", left, right)
    }
}

impl Tester {
    fn compile<Rt: RuntimeType>(
        &mut self,
        output_node: impl ast::TypeEraseable<Rt>,
        config: &driver::Config,
        dir: impl AsRef<Path>,
    ) -> Result<driver::Artifect<Rt>, driver::Error<'static, Rt>> {
        let options = driver::DebugOptions::all(dir.as_ref().to_path_buf()).with_log(true);
        options.prepare_dir();
        driver::FreshType2::from_ast(output_node, &options)?.to_artifect(
            &self.hd_info,
            &mut self.constant_pool,
            config,
            [0].into_iter(),
            dir.as_ref().join("kernels"),
        )
    }

    pub fn big() -> Self {
        Self {
            constant_pool: ast::ConstantPool::with_disk(
                CpuMemoryPool::new(30, std::mem::size_of::<u32>()),
                vec![BuddyDiskPool::new(30, None).unwrap()],
            ),
            hd_info: driver::HardwareInfo::new(MemoryInfo::new(40 * 2u64.pow(30)))
                .with_gpu(MemoryInfo::new(20 * 2u64.pow(30)))
                .with_disk(DiskMemoryInfo::new(None)),
            disk_pool: Arc::new(Mutex::new(vec![BuddyDiskPool::new(30, None).unwrap()])),
            cpu_alloc: CpuMemoryPool::new(30, std::mem::size_of::<u32>()),
        }
    }

    pub fn small() -> Self {
        Self::small_with_gpu_memory_limit(20 * 2u64.pow(30))
    }

    pub fn small_with_gpu_memory_limit(gpu_memory_limit: u64) -> Self {
        Self {
            constant_pool: ast::ConstantPool::only_cpu(CpuMemoryPool::new(
                30,
                std::mem::size_of::<u32>(),
            )),
            hd_info: driver::HardwareInfo::new(MemoryInfo::new(1 * 2u64.pow(30)))
                .with_gpu(MemoryInfo::new(gpu_memory_limit)),
            disk_pool: Arc::new(Mutex::new(Vec::new())),
            cpu_alloc: CpuMemoryPool::new(30, std::mem::size_of::<u32>()),
        }
    }

    pub fn scalar_array<Rt: RuntimeType>(&mut self, deg: usize) -> ScalarArray<Rt::Field> {
        ScalarArray::alloc_cpu(deg, &mut self.cpu_alloc)
    }

    pub fn run<Rt: RuntimeType>(
        &mut self,
        output_node: impl ast::TypeEraseable<Rt>,
        config: &driver::Config,
        inputs: &EntryTable<Rt>,
        dir: impl AsRef<Path>,
    ) -> (Option<Variable<Rt>>, CpuStaticAllocator) {
        let artifect = self.compile(output_node, config, dir).unwrap();
        let pools = driver::artifect::Pools {
            cpu: self.hd_info.cpu_allocator(true),
            gpu: self
                .hd_info
                .gpu_allocators_for(true, [0].into_iter(), config.memory()),
            disk: self.disk_pool.clone(),
        };
        let mut dispatcher = artifect.prepare_dispatcher(
            artifect.versions().next().unwrap(),
            pools,
            AsyncRng::new(1024, rand_core::OsRng),
            Arc::new(|x| x),
        );
        let ((r, _, _), pool) =
            dispatcher.run(inputs, RuntimeDebug::none().with_print_instruction(true));

        (r, pool)
    }
}
