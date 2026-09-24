mod prelude {
    pub(super) use std::{
        collections::{BTreeMap, BTreeSet},
        marker::PhantomData,
    };

    pub(super) use zkpoly_common::{
        define_usize_id,
        heap::{Heap, IdAllocator, UsizeId},
        mm_heap::{self, with_data::MmHeap},
        typ::Slice,
    };

    pub(super) use zkpoly_runtime::{
        args::RuntimeType,
        instructions::{AllocMethod, AllocVariant},
    };

    pub(super) use crate::driver::HardwareInfo;
    pub(super) use crate::transit::{
        type2::{
            self,
            object_analysis::{
                self,
                liveness::{self, NextUses},
                size::{IntegralSize, LogBlockSizes, Size, SmithereenSize},
                template::{
                    self, Ad, ChunkedOp, Constructor, ObjectAtRw, Operation, ResidentalAtom,
                    ResidentalT, ResidentalValue, Seq, SubgraphOperation, Temporaries, Type2Op,
                },
                value::{
                    Object, ObjectBijection, ObjectDefs, ObjectTable, TopNode, Tree, ValueNode,
                },
                Index,
            },
            VertexId,
        },
        type3::{self, Device, DeviceSpecific, RegisterId},
    };

    pub(super) use super::{
        address::Addr,
        allocator::{
            self, Allocator, AllocatorCollection, AsRealizer, Criterion, Epsilon, OomError,
            Pointer, Realizer,
        },
        allocators, phase0, phase1,
        planning::{self, criterion, DevicesStage, Stage},
        realization::{self},
        Error,
    };
}

use std::fmt::Debug;

use prelude::*;

use crate::driver;

pub mod address;
mod allocator;

define_usize_id!(PointerId);

pub mod allocators;
pub mod config;
pub mod planning;
pub mod pretty_print;
pub mod realization;
pub use config::Config;
pub use realization::dependency_analysis;

#[derive(Debug)]
pub enum Error<'s> {
    /// Out of memory when planning for a Type2 operation
    Oom(OomError, Option<(VertexId, type2::SourceInfo<'s>)>),

    /// Out of memory when planning for a sliceable subgraph
    SubgraphOom(OomError, VertexId),

    /// Not enough CPU memory but disk not enabled
    CpuOom(/* overflowed bytes */ u64),
}

impl<'s> Error<'s> {
    pub fn oom_during_type2(oom: OomError, vid: VertexId, src: type2::SourceInfo<'s>) -> Self {
        Self::Oom(oom, Some((vid, src)))
    }

    pub fn oom(oom: OomError) -> Self {
        Self::Oom(oom, None)
    }

    pub fn subgraph_oom(oom: OomError, vid: type2::VertexId) -> Self {
        Self::SubgraphOom(oom, vid)
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    pub realization_statistics: realization::machine::Statistics,
    pub disk_peak_usage: u64,
    pub cpu_peak_usage: u64,
    pub disk_constants_size: u64,
    pub cpu_constants_size: u64,
    pub page_allocator_statistics: Vec<allocators::page_allocator::Statistics>,
}

impl std::fmt::Display for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::utils::human_readable_size;

        writeln!(f, "Realization statistics:")?;
        self.realization_statistics.pretty_print(f, 2)?;
        writeln!(
            f,
            "Disk peak usage           : {}",
            human_readable_size(self.disk_peak_usage)
        )?;
        writeln!(
            f,
            "CPU peak usage            : {}",
            human_readable_size(self.cpu_peak_usage)
        )?;
        writeln!(
            f,
            "Disk constants size       : {}",
            human_readable_size(self.disk_constants_size)
        )?;
        writeln!(
            f,
            "CPU constants size        : {}",
            human_readable_size(self.cpu_constants_size)
        )?;

        writeln!(
            f,
            "Page Allocator Statistics:\n{:#?}",
            self.page_allocator_statistics
        )
    }
}

pub fn plan_subgraphs<'a, 's, Rt: RuntimeType>(
    ops: phase0::OperationSeq<'s, PointerId>,
    objects: &mut ObjectDefs,
    memory_config: &Config,
    hd_info: &HardwareInfo,
) -> Result<
    (
        phase1::OperationSeq<'s, PointerId>,
        planning::subgraph::Statistics,
    ),
    Error<'s>,
> {
    planning::subgraph::plan_subgraphs::<_, Rt>(
        ops,
        objects,
        |d| memory_config.gpu_integral_space(hd_info.gpus().nth(d.unwrap_gpu()).unwrap()),
        hd_info,
    )
}

use allocators::{
    ConstantWrapper, PageAllocator, SlabAllocator, SmithereenWrapper, SuperAllocator,
};

pub trait MemoryPlanners<Rt: RuntimeType> {
    fn plan_driver<'a, 'i>(
        &'a mut self,
        hd_info: &'i driver::HardwareInfo,
    ) -> Box<dyn planning::PlanDrivers<PointerId> + 'a>
    where
        'i: 'a;
    fn realize<'s>(
        self: Box<Self>,
        ops: phase1::OperationSeq<'s, PointerId>,
        hd_info: &driver::HardwareInfo,
        execution_device: Box<dyn Fn(VertexId) -> type2::DeviceCollection + '_>,
        dependencies: &dependency_analysis::Dependencies,
    ) -> Result<(type3::Chunk<'s, Rt>, Statistics), Error<'s>>;
}

pub struct MemoryPlanner<M, Ga, Rt: RuntimeType>
where
    M: criterion::Manager,
{
    lbss: Vec<IntegralSize>,
    gpu_allocators: Vec<SmithereenWrapper<Ga, PointerId>>,
    cpu_allocator: ConstantWrapper<
        SmithereenWrapper<SlabAllocator<PointerId, M::C, M::Cf>, PointerId>,
        PointerId,
    >,
    disk_allocator: ConstantWrapper<SuperAllocator<PointerId>, PointerId>,
    cpu_constants_size: u64,
    disk_constants_size: u64,
    _phantom: PhantomData<(M, Rt)>,
}

impl<M, Rt: RuntimeType> MemoryPlanner<M, PageAllocator<PointerId, M::C, M::Cf>, Rt>
where
    M: criterion::Manager,
    M::C: Debug,
    M::Cf: Debug,
{
    pub fn new(
        objects: &ObjectDefs,
        hd_info: &driver::HardwareInfo,
        analysis: &type2::object_analysis::DefUse,
        gpu_page_number: impl Fn(&driver::MemoryInfo) -> u64,
        gpu_page_size: u64,
        gpu_smithereen_space: u64,
        cpu_integral_space: u64,
        cpu_smithereen_space: u64,
    ) -> Self {
        let lbss = allocators::collect_integral_sizes::<Rt>(objects.iter_with_id());

        let gpu_allocators = hd_info
            .gpus()
            .enumerate()
            .map(|(i, gpu)| {
                SmithereenWrapper::for_filter_smithereen(
                    PageAllocator::new(
                        gpu_page_number(gpu) as usize,
                        gpu_page_size,
                        Device::Gpu(i),
                    ),
                    gpu_smithereen_space,
                    0,
                )
            })
            .collect();

        let cpu_allocator = SmithereenWrapper::for_filter_smithereen(
            SlabAllocator::new(
                cpu_integral_space,
                cpu_smithereen_space,
                *lbss.last().unwrap(),
                Device::Cpu,
            ),
            cpu_smithereen_space,
            0,
        );
        let disk_allocator = SuperAllocator::new(Device::Disk);

        let constants = analysis.constants(objects).collect::<Vec<_>>();
        let cpu_constants = constants
            .iter()
            .filter(|(_, d)| *d == Device::Cpu)
            .map(|(o, _)| o.clone())
            .collect::<Vec<_>>();
        let disk_constants = constants
            .iter()
            .filter(|(_, d)| *d == Device::Disk)
            .map(|(o, _)| o.clone())
            .collect::<Vec<_>>();
        let cpu_constants_size = cpu_constants
            .iter()
            .map(|o| u64::from(o.size::<Rt>()))
            .sum();
        let disk_constants_size = disk_constants
            .iter()
            .map(|o| u64::from(o.size::<Rt>()))
            .sum();

        let cpu_allocator = ConstantWrapper::<_, _>::new(cpu_allocator, cpu_constants.into_iter());
        let disk_allocator = ConstantWrapper::new(disk_allocator, disk_constants.into_iter());

        Self {
            lbss,
            gpu_allocators,
            cpu_allocator,
            disk_allocator,
            cpu_constants_size,
            disk_constants_size,
            _phantom: PhantomData,
        }
    }
}

impl<M, Rt: RuntimeType> MemoryPlanner<M, SlabAllocator<PointerId, M::C, M::Cf>, Rt>
where
    M: criterion::Manager,
    M::C: Debug,
    M::Cf: Debug,
{
    pub fn new(
        objects: &ObjectDefs,
        hd_info: &driver::HardwareInfo,
        analysis: &type2::object_analysis::DefUse,
        gpu_integral_space: impl Fn(&driver::MemoryInfo) -> u64,
        gpu_smithereen_space: u64,
        cpu_integral_space: u64,
        cpu_smithereen_space: u64,
    ) -> Self {
        let lbss = allocators::collect_integral_sizes::<Rt>(objects.iter_with_id());

        let gpu_allocators = hd_info
            .gpus()
            .enumerate()
            .map(|(i, gpu)| {
                SmithereenWrapper::for_filter_smithereen(
                    SlabAllocator::new(
                        gpu_integral_space(gpu),
                        gpu_smithereen_space,
                        *lbss.last().unwrap(),
                        Device::Gpu(i),
                    ),
                    gpu_smithereen_space,
                    0,
                )
            })
            .collect();

        let cpu_allocator = SmithereenWrapper::for_filter_smithereen(
            SlabAllocator::new(
                cpu_integral_space,
                cpu_smithereen_space,
                *lbss.last().unwrap(),
                Device::Cpu,
            ),
            cpu_smithereen_space,
            0,
        );
        let disk_allocator = SuperAllocator::new(Device::Disk);

        let constants = analysis.constants(objects).collect::<Vec<_>>();
        let cpu_constants = constants
            .iter()
            .filter(|(_, d)| *d == Device::Cpu)
            .map(|(o, _)| o.clone())
            .collect::<Vec<_>>();
        let disk_constants = constants
            .iter()
            .filter(|(_, d)| *d == Device::Disk)
            .map(|(o, _)| o.clone())
            .collect::<Vec<_>>();
        let cpu_constants_size = cpu_constants
            .iter()
            .map(|o| u64::from(o.size::<Rt>()))
            .sum();
        let disk_constants_size = disk_constants
            .iter()
            .map(|o| u64::from(o.size::<Rt>()))
            .sum();

        let cpu_allocator = ConstantWrapper::<_, _>::new(cpu_allocator, cpu_constants.into_iter());
        let disk_allocator = ConstantWrapper::new(disk_allocator, disk_constants.into_iter());

        Self {
            lbss,
            gpu_allocators,
            cpu_allocator,
            disk_allocator,
            cpu_constants_size,
            disk_constants_size,
            _phantom: PhantomData,
        }
    }
}

impl<M, Rt: RuntimeType> MemoryPlanners<Rt>
    for MemoryPlanner<M, SlabAllocator<PointerId, M::C, M::Cf>, Rt>
where
    M: criterion::Manager,
    M::C: Debug,
    M::Cf: Debug,
{
    fn plan_driver<'a, 'i>(
        &'a mut self,
        hd_info: &'i driver::HardwareInfo,
    ) -> Box<dyn planning::PlanDrivers<PointerId> + 'a>
    where
        'i: 'a,
    {
        Box::new(planning::PlanDriver::<PointerId, M, Rt>::new(
            &mut self.gpu_allocators,
            &mut self.cpu_allocator,
            &mut self.disk_allocator,
            hd_info,
        ))
    }

    fn realize<'s>(
        self: Box<Self>,
        ops: phase1::OperationSeq<'s, PointerId>,
        hd_info: &driver::HardwareInfo,
        execution_device: Box<dyn Fn(VertexId) -> type2::DeviceCollection + '_>,
        dependencies: &dependency_analysis::Dependencies,
    ) -> Result<(type3::Chunk<'s, Rt>, Statistics), Error<'s>> {
        let realizers = DeviceSpecific {
            gpu: self
                .gpu_allocators
                .iter()
                .map(AsRealizer::new)
                .map(|r| Box::new(r) as Box<dyn Realizer<_>>)
                .collect(),
            cpu: Box::new(AsRealizer::new(&self.cpu_allocator)) as Box<dyn Realizer<_>>,
            disk: Box::new(AsRealizer::<_, _, M::C, M::Cf>::new(&self.disk_allocator))
                as Box<dyn Realizer<_>>,
        };

        let (chunk, realization_statistics) = realization::realize(
            ops,
            hd_info,
            LogBlockSizes::new(self.lbss),
            realizers,
            &execution_device,
            dependencies,
        )?;

        let disk_allocator = self.disk_allocator.unwrap();

        if hd_info.disks_available() == 0 && disk_allocator.peak_memory_usage() > 0 {
            return Err(Error::CpuOom(disk_allocator.peak_memory_usage()));
        }

        let cpu_smithereen_size = self.cpu_allocator.unwrap_ref().capacity();

        Ok((
            chunk,
            Statistics {
                realization_statistics,
                disk_peak_usage: disk_allocator.peak_memory_usage(),
                cpu_peak_usage: self.cpu_allocator.unwrap().unwrap().peak_memory_usage()
                    + cpu_smithereen_size,
                disk_constants_size: self.disk_constants_size,
                cpu_constants_size: self.cpu_constants_size,
                page_allocator_statistics: vec![],
            },
        ))
    }
}

impl<M, Rt: RuntimeType> MemoryPlanners<Rt>
    for MemoryPlanner<M, PageAllocator<PointerId, M::C, M::Cf>, Rt>
where
    M: criterion::Manager,
    M::C: Debug,
    M::Cf: Debug,
{
    fn plan_driver<'a, 'i>(
        &'a mut self,
        hd_info: &'i driver::HardwareInfo,
    ) -> Box<dyn planning::PlanDrivers<PointerId> + 'a>
    where
        'i: 'a,
    {
        Box::new(planning::PlanDriver::<PointerId, M, Rt>::new(
            &mut self.gpu_allocators,
            &mut self.cpu_allocator,
            &mut self.disk_allocator,
            hd_info,
        ))
    }

    fn realize<'s>(
        self: Box<Self>,
        ops: phase1::OperationSeq<'s, PointerId>,
        hd_info: &driver::HardwareInfo,
        execution_device: Box<dyn Fn(VertexId) -> type2::DeviceCollection + '_>,
        dependencies: &dependency_analysis::Dependencies,
    ) -> Result<(type3::Chunk<'s, Rt>, Statistics), Error<'s>> {
        let realizers = DeviceSpecific {
            gpu: self
                .gpu_allocators
                .iter()
                .map(AsRealizer::new)
                .map(|r| Box::new(r) as Box<dyn Realizer<_>>)
                .collect(),
            cpu: Box::new(AsRealizer::new(&self.cpu_allocator)) as Box<dyn Realizer<_>>,
            disk: Box::new(AsRealizer::<_, _, M::C, M::Cf>::new(&self.disk_allocator))
                as Box<dyn Realizer<_>>,
        };

        let (chunk, realization_statistics) = realization::realize(
            ops,
            hd_info,
            LogBlockSizes::new(self.lbss),
            realizers,
            &execution_device,
            dependencies,
        )?;

        let disk_allocator = self.disk_allocator.unwrap();

        if hd_info.disks_available() == 0 && disk_allocator.peak_memory_usage() > 0 {
            return Err(Error::CpuOom(disk_allocator.peak_memory_usage()));
        }

        let cpu_smithereen_size = self.cpu_allocator.unwrap_ref().capacity();

        Ok((
            chunk,
            Statistics {
                realization_statistics,
                disk_peak_usage: disk_allocator.peak_memory_usage(),
                cpu_peak_usage: self.cpu_allocator.unwrap().unwrap().peak_memory_usage()
                    + cpu_smithereen_size,
                disk_constants_size: self.disk_constants_size,
                cpu_constants_size: self.cpu_constants_size,
                page_allocator_statistics: self
                    .gpu_allocators
                    .iter()
                    .map(|ga| ga.unwrap_ref().statistics())
                    .collect(),
            },
        ))
    }
}

pub mod phase0 {
    use super::super::object_analysis::{self, *};
    use super::Pointer;
    pub use object_analysis::Paragraph;

    pub type SubgraphOperation<'s> = template::SubgraphOperation<'s, value::Object, ()>;
    pub type SubgraphOperationSeq<'s> = template::Seq<SubgraphOperation<'s>>;
    pub type Subgraph<'s, P> = template::Subgraph<value::Object, Option<P>, Paragraph<'s>, ()>;

    pub type Operation<'s, P> =
        template::Operation<'s, value::Object, Pointer<P>, Subgraph<'s, Pointer<P>>>;
    pub type OperationSeq<'s, P> = template::Seq<Operation<'s, P>>;
}

pub mod phase1 {
    use super::super::object_analysis::*;
    use super::allocator::Epsilon;
    use super::allocators::{LinearAllocator, SmithereenWrapper};
    use super::prelude::ObjectAtRw;
    use super::{Object, Pointer, ResidentalT};

    pub type SubgraphOperation<'s, P> = template::SubgraphOperation<'s, value::Object, P>;
    pub type LcsAllocator<P> = SmithereenWrapper<LinearAllocator<P, Epsilon, Epsilon>, P>;
    pub type Lcs<'s, P> = template::Lcs<'s, value::Object, Pointer<P>, LcsAllocator<P>>;

    pub type Intermediates<P> = Vec<ResidentalT<Object, Pointer<P>>>;

    pub type Subgraph<'s, P> =
        template::Subgraph<value::Object, Option<Pointer<P>>, Lcs<'s, P>, Intermediates<P>>;

    #[derive(Debug, Hash, Clone)]
    pub struct ResidentalSubgraph<'s, P> {
        pub sg: Subgraph<'s, P>,
        pub allocation: super::planning::subgraph::MemoryAllocation<P>,
    }

    mod pretty_print {
        use super::*;
        use zkpoly_common::indent_formatter::*;

        impl<'s, W, P> IndentDebug<W> for ResidentalSubgraph<'s, P>
        where
            P: std::fmt::Debug,
            W: std::io::Write,
        {
            fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
                f.debug_struct()
                    .entry("sg", &self.sg)
                    .entry_debug("allocation", &self.allocation)
                    .finish()
            }
        }
    }

    impl<'s, P> ResidentalSubgraph<'s, P> {
        pub fn new(
            sg: Subgraph<'s, P>,
            allocation: super::planning::subgraph::MemoryAllocation<P>,
        ) -> Self {
            Self { sg, allocation }
        }
    }

    impl<'s, P> ObjectAtRw<Object, Option<Pointer<P>>> for ResidentalSubgraph<'s, P>
    where
        P: Clone,
    {
        fn object_at_reads<'a>(
            &'a self,
        ) -> Box<dyn Iterator<Item = (&'a Object, Option<Pointer<P>>, super::Device)> + 'a>
        where
            Object: 'a,
        {
            Box::new(self.sg.object_at_reads().chain(std::iter::once((
                &self.allocation.object,
                self.allocation.pointer.clone(),
                self.sg.device,
            ))))
        }

        fn object_at_writes<'a>(
            &'a self,
        ) -> Box<
            dyn Iterator<
                    Item = (
                        value::OutputT<&'a Object, &'a Object>,
                        Option<Pointer<P>>,
                        super::Device,
                    ),
                > + 'a,
        >
        where
            Object: 'a,
        {
            self.sg.object_at_writes()
        }

        fn object_at_allocates<'a>(
            &'a self,
        ) -> Option<(&'a Object, Option<Pointer<P>>, super::Device)> {
            self.sg.object_at_allocates()
        }

        fn object_at_deallocates<'a>(
            &'a self,
        ) -> Option<(&'a Object, Option<Pointer<P>>, super::Device)> {
            self.sg.object_at_deallocates()
        }
    }

    pub type Operation<'s, P> =
        template::Operation<'s, value::Object, Pointer<P>, ResidentalSubgraph<'s, P>>;
    pub type OperationSeq<'s, P> = template::Seq<Operation<'s, P>>;
}

pub type AtomRegisterValue = realization::machine::AtomRegisterValue<Pointer<PointerId>>;
pub type RegisterValue = realization::machine::RegisterValue<Pointer<PointerId>>;
