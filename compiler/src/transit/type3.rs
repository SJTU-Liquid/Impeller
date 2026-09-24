use std::marker::PhantomData;

pub use crate::transit::type2::memory_planning;
use crate::transit::type2::{self};
use zkpoly_common::{
    define_usize_id,
    heap::{IdAllocator, RoHeap},
    typ::Typ,
};
use zkpoly_runtime::args::RuntimeType;

use super::type2::object_analysis::{size::LogBlockSizes, value::Tree};

define_usize_id!(RegisterId);
define_usize_id!(InstructionIndex);

impl std::fmt::Display for InstructionIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

mod device;
pub use device::{Device, DeviceSpecific};

pub mod template;
pub use template::{Begin, Debug, LcsPc, Load, Seq, Store, SubgraphInput};

mod track;
pub use track::{Track, TrackSpecific};

pub type VertexNode = type2::no_subgraph::alt_label::VertexNode<RegisterId>;
pub type Lcs<'s> = template::Lcs<'s, RegisterId, VertexNode>;
pub type SubgraphInstruction<'s> = template::SubgraphInstruction<'s, RegisterId, VertexNode>;
pub type Subgraph<'s> = template::Subgraph<'s, RegisterId, VertexNode>;
pub type InstructionNode<'s> = template::InstructionNode<'s, RegisterId, VertexNode>;
pub type OffstedAllocation = template::OffsetedAllocation<RegisterId>;
pub type RegisterManipulation = template::RegisterManipulation<RegisterId>;
pub type Instruction<'s> = template::Instruction<'s, RegisterId, VertexNode>;
pub type Instructions<'s> = template::Seq<InstructionIndex, Instruction<'s>>;

fn determine_transfer_track(from: Device, to: Device) -> Track {
    use Device::*;
    match (from, to) {
        (Gpu(..), Cpu) => Track::FromGpu,
        (Gpu(i), Gpu(j)) if (i == j) => Track::GpuMemory(i),
        (Gpu(_), Gpu(_)) => Track::ToGpu,
        (Cpu, Gpu(..)) => Track::ToGpu,
        (Cpu, Cpu) => Track::Cpu,
        (_, Disk) => Track::ToDisk,
        (Disk, _) => Track::FromDisk,
    }
}

impl<'s> InstructionNode<'s> {
    pub fn track(&self, memory_device: impl Fn(RegisterId) -> Device) -> Track {
        use template::InstructionNode::*;
        use Track::*;

        match self {
            Type2 {
                vertex: type2::template::VertexNode::Return(..),
                ..
            } => MemoryManagement,
            Type2 { vertex, device, .. } => vertex.track(device),
            Sub(subgraph) => CoProcess(subgraph.device.ensure_gpu().unwrap_gpu()),
            Malloc { .. } | MallocOffseted { .. } => MemoryManagement,
            Free { .. } => MemoryManagement,
            Transfer { from, id: to, .. } => {
                determine_transfer_track(memory_device(*from), memory_device(*to))
            }
            FillPoly { id, .. } => Track::on_device(match memory_device(*id) {
                Device::Cpu => type2::Device::Cpu,
                Device::Gpu(i) => type2::Device::Gpu(i),
                _ => panic!("FillPoly should have its result on either GPU or CPU"),
            }),
        }
    }
}

#[derive(Debug, Hash)]
pub struct Chunk<'s, Rt: RuntimeType> {
    pub(crate) instructions: template::Seq<InstructionIndex, Instruction<'s>>,
    pub(crate) register_value: RoHeap<RegisterId, memory_planning::RegisterValue>,
    pub(crate) reg_id_allocator: IdAllocator<RegisterId>,
    pub(crate) lbss: LogBlockSizes,
    pub(crate) _phantom: PhantomData<Rt>,
}

impl<'s, Rt: RuntimeType> Chunk<'s, Rt> {
    /// The device of the register pointing to if the register is an atom
    pub fn memory_device(&self, reg: RegisterId) -> Option<Device> {
        use type2::object_analysis::value::Tree::*;

        match &self.register_value[reg] {
            Tuple(..) => None,
            Single(ra) => Some(ra.device()),
        }
    }

    pub fn take_reg_id_allocator(&mut self) -> IdAllocator<RegisterId> {
        std::mem::take(&mut self.reg_id_allocator)
    }

    /// Runtime type of the register
    pub fn register_rt_typ(&self, reg: RegisterId) -> Typ {
        match &self.register_value[reg] {
            Tree::Tuple(..) => Typ::Tuple,
            Tree::Single(ra) => ra
                .object()
                .node()
                .map_scalar_array_len(|_, s| s.len() as usize)
                .erase_p(),
        }
    }

    /// Storage device of the register.
    /// The register cannot be a tuple.
    pub fn register_device(&self, reg: RegisterId) -> Device {
        self.register_value[reg].unwrap_leaf().device()
    }
}

pub mod lowering;
pub mod pretty_print;
pub mod track_splitting;
pub mod typ;
