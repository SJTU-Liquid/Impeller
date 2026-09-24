use std::io::Write;

use crate::args::{ConstantId, EntryId, RuntimeType, VariableId};
use crate::devices::{EventId, ThreadId};
use crate::functions::{self, FuncMeta, FunctionId};
use serde::{Deserialize, Serialize};
use zkpoly_common::devices::DeviceType;
use zkpoly_common::typ::{PolySlice, Slice, Typ};

#[derive(Clone, Hash, Serialize, Deserialize, Debug)]
pub enum AllocMethod {
    /// Allocate using page allocator
    Paged {
        /// virtual address ID
        vaid: usize,
        /// size of the virtual address space
        va_size: usize,
        /// physical pages bound to this virtual address
        pa: Vec<usize>,
    },
    /// Allocate using offset and size
    Offset(usize, usize),
    /// Let the runtime decide the location
    Dynamic(usize),
}

impl AllocMethod {
    pub fn unwrap_offset(self) -> (usize, usize) {
        match self {
            Self::Offset(offset, size) => (offset, size),
            _ => panic!("called unwrap_offset on {:?}", self),
        }
    }
}

#[derive(Clone, Hash, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AllocVariant {
    Paged {
        /// virtual address ID
        vaid: usize,
    },
    Offset,
    Dynamic,
}

impl Default for AllocMethod {
    fn default() -> Self {
        Self::Dynamic(0)
    }
}

#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub struct AllocateOffsetInfo {
    pub base_buffer: VariableId,
    pub offset: usize,
    pub typ: Typ,
    pub id: VariableId,
}

#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub struct LoadSlice {
    pub src_reg: VariableId,
    pub alloc_dst: AllocateOffsetInfo,
    /// If this is a poly slice, begin_range is Some
    /// otherwise, it is None
    /// the starting slice in the loop
    /// add chunk_len * stages to get next slice start
    pub begin_range: Option<Slice>,
}

#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub struct StoreSlice {
    pub src_reg: VariableId,
    pub dst_reg: VariableId,
    pub begin_range: Option<Slice>,
}
#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub struct Lcs {
    pub loads: Vec<LoadSlice>,
    pub compute: (Vec<Instruction>, Vec<Option<PolySlice>>), // instructions and their slice info
    pub stores: Vec<StoreSlice>,
}

#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub enum InstructionNode {
    Allocate {
        device: DeviceType,
        typ: Typ,
        id: VariableId,
        alloc_method: AllocMethod,
    },

    /// Allocate a variable at an offset of a base buffer, used inside subgraphs
    AllocateOffset {
        info: AllocateOffsetInfo,
    },

    Deallocate {
        // free the underlying memory
        id: VariableId,
        alloc_method: AllocVariant,
    },

    RemoveRegister {
        // only delete the register file
        id: VariableId,
    },

    Transfer {
        src_device: DeviceType,
        dst_device: DeviceType,
        stream: Option<VariableId>,
        src_id: VariableId,
        dst_id: VariableId,
    },

    FuncCall {
        func_id: FunctionId,
        arg_mut: Vec<VariableId>,
        arg: Vec<VariableId>,
    },

    Wait {
        slave: DeviceType,
        stream: Option<VariableId>,
        event: EventId,
    },

    Record {
        stream: Option<VariableId>,
        event: EventId,
    },

    Fork {
        new_thread: ThreadId,
        instructions: Vec<Instruction>,
    },

    Join {
        thread: ThreadId,
    },

    Rotation {
        src: VariableId,
        dst: VariableId,
        shift: i64,
    },

    Slice {
        src: VariableId,
        dst: VariableId,
        start: usize,
        end: usize,
    },

    LoadConstant {
        src: ConstantId,
        dst: VariableId,
    },

    AssembleTuple {
        vars: Vec<VariableId>,
        dst: VariableId,
    },

    TupleGet {
        dst: VariableId,
        src: VariableId,
        index: usize,
    },

    Blind {
        dst: VariableId,
        start: usize,
        end: usize,
    },

    Return(VariableId),
    SetSliceMeta {
        // for directly operating on the meta data of
        src: VariableId,
        dst: VariableId,
        offset: usize,
        len: usize,
    },

    GetScalarFromArray {
        src: VariableId,
        dst: VariableId,
        idx: usize,
        stream: Option<VariableId>,
    },

    MoveRegister {
        src: VariableId,
        dst: VariableId,
    },

    LoadInput {
        src: EntryId,
        dst: VariableId,
    },

    AssertEq {
        value: VariableId,
        expected: VariableId,
        msg: Option<String>,
    },

    Print(VariableId, String),

    CopyRegister {
        src: VariableId,
        dst: VariableId,
    },

    SliceBuffer {
        src: VariableId,
        dst: VariableId,
        offset: usize,
        len: usize,
    },

    SubGraphFor {
        prologue: Lcs,
        body: Vec<Lcs>,
        total_iters: usize,
        to_gpu_stream: VariableId,
        from_gpu_stream: VariableId,
        device_types: Vec<DeviceType>, // where are the origin data located
        compute_stream_id: VariableId,
    },
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Track {
    MemoryManagement,
    RegisterManipulation,
    CoProcess(usize),
    Gpu(usize),
    Cpu,
    ToGpu,
    FromGpu,
    GpuMemory(usize),
    ToDisk,
    FromDisk,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Stream {
    ToGpu,
    FromGpu,
    Gpu,
    GpuMemory,
}

#[derive(Debug, Hash, Clone, Serialize, Deserialize)]
pub struct Instruction {
    node: InstructionNode,
    track: Track,
    stream: Option<Stream>,
}

impl Instruction {
    pub fn new(node: InstructionNode, track: Track, stream: Option<Stream>) -> Self {
        Self {
            node,
            track,
            stream,
        }
    }

    pub fn node(&self) -> &InstructionNode {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut InstructionNode {
        &mut self.node
    }

    pub fn track(&self) -> Track {
        self.track
    }

    pub fn is_gpu(&self) -> bool {
        self.stream.is_some()
    }

    /// The stream related to this instruction
    pub fn stream(&self) -> Option<Stream> {
        self.stream
    }

    /// Only returns some if this instruction need timing using stream
    pub fn need_timing_stream(&self) -> Option<Stream> {
        match self.node() {
            // Allocating a stream is not timed using stream
            InstructionNode::Allocate { .. } => None,
            _ => self.stream(),
        }
    }

    pub fn unwrap_stream(&self) -> Stream {
        self.stream.unwrap()
    }
}

pub fn instruction_label<Rt: RuntimeType>(
    inst: &InstructionNode,
    ftab: &functions::FunctionTable<Rt>,
) -> String {
    use InstructionNode::*;
    match inst {
        FuncCall { func_id, .. } => instruction_label_by_meta(inst, Some(&ftab[*func_id].meta)),
        _ => instruction_label_by_meta(inst, None),
    }
}

pub fn instruction_label_by_meta(inst: &InstructionNode, meta: Option<&FuncMeta>) -> String {
    use InstructionNode::*;
    match inst {
        Allocate { device, .. } => format!("Allocate({:?})", device),
        Deallocate { .. } => "Deallocate".to_string(),
        RemoveRegister { .. } => "RemoveRegister".to_string(),
        Transfer {
            src_device,
            dst_device,
            ..
        } => {
            format!("Transfer({:?}->{:?})", src_device, dst_device)
        }
        FuncCall { func_id, .. } => {
            format!("Call({}: {})", usize::from(*func_id), &meta.unwrap().name)
        }
        Wait { stream, .. } => {
            stream.map_or_else(|| format!("WaitThread"), |_| format!("WaitStream"))
        }
        Record { stream, .. } => {
            stream.map_or_else(|| format!("Record"), |_| format!("RecordStream"))
        }
        Fork { .. } => "Fork".to_string(),
        Join { .. } => "Join".to_string(),
        Rotation { shift, .. } => format!("Rotation({})", shift),
        Slice { start, end, .. } => format!("Slice({},{})", start, end),
        LoadConstant { .. } => "LoadConstant".to_string(),
        AssembleTuple { .. } => "AssembleTuple".to_string(),
        TupleGet {..} => "TupleGet".to_string(),
        Blind { start, end, .. } => format!("Blind({}, {})", start, end),
        Return(..) => "Return".to_string(),
        SetSliceMeta { offset, len, .. } => format!("SetSliceMeta({}, {})", offset, len),
        GetScalarFromArray { idx, stream, .. } => stream.map_or_else(
            || format!("IndexPoly({})", idx),
            |_| format!("IndexPolyGpu({})", idx),
        ),
        MoveRegister { .. } => "Move".to_string(),
        LoadInput { .. } => "LoadInput".to_string(),
        AssertEq { .. } => "AssertEq".to_string(),
        Print(_, label) => format!("Print({})", label),
        CopyRegister { .. } => "CopyRegister".to_string(),
        SliceBuffer { offset, len, .. } => format!("SliceBuffer({}, {})", offset, len),
        AllocateOffset {
            info:
                AllocateOffsetInfo {
                    base_buffer,
                    offset,
                    typ,
                    id,
                },
        } => format!(
            "AllocateOffset(base_buffer: {:?}, offset: {}, typ: {:?}, id: {:?})",
            base_buffer, offset, typ, id
        ),
        SubGraphFor {
            prologue,
            body,
            from_gpu_stream,
            to_gpu_stream,
            total_iters,
            device_types,
            compute_stream_id,
        } => format!(
            "SubGraphFor(to gpu stream: {:?}, from gpu stream: {:?}, prologue loads: {}, body loops: {}, total_iters: {}, device_types: {:?}, compute_stream_id: {:?})",
            to_gpu_stream,
            from_gpu_stream,
            prologue.loads.len(),
            body.len(),
            total_iters,
            device_types,
            compute_stream_id,
        ),
    }
}

pub fn static_args(inst: &InstructionNode) -> Option<String> {
    use InstructionNode::*;
    match inst {
        Allocate {
            typ, alloc_method, ..
        } => Some(format!("{:?}, {:?}", typ, alloc_method)),
        Wait { event, slave, .. } => Some(format!("{:?}, {:?}", event, slave)),
        Record { event, .. } => Some(format!("{:?}", event)),
        Fork { new_thread, .. } => Some(format!("{:?}", new_thread)),
        Join { thread } => Some(format!("{:?}", thread)),
        LoadConstant { src, .. } => Some(format!("{:?}", src)),
        LoadInput { src, .. } => Some(format!("{:?}", src)),
        _ => None,
    }
}

pub fn labeled_mutable_uses(inst: &InstructionNode) -> Vec<(VariableId, String)> {
    use InstructionNode::*;
    match inst {
        Allocate { id, .. } => vec![(*id, "".to_string())],
        AllocateOffset { info, .. } => vec![(info.id, "".to_string())],
        Deallocate { id, .. } => vec![(*id, "".to_string())],
        RemoveRegister { id, .. } => vec![(*id, "".to_string())],
        Transfer { dst_id, .. } => vec![(*dst_id, "".to_string())],
        FuncCall { arg_mut, .. } => arg_mut.iter().map(|id| (*id, "".to_string())).collect(),
        Wait { .. } => vec![],
        Record { .. } => vec![],
        Fork { .. } => vec![],
        Join { .. } => vec![],
        Rotation { dst, .. } | TupleGet { dst, .. } | Slice { dst, .. } => {
            vec![(*dst, "".to_string())]
        }
        LoadConstant { dst, .. } => vec![(*dst, "".to_string())],
        AssembleTuple { dst, .. } => vec![(*dst, "".to_string())],
        Blind { dst, .. } => vec![(*dst, "".to_string())],
        Return(..) => vec![],
        SetSliceMeta { dst, .. } => vec![(*dst, "".to_string())],
        GetScalarFromArray { dst, .. } => vec![(*dst, "".to_string())],
        MoveRegister { dst, .. } => vec![(*dst, "".to_string())],
        LoadInput { dst, .. } => vec![(*dst, "".to_string())],
        AssertEq { .. } => vec![],
        Print(..) => vec![],
        CopyRegister { dst, .. } => vec![(*dst, "".to_string())],
        SliceBuffer { dst, .. } => vec![(*dst, "".to_string())],
        SubGraphFor { .. } => vec![],
    }
}

pub fn labeled_uses(inst: &InstructionNode) -> Vec<(VariableId, String)> {
    use InstructionNode::*;
    match inst {
        Allocate { .. } | Deallocate { .. } | RemoveRegister { .. } => vec![],
        Transfer { src_id, .. } => vec![(*src_id, "".to_string())],
        FuncCall { arg, .. } => arg.iter().map(|id| (*id, "".to_string())).collect(),
        Wait { .. } | Record { .. } | Fork { .. } | Join { .. } => vec![],
        Rotation { src, .. } => vec![(*src, "".to_string())],
        Slice { src, .. } => vec![(*src, "".to_string())],
        LoadConstant { .. } => vec![],
        AssembleTuple { vars, .. } => vars.iter().map(|id| (*id, "".to_string())).collect(),
        Blind { .. } => vec![],
        Return(src) | TupleGet { src, .. } => vec![(*src, "".to_string())],
        SetSliceMeta { src, .. } => vec![(*src, "".to_string())],
        GetScalarFromArray { src, .. } => vec![(*src, "".to_string())],
        MoveRegister { src, .. } => vec![(*src, "".to_string())],
        LoadInput { .. } => vec![],
        AssertEq {
            value, expected, ..
        } => vec![
            (*value, "value".to_string()),
            (*expected, "expected".to_string()),
        ],
        Print(value, _) => vec![(*value, "".to_string())],
        CopyRegister { dst, .. } => vec![(*dst, "".to_string())],
        SliceBuffer { src, .. } => vec![(*src, "".to_string())],
        SubGraphFor { .. } => vec![],
        AllocateOffset { .. } => vec![],
    }
}

pub fn stream(inst: &InstructionNode) -> Option<VariableId> {
    use InstructionNode::*;
    match inst {
        Transfer { stream, .. }
        | Wait { stream, .. }
        | Record { stream, .. }
        | GetScalarFromArray { stream, .. } => *stream,
        _ => None,
    }
}

fn print_lcs_indented(lcs: &Lcs, spaces: usize, writer: &mut impl Write) -> std::io::Result<()> {
    let prefix = " ".repeat(spaces);
    let prefix_2 = " ".repeat(spaces + 2);
    writeln!(writer, "{}Load:", prefix)?;
    for load in lcs.loads.iter() {
        writeln!(writer, "{}{:?}", prefix_2, load)?;
    }

    writeln!(writer, "{}Compute:", prefix)?;
    for (idx, (instruct, begin)) in lcs.compute.0.iter().zip(lcs.compute.1.iter()).enumerate() {
        writeln!(writer, "{}{}: {:?}  {:?}", prefix_2, idx, begin, instruct)?;
    }

    writeln!(writer, "{}Store:", prefix)?;
    for store in lcs.stores.iter() {
        writeln!(writer, "{}{:?}", prefix_2, store)?;
    }

    Ok(())
}

fn print_instructions_indented<'a>(
    instructions: impl Iterator<Item = &'a Instruction>,
    spaces: usize,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let prefix = " ".repeat(spaces);
    for (idx, instruct) in instructions.enumerate() {
        match instruct.node() {
            InstructionNode::Fork {
                new_thread,
                instructions,
            } => {
                writeln!(writer, "{}{}: Fork: {:?}", prefix, idx, new_thread)?;
                print_instructions_indented(instructions.iter(), spaces + 2, writer)?;
            }
            InstructionNode::SubGraphFor {
                prologue,
                body,
                total_iters,
                to_gpu_stream,
                from_gpu_stream,
                device_types,
                compute_stream_id,
            } => {
                writeln!(
                    writer,
                    "{}{}: SubgraphFor {{ to_gpu_stream = {:?}, from_gpu_stream = {:?}, total_iters = {}, device_types = {:?}, compute_stream_id = {:?} }}",
                    prefix, idx, to_gpu_stream, from_gpu_stream, total_iters, device_types, compute_stream_id
                )?;
                let prefix = " ".repeat(spaces + 2);
                writeln!(writer, "{}Prologue:", prefix)?;
                print_lcs_indented(prologue, spaces + 4, writer)?;

                for (i, body) in body.iter().enumerate() {
                    writeln!(writer, "{}Body{}:", prefix, i)?;
                    print_lcs_indented(body, spaces + 4, writer)?;
                }
            }
            _ => writeln!(writer, "{}{}: {:?}", prefix, idx, instruct)?,
        }
    }
    Ok(())
}

pub fn print_instructions(
    instructions: &[Instruction],
    writer: &mut impl Write,
) -> std::io::Result<()> {
    print_instructions_indented(instructions.iter(), 0, writer)
}
