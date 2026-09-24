use super::super::InstructionIndex as T3Index;
use super::prelude::*;
use zkpoly_common::define_usize_id;

define_usize_id!(InstructionId);

#[derive(Debug, Hash, Clone)]
pub(super) struct Cell {
    thread: ThreadId,
    pub(super) inst: Instruction,
    pub(super) tail: Vec<Instruction>,
}

impl Cell {
    pub fn new(inst: Instruction, tid: ThreadId) -> Self {
        Cell {
            inst,
            tail: vec![],
            thread: tid,
        }
    }
}

#[derive(Debug, Hash, Clone)]
pub struct MultithreadChunk {
    pub(super) instructions: Heap<InstructionId, Cell>,
    pub(super) threads: Heap<ThreadId, Vec<InstructionId>>,
    pub(super) primary_thread_id: ThreadSpecific<ThreadId>,
    pub(super) t3idx2id: BTreeMap<T3Index, InstructionId>,
    forks: BTreeMap<ThreadId, (ThreadId, InstructionId)>,
}

impl MultithreadChunk {
    pub fn new() -> Self {
        let mut threads = Heap::new();
        let primary_thread_id = ThreadSpecific::new(|| threads.push(vec![]));
        MultithreadChunk {
            instructions: Heap::new(),
            threads,
            primary_thread_id,
            t3idx2id: BTreeMap::new(),
            forks: BTreeMap::new(),
        }
    }

    pub fn emit(&mut self, thread_id: ThreadId, instruction: Instruction) -> InstructionId {
        let inst_id = self.instructions.push(Cell::new(instruction, thread_id));
        self.threads[thread_id].push(inst_id);
        inst_id
    }

    pub fn emit_primary(
        &mut self,
        thread: PrimaryThread,
        instruction: Instruction,
    ) -> InstructionId {
        self.emit(*self.primary_thread_id.get(thread), instruction)
    }

    pub fn emit_with_idx(&mut self, t3idx: T3Index, thread_id: ThreadId, instruction: Instruction) {
        let inst_id: InstructionId = self.instructions.push(Cell::new(instruction, thread_id));
        self.t3idx2id.insert(t3idx, inst_id);
        self.threads[thread_id].push(inst_id);
    }

    pub fn emit_primary_with_idx(
        &mut self,
        t3idx: T3Index,
        thread: PrimaryThread,
        instruction: Instruction,
    ) {
        self.emit_with_idx(t3idx, *self.primary_thread_id.get(thread), instruction);
    }

    pub fn append_at(&mut self, t3idx: T3Index, tail: Instruction) {
        self.instructions[*self
            .t3idx2id
            .get(&t3idx)
            .unwrap_or_else(|| panic!("instruction for type3 {:?} not found", t3idx))]
        .tail
        .push(tail);
    }

    pub fn new_auxilary_thread(&mut self) -> ThreadId {
        self.threads.push(vec![])
    }

    pub fn emit_fork(&mut self, pthread: PrimaryThread, fork_to: ThreadId) {
        let fork_inst_id = self.emit_primary(
            pthread,
            Instruction::new(
                InstructionNode::Fork {
                    new_thread: fork_to,
                    instructions: Vec::new(),
                },
                Track::Cpu.into(),
                None,
            ),
        );
        self.forks.insert(
            fork_to,
            (*self.primary_thread_id.get(pthread), fork_inst_id),
        );
    }

    pub fn thread_instructions(&self, thread_id: ThreadId) -> impl Iterator<Item = &Instruction> {
        let inst_ids = &self.threads[thread_id];
        inst_ids.iter().flat_map(|&inst_id| {
            std::iter::once(&self.instructions[inst_id].inst)
                .chain(self.instructions[inst_id].tail.iter())
        })
    }

    pub fn primary_thread_instructions(
        &self,
        pthread: PrimaryThread,
    ) -> impl Iterator<Item = &Instruction> {
        self.thread_instructions(*self.primary_thread_id.get(pthread))
    }

    pub fn fillback_auxiliary_instructions(&mut self) {
        for (&fork_to, &(_, fork_inst_id)) in self.forks.iter() {
            let aux_inst = self.thread_instructions(fork_to).cloned().collect();

            let fork_inst = &mut self.instructions[fork_inst_id];
            if let InstructionNode::Fork { instructions, .. } = fork_inst.inst.node_mut() {
                *instructions = aux_inst;
            } else {
                panic!("expect to fillback to a fork instruction")
            }
        }
    }

    pub fn thread_of(&self, t3idx: T3Index) -> ThreadId {
        let inst_id = self.t3idx2id[&t3idx];
        self.instructions[inst_id].thread
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrimaryThread {
    MemoryManagement,
    Register,
    Gpu,
    Cpu,
    ToDisk,
    FromDisk,
}

impl PrimaryThread {
    pub fn main() -> Self {
        PrimaryThread::MemoryManagement
    }
}

#[derive(Debug, Hash, Clone)]
pub struct ThreadSpecific<T> {
    memory_management: T,
    register: T,
    gpu: T,
    cpu: T,
    to_disk: T,
    from_disk: T,
}

impl<T> ThreadSpecific<T> {
    pub fn get(&self, thread: PrimaryThread) -> &T {
        match thread {
            PrimaryThread::MemoryManagement => &self.memory_management,
            PrimaryThread::Register => &self.register,
            PrimaryThread::Gpu => &self.gpu,
            PrimaryThread::Cpu => &self.cpu,
            PrimaryThread::FromDisk => &self.from_disk,
            PrimaryThread::ToDisk => &self.to_disk,
        }
    }

    #[allow(unused)]
    pub fn get_mut(&mut self, thread: PrimaryThread) -> &mut T {
        match thread {
            PrimaryThread::MemoryManagement => &mut self.memory_management,
            PrimaryThread::Register => &mut self.register,
            PrimaryThread::Gpu => &mut self.gpu,
            PrimaryThread::Cpu => &mut self.cpu,
            PrimaryThread::FromDisk => &mut self.from_disk,
            PrimaryThread::ToDisk => &mut self.to_disk,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (PrimaryThread, &T)> {
        [
            (PrimaryThread::MemoryManagement, &self.memory_management),
            (PrimaryThread::Register, &self.register),
            (PrimaryThread::Gpu, &self.gpu),
            (PrimaryThread::Cpu, &self.cpu),
            (PrimaryThread::ToDisk, &self.to_disk),
            (PrimaryThread::FromDisk, &self.from_disk),
        ]
        .into_iter()
    }

    pub fn new(mut f: impl FnMut() -> T) -> Self {
        ThreadSpecific {
            memory_management: f(),
            register: f(),
            gpu: f(),
            cpu: f(),
            to_disk: f(),
            from_disk: f(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    ToGpu,
    FromGpu,
    Gpu,
    GpuMemory,
}

impl From<Stream> for zkpoly_runtime::instructions::Stream {
    fn from(value: Stream) -> Self {
        use Stream::*;
        match value {
            ToGpu => Self::ToGpu,
            FromGpu => Self::FromGpu,
            Gpu => Self::Gpu,
            GpuMemory => Self::GpuMemory,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamSpecific<T> {
    to_gpu: T,
    from_gpu: T,
    gpu: T,
    gpu_memory: T,
}

impl<T> StreamSpecific<T> {
    pub fn get(&self, stream: Stream) -> &T {
        match stream {
            Stream::ToGpu => &self.to_gpu,
            Stream::FromGpu => &self.from_gpu,
            Stream::Gpu => &self.gpu,
            Stream::GpuMemory => &self.gpu_memory,
        }
    }

    pub fn get_mut(&mut self, stream: Stream) -> &mut T {
        match stream {
            Stream::ToGpu => &mut self.to_gpu,
            Stream::FromGpu => &mut self.from_gpu,
            Stream::Gpu => &mut self.gpu,
            Stream::GpuMemory => &mut self.gpu_memory,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Stream, &T)> {
        [
            (Stream::ToGpu, &self.to_gpu),
            (Stream::FromGpu, &self.from_gpu),
            (Stream::Gpu, &self.gpu),
            (Stream::GpuMemory, &self.gpu_memory),
        ]
        .into_iter()
    }

    pub fn new(mut f: impl FnMut() -> T) -> Self
    where
        T: Clone,
    {
        StreamSpecific {
            to_gpu: f(),
            from_gpu: f(),
            gpu: f(),
            gpu_memory: f(),
        }
    }
}

impl PrimaryThread {
    pub fn for_track(track: Track) -> Self {
        match track {
            Track::CoProcess(_) => PrimaryThread::Gpu,
            Track::Gpu(_) => PrimaryThread::Gpu,
            Track::MemoryManagement => PrimaryThread::MemoryManagement,
            Track::ToGpu => PrimaryThread::MemoryManagement,
            Track::FromGpu => PrimaryThread::MemoryManagement,
            Track::GpuMemory(_) => PrimaryThread::MemoryManagement,
            Track::Cpu => PrimaryThread::Cpu,
            Track::FromDisk => PrimaryThread::FromDisk,
            Track::ToDisk => PrimaryThread::ToDisk,
        }
    }
}

impl Stream {
    pub fn of_track(track: Track) -> Option<Self> {
        match track {
            Track::Gpu(_) => Some(Stream::Gpu),
            Track::ToGpu => Some(Stream::ToGpu),
            Track::FromGpu => Some(Stream::FromGpu),
            Track::GpuMemory(_) => Some(Stream::GpuMemory),
            _ => None,
        }
    }

    pub fn of_coprocess(track: Track) -> Self {
        match track {
            Track::CoProcess(_i) => Stream::Gpu,
            _ => panic!("called Stream::of_coprocess on {:?}", track),
        }
    }
}
