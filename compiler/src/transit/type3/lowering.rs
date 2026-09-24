mod prelude {

    pub(super) use std::collections::{BTreeMap, BTreeSet};
    pub(super) use std::path::PathBuf;

    pub(super) use crate::transit::type2;
    pub(super) use crate::transit::type2::object_analysis::size::LogBlockSizes;
    pub(super) use crate::transit::type3;

    pub(super) use super::super::track_splitting::TrackTasks;
    pub(super) use super::super::{Track, VertexNode};
    pub(super) use super::kernel_gen::GeneratedFunctions;
    pub(super) use zkpoly_common::devices::DeviceType;
    pub(super) use zkpoly_common::heap::{Heap, IdAllocator};
    pub(super) use zkpoly_common::load_dynamic::Libs;
    pub(super) use zkpoly_common::typ::Typ;
    pub(super) use zkpoly_runtime::args::{RuntimeType, VariableId};
    pub(super) use zkpoly_runtime::devices::{EventId, EventType, EventTypeTable, ThreadId};
    pub(super) use zkpoly_runtime::functions::FunctionTable;
    pub(super) use zkpoly_runtime::instructions::{AllocMethod, Instruction, InstructionNode};
}
pub use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;

use prelude::*;

mod emit_func;
mod kernel_gen;
mod threads;
pub use threads::Stream;
use threads::{InstructionId, MultithreadChunk, PrimaryThread, StreamSpecific};

mod instructions;
use instructions::lower_instruction;

impl From<super::Device> for DeviceType {
    fn from(value: super::Device) -> Self {
        match value {
            super::Device::Cpu => DeviceType::CPU,
            super::Device::Gpu(i) => DeviceType::GPU {
                device_id: i as i32,
            },
            super::Device::Disk => DeviceType::Disk,
        }
    }
}

/// A CPU thread `thread` waits for instruction `depended_t3idx` at `depended_track`
fn lower_cpu_waits_any(
    depended_t3idx: super::InstructionIndex,
    thread: ThreadId,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2cpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    match Stream::of_track(depended_track) {
        Some(depended_stream) => {
            let event_id = instruct2gpu_event
                .get(&depended_t3idx)
                .copied()
                .unwrap_or_else(|| {
                    let event_id = event_table.push(EventType::new_gpu(0)); // one card for now
                    instruct2gpu_event.insert(depended_t3idx, event_id);

                    // Record/Wait are not considered to be on GPU
                    chunk.append_at(
                        depended_t3idx,
                        Instruction::new(
                            InstructionNode::Record {
                                stream: Some(*stream2variable_id.get(depended_stream)),
                                event: event_id,
                            },
                            Track::Cpu.into(),
                            None,
                        ),
                    );
                    event_id
                });

            chunk.emit(
                thread,
                Instruction::new(
                    InstructionNode::Wait {
                        slave: DeviceType::CPU,
                        stream: None,
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
        }
        None => {
            let event_id = instruct2cpu_event
                .get(&depended_t3idx)
                .copied()
                .unwrap_or_else(|| {
                    let event_id = event_table.push(EventType::new_thread());
                    instruct2cpu_event.insert(depended_t3idx, event_id);

                    chunk.append_at(
                        depended_t3idx,
                        Instruction::new(
                            InstructionNode::Record {
                                stream: None,
                                event: event_id,
                            },
                            Track::Cpu.into(),
                            None,
                        ),
                    );
                    event_id
                });

            chunk.emit(
                thread,
                Instruction::new(
                    InstructionNode::Wait {
                        slave: DeviceType::CPU,
                        stream: None,
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
        }
    }
}

/// A GPU stream `stream` waits for instruction `depended_t3idx` at `depended_track`, which must be a GPU stream.
/// The event wait function is issued at the thread `thread`.
fn lower_gpu_waits_gpu(
    depended_t3idx: super::InstructionIndex,
    launch_thread: ThreadId,
    stream: Stream,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    let event_id = instruct2gpu_event
        .get(&depended_t3idx)
        .copied()
        .unwrap_or_else(|| {
            let event_id = event_table.push(EventType::new_gpu(0));
            instruct2gpu_event.insert(depended_t3idx, event_id);

            chunk.append_at(
                depended_t3idx,
                Instruction::new(
                    InstructionNode::Record {
                        stream: Some(
                            *stream2variable_id.get(Stream::of_track(depended_track).unwrap()),
                        ),
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
            event_id
        });

    chunk.emit(
        launch_thread,
        Instruction::new(
            InstructionNode::Wait {
                slave: DeviceType::GPU { device_id: 0 },
                stream: Some(*stream2variable_id.get(stream)),
                event: event_id,
            },
            Track::Cpu.into(),
            None,
        ),
    );
}

fn lower_dependency(
    depended_t3idx: super::InstructionIndex,
    thread: PrimaryThread,
    track: Track,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2cpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    let thread = *chunk.primary_thread_id.get(thread);
    match (Stream::of_track(track), Stream::of_track(depended_track)) {
        (None, _) => lower_cpu_waits_any(
            depended_t3idx,
            thread,
            depended_track,
            stream2variable_id,
            event_table,
            chunk,
            instruct2cpu_event,
            instruct2gpu_event,
        ),
        (Some(stream), Some(..)) => lower_gpu_waits_gpu(
            depended_t3idx,
            thread,
            stream,
            depended_track,
            stream2variable_id,
            event_table,
            chunk,
            instruct2gpu_event,
        ),
        (a, b) => panic!(
            "cannot handle {:?}@{:?} waits {:?}@{:?} here",
            track, a, depended_track, b
        ),
    }
}

#[derive(Clone, Hash)]
pub struct Chunk<Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) f_table: FunctionTable<Rt>,
    pub(crate) event_table: EventTypeTable,
    pub(crate) n_variables: usize,
    pub(crate) n_threads: usize,
    pub(crate) lbss: LogBlockSizes,
}

pub fn emit_multithread_instructions<'s, Rt: RuntimeType>(
    track_tasks: &TrackTasks,
    mut t3chunk: super::Chunk<'s, Rt>,
    t2uf_table: type2::user_function::Table<Rt>,
    libs: &mut Libs,
    kernel_dir: PathBuf,
    ag_scheduler: ArithGraphScheduler,
) -> (
    MultithreadChunk,
    FunctionTable<Rt>,
    EventTypeTable,
    StreamSpecific<VariableId>,
    IdAllocator<VariableId>,
    LogBlockSizes,
) {
    let mut event_table = EventTypeTable::new();
    let mut f_table = FunctionTable::<Rt>::new();
    let (mut variable_id_allcoator, reg_id2var_id) = t3chunk.take_reg_id_allocator().decompose();

    std::fs::create_dir_all(&kernel_dir).expect("failed to create directory for kernels");
    let generated_functions = kernel_gen::get_function_id(
        &mut f_table,
        &t3chunk,
        t2uf_table,
        &|i| reg_id2var_id.on(i),
        libs,
        kernel_dir,
        ag_scheduler,
    );

    let stream2variable_id = StreamSpecific::new(|| variable_id_allcoator.alloc());

    let mut chunk = MultithreadChunk::new();

    let mut instruct2cpu_event = BTreeMap::new();
    let mut instruct2gpu_event = BTreeMap::new();

    for (t3idx, t3inst) in t3chunk.instructions.iter() {
        let track = track_tasks.inst_track[&t3idx];
        let pthread = PrimaryThread::for_track(track);

        if track.is_gpu()
            && track_tasks.inst_depend[&t3idx]
                .iter()
                .any(|&depended_t3idx| track_tasks.inst_track[&depended_t3idx].is_cpu())
        {
            // GPU waits for some CPU tasks
            let depended_threads: BTreeSet<_> = track_tasks.inst_depend[&t3idx]
                .iter()
                .copied()
                .filter(|&t3idx| track_tasks.inst_track[&t3idx].is_cpu())
                .map(|depended_t3idx| chunk.thread_of(depended_t3idx))
                .collect();

            if depended_threads.len() > 1 {
                // A GPU task waits for CPU tasks on multiple threads.
                // In this case, we need a new CPU thread that first waits for those CPU tasks,
                // then the CPU thread launches the GPU task.
                let aux_thread = chunk.new_auxilary_thread();

                for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                    let depended_track = track_tasks.inst_track[&depended_t3idx];
                    lower_cpu_waits_any(
                        depended_t3idx,
                        aux_thread,
                        depended_track,
                        &stream2variable_id,
                        &mut event_table,
                        &mut chunk,
                        &mut instruct2cpu_event,
                        &mut instruct2gpu_event,
                    );
                }

                lower_instruction(
                    t3idx,
                    t3inst,
                    track,
                    reg_id2var_id,
                    &stream2variable_id,
                    &t3chunk,
                    &generated_functions,
                    &mut |inst| chunk.emit_with_idx(t3idx, aux_thread, inst),
                );

                chunk.emit_fork(pthread, aux_thread);
            } else {
                // A GPU task waits for CPU tasks on a single thread.
                // In this case, we can just launch the GPU task on the same thread.
                let thread = depended_threads.first().copied().unwrap();
                let stream = Stream::of_track(track).unwrap();

                // First waits for GPU tasks
                for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                    let depended_track = track_tasks.inst_track[&depended_t3idx];
                    if Stream::of_track(depended_track).is_some() {
                        lower_gpu_waits_gpu(
                            depended_t3idx,
                            thread,
                            stream,
                            depended_track,
                            &stream2variable_id,
                            &mut event_table,
                            &mut chunk,
                            &mut instruct2gpu_event,
                        );
                    }
                }

                lower_instruction(
                    t3idx,
                    t3inst,
                    track,
                    reg_id2var_id,
                    &stream2variable_id,
                    &t3chunk,
                    &generated_functions,
                    &mut |inst| chunk.emit_with_idx(t3idx, thread, inst),
                );
            }
        } else {
            // GPU waits for only GPU tasks, or CPU waits.
            for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                let depended_track = track_tasks.inst_track[&depended_t3idx];
                lower_dependency(
                    depended_t3idx,
                    pthread,
                    track,
                    depended_track,
                    &stream2variable_id,
                    &mut event_table,
                    &mut chunk,
                    &mut instruct2cpu_event,
                    &mut instruct2gpu_event,
                );
            }

            lower_instruction(
                t3idx,
                t3inst,
                track,
                reg_id2var_id,
                &stream2variable_id,
                &t3chunk,
                &generated_functions,
                &mut |inst| chunk.emit_primary_with_idx(t3idx, pthread, inst),
            );
        }
    }

    (
        chunk,
        f_table,
        event_table,
        stream2variable_id,
        variable_id_allcoator,
        t3chunk.lbss,
    )
}

pub fn lower<'s, Rt: RuntimeType>(
    mut mt_chunk: MultithreadChunk,
    f_table: FunctionTable<Rt>,
    event_table: EventTypeTable,
    stream2variable_id: StreamSpecific<VariableId>,
    variable_id_allocator: IdAllocator<VariableId>,
    lbss: LogBlockSizes,
) -> Chunk<Rt> {
    let mut instructions = Vec::new();

    // Create streams
    stream2variable_id.iter().for_each(|(stream, &var_id)| {
        instructions.push(Instruction::new(
            InstructionNode::Allocate {
                device: DeviceType::GPU { device_id: 0 },
                typ: Typ::Stream,
                id: var_id,
                alloc_method: AllocMethod::default(),
            },
            Track::Cpu.into(),
            Some(stream.into()),
        ))
    });

    // Fillback auxiliary threads
    mt_chunk.fillback_auxiliary_instructions();

    // Fork primary threads
    mt_chunk
        .primary_thread_id
        .iter()
        .filter(|(pthread, _)| *pthread != PrimaryThread::main())
        .for_each(|(_pthread, &thread)| {
            instructions.push(Instruction::new(
                InstructionNode::Fork {
                    new_thread: thread,
                    instructions: mt_chunk.thread_instructions(thread).cloned().collect(),
                },
                Track::Cpu.into(),
                None,
            ))
        });

    // Emit main thread instructions
    instructions.extend(
        mt_chunk
            .primary_thread_instructions(PrimaryThread::main())
            .cloned(),
    );

    Chunk {
        instructions,
        f_table,
        event_table,
        n_threads: mt_chunk.threads.len(),
        n_variables: variable_id_allocator.n_allocated(),
        lbss,
    }
}

pub mod pretty_print;
pub mod serialization;
