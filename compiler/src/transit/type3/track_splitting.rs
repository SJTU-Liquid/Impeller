use std::collections::BTreeMap;

use zkpoly_runtime::args::RuntimeType;

use crate::driver;

use super::{Chunk, InstructionIndex, Track, TrackSpecific};

#[derive(Debug, Hash, Clone)]
pub struct TrackTasks {
    pub inst_track: BTreeMap<InstructionIndex, Track>,
    pub inst_depend: BTreeMap<InstructionIndex, Vec<InstructionIndex>>,
}

impl TrackTasks {
    pub fn new() -> Self {
        Self {
            inst_track: BTreeMap::new(),
            inst_depend: BTreeMap::new(),
        }
    }
}
pub fn split<'s, Rt: RuntimeType>(
    chunk: &Chunk<'s, Rt>,
    hd_info: &driver::HardwareInfo,
) -> TrackTasks {
    let mut track_tasks = TrackTasks::new();

    for (i, inst) in chunk.instructions.iter() {
        let track = inst.op.track(|reg_id| {
            chunk
                .memory_device(reg_id)
                .expect("to only query memory device of register for transfer instructions")
        });

        let depended_instructions = chunk.instructions.dependencies(i);

        // Collect last dependency of each track
        let mut depended_of_each_track =
            TrackSpecific::<Vec<InstructionIndex>>::default(hd_info.n_gpus());

        for depended_inst in depended_instructions {
            let depended_track = chunk.instructions[depended_inst].op.track(|reg_id| {
                chunk
                    .memory_device(reg_id)
                    .expect("to only query memory device of register for transfer instructions")
            });

            if depended_track.is_cpu() {
                // CPU tracks are strictly serial, so depending on last one would do
                let last = depended_of_each_track
                    .get_track(depended_track)
                    .get(0)
                    .cloned()
                    .unwrap_or_else(|| depended_inst);
                *depended_of_each_track.get_track_mut(depended_track) =
                    vec![std::cmp::max(last, depended_inst)];
            } else {
                // Instructions on each GPU track can execute out-of-order, so this instruction must depend on all of them
                depended_of_each_track
                    .get_track_mut(depended_track)
                    .push(depended_inst);
            }
        }

        // Insert synchronization points
        track_tasks.inst_depend.insert(i, Vec::new());

        depended_of_each_track
            .iter(hd_info.n_gpus())
            .for_each(|(depended_track, depended)| {
                depended.iter().for_each(|&depended| {
                    let same_cpu_track = depended_track.is_cpu() && track == depended_track;
                    if depended != i && !same_cpu_track {
                        track_tasks.inst_depend.get_mut(&i).unwrap().push(depended);
                    }
                });
            });

        track_tasks.inst_track.insert(i, track);
    }

    track_tasks
}
