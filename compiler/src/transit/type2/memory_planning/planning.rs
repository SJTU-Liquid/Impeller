use std::collections::VecDeque;

use super::prelude::*;
pub use machine::*;

pub mod criterion;
pub mod machine;
mod stages;
pub use stages::{DevicesStage, Stage};

fn plan_type2<'m, 's, 'a, P, M, Rt: RuntimeType>(
    t2op: Type2Op<'s, Object, Option<Pointer<P>>>,
    machine: &mut MachineAt<'m, 'a, phase1::Operation<'s, P>, P, M, Rt>,
) -> Result<(), OomError>
where
    M: criterion::Manager,
    P: UsizeId + 'static,
{
    let input_objects = t2op
        .node
        .uses_ref()
        .map(|vi| vi.iter().map(|ira| ira.object().clone()))
        .flatten()
        .collect::<ObjectTable<()>>();
    // Obtain pointer for inputs.
    // We only care about objects on planning devices, and pointers to unplanned devices will be assigned
    // in later phases.
    let node = t2op.node.try_relabeled(|ri| {
        ri.try_map(|ira| {
            if machine.current_stage()[ira.device()].is_planning_or_unplanned() {
                let pointer = machine.claim_and_queue_unlock(ira.object(), ira.device())?;
                if machine.current_stage()[ira.device()].is_planning() {
                    Ok(ira.map_v(|ra| ra.set_pointer(Some(pointer))))
                } else {
                    Ok(ira)
                }
            } else {
                Ok(ira)
            }
        })
    })?;

    // Allocate outputs and temporaries.
    // We only care about objects on planning devices and unplanned devices.
    // - If the output took place of some input object, then later reuse space of the input object
    // - If the output object is the same as input, do nothing
    // - Otherwise, allocate new space
    for ora in t2op.output.iter() {
        if machine.current_stage()[ora.device()].is_planning_or_unplanned() {
            if let Some(inplace_of) = ora.inplace_of() {
                machine.reuse(ora.device(), ora.object(), inplace_of.object());
            } else if input_objects.contains(&ora.object()) {
                // do nothing
            } else {
                machine.allocate_and_queue_unlock(ora.device(), ora.object())?;
            }
        }
    }

    if let Some(t) = &t2op.temporaries {
        let vo = &t.buf;
        if machine.current_stage()[vo.device()].is_planning() {
            machine.allocate_and_queue_unlock(vo.device(), vo.object())?;
        }
    }

    // Obtain pointer for outputs and temporaries.
    let output = t2op.output.map(|ora| {
        if machine.current_stage()[ora.device()].is_planning() {
            let pointer = machine
                .access(ora.device(), ora.object())
                .expect("to find just allocated output");
            ora.map_v(|ra| ra.set_pointer(Some(pointer)))
                .map_i(|ra| ra.map(|ra| ra.set_pointer(Some(pointer))))
        } else {
            ora
        }
    });

    let temps = t2op.temporaries.map(|t| {
        let ra = &t.buf;
        if machine.current_stage()[ra.device()].is_planning() {
            let pointer = machine
                .access(ra.device(), ra.object())
                .expect("to find just allocated temporary");
            Temporaries {
                buf: t.buf.set_pointer(Some(pointer)),
                slices: t.slices,
            }
        } else {
            t
        }
    });

    machine.emit(Type2Op::new(
        t2op.vid,
        output,
        temps.clone(),
        node,
        t2op.src,
    ));

    // Deallocate temporary spaces right away, for clearity
    temps.inspect(|t| {
        let ra = &t.buf;
        if machine.current_stage()[ra.device()].is_planning() {
            machine.deallocate(ra.device(), ra.object());
        }
    });

    Ok(())
}

pub mod subgraph;

fn plan_op<'m, 's, 'a, P, M, Rt: RuntimeType>(
    op: phase1::Operation<'s, P>,
    machine: &mut MachineAt<'m, 'a, phase1::Operation<'s, P>, P, M, Rt>,
) -> Result<(), Error<'s>>
where
    M: criterion::Manager,
    P: UsizeId + 'static,
{
    use Operation::*;
    match op {
        Type2(t2op) => {
            let vid = t2op.vid;
            let src = t2op.src.clone();
            plan_type2(t2op, machine).map_err(|e| Error::oom_during_type2(e, vid, src))?;
        }
        Sub(mut rsg) => {
            let sg = &mut rsg.sg;
            // - Eject objects from `sg.device` if space is insufficient for the subgraph
            //
            //   Double buffer here need two instances of buffer
            //   (Lp1 for load prologue to buffer 1, 1Cp1 for compute prologue from buffer 1 to buffer 1, etc.)
            //
            //    L    C   S    |       Buffers
            //   Lp1            |   Bp1   |
            //   Lb1 1Cp1       |   Bp1   |  Bb1  |Bi|
            //   Lb2 2Cb2 1Sp   |  Bb2  | |  Bb1  |Bi|
            //       1Cb1 2Sb   |  Bb2  | |  Bb1  |Bi|
            //            1Sb   |  Bb2  |
            if machine.current_stage()[sg.device].is_planning() {
                let vid = sg.vid;

                let object = rsg.allocation.object.clone();
                let addr = machine
                    .allocate_and_queue_unlock(sg.device, &object)
                    .map_err(|e| Error::subgraph_oom(e, vid))?;
                rsg.allocation = rsg.allocation.with_pointer(addr);
            }

            // - Claim on-device inputs of each paragraph to device, and fill in pointers
            if machine.current_stage()[sg.device].is_planning() {
                for lcs in [&mut sg.prologue, &mut sg.body] {
                    for (_, op) in lcs.compute.iter_mut() {
                        for (o, p) in op.maybe_unfulfilled_pointer_mut() {
                            if p.is_none() {
                                let pointer = machine
                                    .claim_and_queue_unlock(o, sg.device)
                                    .map_err(|e| Error::subgraph_oom(e, sg.vid))?;
                                *p = Some(pointer);

                                *sg.on_device_uses
                                    .iter_mut()
                                    .map(|irv| irv.unwrap_leaf_mut())
                                    .find(|ira| ira.object() == o)
                                    .unwrap()
                                    .pointer_mut() = Some(pointer);
                            }
                        }
                    }
                }
            }

            // - Correct loads' devices to where the objects are now
            //   - If there is still space left on planning device, claim object to it
            //   - Only set pointer if the object is on planning device since this ensure finding the object on a planning device
            let mut loads_pointer = ObjectTable::new();
            for irv in sg.uses.iter_mut() {
                irv.iter_mut().for_each(|ira| {
                    *ira = ira.clone().map_v(|ra| {
                        if ra.pointer().is_none() {
                            let (d, p) = machine.claim_where_you_can(ra.object(), ra.device());

                            let p = if machine.current_stage()[d].is_planning() {
                                Some(p)
                            } else {
                                None
                            };

                            loads_pointer.insert(ra.object().clone(), (d, p));
                            ra.set_pointer(p).map_v(|a| a.with_device(d))
                        } else {
                            ra
                        }
                    })
                });
            }

            for lcs in [&mut sg.prologue, &mut sg.body] {
                lcs.load.iter_mut().for_each(|(_, src)| {
                    if let Some((d, p)) = loads_pointer.get(src.object()) {
                        *src = src.clone().set_pointer(*p).map_v(|a| a.with_device(*d));
                    }
                });
            }

            // - Allocate output first on `sg.device`, then on its parent, grandparent devices if space insufficient
            let mut stores_pointer = ObjectTable::new();
            sg.output.iter_mut().for_each(|ora| {
                *ora = ora.clone().map_v(|ra| {
                    if ora.pointer().is_none() {
                        let (d, p) = machine.allocate_where_you_can(ora.object(), ora.device());

                        let p = if machine.current_stage()[d].is_planning() {
                            Some(p)
                        } else {
                            None
                        };

                        stores_pointer.insert(ra.object().clone(), (d, p));
                        ra.set_pointer(p).map_v(|a| a.with_device(d))
                    } else {
                        ra
                    }
                })
            });

            for lcs in [&mut sg.prologue, &mut sg.body] {
                lcs.store.iter_mut().for_each(|(dst, _)| {
                    if let Some((d, p)) = stores_pointer.get(dst.object()) {
                        *dst = dst.clone().set_pointer(*p).map_v(|a| a.with_device(*d));
                    }
                });
            }

            let device = sg.device;
            let object = rsg.allocation.object.clone();
            machine.emit(Sub(rsg));

            // Deallocate buffer
            if machine.current_stage()[device].is_planning() {
                machine.deallocate(device, &object);
            }
        }
        Clone(dst, src) => {
            machine.clone_object(dst, src).map_err(|e| Error::oom(e))?;
        }
        Transfer(dst, dst_device, src) => {
            machine
                .transfer(dst, dst_device, src)
                .map_err(|e| Error::oom(e))?;
        }
        op @ Ad(..) => machine.emit(op),
    }

    Ok(())
}

/// Run memory planning on `current_stage.planning_devices()`, where `planned_devices` has been planned and `unplanned_devices`
/// has not.
///
/// The invariant is that all needed reclaim/reject from and to planned devices are already in the
/// operation sequence `source_ops`.
/// Also, before an object dies on all devices, i.e., will not be used anywhere,
/// it must can be found on some device.
///
/// `allcators` must contains allocators for all planning devices.
///
/// Note that for inplace operation
///   (v1 <- io2, ... ) = type2_vertex (u1, u2, ...)
/// where v1, u1, u2 are [`ResidentalAtom`]'s,  u2.object() is io2, we'll assign the same pointer to v1 and u2
fn plan_graph<'s, 'a, P, M, Rt: RuntimeType>(
    source_ops: phase1::OperationSeq<'s, P>,
    allocators: AllocatorCollection<'a, P, M::C, M::Cf>,
    next_uses: liveness::NextUses,
    current_stage: DevicesStage,
    hd_info: &HardwareInfo,
) -> Result<
    (
        phase1::OperationSeq<'s, P>,
        AllocatorCollection<'a, P, M::C, M::Cf>,
        machine::EjectionStatistics,
    ),
    Error<'s>,
>
where
    M: criterion::Manager,
    P: UsizeId + 'static,
{
    let mut machine = Machine::<_, _, _, Rt>::new(current_stage, allocators, next_uses, hd_info);

    for (index, op) in source_ops.into_iter() {
        let mut machine = machine.at(index);
        plan_op::<_, M, Rt>(op, &mut machine)?;
        machine.finish().map_err(|e| Error::oom(e))?;
    }

    Ok(machine.finish())
}

pub struct PlanDriver<'a, 'i, P, M, Rt: RuntimeType>
where
    M: criterion::Manager,
{
    allocators: AllocatorCollection<'a, P, M::C, M::Cf>,
    hd_info: &'i HardwareInfo,
    plan_phases: VecDeque<BTreeSet<Device>>,
    current_stage: DevicesStage,
    _phantom: PhantomData<(M, Rt)>,
}

pub trait PlanDrivers<P> {
    fn analyze_next_uses<'s>(&self, source_ops: &phase1::OperationSeq<'s, P>)
        -> liveness::NextUses;
    fn advance<'s>(
        &mut self,
        ops: phase1::OperationSeq<'s, P>,
        next_uses: liveness::NextUses,
    ) -> Result<
        (
            phase1::OperationSeq<'s, P>,
            bool,
            machine::EjectionStatistics,
        ),
        Error<'s>,
    >;
}

impl<'a, 'i, P, M, Rt: RuntimeType> PlanDriver<'a, 'i, P, M, Rt>
where
    M: criterion::Manager,
    P: UsizeId + 'static,
{
    pub fn new<Ca, Ga, Da>(
        gpu_allocators: &'a mut [Ga],
        cpu_allocator: &'a mut Ca,
        disk_allocator: &'a mut Da,
        hd_info: &'i HardwareInfo,
    ) -> Self
    where
        Ca: Allocator<P, M::C, M::Cf>,
        Ga: Allocator<P, M::C, M::Cf>,
        Da: Allocator<P, M::C, M::Cf>,
    {
        if gpu_allocators.len() != hd_info.n_gpus() {
            panic!(
                "expected {} allocators because we have that many GPUs, got {}",
                hd_info.n_gpus(),
                gpu_allocators.len()
            );
        }

        let plan_phases: VecDeque<BTreeSet<Device>> = vec![
            (0..hd_info.n_gpus()).map(|i| Device::Gpu(i)).collect(),
            [Device::Cpu].into_iter().collect(),
            [Device::Disk].into_iter().collect(),
        ]
        .into();
        let current_stage = DevicesStage::all_unplanned(gpu_allocators.len());

        let allocators = AllocatorCollection::new_all(DeviceSpecific {
            cpu: Some(cpu_allocator),
            gpu: gpu_allocators
                .iter_mut()
                .map(|ga| Some(ga as &mut dyn Allocator<_, _, _>))
                .collect(),
            disk: Some(disk_allocator),
        });

        Self {
            allocators,
            hd_info,
            plan_phases,
            current_stage,
            _phantom: PhantomData,
        }
    }
}

impl<'a, 'i, P, M, Rt: RuntimeType> PlanDrivers<P> for PlanDriver<'a, 'i, P, M, Rt>
where
    M: criterion::Manager,
    P: UsizeId + 'static,
{
    fn analyze_next_uses<'s>(
        &self,
        source_ops: &phase1::OperationSeq<'s, P>,
    ) -> liveness::NextUses {
        let read_by = liveness::ReadBy::analyze::<phase1::Operation<'s, P>, _>(
            source_ops.iter(),
            self.hd_info,
        );
        let write_at = liveness::WriteAt::analyze::<phase1::Operation<'s, P>, _>(
            source_ops.iter(),
            self.hd_info,
        );

        liveness::NextUses::new(read_by, write_at)
    }

    fn advance<'s>(
        &mut self,
        mut ops: phase1::OperationSeq<'s, P>,
        next_uses: liveness::NextUses,
    ) -> Result<(phase1::OperationSeq<'s, P>, bool, EjectionStatistics), Error<'s>> {
        let planning_devices = self.plan_phases.pop_front().unwrap();
        self.current_stage.advance(planning_devices.into_iter());
        let allocators = std::mem::take(&mut self.allocators);

        let ejection_statistics: EjectionStatistics;
        (ops, self.allocators, ejection_statistics) = plan_graph::<P, M, Rt>(
            ops,
            allocators,
            next_uses,
            self.current_stage.clone(),
            &self.hd_info,
        )?;
        Ok((ops, self.plan_phases.is_empty(), ejection_statistics))
    }
}
