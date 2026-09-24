use crate::{ast::PolyInit, transit::type2::memory_planning::prelude::*};
use std::ops::Deref;
use type3::InstructionNode;

pub mod dependency_analysis;
pub mod machine;
use machine::MachineAt;
pub use machine::{AdEmitter, Machine, RegBooking, RegGrabber, Statistics, SubgraphAdEmitter};

use super::object_analysis::size::LogBlockSizes;

type Ma<'s, 'r, P, Rt> =
    MachineAt<'s, 'r, type3::InstructionIndex, Pointer<P>, type3::Instruction<'s>, Rt>;
type Sma<'s, 'r, P, Rt> =
    MachineAt<'s, 'r, type3::LcsPc, Pointer<P>, type3::SubgraphInstruction<'s>, Rt>;

fn realize_subgraph_op<'s, 'm, 'r, P, Rt: RuntimeType>(
    machine: &mut Sma<'s, 'r, P, Rt>,
    realizer: &AsRealizer<phase1::LcsAllocator<P>, P, Epsilon, Epsilon>,
    op: phase1::SubgraphOperation<'s, Pointer<P>>,
    device: Device,
    t2device: type2::DeviceCollection,
    t2vid: type2::VertexId,
    buffer_base: RegisterId,
    rotation: i64,
) where
    P: UsizeId + 'static,
{
    use SubgraphOperation::*;
    match op {
        Type2(ChunkedOp {
            node:
                type2::sliceable_subgraph::template::VertexNode::Inner(
                    type2::template::SliceableNode::NewPoly(_, init, pty),
                ),
            output,
            ..
        }) => {
            let output = output.unwrap_leaf();
            let (deg, s) = output.object().node().unwrap_poly();
            let output_reg = machine.reg_for_atom(&output.deref().clone().assume_pointed());

            machine.emit_with_slice_begin(
                InstructionNode::FillPoly {
                    id: output_reg,
                    deg: s.len() as usize,
                    init,
                    pty,
                },
                Some(s.rotated(rotation, deg as u64).full(deg as u64)),
            )
        }
        Type2(cop) => {
            let node = cop.node.relabeled::<(), _>(
                |vi| {
                    let rv = vi.map(|ira| ira.v().assume_pointed());
                    machine.reg_for(&rv)
                },
                |_| panic!("input vertex not expected"),
            );

            let begin = cop.output.iter().fold(None, |mut acc, ora| {
                ora.object().node().map_p(|deg, slice| {
                    acc = Some(slice.rotated(rotation, deg as u64).full(deg as u64))
                });
                acc
            });

            let output_regs = cop
                .output
                .into_iter1()
                .map(|orv| match orv {
                    Tree::Single(ora) => {
                        if let Some(inplace_of) = ora.inplace_of() {
                            let inplace_reg = machine.reg_for_atom(&inplace_of.assume_pointed());
                            (
                                machine.new_atom_reg(ora.deref().clone().assume_pointed()),
                                Some(inplace_reg),
                            )
                        } else {
                            (machine.reg_for_atom(&ora.v().assume_pointed()), None)
                        }
                    }
                    Tree::Tuple(..) => (
                        machine.reg_for(&orv.map(|ora| ora.v().assume_pointed())),
                        None,
                    ),
                })
                .collect::<Vec<_>>();

            let temp_regs = cop
                .temporaries
                .map(|t| {
                    let temp_primary_reg = machine.reg_for_atom(&t.buf.clone().assume_pointed());
                    t.slices
                        .into_iter()
                        .map(|slice| {
                            let reg = machine.new_atom_reg(
                                t.buf
                                    .clone()
                                    .assume_pointed()
                                    .map_v(|a| a.map_node(|n| n.map_gpu_buffer(|_, _| slice))),
                            );

                            machine.emit_pre(type3::RegisterManipulation::SliceBuffer {
                                id: reg,
                                operand: temp_primary_reg,
                                offset: slice.begin() as usize,
                                len: slice.len() as usize,
                            });

                            reg
                        })
                        .collect()
                })
                .unwrap_or_default();

            machine.emit_with_slice_begin(
                InstructionNode::Type2 {
                    ids: output_regs.clone(),
                    temp: temp_regs,
                    vertex: node.wrap_type2(),
                    debug: type3::Debug::Subgraph(t2vid, cop.vid, cop.src),
                    device: t2device,
                },
                begin,
            );

            for r in output_regs.iter().filter_map(|(_, x)| *x) {
                machine.reg_booking_mut().free_reg(r);
            }
        }
        Clone(dst, src, slice) => {
            let ra_to = dst.map_v(|o| o.whole(device));
            let ra_from = src.map_v(|o| {
                if let Some(slice) = slice {
                    o.whole(device).slice(slice.begin(), slice.end())
                } else {
                    o.whole(device)
                }
            });

            if let Some((_, slice_to)) = ra_to.object().node().poly() {
                let (_, slice_from) = ra_from.node().unwrap_poly();
                if slice_to.len() != slice_from.len() {
                    panic!(
                        "cannot clone {:?} from {:?} due to length mismatch",
                        &ra_to, &ra_from
                    );
                }
            }

            machine.transfer(ra_to.assume_pointed(), ra_from.assume_pointed());
        }
        Concat(dst, [src1, src2]) => {
            let (_, src1_is_slice) = src1.node().unwrap_poly();
            let (_, src2_is_slice) = src2.node().unwrap_poly();
            let (_, dst_is_slice) = dst.node().unwrap_poly();
            let src1_len = src1_is_slice.len();
            let src2_len = src2_is_slice.len();
            let dst_len = dst_is_slice.len();

            if dst_len != src1_len + src2_len {
                panic!(
                    "concating object {:?} and {:?} cannot produce length of object {:?}",
                    src1, src2, dst
                );
            }

            machine.transfer(
                dst.clone().map_v(|o| o.whole(device).slice(0, src1_len)),
                src1.map_v(|o| o.whole(device)),
            );
            machine.transfer(
                dst.map_v(|o| o.whole(device).slice(src1_len, dst_len)),
                src2.map_v(|o| o.whole(device)),
            );
        }
        Ad(template::Ad::Allocate(o, _d, p)) => {
            realizer.allocate(&o, &p, &mut SubgraphAdEmitter(machine, buffer_base));
        }
        Ad(template::Ad::Deallocate(o, _d, p)) => {
            realizer.deallocate(&o, &p, &mut SubgraphAdEmitter(machine, buffer_base));
        }
    }
}

fn realize_lcs<'s, P, Rt: RuntimeType>(
    lcs: phase1::Lcs<'s, P>,
    rotation: i64,
    itms_offset: u64,
    compute_allocator_offset: u64,
    hd_info: &HardwareInfo,
    t2vid: type2::VertexId,
    device: type2::DeviceCollection,
    input_regs: &ObjectTable<RegisterId>,
    output_regs: &ObjectTable<RegisterId>,
    reg_booking: &mut RegBooking<Pointer<P>>,
    buffer_base: RegisterId,
) -> (type3::Lcs<'s>, Statistics)
where
    P: UsizeId + 'static,
{
    let mut machine = Machine::new(hd_info.n_gpus(), reg_booking);

    let mut allocator = lcs.allocator;
    allocator.unwrap_mut().set_offset(compute_allocator_offset);
    allocator.set_offset(itms_offset);
    let realizer = AsRealizer::new(&allocator);

    let mut t3loads = Vec::new();
    for (dst, src) in lcs.load.into_iter() {
        let (addr, _) = realizer
            .allocated(dst.deref(), dst.pointer())
            .unwrap_offset();

        let src_reg = *input_regs
            .get(src.object())
            .unwrap_or_else(|| panic!("{:?} is not a input", src.object()));
        let begin = dst
            .node()
            .poly()
            .map(|(deg, slice)| slice.rotated(rotation, *deg as u64));
        let dst_reg = machine
            .reg_booking_mut()
            .new_atom_reg(dst.map_v(|o| o.whole(lcs.device)));

        t3loads.push(type3::Load {
            id: dst_reg,
            offset: addr,
            input: type3::SubgraphInput {
                src: src_reg,
                begin,
            },
        })
    }

    for (op_idx, op) in lcs.compute.into_iter() {
        let mut ma = machine.at(op_idx);
        realize_subgraph_op::<_, Rt>(
            &mut ma,
            &realizer,
            op,
            lcs.device,
            device.clone(),
            t2vid,
            buffer_base,
            rotation,
        );
        machine = ma.finish();
    }

    {
        let mut ma = machine.at_append();
        for (dst, src) in lcs.save_itms.into_iter() {
            let ra_from = src.map_v(|o| o.whole(lcs.device));
            let ra_to = dst.map_v(|o| o.whole(lcs.device));
            ma.transfer(ra_to, ra_from);
        }
        machine = ma.finish();
    }

    let mut t3pre_store = Vec::new();
    let mut t3stores = Vec::new();
    for (dst, src) in lcs.store.into_iter() {
        let begin = src
            .node()
            .poly()
            .map(|(deg, slice)| slice.rotated(rotation, *deg as u64));

        let src_reg = RegGrabber::new(|m| t3pre_store.push(m)).reg_for_atom(
            &src.clone().map_v(|o| o.whole(lcs.device)),
            machine.reg_booking_mut(),
        );
        let dst_reg = output_regs[dst.object()];

        t3stores.push(type3::Store {
            dst: dst_reg,
            src: src_reg,
            begin,
        });
    }

    for to_free in lcs.post_store {
        machine.reg_booking_mut().free_reg_pointing_to(
            &to_free,
            to_free.pointer().clone(),
            lcs.device,
        );
    }

    let (t3seq, statistics) = machine.finish();

    (
        type3::Lcs {
            loads: t3loads,
            compute: t3seq,
            pre_store: t3pre_store,
            store: t3stores,
        },
        statistics,
    )
}

fn realize_subgraph<'s, 'a, 'r, P, Rt: RuntimeType>(
    machine: &mut Ma<'s, 'r, P, Rt>,
    rsg: phase1::ResidentalSubgraph<'s, P>,
    hd_info: &HardwareInfo,
    device: type2::DeviceCollection,
) -> type3::Subgraph<'s>
where
    P: UsizeId + 'static,
{
    let sg = rsg.sg;

    let io_devices = sg
        .prologue
        .load
        .iter()
        .map(|(_, src)| src.device())
        .chain(sg.prologue.store.iter().map(|(dst, _)| dst.device()))
        .fold(
            type3::DeviceSpecific::new(hd_info.n_gpus(), |_| false),
            |mut acc, d| {
                *acc.get_mut(d) = true;
                acc
            },
        );

    let sg_allocation = rsg.allocation;
    let sg_allocation_reg = machine.reg_for_atom(&ResidentalT::new(
        sg_allocation.object.whole(sg.device),
        sg_allocation.pointer.unwrap(),
    ));

    // Allocate intermediates
    let scheme = sg_allocation.scheme;
    let mut sg = sg;
    sg.prologue
        .allocator
        .set_offset(scheme.intermediates_offset());
    let mut itms_machine =
        machine::IntermediatesMachine::new(sg_allocation_reg, machine.reg_booking_mut());
    let itms_registers = sg
        .itms
        .iter()
        .map(|ro| {
            AsRealizer::new(&sg.prologue.allocator).allocate(
                ro.deref(),
                ro.pointer(),
                &mut itms_machine,
            )
        })
        .collect::<Vec<_>>();
    let itms_allocations = itms_machine.finish();
    let sg = sg;

    // Obtain input/output registers
    let input_registers = sg
        .uses
        .iter()
        .map(|irv| {
            irv.iter()
                .map(|ira| {
                    let reg = machine.reg_for_atom(&ira.deref().clone().assume_pointed());
                    (ira.object().clone(), reg)
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .flatten()
        .collect::<ObjectTable<RegisterId>>();
    let output_registers = sg
        .output
        .iter()
        .map(|ora| {
            let reg = machine.reg_for_atom(&ora.deref().clone().assume_pointed());
            (ora.object().clone(), reg)
        })
        .collect::<ObjectTable<RegisterId>>();

    let (prologue1, prologue_statistics) = realize_lcs::<_, Rt>(
        sg.prologue,
        0,
        scheme.intermediates_offset(),
        scheme.prologue_buffer_offset(),
        hd_info,
        sg.vid,
        device.clone(),
        &input_registers,
        &output_registers,
        machine.reg_booking_mut(),
        sg_allocation_reg,
    );

    let (body1, body1_statistics) = realize_lcs::<_, Rt>(
        sg.body.clone(),
        0,
        scheme.intermediates_offset(),
        scheme.body_buffer1_offset(),
        hd_info,
        sg.vid,
        device.clone(),
        &input_registers,
        &output_registers,
        machine.reg_booking_mut(),
        sg_allocation_reg,
    );
    let (body2, _) = realize_lcs::<_, Rt>(
        sg.body,
        sg.chunker.chunk_len().try_into().unwrap(),
        scheme.intermediates_offset(),
        scheme.body_buffer2_offset(),
        hd_info,
        sg.vid,
        device.clone(),
        &input_registers,
        &output_registers,
        machine.reg_booking_mut(),
        sg_allocation_reg,
    );

    machine.statistics_mut().merge(&prologue_statistics);
    machine
        .statistics_mut()
        .merge(&body1_statistics.multiple(sg.chunker.chunks() as usize));

    let itms_deallocation = itms_registers
        .iter()
        .copied()
        .map(|id| type3::RegisterManipulation::StackFree { id })
        .collect();

    // Intermediates are local to this subgraph invocation. Keeping their
    // registers marked as live lets a later subgraph accidentally reuse one
    // when it has the same logical ObjectId and allocator-local PointerId.
    // The underlying scratch buffer is different (and may already have been
    // released), so such a register denotes stale chunk data.
    for reg in itms_registers {
        machine.reg_booking_mut().free_reg(reg);
    }

    type3::Subgraph {
        vid: sg.vid,
        itms_allocations,
        chunker: sg.chunker,
        prologue1,
        bodies: vec![body1, body2],
        device,
        buffer: sg_allocation_reg,
        itms_deallocation,
        io_devices,
    }
}

fn realize_type2op<'s, 'a, 'r, Rt: RuntimeType, P>(
    ma: &mut Ma<'s, 'r, P, Rt>,
    t2op: Type2Op<'s, Object, Option<Pointer<P>>>,
    execution_device: impl Fn(VertexId) -> type2::DeviceCollection,
) where
    P: UsizeId + 'static,
{
    // Get registers for inputs
    let node = t2op.node.relabeled(|vi| {
        let rv = vi.map(|ira| ira.v().assume_pointed());
        ma.reg_for(&rv)
    });

    // Get registers for outputs.
    // - If the output is inplace, or if the output is immortal,
    //   its space need no allocation, so the currespounding output register
    //   is not created yet.
    // - If the output is not inplace, the currespounding output register must have been created by
    //   previous memory allocation instruction.
    let output_regs = t2op
        .output
        .into_iter1()
        .map(|orv| match orv {
            Tree::Single(ora) => {
                if let Some(inplace_of) = ora.inplace_of() {
                    let inplace_reg = ma.reg_for_atom(&inplace_of.assume_pointed());
                    (
                        ma.new_atom_reg(ora.deref().clone().assume_pointed()),
                        Some(inplace_reg),
                    )
                } else {
                    (ma.reg_for_atom(&ora.v().assume_pointed()), None)
                }
            }
            Tree::Tuple(..) => (ma.reg_for(&orv.map(|ora| ora.v().assume_pointed())), None),
        })
        .collect::<Vec<_>>();

    let temp_regs = t2op
        .temporaries
        .map(|t| {
            let temp_primary_reg = ma.reg_for_atom(&t.buf.clone().assume_pointed());
            t.slices
                .into_iter()
                .map(|slice| {
                    let reg = ma.new_atom_reg(
                        t.buf
                            .clone()
                            .map_v(|a| a.map_node(|n| n.map_gpu_buffer(|_, _| slice)))
                            .assume_pointed(),
                    );

                    ma.emit_pre(type3::RegisterManipulation::SliceBuffer {
                        id: reg,
                        operand: temp_primary_reg,
                        offset: slice.begin() as usize,
                        len: slice.len() as usize,
                    });

                    reg
                })
                .collect()
        })
        .unwrap_or_default();

    ma.emit(InstructionNode::Type2 {
        ids: output_regs.clone(),
        temp: temp_regs,
        vertex: node,
        debug: type3::Debug::Type2(t2op.vid, t2op.src),
        device: execution_device(t2op.vid),
    });

    for r in output_regs.iter().filter_map(|(_, x)| *x) {
        ma.reg_booking_mut().free_reg(r);
    }
}

pub fn realize<'s, 'a, Rt: RuntimeType>(
    ops: phase1::OperationSeq<'s, super::PointerId>,
    hd_info: &HardwareInfo,
    lbss: LogBlockSizes,
    realizers: DeviceSpecific<Box<dyn Realizer<super::PointerId> + 'a>>,
    execution_device: impl Fn(VertexId) -> type2::DeviceCollection,
    dependencies: &dependency_analysis::Dependencies,
) -> Result<(type3::Chunk<'s, Rt>, machine::Statistics), Error<'s>> {
    let mut reg_booking = RegBooking::empty();
    let mut machine = Machine::new(hd_info.n_gpus(), &mut reg_booking);

    for (op_idx, op) in ops.into_iter() {
        use Operation::*;

        let mut ma = machine.at(op_idx);

        match op {
            Type2(Type2Op {
                node:
                    type2::template::VertexNode::Sliceable(type2::template::SliceableNode::NewPoly(
                        ..,
                        init,
                        typ,
                    )),
                output,
                ..
            }) => {
                let output = output.unwrap_leaf();
                let (_, s) = output.object().node().unwrap_poly();
                let output_reg = ma.reg_for_atom(&output.deref().clone().assume_pointed());

                ma.emit(InstructionNode::FillPoly {
                    id: output_reg,
                    deg: s.len() as usize,
                    init,
                    pty: typ,
                })
            }
            Type2(Type2Op {
                node: type2::template::VertexNode::Extend(src, to_deg),
                output,
                ..
            }) => {
                let (_, src_os) = src.unwrap_leaf().object().node().unwrap_poly();
                let output = output.unwrap_leaf();
                let output_reg1 =
                    ma.reg_for_atom(&output.deref().clone().assume_pointed().map_v(|a| {
                        a.map_node(|n| {
                            n.map_p(|deg, slice| {
                                assert!(slice.len() == to_deg);
                                assert!(deg as u64 == to_deg);
                                Slice::new(0, src_os.len())
                            })
                        })
                    }));
                let output_reg2 =
                    ma.reg_for_atom(&output.deref().clone().assume_pointed().map_v(|a| {
                        a.map_node(|n| {
                            n.map_p(|deg, _slice| {
                                Slice::new(src_os.len(), deg as u64 - src_os.len())
                            })
                        })
                    }));
                let src_reg = ma.reg_for_atom(&src.unwrap_leaf().deref().clone().assume_pointed());

                ma.emit(InstructionNode::Transfer {
                    id: output_reg1,
                    from: src_reg,
                });
                ma.emit(InstructionNode::FillPoly {
                    id: output_reg2,
                    deg: (to_deg - src_os.len()) as usize,
                    init: PolyInit::Zeros,
                    pty: zkpoly_common::typ::PolyType::Coef,
                });
            }
            Type2(t2op) => {
                realize_type2op(&mut ma, t2op, &execution_device);
            }
            Clone(dst, src) => {
                let ra_to = dst.assume_pointed().map_v(|o| o.whole(src.device()));
                ma.transfer(ra_to, src.assume_pointed());
            }
            Transfer(dst, d, src) => {
                let ra_to = dst.assume_pointed().map_v(|o| o.whole(d));
                ma.transfer(ra_to, src.assume_pointed());
            }
            Ad(template::Ad::Allocate(object, device, pointer)) => {
                realizers.get(device).allocate(&object, &pointer, &mut ma);
            }
            Ad(template::Ad::Deallocate(object, device, pointer)) => {
                realizers.get(device).deallocate(&object, &pointer, &mut ma);
            }
            Sub(rsg) => {
                let vid = rsg.sg.vid;

                let sg = realize_subgraph::<_, Rt>(&mut ma, rsg, hd_info, execution_device(vid));

                ma.emit(InstructionNode::Sub(sg));
            }
        }

        machine = ma.finish(dependencies.of(op_idx));
    }

    let register_value = machine.reg_booking().export_reg_values();
    let (instructions, statistics) = machine.finish();

    Ok((
        type3::Chunk {
            instructions,
            reg_id_allocator: reg_booking.reg_id_allocator(),
            register_value,
            lbss,
            _phantom: PhantomData,
        },
        statistics,
    ))
}
