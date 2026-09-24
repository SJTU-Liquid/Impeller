use super::emit_func::emit_func;
use super::kernel_gen::CallSite;
use super::prelude::*;
use super::threads::{Stream, StreamSpecific};
use type3::{RegisterId, RegisterManipulation};

use zkpoly_common::heap::Remap;
use zkpoly_runtime::instructions::{self as rt_inst, Lcs, LoadSlice, StoreSlice};

struct InstructionLower<'c, 's, Rt: RuntimeType> {
    t3chunk: &'c type3::Chunk<'s, Rt>,
    generated_functions: &'c GeneratedFunctions,
    stream2variable_id: &'c StreamSpecific<VariableId>,
    remap: Remap<RegisterId, VariableId>,
}

trait Emitter {
    fn emit(&mut self, inode: InstructionNode, track: Track);
}

struct EmitFnMut<F>(F);

impl<F> Emitter for EmitFnMut<F>
where
    F: FnMut(Instruction),
{
    fn emit(&mut self, inode: InstructionNode, track: Track) {
        (self.0)(wrap_instruction_node(inode, track))
    }
}

impl<'c, 's, Rt: RuntimeType> InstructionLower<'c, 's, Rt> {
    pub fn reg_id2var_id(&self, reg: RegisterId) -> VariableId {
        self.remap.on(reg)
    }

    fn lower_register_manip(&self, manip: &RegisterManipulation, emit: &mut impl Emitter) {
        use type3::template::RegisterManipulation::*;

        match manip {
            StackFree { id } => emit.emit(
                InstructionNode::RemoveRegister {
                    id: self.reg_id2var_id(*id),
                },
                Track::Cpu,
            ),
            Tuple { id, oprands } => {
                let dst = self.reg_id2var_id(*id);
                let vars = oprands.iter().map(|&id| self.reg_id2var_id(id)).collect();
                emit.emit(InstructionNode::AssembleTuple { vars, dst }, Track::Cpu);
            }
            SetPolyMeta {
                id,
                from,
                offset,
                len,
            } => emit.emit(
                InstructionNode::SetSliceMeta {
                    src: self.reg_id2var_id(*from),
                    dst: self.reg_id2var_id(*id),
                    offset: *offset as usize,
                    len: *len as usize,
                },
                Track::Cpu,
            ),
            SliceBuffer {
                id,
                operand,
                offset,
                len,
            } => emit.emit(
                InstructionNode::SliceBuffer {
                    src: self.reg_id2var_id(*operand),
                    dst: self.reg_id2var_id(*id),
                    offset: *offset,
                    len: *len,
                },
                Track::Cpu,
            ),
            TupleGet { id, target, index } => emit.emit(
                InstructionNode::TupleGet {
                    dst: self.reg_id2var_id(*id),
                    src: self.reg_id2var_id(*target),
                    index: *index,
                },
                Track::Cpu,
            ),
        }
    }

    fn lower_lcs(
        &self,
        lcs: &type3::Lcs<'s>,
        call_site: impl Fn(type3::LcsPc) -> CallSite,
        track: Track,
        base: RegisterId,
    ) -> Lcs {
        let loads = lcs
            .loads
            .iter()
            .map(
                |type3::Load {
                     id,
                     offset,
                     input: type3::SubgraphInput { src, begin },
                 }| {
                    LoadSlice {
                        src_reg: self.reg_id2var_id(*src),
                        alloc_dst: rt_inst::AllocateOffsetInfo {
                            base_buffer: self.reg_id2var_id(base),
                            offset: *offset,
                            typ: self.t3chunk.register_rt_typ(*id),
                            id: self.reg_id2var_id(*id),
                        },
                        begin_range: begin.clone(),
                    }
                },
            )
            .collect();

        struct VecEmitter<'v>(&'v mut Vec<(Instruction, type3::Begin)>, type3::Begin);

        impl<'v> Emitter for VecEmitter<'v> {
            fn emit(&mut self, inode: InstructionNode, track: Track) {
                self.0
                    .push((wrap_instruction_node(inode, track), self.1.clone()))
            }
        }

        let mut compute = Vec::new();
        for (pc, sinst) in lcs.compute.iter() {
            self.lower_instruction(
                call_site(pc),
                &sinst.inst,
                track.coprocess_to_gpu(),
                &mut VecEmitter(&mut compute, sinst.begin.clone()),
            );
        }

        for manip in lcs.pre_store.iter() {
            self.lower_register_manip(manip, &mut VecEmitter(&mut compute, None));
        }

        let stores = lcs
            .store
            .iter()
            .map(|type3::Store { dst, src, begin }| StoreSlice {
                src_reg: self.reg_id2var_id(*src),
                dst_reg: self.reg_id2var_id(*dst),
                begin_range: *begin,
            })
            .collect();

        Lcs {
            loads,
            compute: compute.into_iter().unzip(),
            stores,
        }
    }

    fn lower_subgraph(
        &self,
        t3idx: type3::InstructionIndex,
        track: Track,
        sg: &type3::Subgraph<'s>,
        emit: &mut impl Emitter,
    ) {
        let itms_allocations = sg.itms_allocations.iter().map(|oa| {
            wrap_instruction_node(
                InstructionNode::AllocateOffset {
                    info: rt_inst::AllocateOffsetInfo {
                        base_buffer: self.reg_id2var_id(oa.base),
                        offset: oa.offset,
                        typ: self.t3chunk.register_rt_typ(oa.id),
                        id: self.reg_id2var_id(oa.id),
                    },
                },
                track,
            )
        });

        let mut prologue = self.lower_lcs(
            &sg.prologue1,
            |pc| CallSite::Prologue(t3idx, pc),
            track,
            sg.buffer,
        );

        prologue.compute = itms_allocations
            .into_iter()
            .map(|a| (a, None))
            .chain(
                prologue
                    .compute
                    .0
                    .into_iter()
                    .zip(prologue.compute.1.into_iter()),
            )
            .unzip();

        let bodies = sg
            .bodies
            .iter()
            .enumerate()
            .map(|(i, body)| {
                self.lower_lcs(body, |pc| CallSite::Body(t3idx, i, pc), track, sg.buffer)
            })
            .collect();

        emit.emit(
            InstructionNode::SubGraphFor {
                prologue,
                body: bodies,
                total_iters: sg.chunker.chunks() as usize,
                from_gpu_stream: *self.stream2variable_id.get(Stream::FromGpu),
                to_gpu_stream: *self.stream2variable_id.get(Stream::ToGpu),
                device_types: sg
                    .io_devices
                    .iter()
                    .filter(|(_, s)| **s)
                    .map(|(d, _)| d.into())
                    .collect(),
                compute_stream_id: *self.stream2variable_id.get(Stream::of_coprocess(track)),
            },
            track,
        );

        for manip in &sg.itms_deallocation {
            self.lower_register_manip(manip, emit);
        }
    }

    fn lower_instruction(
        &self,
        call_site: CallSite,
        inst: &type3::Instruction<'s>,
        track: Track,
        emitter: &mut impl Emitter,
    ) {
        for rm in inst.pre.iter() {
            self.lower_register_manip(rm, emitter);
        }

        use type3::template::InstructionNode::*;
        let mut emit = |inode| emitter.emit(inode, track);

        match &inst.op {
            Type2 {
                ids, temp, vertex, ..
            } => {
                ids.iter()
                    .filter_map(|(id, inplace)| Some((*id, inplace.as_ref()?)))
                    .for_each(|(id, &inplace)| {
                        emit(InstructionNode::MoveRegister {
                            src: self.reg_id2var_id(inplace),
                            dst: self.reg_id2var_id(id),
                        })
                    });
                let mut vertex = vertex.clone();
                vertex.uses_mut().for_each(|u| {
                    if let Some((id, _)) = ids
                        .iter()
                        .find(|(_, inplace)| inplace.is_some_and(|i| i == *u))
                    {
                        *u = *id;
                    }
                });
                let ids: Vec<_> = ids.iter().map(|x| x.0).collect();
                match &vertex {
                    VertexNode::Constant(constant_id) => {
                        assert!(ids.len() == 1);
                        emit(InstructionNode::LoadConstant {
                            src: *constant_id,
                            dst: self.reg_id2var_id(ids[0]),
                        })
                    }
                    VertexNode::Blind(id, start_pos, end_pos) => {
                        assert!(ids.len() == 1);
                        assert_eq!(ids[0], *id);
                        let dst = self.reg_id2var_id(*id);
                        emit(InstructionNode::Blind {
                            dst,
                            start: *start_pos as usize,
                            end: *end_pos as usize,
                        });
                    }
                    VertexNode::Entry(idx) => {
                        assert!(ids.len() == 1);
                        emit(InstructionNode::LoadInput {
                            src: *idx,
                            dst: self.reg_id2var_id(ids[0]),
                        })
                    }
                    VertexNode::Return(id) => {
                        emit(InstructionNode::Return(self.reg_id2var_id(*id)))
                    }
                    VertexNode::IndexPoly(operand, idx) => {
                        let operand = self.reg_id2var_id(*operand);
                        let target = self.reg_id2var_id(ids[0]);
                        let stream =
                            Stream::of_track(track).map(|t| self.stream2variable_id.get(t).clone());
                        emit(InstructionNode::GetScalarFromArray {
                            src: operand,
                            dst: target,
                            idx: *idx as usize,
                            stream,
                        })
                    }
                    VertexNode::AssertEq(src, truth, msg) => {
                        assert!(ids.len() == 0);
                        let src = self.reg_id2var_id(*src);
                        let truth = self.reg_id2var_id(*truth);
                        emit(InstructionNode::AssertEq {
                            value: src,
                            expected: truth,
                            msg: msg.clone(),
                        });
                    }
                    VertexNode::Print(id, s) => {
                        assert!(ids.len() == 0);
                        emit(InstructionNode::Print(self.reg_id2var_id(*id), s.clone()));
                    }
                    _ => {
                        emit_func(
                            call_site,
                            &ids,
                            temp,
                            track,
                            &vertex,
                            &|i| self.reg_id2var_id(i),
                            &self.stream2variable_id,
                            self.generated_functions.at(call_site),
                            &self.t3chunk,
                            &mut emit,
                        );
                    }
                }
            }
            Malloc { id, addr, device } => {
                let var_id = self.reg_id2var_id(*id);

                emit(InstructionNode::Allocate {
                    device: DeviceType::from(device.clone()),
                    typ: self.t3chunk.register_rt_typ(*id),
                    id: var_id,
                    alloc_method: addr.clone(),
                });
            }
            Free { id, variant, .. } => emit(InstructionNode::Deallocate {
                id: self.reg_id2var_id(*id),
                alloc_method: *variant,
            }),
            Transfer { id, from } => emit(InstructionNode::Transfer {
                src_device: DeviceType::from(self.t3chunk.register_device(*from)),
                dst_device: DeviceType::from(self.t3chunk.register_device(*id)),
                stream: Stream::of_track(track).map(|s| *self.stream2variable_id.get(s)),
                src_id: self.reg_id2var_id(*from),
                dst_id: self.reg_id2var_id(*id),
            }),
            FillPoly { id, .. } => {
                let (f_id, _) = self.generated_functions.at(call_site);
                let device = self.t3chunk.register_device(*id);
                let dst = self.reg_id2var_id(*id);
                if device == type3::Device::Cpu {
                    emit(InstructionNode::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![dst],
                        arg: vec![],
                    });
                } else {
                    let stream =
                        Stream::of_track(track).map(|t| self.stream2variable_id.get(t).clone());
                    emit(InstructionNode::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![dst],
                        arg: vec![stream.unwrap()],
                    });
                }
            }
            MallocOffseted { id, base, offset } => {
                let base_buffer = self.reg_id2var_id(*base);
                let typ = self.t3chunk.register_rt_typ(*id);
                let id = self.reg_id2var_id(*id);

                emit(InstructionNode::AllocateOffset {
                    info: rt_inst::AllocateOffsetInfo {
                        base_buffer,
                        offset: *offset,
                        typ,
                        id,
                    },
                })
            }
            Sub(sg) => match call_site {
                CallSite::Main(t3idx) => self.lower_subgraph(t3idx, track, sg, emitter),
                _ => panic!("subgraph not expected in subgraph instructions"),
            },
        };

        for rm in inst.post.iter() {
            self.lower_register_manip(rm, emitter);
        }
    }
}

fn wrap_instruction_node(inode: InstructionNode, track: Track) -> Instruction {
    let rt_track = match &inode {
        InstructionNode::MoveRegister { .. } | InstructionNode::CopyRegister { .. } => {
            rt_inst::Track::RegisterManipulation
        }
        _ => track.into(),
    };

    Instruction::new(inode, rt_track, Stream::of_track(track).map(|x| x.into()))
}

pub fn lower_instruction<'s, Rt: RuntimeType>(
    t3idx: type3::InstructionIndex,
    inst: &type3::Instruction<'s>,
    track: Track,
    reg_id2var_id: Remap<RegisterId, VariableId>,
    stream2variable_id: &StreamSpecific<VariableId>,
    t3chunk: &type3::Chunk<'s, Rt>,
    generated_functions: &GeneratedFunctions,
    emit_inst: &mut impl FnMut(Instruction),
) {
    let lowerer = InstructionLower {
        t3chunk,
        generated_functions,
        stream2variable_id,
        remap: reg_id2var_id,
    };

    lowerer.lower_instruction(
        CallSite::Main(t3idx),
        inst,
        track,
        &mut EmitFnMut(emit_inst),
    );
}
