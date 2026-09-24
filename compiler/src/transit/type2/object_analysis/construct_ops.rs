use super::*;
use cg_def_use::DefUse;
use subgraph_elaboration::{Paragraph, ParagraphBuilder};
use template::{Atom, Subgraph, Type2Op};
use type2::memory_planning::phase0;
use value::Object;

mod last_use_analyze {
    use super::*;

    #[derive(Debug)]
    pub struct ObjectLastUse {
        lu: BTreeMap<ObjectId, VertexId>,
    }

    impl ObjectLastUse {
        pub fn analyze<'a, 's, Rt: RuntimeType>(
            g: &type2::CgSubgraph<'a, 's, Rt>,
            def_use: &DefUse,
            schedule: &type2::graph_scheduling::WholeSchedule,
        ) -> Self {
            let mut olu = ObjectLastUse {
                lu: BTreeMap::new(),
            };

            for vid in schedule
                .cg_schedule
                .seq()
                .iter()
                .copied()
                .filter(|vid| !g.vertex(*vid).node().is_virtual())
            {
                for ia in def_use.input_of(vid).flat_map(|(_, iv)| iv.iter()) {
                    olu.lu.insert(*ia.object(), vid);
                }
            }

            olu
        }

        pub fn of(&self, object: ObjectId) -> Option<VertexId> {
            self.lu.get(&object).copied()
        }
    }
}
pub use last_use_analyze::ObjectLastUse;

fn make_object(id: ObjectId, objects: &ObjectDefs) -> Object {
    Object::new(id, objects[id].with_normalized_p())
}

fn make_cloned_slice_if_needed<'s, P>(
    object: ObjectId,
    device: type3::Device,
    force_on_device: bool,
    def_use: &DefUse,
    sliced_is_cloned: &mut BTreeSet<ObjectId>,
    objects: &ObjectDefs,
    ops: &mut Seq<Operation<'s, Object, P, phase0::Subgraph<'s, P>>>,
) {
    if let Some((src_object_id, slice)) = def_use.object_is_cloned_slice(object) {
        if !sliced_is_cloned.contains(&object) {
            sliced_is_cloned.insert(object);

            let cloned_obj = make_object(object, &objects);
            let src_obj = make_object(src_object_id, &objects);

            if force_on_device {
                ops.emit(Operation::Transfer(
                    ResidentalT::with_no_pointer(cloned_obj),
                    device,
                    {
                        let node = src_obj.node().map_p(|_, _| slice);
                        ResidentalT::with_no_pointer(Atom::new(src_obj, device, node))
                    },
                ))
            } else {
                ops.emit(Operation::Clone(
                    ResidentalT::with_no_pointer(cloned_obj),
                    {
                        let node = src_obj.node().map_p(|_, _| slice);
                        ResidentalT::with_no_pointer(Atom::new(src_obj, device, node))
                    },
                ))
            }
        }
    }
}

fn relabel_node<'s, P>(
    vid: VertexId,
    node: type2::no_subgraph::VertexNode,
    def_use: &DefUse,
    objects: &mut ObjectDefs,
    ops: &mut Seq<Operation<'s, Object, P, phase0::Subgraph<'s, P>>>,
    vid2mutated_obj: &mut BTreeMap<VertexId, template::InputAtom<Object>>,
    sliced_is_cloned: &mut BTreeSet<ObjectId>,
    object_last_use: &ObjectLastUse,
) -> template::VertexNode<Object, Option<P>>
where
    P: 'static + Ord + std::fmt::Debug + Clone,
{
    let r = node.relabeled(|u| {
        def_use.input[&vid]
            .iter()
            .find(|(vid, _)| u == *vid)
            .map(|(input_vid, vi)| {
                // Clone slices, if needed
                vi.iter().for_each(|v| {
                    make_cloned_slice_if_needed(
                        *v.object(),
                        v.device(),
                        true,
                        def_use,
                        sliced_is_cloned,
                        objects,
                        ops,
                    );
                });

                // If input is mutated
                // - If the inputed object dies after this vertex on all devices,
                //   it's space will be simply reused.
                // - Otherwise, we make a clone for it
                let vi = vi.map_ref(|ai| {
                    let object = make_object(*ai.object(), objects);

                    if ai.mutability() == Mutability::Mut {
                        match (
                            def_use.is_whole_object_immortal(*ai.object()),
                            object_last_use.of(*ai.object()),
                        ) {
                            (false, Some(x)) if x == vid => {
                                let ai = ai.clone().map_v(|a| a.map_object(|_| object));
                                vid2mutated_obj.insert(*input_vid, ai.clone());
                                ai
                            }
                            (true, Some(_)) | (false, Some(_)) => {
                                let tnode = objects[*ai.object()].clone();
                                let temp_object_id = objects.push(tnode.clone());
                                let cloned_object = object.with_object_id(temp_object_id);

                                ops.emit(Operation::Transfer(
                                    ResidentalT::with_no_pointer(cloned_object.clone()),
                                    ai.device(),
                                    ResidentalT::with_no_pointer(object.whole(ai.device())),
                                ));

                                let ai = ai.clone().map_v(|a| a.map_object(|_| cloned_object));
                                vid2mutated_obj.insert(*input_vid, ai.clone());
                                ai
                            }
                            (_, None) => panic!(
                                "{:?} cannot be unused on device {:?} because we are using it now",
                                ai.object(),
                                ai.device()
                            ),
                        }
                    } else {
                        ai.clone()
                            .map_v(|a| a.map_object(|o| make_object(*o, &objects)))
                    }
                });

                vi.map(|ai| ai.map_v(ResidentalT::with_no_pointer))
            })
            .unwrap()
    });

    r
}

fn build_subgraph_ops<'s, Rt: RuntimeType>(
    cg: &sliceable_subgraph::Cg<'s, Rt>,
    analysis: &SubgraphAnalysis,
    seq: &[sliceable_subgraph::VertexId],
    objects: &mut ObjectDefs,
    hd_info: &HardwareInfo,
    gpu: type2::Device,
    libs: &mut Libs,
    ag_scheduler: ArithGraphScheduler,
) -> (
    Paragraph<'s>,
    Paragraph<'s>,
    BTreeSet<ObjectId>,
    BTreeSet<ObjectId>,
    BTreeSet<ObjectId>,
) {
    let mut prologue_builder = ParagraphBuilder::new();
    let prologue: Paragraph = prologue_builder.construct(
        cg,
        seq,
        &analysis.prologue_def_use,
        &analysis.prologue_itms,
        objects,
        hd_info,
        gpu,
        ag_scheduler,
        libs,
    );

    let mut body_builder = ParagraphBuilder::new();
    let body: Paragraph = body_builder.construct(
        cg,
        seq,
        &analysis.body_def_use,
        &analysis.body_itms,
        objects,
        hd_info,
        gpu,
        ag_scheduler,
        libs,
    );

    let [prologue_input_oids, body_input_oids] =
        [&prologue, &body].map(|p| p.inputs.iter().map(|o| o.id()).collect::<BTreeSet<_>>());
    if prologue_input_oids != body_input_oids {
        panic!(
            "inconsistent inputs for prologue and body: {:?} and {:?}",
            &prologue_input_oids, &body_input_oids
        );
    }

    let [prologue_output_oids, body_output_oids] =
        [&prologue, &body].map(|p| p.outputs.iter().map(|o| o.id()).collect::<BTreeSet<_>>());
    if prologue_output_oids != body_output_oids {
        panic!(
            "inconsistent outputs for prologue and body: {:?} and {:?}",
            &prologue_output_oids, &body_output_oids
        );
    }

    let [prologue_as_is_input_oids, body_as_is_input_oids] = [&prologue, &body].map(|p| {
        p.as_is_inputs
            .iter()
            .map(|o| o.id())
            .collect::<BTreeSet<_>>()
    });
    let on_device_input_oids = prologue_as_is_input_oids
        .union(&body_as_is_input_oids)
        .cloned()
        .collect();

    (
        prologue,
        body,
        prologue_input_oids,
        prologue_output_oids,
        on_device_input_oids,
    )
}

use type2::VertexId;
use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;
impl<'s, P> phase0::OperationSeq<'s, P>
where
    P: 'static + Ord + std::fmt::Debug + Clone,
{
    pub fn construct<'a, Rt: RuntimeType>(
        g: &type2::CgSubgraph<'a, 's, Rt>,
        schedule: &type2::graph_scheduling::WholeSchedule,
        def_use: &super::DefUse,
        execution_device: impl Fn(VertexId) -> &'a type2::DeviceCollection,
        hd_info: &HardwareInfo,
        objects: &mut ObjectDefs,
        libs: &mut Libs,
        object_last_use: &ObjectLastUse,
        ag_scheduler: ArithGraphScheduler,
    ) -> Self {
        let mut ops_x = Seq::empty();
        let ops = &mut ops_x;

        // This closure emits an operation that makes a slice clone if it no clone been made
        let sliced_is_cloned = BTreeSet::new();

        let mut ctx = (libs, objects, sliced_is_cloned, ops);

        for &vid in schedule
            .cg_schedule
            .seq()
            .iter()
            .filter(|&vid| !g.vertex(*vid).is_virtual())
        {
            let v = g.vertex(vid);

            let mut vid2mutated_obj = BTreeMap::new();

            let op: phase0::Operation<'s, P> = v.node().clone().branch_s(
                &mut ctx,
                |node, (libs, objects, sliced_is_cloned, ops)| {
                    let node = relabel_node(
                        vid,
                        node,
                        def_use,
                        objects,
                        ops,
                        &mut vid2mutated_obj,
                        sliced_is_cloned,
                        object_last_use,
                    );
                    use type2::template::VertexNode;

                    let output = match &node {
                        VertexNode::AssertEq(..) | VertexNode::Print(..) => {
                            value::Tree::Tuple(Vec::new())
                        }
                        _ => def_use.value[&vid].map_ref(|v| {
                            value::OutputT::new(
                                ResidentalT::with_no_pointer(
                                    v.deref().map_object(|&oid| make_object(oid, &objects)),
                                ),
                                v.inplace_of().map(|inplaced_vid| {
                                    let ai = vid2mutated_obj
                                        .get(&inplaced_vid)
                                        .unwrap_or_else(|| {
                                            panic!(
                                                "inplace object not found for input {:?} at {:?}",
                                                inplaced_vid, vid
                                            )
                                        })
                                        .clone();
                                    ResidentalT::with_no_pointer(ai.deref().clone())
                                }),
                            )
                        }),
                    };

                    let temp = g
                        .vertex(vid)
                        .node()
                        .temporary_space_needed(
                            execution_device(vid),
                            libs,
                            |i| g.vertex(*i).typ(),
                            ag_scheduler,
                        )
                        .map(|t| {
                            let (size, slices) =
                                super::merge_temporary_buffer(t.sizes.into_iter(), 32);
                            let temp_buffer_obj_id = objects
                                .push(TopNode::GpuBuffer(size as usize, Slice::new(0, size)));

                            Temporaries {
                                buf: ResidentalT::with_no_pointer(value::Atom::new(
                                    Object::new(
                                        temp_buffer_obj_id,
                                        value::ValueNode::GpuBuffer(
                                            size as usize,
                                            Slice::new(0, size),
                                        ),
                                    ),
                                    t.device,
                                    value::ValueNode::GpuBuffer(size as usize, Slice::new(0, size)),
                                )),
                                slices,
                            }
                        });

                    Type2Op::new(vid, output, temp, node, v.src().clone()).into()
                },
                |subgraph, (libs, objects, sliced_is_cloned, ops)| {
                    let (prologue, body, input_oids, output_oids, on_device_input_oids) =
                        build_subgraph_ops(
                            &subgraph,
                            def_use.subgraph_analysis(vid),
                            &schedule.subgraphs_schedule.for_subgraph(vid).seq(),
                            objects,
                            hd_info,
                            execution_device(vid).ensure_single(),
                            libs,
                            ag_scheduler,
                        );

                    let memory_device = type3::Device::for_execution_on(prologue.device.clone());

                    for oid in input_oids.iter().cloned() {
                        make_cloned_slice_if_needed(
                            oid,
                            memory_device,
                            false,
                            def_use,
                            sliced_is_cloned,
                            objects,
                            ops,
                        );
                    }

                    let [uses, on_device_uses] = [&input_oids, &on_device_input_oids].map(|os| {
                        os.iter()
                            .map(|oid| {
                                let object = make_object(*oid, &objects);
                                let node = object.node().normalized();
                                value::Tree::Single(value::InputT::immutable(
                                    ResidentalT::with_no_pointer(value::Atom::new(
                                        object,
                                        memory_device,
                                        node,
                                    )),
                                ))
                            })
                            .collect::<Vec<_>>()
                    });

                    let outputs = output_oids
                        .iter()
                        .map(|oid| {
                            let object = make_object(*oid, &objects);
                            let node = object.node().normalized();
                            value::Tree::Single(value::OutputT::non_inplace(
                                ResidentalT::with_no_pointer(value::Atom::new(
                                    object,
                                    memory_device,
                                    node,
                                )),
                            ))
                        })
                        .collect::<Vec<_>>();

                    Operation::Sub(Subgraph::new(
                        uses,
                        on_device_uses,
                        memory_device,
                        prologue,
                        body,
                        value::Tree::Tuple(outputs),
                        vid,
                        def_use.subgraph_analysis(vid).chunker.clone(),
                    ))
                },
            );

            ctx.3.emit(op);
        }

        ops_x
    }
}
