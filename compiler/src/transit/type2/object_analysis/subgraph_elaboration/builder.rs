use super::super::template::{InputRv, ResidentalT};
use super::prelude::*;
use super::slice_analysis;
use sliceable_subgraph::VertexId;
use type2::memory_planning::phase0;
use type2::DeviceCollection;
use value::Object;
use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;

type Atom = slice_analysis::Atom<slice_analysis::RotatedNode>;

#[derive(Debug, Clone)]
struct ObjectAvailability {
    biggest: Slice,
    others: BTreeSet<Slice>,
}

impl ObjectAvailability {
    pub fn with_biggest(biggest: Slice) -> Self {
        ObjectAvailability {
            biggest,
            others: BTreeSet::new(),
        }
    }

    pub fn set_biggest(&mut self, slice: Slice) {
        self.others.insert(self.biggest);
        self.biggest = slice;
    }

    pub fn add(&mut self, slice: Slice) {
        self.others.insert(slice);
    }

    pub fn contains(&self, slice: Slice) -> bool {
        self.biggest == slice || self.others.contains(&slice)
    }
}

pub struct ParagraphBuilder {
    /// The exact slice that the object ID is available in
    available_slices: BTreeMap<ObjectId, ObjectAvailability>,
}

/// Either loop prologue or loop body
#[derive(Debug, Hash, Clone)]
pub struct Paragraph<'s> {
    /// The operations carried in this paragraph
    pub ops: phase0::SubgraphOperationSeq<'s>,

    /// Inputed objects (being slices of larger objects)
    pub inputs: Vec<Object>,

    /// Inputed objects that are the same between loops
    pub as_is_inputs: Vec<Object>,

    /// Inputed intermediates.
    ///
    /// For prologue, there are no inputed intermediates;
    /// For body, inputed intermediates should be exactly outputed by the prologue.
    pub itms_in: Vec<Object>,

    /// Output intermediates.
    ///
    /// The intermediates outputed by body have same [`ObjectId`] as those outputed by prologue,
    /// but are rotated right by a chunk, if the object ID curresponds to a polynomial.
    pub itms_out: Vec<Object>,

    /// Put out-intermediates to where next loop can read.
    pub itms_write_back: Vec<(/* dst */ Object, /* src */ Object)>,
    pub device: type2::Device,

    /// Output objects (being slices of larger objects)
    pub outputs: value::Tree<Object>,
}

mod pretty_print {
    use super::*;
    use std::io::Write;
    use zkpoly_common::indent_formatter::*;

    impl<'s, W> IndentDebug<W> for Paragraph<'s>
    where
        W: Write,
    {
        fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
            f.debug_struct()
                .entry_debug("inputs", &self.inputs)
                .entry_debug("as_is_inputs", &self.as_is_inputs)
                .entry_debug("itms_in", &self.itms_in)
                .entry_debug("itms_out", &self.itms_out)
                .entry_debug("itms_write_back", &self.itms_write_back)
                .entry_debug("device", &self.device)
                .entry_debug("outputs", &self.outputs)
                .entry("ops", &self.ops)
                .finish()
        }
    }
}

/// Produce an object for object analysis. Refer to `make_value` for detail.
fn make_object(saa: &Atom) -> Object {
    Object::new(saa.object().id(), saa.object().node().map_p(|_deg, s| *s))
}

/// Produce an atom for object analysis.
/// The produce atom has normalized node, the provided object ID and the desired device from slice analysis.
///
/// The object from slice analysis must has one single continuous range
fn make_value<P>(saa: &Atom) -> ResidentalAtom<Object, Option<P>> {
    ResidentalT::new(
        value::Atom::new(
            make_object(saa),
            saa.device(),
            saa.object()
                .node()
                .map_scalar_array_len(|_, s| s.len() as usize)
                .map_p(|_deg, s| Slice::new(0, s.len())),
        ),
        None,
    )
}

impl ParagraphBuilder {
    pub fn new() -> Self {
        Self {
            available_slices: BTreeMap::new(),
        }
    }

    fn prepare_input_for<'s, P>(
        &mut self,
        vid: VertexId,
        u: VertexId,
        vi: &slice_analysis::InputValue<slice_analysis::SlicesNode>,
        objects: &mut ObjectDefs,
        mutable_inputs: &mut BTreeMap<VertexId, ResidentalAtom<Object, Option<P>>>,
        ops: &mut Seq<phase0::SubgraphOperation<'s>>,
    ) -> InputRv<Object, Option<P>>
    where
        P: Clone,
    {
        // Make cloned slice if needed
        // - If the input can be sliced
        //   - If the needed slice is same as slice object we already have, use it
        //   - Otherwise, emit a clone operation to obtain the desired input slice
        // - Otherwise, trivially produce value (scalar, etc.)
        vi.map_ref(|v| {
            let v = v.clone().map_v(|a| a.into_rotated_atom());

            let (rv, cloned) = if let Some(input_slice) = v.object().range() {
                let availabele_slices =
                    self.available_slices
                        .get(&v.object().id())
                        .unwrap_or_else(|| {
                            panic!(
                                "cannot find slice for object {:?} as input to {:?}",
                                v.object(),
                                vid
                            )
                        });

                if availabele_slices.contains(input_slice) {
                    (make_value(&v), false)
                } else {
                    let availabele_slice = availabele_slices.biggest;
                    let cloned_value = make_value(&v);
                    let deg = v.object().node().unwrap_poly().0 as u64;

                    ops.emit(SubgraphOperation::Clone(
                        ResidentalT::with_no_pointer(cloned_value.object().clone()),
                        ResidentalT::with_no_pointer(
                            make_object(&v).map_node(|t| t.map_p(|_, _| availabele_slice)),
                        ),
                        Some(input_slice.relative_to(&availabele_slice, deg)),
                    ));

                    self.available_slices
                        .get_mut(&v.object().id())
                        .unwrap()
                        .add(input_slice);

                    (cloned_value, true)
                }
            } else {
                (make_value(&v), false)
            };

            match v.mutability() {
                Mutability::Mut => {
                    if cloned {
                        value::InputT::new(rv, Mutability::Mut)
                    } else {
                        let new_obj_id = objects.push(rv.object().node().erase_p());
                        let cloned_obj = Object::new(new_obj_id, rv.object().node().clone());

                        ops.emit(SubgraphOperation::Clone(
                            ResidentalT::new(cloned_obj.clone(), None),
                            ResidentalT::new(rv.object().clone(), None),
                            None,
                        ));

                        let cloned_rv = rv.clone().map_v(|a| a.map_object(|_| cloned_obj.clone()));

                        mutable_inputs.insert(u, cloned_rv.clone());

                        value::InputT::new(cloned_rv, Mutability::Mut)
                    }
                }
                _ => value::InputT::new(rv, Mutability::Const),
            }
        })
    }

    pub fn construct<'s, 'a, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        seq: &[VertexId],
        def_use: &slice_analysis::DefUse,
        itms: &impl slice_analysis::Intermediates,
        objects: &mut ObjectDefs,
        hd_info: &HardwareInfo,
        gpu: type2::Device,
        ag_scheduler: ArithGraphScheduler,
        libs: &mut Libs,
    ) -> Paragraph<'s> {
        let mut ops = Seq::empty();
        let mut itms_in = Vec::new();
        let mut itms_out = Vec::new();
        let mut outputs = None;
        let mut inputs = Vec::new();
        let mut as_is_inputs = Vec::new();

        let mut visited_input = BTreeSet::new();

        for &vid in seq.iter() {
            let v = cg.g.vertex(vid);
            let mut postprocess_vid = Some(vid);

            use sliceable_subgraph::template::VertexNode::*;
            use type2::template::SliceableNode::*;
            match v.node() {
                Inner(RotateIdx(..)) => {
                    // Their information are already embeded in [`Object`]'s
                }
                TupleGet(..) => {}
                Input(_) | InputAsIs(_) => {
                    let vid = match v.node() {
                        Input(..) => def_use.representing_input(vid, &mut visited_input),
                        InputAsIs(..) => Some(vid),
                        _ => unreachable!(),
                    };
                    postprocess_vid = vid;

                    if let Some(vid) = vid {
                        let oa = def_use.value(vid).unwrap_leaf();
                        let inner_object =
                            make_object(&oa.clone().map_v(|a| a.into_rotated_atom()));

                        if let Some(slice) = oa.object().range1() {
                            self.available_slices
                                .insert(inner_object.id(), ObjectAvailability::with_biggest(slice));
                        }
                        match v.node() {
                            Input(_) => inputs.push(inner_object),
                            InputAsIs(_) => as_is_inputs.push(inner_object),
                            _ => unreachable!(),
                        }
                    }
                }
                Return(..) => {
                    if outputs.is_some() {
                        panic!("mutiple outputs");
                    }

                    let return_values = def_use
                        .input(vid)
                        .iter()
                        .map(|(_, x)| x.unwrap_leaf().deref())
                        .map(|oa| make_object(&oa.clone().map_object(|o| o.into_rotated_object())));

                    outputs = Some(return_values.collect::<Vec<_>>());
                }
                _ => {
                    let mut mutable_inputs = BTreeMap::new();

                    // Rewrite inputs to [`InputRv`]'s for object analysis
                    let mut relabel_cache = BTreeMap::new();
                    let node: ChunkedNode<Object, Option<()>> = v.node().clone().relabeled(
                        |u| {
                            relabel_cache
                                .entry(u)
                                .or_insert_with(|| {
                                    def_use
                                        .input(vid)
                                        .iter()
                                        .find(|(vid, _)| u == *vid)
                                        .map(|(_, vi)| {
                                            self.prepare_input_for(
                                                vid,
                                                u,
                                                vi,
                                                objects,
                                                &mut mutable_inputs,
                                                &mut ops,
                                            )
                                        })
                                        .unwrap()
                                })
                                .deref()
                                .clone()
                        },
                        |_| panic!("Input vertex not expected here"),
                    );

                    // The output of the vertex should have slice the same as `the_slice`
                    let output = def_use.value(vid).map_ref(|ov| {
                        value::OutputT::new(
                            make_value(&ov.clone().map_v(|a| a.into_rotated_atom())),
                            ov.inplace_of().map(|src_vid| {
                                def_use
                                    .input(vid)
                                    .iter()
                                    .find(|(u, _)| *u == src_vid)
                                    .map(|(u, _)| mutable_inputs[u].clone())
                                    .unwrap()
                            }),
                        )
                    });

                    output.iter().for_each(|ora| {
                        if let Some((_, s)) = ora.object().node().poly() {
                            self.available_slices
                                .insert(ora.object().id(), ObjectAvailability::with_biggest(*s));
                        }
                    });

                    let temp = cg
                        .g
                        .vertex(vid)
                        .node()
                        .temporary_space_needed(
                            &DeviceCollection::default(hd_info.n_gpus()).with(gpu),
                            libs,
                            |i| cg.g.vertex(*i).typ().inner(),
                            ag_scheduler,
                        )
                        .map(|t| {
                            let (size, slices) =
                                super::super::merge_temporary_buffer(t.sizes.into_iter(), 32);
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

                    let chunked_op = ChunkedOp::new(vid, output, temp, node, v.src().clone());
                    ops.emit(chunked_op);
                }
            }

            if let Some(vid) = postprocess_vid {
                // Use in-intermediates to concat into needed object for latter uses
                for concat in itms.concat_ins(vid) {
                    let resulting = concat.resulting();
                    ops.emit(SubgraphOperation::Concat(
                        ResidentalT::new(make_object(&resulting), ()),
                        [
                            ResidentalT::new(make_object(&concat.former), ()),
                            ResidentalT::new(make_object(&concat.latter), ()),
                        ],
                    ));

                    itms_in.push(make_object(&concat.former));

                    self.available_slices
                        .get_mut(&resulting.object().id())
                        .unwrap()
                        .set_biggest(resulting.object().unwrap_range());
                }

                // Produce out-intermediates
                for sliced_out in itms.slice_outs(vid) {
                    let being_sliced = Object::new(
                        sliced_out.object().id(),
                        sliced_out
                            .object()
                            .node()
                            .map_p(|_, _| self.available_slices[&sliced_out.object().id()].biggest),
                    );
                    ops.emit(SubgraphOperation::Clone(
                        ResidentalT::with_no_pointer(make_object(sliced_out)),
                        ResidentalT::with_no_pointer(being_sliced.clone()),
                        {
                            let (deg, holding_slice) = being_sliced.node().unwrap_poly();
                            Some(
                                sliced_out
                                    .object()
                                    .unwrap_range()
                                    .relative_to(holding_slice, deg as u64),
                            )
                        },
                    ));

                    self.available_slices
                        .get_mut(&sliced_out.object().id())
                        .unwrap()
                        .add(sliced_out.object().unwrap_range());

                    itms_out.push(make_object(sliced_out));
                }
            }
        }

        let itms_write_back = itms
            .write_backs()
            .map(|(dst, src)| (make_object(dst), make_object(src)))
            .collect();

        Paragraph {
            ops,
            inputs,
            as_is_inputs,
            itms_in,
            itms_out,
            device: gpu,
            itms_write_back,
            outputs: value::Tree::Tuple(
                outputs
                    .unwrap()
                    .into_iter()
                    .map(|o| value::Tree::Single(o))
                    .collect(),
            ),
        }
    }
}
