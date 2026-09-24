use super::super::cg_def_use;
use super::prelude::*;
use super::Chunker;
use sliceable_subgraph::{Vertex, VertexId, VertexNode};

/// The degree of ScalarArray's here is the degree of the polynomial containing the object
pub type SlicesNode = zkpoly_common::typ::template::Typ<Vec<Slice>>;
pub type RotatedNode = zkpoly_common::typ::template::Typ<Slice>;

#[derive(Debug, Hash, Clone)]
pub struct Object<N> {
    id: ObjectId,
    node: N,
}

impl<N> Object<N> {
    fn new(id: ObjectId, node: N) -> Self {
        Self { id, node }
    }

    pub fn id(&self) -> ObjectId {
        self.id
    }

    pub fn node(&self) -> &N {
        &self.node
    }
}

impl Object<SlicesNode> {
    /// Append slice to ranges this object refers.
    /// Only effective if this object is a polynomial.
    fn add_range(&mut self, range: Slice) {
        self.node.modify_p(|_, ranges| ranges.push(range));
    }

    /// Set ranges this object refers to slices.
    /// Only effective if this object is a polynomial.
    fn set_ranges(&mut self, ranges: Vec<Slice>) {
        self.node.modify_p(|_, rs| *rs = ranges);
    }

    /// Set ranges this object refers to one or no slice.
    /// Only effective if this object is a polynomial.
    fn set_range(&mut self, range: Option<Slice>) {
        self.node
            .modify_p(|_, rs| *rs = range.map_or_else(Vec::new, |x| vec![x]));
    }

    fn unwrap_range(&self) -> &[Slice] {
        &self.node.unwrap_poly().1
    }

    fn with_range(&self, range: Slice) -> Object<RotatedNode> {
        Object {
            id: self.id,
            node: self.node.map_p(|_, _| range),
        }
    }

    /// Assuming the object is a polynomial and refers to only one range.
    pub fn unwrap_range1(&self) -> Slice {
        let ranges = self.node.unwrap_poly().1;
        assert_eq!(ranges.len(), 1);
        ranges[0]
    }

    /// Assuming the object refers to only one range and turn it into an according [`RotatedNode`]
    pub fn into_rotated_object(&self) -> Object<RotatedNode> {
        Object::new(
            self.id,
            self.node.map_p(|_, r| {
                if r.len() != 1 {
                    panic!("{:?} cannot be converted into rotated object", self);
                }
                r[0]
            }),
        )
    }

    /// Assuming the object refers to only one range if it is a polynomial.
    pub fn range1(&self) -> Option<Slice> {
        self.node().poly().map(|(_, rs)| {
            assert_eq!(rs.len(), 1);
            rs[0]
        })
    }

    /// Ranges the object refers to.
    /// Returns empty iterator if object is not polynomial.
    pub fn ranges<'a>(&'a self) -> impl Iterator<Item = Slice> + 'a {
        self.node
            .poly()
            .into_iter()
            .map(|(_, rs)| rs.iter())
            .flatten()
            .cloned()
    }

    pub fn can_be_sliced(&self) -> bool {
        matches!(
            self.node(),
            SlicesNode::ScalarArray { .. } | SlicesNode::PointBase { .. }
        )
    }
}

fn rotation_of_slice(deg: u64, slice: Slice) -> i64 {
    if slice.begin() > deg / 2 {
        -((deg - slice.begin()) as i64)
    } else {
        slice.begin() as i64
    }
}

impl Object<RotatedNode> {
    pub fn unwrap_range(&self) -> Slice {
        self.node.unwrap_poly().1.clone()
    }

    pub fn with_range(&self, range: Slice) -> Self {
        Self {
            id: self.id,
            node: self.node.map_p(|_, _| range),
        }
    }

    pub fn rotated(&self, delta: i64) -> Self {
        Self {
            id: self.id,
            node: self.node.map_p(|deg, s| s.rotated(delta, deg as u64)),
        }
    }

    pub fn range(&self) -> Option<Slice> {
        self.node().poly().map(|(_, s)| *s)
    }

    pub fn into_slices(&self) -> Object<SlicesNode> {
        Object {
            id: self.id,
            node: self.node.map_p(|_, s| vec![*s]),
        }
    }
}

pub type Atom<N> = value::Atom<Object<N>, ()>;
pub type OutputValue<N> = value::OutputValue<Atom<N>, VertexId>;
pub type InputValue<N> = value::InputValue<Atom<N>>;

impl Atom<SlicesNode> {
    pub fn into_rotated_atom(&self) -> Atom<RotatedNode> {
        self.map_object(|o| o.into_rotated_object())
    }
}

#[derive(Debug, Hash)]
pub struct Concat {
    pub former: Atom<RotatedNode>,
    pub latter: Atom<RotatedNode>,
}

impl Concat {
    pub fn resulting(&self) -> Atom<RotatedNode> {
        self.former.map_object(|obj| {
            obj.with_range(Slice::new(
                self.former.object().unwrap_range().begin(),
                self.former.object().unwrap_range().len()
                    + self.latter.object().unwrap_range().len(),
            ))
        })
    }
}

/// Keep tracks of the intermediates between prologue and body,
/// or between body and body.
///
/// All intermediates must not be zero-sized
pub trait Intermediates {
    /// After execution of some vertex in prologue/body, we need to slice some objects out of its outputs,
    /// and output the produced object as an intermediate
    fn slice_outs(&self, vid: VertexId) -> impl Iterator<Item = &Atom<RotatedNode>>;
    /// After execution of some vertex in body, we need to concat two slices to get output that can be used by successive vertices.
    fn concat_ins(&self, vid: VertexId) -> impl Iterator<Item = &Concat>;
    /// After execution of all vertices, we need put produced out-intermediates to place where next loop can read.
    fn write_backs(&self) -> impl Iterator<Item = (&Atom<RotatedNode>, &Atom<RotatedNode>)>;
}

#[derive(Hash)]
pub struct PrologueIntermediates {
    slice_out: BTreeMap<VertexId, Vec<Atom<RotatedNode>>>,
}

#[derive(Hash)]
pub struct BodyIntermediates {
    slice_out: BTreeMap<VertexId, Vec<Atom<RotatedNode>>>,
    concat_in: BTreeMap<VertexId, Vec<Concat>>,
    write_back: Vec<(
        /* dst */ Atom<RotatedNode>,
        /* src */ Atom<RotatedNode>,
    )>,
}

impl Intermediates for PrologueIntermediates {
    fn slice_outs(&self, vid: VertexId) -> impl Iterator<Item = &Atom<RotatedNode>> {
        self.slice_out
            .get(&vid)
            .map(|x| x.iter())
            .into_iter()
            .flatten()
    }

    fn concat_ins(&self, _vid: VertexId) -> impl Iterator<Item = &Concat> {
        std::iter::empty()
    }

    fn write_backs(&self) -> impl Iterator<Item = (&Atom<RotatedNode>, &Atom<RotatedNode>)> {
        std::iter::empty()
    }
}

impl Intermediates for BodyIntermediates {
    fn slice_outs(&self, vid: VertexId) -> impl Iterator<Item = &Atom<RotatedNode>> {
        self.slice_out
            .get(&vid)
            .map(|x| x.iter())
            .into_iter()
            .flatten()
    }

    fn concat_ins(&self, vid: VertexId) -> impl Iterator<Item = &Concat> {
        self.concat_in
            .get(&vid)
            .map(|x| x.iter())
            .into_iter()
            .flatten()
    }

    fn write_backs(&self) -> impl Iterator<Item = (&Atom<RotatedNode>, &Atom<RotatedNode>)> {
        self.write_back.iter().map(|(o1, o2)| (o1, o2))
    }
}

pub trait Planner {
    /// Plan the range that must be calculated at `vid` given `needed_ranges`.
    /// Also, output `atoms` of `vid` must be provided so as to plan intermediates.
    fn calculated_range(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice>;
    fn inputed_range(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice>;
    fn output_range(&self) -> Slice;
}

pub struct ProloguePlanner<'a> {
    chunker: &'a Chunker,
}

impl<'a> ProloguePlanner<'a> {
    pub fn new(chunker: &'a Chunker) -> Self {
        Self { chunker }
    }
}

impl<'b> Planner for ProloguePlanner<'b> {
    fn calculated_range(
        &mut self,
        _vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        _atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice> {
        self.chunker.including_slice(needed_range)
    }

    fn inputed_range(
        &mut self,
        _vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        _atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice> {
        self.chunker.including_chunks(needed_range)
    }

    fn output_range(&self) -> Slice {
        self.chunker.output_range()
    }
}

/// For a vertex with successors need range x,
/// we determine the range to calculate and intermediates in the following manner:
///   |--------------------|    x
///                |-------|    the calculated range is exactly the last chunk_size
///   |------------|            we need this range taken from last iteration
///           |------------|    and this range is fed to next iteration
pub struct BodyPlanner<'a> {
    chunker: &'a Chunker,
    prologue_itm: PrologueIntermediates,
    body_itm: BodyIntermediates,
}

impl<'b> BodyPlanner<'b> {
    pub fn new(chunker: &'b Chunker) -> Self {
        Self {
            chunker,
            prologue_itm: PrologueIntermediates {
                slice_out: BTreeMap::new(),
            },
            body_itm: BodyIntermediates {
                slice_out: BTreeMap::new(),
                concat_in: BTreeMap::new(),
                write_back: Vec::new(),
            },
        }
    }

    pub fn finish(self) -> (PrologueIntermediates, BodyIntermediates) {
        (self.prologue_itm, self.body_itm)
    }

    fn plan_itm_for_desired_range(
        &mut self,
        vid: VertexId,
        including_range: Slice,
        atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Slice {
        let itm_in_range = self.chunker.new_slice(
            including_range.begin(),
            including_range.len() - self.chunker.chunk_size(),
        );
        let itm_out_range = self
            .chunker
            .rotate(&itm_in_range, self.chunker.chunk_size() as i64);
        let calculated_range = self
            .chunker
            .new_slice(itm_in_range.end(), self.chunker.chunk_size());

        if itm_in_range.len() != 0 {
            atoms.for_each(|atm| {
                self.prologue_itm
                    .slice_out
                    .entry(vid)
                    .or_default()
                    .push(atm.map_object(|obj| obj.with_range(itm_in_range)));
                self.body_itm
                    .slice_out
                    .entry(vid)
                    .or_default()
                    .push(atm.map_object(|obj| obj.with_range(itm_out_range)));
                self.body_itm
                    .concat_in
                    .entry(vid)
                    .or_default()
                    .push(Concat {
                        former: atm.map_object(|obj| obj.with_range(itm_in_range)),
                        latter: atm.map_object(|obj| obj.with_range(calculated_range)),
                    });
                self.body_itm.write_back.push((
                    atm.map_object(|obj| obj.with_range(itm_in_range)),
                    atm.map_object(|obj| obj.with_range(itm_out_range)),
                ))
            });
        }
        calculated_range
    }
}

impl<'b> Planner for BodyPlanner<'b> {
    fn calculated_range(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice> {
        let including_range = self.chunker.including_slice(needed_range)?;
        Some(self.plan_itm_for_desired_range(vid, including_range, atoms))
    }

    fn inputed_range(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        atoms: impl Iterator<Item = Atom<SlicesNode>>,
    ) -> Option<Slice> {
        let including_range = self.chunker.including_chunks(needed_range)?;
        Some(self.plan_itm_for_desired_range(vid, including_range, atoms))
    }

    fn output_range(&self) -> Slice {
        self.chunker.rotate(
            &self.chunker.output_range(),
            self.chunker.chunk_size() as i64,
        )
    }
}

#[derive(Clone, Debug)]
pub struct RotationAnalysis {
    value: BTreeMap<VertexId, OutputValue<RotatedNode>>,
    input: BTreeMap<VertexId, Vec<(VertexId, InputValue<RotatedNode>)>>,
}

impl RotationAnalysis {
    fn empty() -> Self {
        Self {
            value: BTreeMap::new(),
            input: BTreeMap::new(),
        }
    }

    pub fn subgraph_outputs<'s, Rt: RuntimeType>(
        &self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        _objects: &ObjectDefs,
    ) -> cg_def_use::OutputValue {
        self.value(cg.output).map_ref(|oa| {
            let object = oa.deref().object();
            value::OutputT::non_inplace(value::Atom::new(
                object.id(),
                oa.device(),
                object.node().clone(),
            ))
        })
    }

    pub fn value(&self, v: VertexId) -> &OutputValue<RotatedNode> {
        self.value.get(&v).unwrap()
    }

    pub fn input(&self, v: VertexId) -> &Vec<(VertexId, InputValue<RotatedNode>)> {
        self.input.get(&v).unwrap()
    }

    pub fn input_from(&self, v: VertexId, from: VertexId) -> &InputValue<RotatedNode> {
        &self
            .input(v)
            .iter()
            .find(|(vid, _)| *vid == from)
            .unwrap_or_else(|| panic!("vertex {:?} does not have input from {:?}", v, from))
            .1
    }

    fn output_of<'s, 'a, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
        objects: &mut ObjectDefs,
        inputs_mapping: impl Fn(type2::VertexId) -> &'a cg_def_use::InputValue,
    ) -> OutputValue<RotatedNode> {
        use sliceable_subgraph::template::VertexNode::*;
        match v.node() {
            TupleGet(t, i) => {
                let pred_value = &self.value[t];
                match pred_value {
                    OutputValue::Tuple(tuple) => tuple[*i].clone(),
                    OutputValue::Single(..) => panic!("expected tuple or array"),
                }
            }
            Inner(SliceableNode::RotateIdx(src, delta)) => {
                let src_value = self.value[src].unwrap_leaf();
                OutputValue::Single(
                    src_value
                        .with_non_inplace()
                        .map_v(|a| a.map_object(|o| o.rotated(*delta as i64))),
                )
            }
            Return(vids) => {
                let pred_values = vids
                    .iter()
                    .map(|vid| self.value[vid].map_ref(|osv| osv.with_non_inplace()))
                    .collect::<Vec<_>>();
                OutputValue::Tuple(pred_values)
            }
            Input(outer_vid) | InputAsIs(outer_vid) => {
                let ov = inputs_mapping(*outer_vid);
                ov.map_ref(|ao| {
                    value::OutputT::non_inplace(
                        ao.map_object(|oid| Object::new(*oid, ao.node().clone()))
                            .map_node(|_| ()),
                    )
                })
            }
            _ => lower_typ(v.typ().outer(), &mut |vnode| {
                let object_id = objects.push(vnode.erase_p());

                value::OutputT::non_inplace(Atom::new(
                    Object::new(object_id, vnode),
                    memory_device,
                    (),
                ))
            }),
        }
    }

    fn input_of<'s, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
    ) -> Vec<(VertexId, InputValue<RotatedNode>)> {
        let mutable_used_vids = v.node().mutable_uses().collect::<Vec<_>>();
        let uses = v.node().uses().map(|u| {
            (
                u,
                if mutable_used_vids.contains(&u) {
                    Mutability::Mut
                } else {
                    Mutability::Const
                },
            )
        });

        let with_device = |val: Atom<RotatedNode>| {
            if matches!(v.node(), VertexNode::TupleGet(..)) {
                val
            } else {
                val.with_device(memory_device)
            }
        };

        let inputs = uses
            .map(|(used_vid, mutability)| {
                let iv = self.value[&used_vid].map_ref(|osv| {
                    value::InputT::new(with_device(osv.deref().clone()), mutability)
                });
                (used_vid, iv)
            })
            .collect::<Vec<_>>();

        inputs
    }

    // Forward propagation from inputs to return vertex to acquire object information for each vertex
    pub fn infer_objects<'s, 'a, Rt: RuntimeType>(
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        execution_device: impl Fn(VertexId) -> type2::Device,
        objects: &mut ObjectDefs,
        inputs_mapping: impl Fn(type2::VertexId) -> &'a cg_def_use::InputValue,
    ) -> Self {
        let mut def_use = Self::empty();

        for (vid, v) in cg.g.topology_sort() {
            let memory_device = match execution_device(vid) {
                type2::Device::Gpu(i) => type3::Device::Gpu(i),
                type2::Device::Cpu => type3::Device::Cpu,
            };

            let input = def_use.input_of(v, memory_device);
            let mut output = def_use.output_of(v, memory_device, objects, &inputs_mapping);

            output
                .iter1_mut()
                .zip(v.node().outputs_inplace())
                .for_each(|(ov, inplace_of)| {
                    if let Some(inplace_of) = inplace_of {
                        *ov.unwrap_leaf_mut().inplace_of_mut() = Some(inplace_of);
                    }
                });

            def_use.input.insert(vid, input);
            def_use.value.insert(vid, output);
        }

        def_use
    }
}

/// The desired inputed and outputed slices for each vertex in the sliceable subgraph.
/// That is, foreach vertex, the inputs are ensured to produce all slices needed for all outputs,
/// but the actual outputs may contain more slices then outputs really needs.
///
/// After analysis, all output and input objects for computation vertices
/// (that is, not including TupleGet stuff) are expected to contain one and only slice.
#[derive(Clone, Debug, Hash)]
pub struct DefUse {
    /// For input vertices, since there may be multiple input vertices taking the same object ID,
    /// In this case, we represent all of them with the first one.
    /// InputAsIs vertices are not counted here.
    object_inputs_at: BTreeMap<ObjectId, Vec<VertexId>>,
    /// The desired output of each vertex.
    value: BTreeMap<VertexId, OutputValue<SlicesNode>>,
    /// The desired input of each vertex.
    /// For vertices that do computations (excluding Return, RotateIdx and TupleGet),
    /// We take the smallest slice that contains the desired outputed slices.
    ///
    /// e.g.
    ///          |------|     |------|       desired outputed slices
    ///       |     |      |      |      |   chunk boundaries
    ///          |-------------------|       input slice
    ///
    /// Furthermore, subgraph inputs must be aligned to chunk boundaries.
    input: BTreeMap<VertexId, Vec<(VertexId, InputValue<SlicesNode>)>>,
}

struct InferScratchPad {
    /// Mapping from input vertices to the slices they output
    inputs_output_slices: BTreeMap<VertexId, Vec<Slice>>,
}

impl DefUse {
    fn empty(rotation_analysis: &RotationAnalysis) -> Self {
        Self {
            object_inputs_at: BTreeMap::new(),
            value: rotation_analysis
                .value
                .iter()
                .map(|(vid, v)| {
                    (
                        *vid,
                        v.map_ref(|oa| {
                            oa.clone().map_v(|a| {
                                a.map_object(|o| Object::new(o.id, o.node.map_p(|_, _| Vec::new())))
                            })
                        }),
                    )
                })
                .collect(),
            input: rotation_analysis
                .input
                .iter()
                .map(|(vid, inputs)| {
                    (
                        *vid,
                        inputs
                            .iter()
                            .map(|(uid, v)| {
                                (
                                    *uid,
                                    v.map_ref(|ia| {
                                        ia.clone().map_v(|a| {
                                            a.map_object(|o| {
                                                Object::new(o.id, o.node.map_p(|_, _| Vec::new()))
                                            })
                                        })
                                    }),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }

    pub fn value(&self, v: VertexId) -> &OutputValue<SlicesNode> {
        self.value.get(&v).unwrap()
    }

    pub fn input(&self, v: VertexId) -> &Vec<(VertexId, InputValue<SlicesNode>)> {
        self.input.get(&v).unwrap()
    }

    /// The schedule produced by a later pass does not know which input vertex is the representing one
    /// among those inputing the same object,
    /// so when dealing with input vertices, we must query this method for the representing one.
    /// A `visited` set filters out vertices already handled.
    pub fn representing_input(
        &self,
        v: VertexId,
        visited: &mut BTreeSet<VertexId>,
    ) -> Option<VertexId> {
        if visited.contains(&v) {
            return None;
        }

        let oid = self.value(v).unwrap_leaf().object().id();
        let vids = &self.object_inputs_at[&oid];
        for &vid in vids.iter() {
            visited.insert(vid);
        }

        Some(vids[0])
    }

    pub fn input_from(&self, v: VertexId, from: VertexId) -> &InputValue<SlicesNode> {
        &self
            .input(v)
            .iter()
            .find(|(vid, _)| *vid == from)
            .unwrap_or_else(|| panic!("vertex {:?} does not have input from {:?}", v, from))
            .1
    }

    fn value_mut(&mut self, v: VertexId) -> &mut OutputValue<SlicesNode> {
        self.value.get_mut(&v).unwrap()
    }

    fn input_mut(&mut self, v: VertexId) -> &mut Vec<(VertexId, InputValue<SlicesNode>)> {
        self.input.get_mut(&v).unwrap()
    }

    pub fn on<'s, Rt: RuntimeType>(
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        rotation_analysis: &RotationAnalysis,
        planner: &mut impl Planner,
    ) -> Self {
        let mut r = Self::empty(rotation_analysis);
        r.infer_slice(cg, rotation_analysis, planner);
        r
    }

    fn infer_slice_at<'s, Rt: RuntimeType>(
        &mut self,
        vid: VertexId,
        v: &sliceable_subgraph::Vertex<'s, Rt>,
        rotation_analysis: &RotationAnalysis,
        planner: &mut impl Planner,
        ctx: &mut InferScratchPad,
    ) {
        // Return vertex's output range is set to first chunk
        if let VertexNode::Return(..) = v.node() {
            self.value_mut(vid)
                .iter_mut()
                .for_each(|vo| vo.object_mut().add_range(planner.output_range()))
        }

        // Propagate needed slices from output to inputs.
        match v.node() {
            // Rotations should only have one input and one output
            VertexNode::Inner(SliceableNode::RotateIdx(..)) => {
                let output_object = &self.value[&vid].unwrap_leaf().object();
                let output_slices = output_object.unwrap_range().to_vec();

                assert!(self.input(vid).len() == 1);
                self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                    iv.unwrap_leaf_mut()
                        .object_mut()
                        .set_ranges(output_slices.clone());
                });
            }
            // As-is input's value follows that of ratation analysis
            VertexNode::InputAsIs(..) => {
                *self.value_mut(vid) = rotation_analysis.value(vid).map_ref(|oa| {
                    oa.clone()
                        .map_v(|a| a.map_node(|_| ()).map_object(|o| o.into_slices()))
                })
            }
            // For inner vertices that take slices as inputs and outputs,
            // we need its output to include all desired inputs of its successors.
            //
            // Then, the `planner` decides and keeps track of which slice to compute.
            // Finally, we rotate the computed slice to each of the input rotation as discovered by rotation analysis.
            VertexNode::Inner(..) => {
                let output_slices = self.value[&vid]
                    .iter()
                    .map(|ov| ov.deref().object().ranges())
                    .flatten();
                // Some part of the desired output can be taken from last iteration.
                // `planner` is responsible for deciding and keeping track of that.
                let out_slice = planner.calculated_range(
                    vid,
                    output_slices,
                    self.value[&vid].iter().map(|oa| oa.deref().clone()),
                );

                self.value_mut(vid).iter_mut().for_each(|ov| {
                    ov.object_mut().set_range(out_slice);
                });

                let sliceable_uses = v
                    .node()
                    .sliceable_uses_ref()
                    .cloned()
                    .collect::<BTreeSet<_>>();

                self.input_mut(vid).iter_mut().for_each(|(u, iv)| {
                    if sliceable_uses.contains(&u) {
                        let input_rotation = rotation_analysis.input_from(vid, *u);
                        iv.iter_mut()
                            .zip(input_rotation.iter())
                            .for_each(|(ia, rotation_a)| {
                                // Rotate input to its desired position according to rotation analysis
                                let rotation = rotation_a
                                    .object()
                                    .node()
                                    .poly()
                                    .map(|(deg, s)| (*deg, rotation_of_slice(*deg as u64, *s)));
                                let out_slice = match (out_slice, rotation) {
                                    // If some output has a range, `out_slice` must be non-empty
                                    (s, Some((deg, r))) => Some(s.unwrap().rotated(r, deg as u64)),
                                    _ => None,
                                };
                                ia.object_mut().set_range(out_slice)
                            });
                    } else {
                        *iv = rotation_analysis.input_from(vid, *u).map_ref(|ia| {
                            ia.clone()
                                .map_v(|a| a.map_object(|o| o.into_slices()).map_node(|_| ()))
                        });
                    }
                });
            }
            // For input vertices, we also need its output to include all desired inputs of its successors.
            VertexNode::Input(..) => {
                let output_slices = self.value[&vid].unwrap_leaf().object().ranges();

                self.object_inputs_at
                    .entry(self.value[&vid].unwrap_leaf().object().id())
                    .or_default()
                    .push(vid);
                ctx.inputs_output_slices
                    .insert(vid, output_slices.collect());
            }
            // Return is actually tuple packing
            VertexNode::Return(..) => {
                let output_values = self.value[&vid]
                    .iter1()
                    .map(|ov| ov.map_ref(|osv| value::InputT::immutable(osv.deref().clone())))
                    .collect::<Vec<_>>();
                self.input_mut(vid)
                    .iter_mut()
                    .zip(output_values.into_iter())
                    .for_each(|((_, iv), ov)| *iv = ov)
            }
            // For vertices that outputs partial scalar/point results, we enforce their inputs to include the output range
            // [0, chunk_size)
            VertexNode::Last(..) => self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                iv.iter_mut()
                    .for_each(|v| v.object_mut().set_ranges(vec![planner.output_range()]))
            }),
            // For tuple gets, we propagate slice requirement to the currespounding element in the input tuple
            VertexNode::TupleGet(_, i) => {
                let output_value =
                    self.value[&vid].map_ref(|osv| value::InputT::immutable(osv.deref().clone()));
                assert_eq!(self.input(vid).len(), 1);
                let iv = self.input_mut(vid)[0].1.iter1_mut().nth(*i).unwrap();
                *iv = output_value;
            }
        }

        // Propagate neede slices from inputs to outputs
        self.input[&vid]
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(uid, vi)| {
                self.value_mut(uid)
                    .iter_mut()
                    .zip(vi.iter())
                    .for_each(|(u_out, v_in)| {
                        v_in.object()
                            .ranges()
                            .for_each(|s| u_out.object_mut().add_range(s));
                    });
            });
    }

    fn finalize_inputs(&mut self, planner: &mut impl Planner, ctx: InferScratchPad) {
        for (oid, vids) in self.object_inputs_at.clone().iter() {
            let slices = vids
                .iter()
                .map(|vid| ctx.inputs_output_slices[vid].iter().cloned())
                .flatten();

            let vid = vids[0];
            let a = self.value[&vid].unwrap_leaf().deref().clone();
            assert_eq!(a.object().id(), *oid);

            let out_slice = planner.inputed_range(vid, slices, std::iter::once(a));

            self.value_mut(vid).iter_mut().for_each(|ov| {
                ov.object_mut().set_range(out_slice);
            });
        }
    }

    // Back propagation from return vertex to acquire needed slice for each vertex
    fn infer_slice<'s, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        rotation_analysis: &RotationAnalysis,
        planner: &mut impl Planner,
    ) {
        let mut ctx = InferScratchPad {
            inputs_output_slices: BTreeMap::new(),
        };

        // # Loop Invariants
        // Let's call inner vertices except rotations and tuple-gets computation vertices.
        //
        // - Processed computation vertices have their output and input objects refering to one and only one
        //   range, if it is a polynomial. So does input vertices.
        // - Unprecessed predecessors of processed vertices have their output objects refering to at least one
        //   range, if it is a polynomial.
        for (vid, v) in cg.g.topology_sort_inv() {
            self.infer_slice_at(vid, v, rotation_analysis, planner, &mut ctx);
        }

        self.finalize_inputs(planner, ctx);
    }
}
