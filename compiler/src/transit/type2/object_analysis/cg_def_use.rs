use super::prelude::*;
use super::subgraph_elaboration::{self, slice_analysis};
use type2::VertexId;
use type3::{Device, DeviceSpecific};
use SliceableNode::*;

pub type Atom = value::Atom<ObjectId, ValueNode>;
pub type Value = value::Tree<Atom>;
pub type OutputValue = value::OutputValue<Atom, VertexId>;
pub type InputValue = value::InputValue<Atom>;

#[derive(Hash)]
pub struct SubgraphAnalysis {
    pub prologue_def_use: slice_analysis::DefUse,
    pub body_def_use: slice_analysis::DefUse,
    pub prologue_itms: slice_analysis::PrologueIntermediates,
    pub body_itms: slice_analysis::BodyIntermediates,
    pub chunker: subgraph_elaboration::Chunker,
}

/// Results of analyzing value definitions of each vertex
#[derive(Hash)]
pub struct DefUse {
    /// Value output of each vertex.
    /// All vertices should appear in this mapping.
    pub(super) value: BTreeMap<VertexId, OutputValue>,

    /// At which vertex an object is defined.
    /// All allocated objects should appear in this mapping.
    pub(super) def_at: BTreeMap<ObjectId, VertexId>,

    /// On which device an object is defined
    /// All allocated objects should appear in this mapping.
    ///
    /// For cloned slice objects, this is the device where they are first used.
    pub(super) def_on: BTreeMap<ObjectId, Device>,

    /// Value input of each vertex.
    /// All vertices should appear in this mapping.
    pub(super) input: BTreeMap<VertexId, Vec<(VertexId, InputValue)>>,

    /// Sometimes, we need to clone a slice out of a polynomial.
    /// When a slice is rotated, it is no longer a continuous (and allowing wrapping around) piece of memory.
    /// Some `((cloned, slice), new)` in this map means that `new` is cloned from `cloned` with `slice`.
    pub(super) cloned_slices: Bijection<(ObjectId, Slice), ObjectId>,

    /// The set of objects that should not be deallocated on CPU.
    /// For example, constants and computation graph inputs.
    pub(super) immortal_objects: BTreeSet<ObjectId>,

    pub(super) subgraph_analysis: BTreeMap<VertexId, super::SubgraphAnalysis>,
}

/// We are only debugging fields other than `value` and `input` since they will be debugged through visualization.
impl std::fmt::Debug for DefUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefUse")
            .field("def_at", &self.def_at)
            .field("def_on", &self.def_on)
            .field("cloned_slices", &self.cloned_slices)
            .field("immortal_objects", &self.immortal_objects)
            .finish()
    }
}

/// Mark that `object_id` is used at vertex `vid` on `device`.
/// Used for obtaining last use of each object on each device,
/// which is then used to decide whether an operation can be carried out inplace without making
/// object clones.
///
/// This analysis does not have to be tight,
/// that is, we can extend liveness longer then actually needed,
/// as it does not affect correctness.
fn mark_use(
    object_id: ObjectId,
    device: Device,
    hd_info: &HardwareInfo,
    def_on: &mut BTreeMap<ObjectId, Device>,
    used_on: &mut BTreeMap<ObjectId, DeviceSpecific<bool>>,
    cloned_slices: &ClonedSliceRegistry,
) {
    let is_used_on = |object_id, device| {
        used_on
            .get(&object_id)
            .is_some_and(|places| *places.get(device))
    };

    // If
    // - The object is sliced from another object,
    // - And is used for the first time for all devices
    // We follow the proceeding simple rules
    // - If the sliced object is defined on CPU, and it is sliced to GPU,
    //   - If the object has been used on GPU, then we extend its lifetime on GPU
    //   - Otherwise, extend its lifetime on CPU
    // - If the sliced object is defined and used on the same device, we extend its lifetime on it
    // - If the sliced object is defined on GPU, and it is sliced to CPU,
    //   similar to the first case
    if !Device::iter(hd_info.n_gpus()).any(|d| is_used_on(object_id, d)) {
        if let Some(..) = cloned_slices.0.get_backward(&object_id) {
            // We must use `device` for `def_on` of `object_id`, is `object_id` itself can later be clone-sliced,
            // and we don't want its `def_on` to be memory device of the [`VertexNode::RotateIdx`] vertex.
            def_on.insert(object_id, device);
        }
    }

    *used_on
        .entry(object_id)
        .or_insert_with(|| DeviceSpecific::default(hd_info.n_gpus()))
        .get_mut(device) = true;
}

/// Collects input values of a vertex.
/// This function basically just take output values of used vertices, and alters their device to `memory_device`.
/// It is also responsible for cloning slices out of polynomials for kernels that do not support sliced polynomials.
fn vertex_input_of<'s, Rt: RuntimeType>(
    vid: VertexId,
    v: &type2::Vertex<'s, Rt>,
    memory_device: Device,
    value: &BTreeMap<VertexId, OutputValue>,
    hd_info: &HardwareInfo,
    cloned_slices: &mut ClonedSliceRegistry,
    objects: &mut ObjectDefs,
    def_at: &mut BTreeMap<ObjectId, VertexId>,
    def_on: &mut BTreeMap<ObjectId, Device>,
    used_on: &mut BTreeMap<ObjectId, DeviceSpecific<bool>>,
) -> Vec<(VertexId, InputValue)> {
    let mutable_used_vids = v.mutable_uses().collect::<Vec<_>>();
    let uses = v.uses().map(|u| {
        (
            u,
            if mutable_used_vids.contains(&u) {
                Mutability::Mut
            } else {
                Mutability::Const
            },
        )
    });

    let with_device = |val: Atom| {
        if v.node().is_virtual() {
            val
        } else {
            val.with_device(memory_device)
        }
    };

    let inputs = uses
        .map(|(used_vid, mutability)| {
            let iv = value[&used_vid].map_ref(|ov| {
                if v.node().no_supports_shorten_slice() && ov.is_shorten_sliced()
                    || v.node().no_supports_pure_rotation() && ov.is_pure_rotated()
                {
                    // Some vertices' currespounding kernel does not support sliced polynomials,
                    // so we make clones for them
                    let (_, slice) = ov.node().unwrap_poly();
                    let new_object =
                        cloned_slices.register(vid, *ov.object(), slice.clone(), objects, def_at);
                    let new_value = with_device(
                        ov.with_object(new_object)
                            .with_node(ValueNode::plain_scalar_array(slice.len() as usize)),
                    );
                    value::InputT::new(new_value, mutability)
                } else {
                    value::InputT::new(with_device(ov.deref().clone()), mutability)
                }
            });
            (used_vid, iv)
        })
        .collect::<Vec<_>>();

    // Stuff like TupleGet or Rotate does not count as a use
    if !v.is_virtual() {
        for object in inputs
            .iter()
            .map(|(_, vi)| vi.iter().map(|v| *v.object()))
            .flatten()
        {
            mark_use(
                object,
                memory_device,
                hd_info,
                def_on,
                used_on,
                cloned_slices,
            );
        }
    }

    inputs
}

/// Collect output values of a vertex.
///
/// Most vertices in the graph define an object, while some vertices only modifies the register.
/// For exmaple, by rotating or slicing a polynomial, we only care about the offset and length information
/// stored in the register, but not their data (i.e., the object the register value points to).
///
/// For the former case, only slice information of the value is not inherited from currespounding input value;
///
/// For the latter case, a new value is defined based on `memory_device` and the vertex's Type2 type,
/// plus a newly allocated [`ObjectId`].
/// Also, if some output took memory of a inplace input, the currespounding output value inherits
/// the input's slice information.
/// This behaviour is determined by how kernels are written.
fn vertex_output_of<'s, Rt: RuntimeType>(
    vid: VertexId,
    v: &type2::Vertex<'s, Rt>,
    memory_device: Device,
    execution_device: &type2::DeviceCollection,
    uf_table: &type2::user_function::Table<Rt>,
    input: &[(VertexId, InputValue)],
    value: &BTreeMap<VertexId, OutputValue>,
    cloned_slices: &mut ClonedSliceRegistry,
    objects: &mut ObjectDefs,
    immortal_objects: &mut BTreeSet<ObjectId>,
    def_at: &mut BTreeMap<ObjectId, VertexId>,
    def_on: &mut BTreeMap<ObjectId, Device>,
    subgraph_analysis: &mut BTreeMap<VertexId, super::SubgraphAnalysis>,
) -> OutputValue {
    use type2::template::VertexNode::*;
    match v.node() {
        // These vertices only modify registers
        TupleGet(pred, i) | ArrayGet(pred, i) => {
            let pred_value = value[pred].clone();

            pred_value.unwrap_tuple()[*i].clone()
        }
        Array(elements) => OutputValue::Tuple(
            elements
                .iter()
                .map(|e| value[e].map_ref(|x| x.with_non_inplace()))
                .collect(),
        ),
        Sliceable(RotateIdx(pred, delta)) => {
            let pred_atom = value[pred].unwrap_leaf().deref();
            if let Some(rotated_atom) = pred_atom.rotate(*delta) {
                OutputValue::Single(value::OutputT::non_inplace(rotated_atom))
            } else {
                // Rotating a incomplete slice produces non-continuous data, so we need to clone the slice
                let (deg, slice) = pred_atom.node().unwrap_poly();
                let new_obj_id =
                    cloned_slices.register(vid, *pred_atom.object(), *slice, objects, def_at);
                // We don't record `def_on` here, as a [`RotateIdx`] vertex must be later used for it to appear in the graph
                let new_atom = pred_atom
                    .with_object(new_obj_id)
                    .with_node(ValueNode::plain_scalar_array(deg))
                    .rotate(*delta)
                    .unwrap();
                OutputValue::Single(value::OutputT::non_inplace(new_atom))
            }
        }
        Slice(pred, begin, end) => {
            let pred_atom = value[pred].unwrap_leaf().deref();
            let new_atom = pred_atom.slice(*begin, *end);
            OutputValue::Single(value::OutputT::non_inplace(new_atom))
        }
        // Output value of [`Return`] vertex is the desired output value of the computation graph
        AssertEq(a, _, _) | Print(a, _) => {
            value[a].map_ref(|x| value::OutputT::non_inplace(x.deref().clone()))
        }
        // Following vertices define new objects
        otherwise => {
            assert!(!otherwise.is_virtual());

            let value = if let Subgraph(subgraph) = otherwise {
                let rot_analysis = slice_analysis::RotationAnalysis::infer_objects(
                    subgraph,
                    |_| execution_device.ensure_gpu(),
                    objects,
                    |u| input.iter().find(|(x, _)| u == *x).map(|(_, r)| r).unwrap(),
                );

                let chunker = subgraph_elaboration::Chunker::new(subgraph.chunk_len, subgraph.deg);

                let mut prologue_planner = slice_analysis::ProloguePlanner::new(&chunker);
                let mut body_planner = slice_analysis::BodyPlanner::new(&chunker);

                let prologue_def_use =
                    slice_analysis::DefUse::on(subgraph, &rot_analysis, &mut prologue_planner);
                let body_def_use =
                    slice_analysis::DefUse::on(subgraph, &rot_analysis, &mut body_planner);

                let (prologue_itms, body_itms) = body_planner.finish();

                subgraph_analysis.insert(
                    vid,
                    SubgraphAnalysis {
                        prologue_def_use,
                        body_def_use,
                        prologue_itms,
                        body_itms,
                        chunker,
                    },
                );

                let output_value = rot_analysis.subgraph_outputs(subgraph, objects);
                output_value
            } else {
                let mut outputs_inplace = v.outputs_inplace(uf_table);
                let value = lower_typ(v.typ(), &mut |vnode| {
                    Atom::new(objects.push(vnode.erase_p()), memory_device, vnode)
                });

                value.map(&mut |atom: Atom| {
                    let output_inplace = outputs_inplace
                        .next()
                        .expect("to know whether this output took the place of some input");

                    if let Some(inplace_input) = output_inplace {
                        // For inplace input, if it is a slice, then output is also a slice of the same shape
                        let i = input
                            .iter()
                            .position(|(vid, _)| vid == &inplace_input)
                            .unwrap();
                        let (_, input_value) = &input[i];
                        let input_atom = input_value.unwrap_leaf();

                        if input_atom.mutability() != Mutability::Mut {
                            panic!("inplace input must be mut at {:?}", vid);
                        }

                        if !atom.node().compatible(input_atom.node()) {
                            panic!(
                                "incompatible value at {:?}: from typ: {:?}, input value: {:?}",
                                vid,
                                atom,
                                input_atom.node()
                            );
                        }

                        value::OutputT::new(atom, Some(inplace_input))
                    } else {
                        value::OutputT::non_inplace(atom)
                    }
                })
            };

            for object_id in value.iter().map(|v| *v.object()) {
                def_at.insert(object_id, vid);
                def_on.insert(object_id, memory_device);

                if v.node().immortal() {
                    immortal_objects.insert(object_id);
                }
            }

            value
        }
    }
}

struct ClonedSliceRegistry(Bijection<(ObjectId, Slice), ObjectId>);

impl ClonedSliceRegistry {
    /// Register that a new object is cloned fro m `cloned_object` with `slice`at vertex `vid`.
    /// In this case, we consider the object to be defined at the vertex where it is first used.
    pub fn register(
        &mut self,
        vid: VertexId,
        cloned_object: ObjectId,
        slice: Slice,
        objects: &mut ObjectDefs,
        def_at: &mut BTreeMap<ObjectId, VertexId>,
    ) -> ObjectId {
        if let Some(new_obj) = self.0.get_forward(&(cloned_object, slice)) {
            *new_obj
        } else {
            let new_obj = objects.push(TopNode::ScalarArray {
                len: slice.len() as usize,
                meta: (),
            });
            self.0.insert((cloned_object, slice), new_obj);
            def_at.insert(new_obj, vid);
            new_obj
        }
    }
}

impl DefUse {
    /// Analyze a computation graph for a [`DefUse`].
    pub fn analyze<'s, 'a, Rt: RuntimeType>(
        g: &type2::CgSubgraph<'_, 's, Rt>,
        uf_table: &type2::user_function::Table<Rt>,
        execution_device: impl Fn(VertexId) -> &'a type2::DeviceCollection,
        hd_info: &HardwareInfo,
        constants_device: &Heap<ConstantId, Device>,
        objects: &mut ObjectDefs,
    ) -> Self {
        // These are the results to be obtained
        let mut value: BTreeMap<VertexId, OutputValue> = BTreeMap::new();
        let mut def_at = BTreeMap::new();
        let mut input = BTreeMap::new();
        let mut cloned_slices = ClonedSliceRegistry(Bijection::new());
        let mut cpu_immortal_objects = BTreeSet::new();
        let mut def_on = BTreeMap::new();
        let mut subgraph_analysis = BTreeMap::new();

        // Intermediate keeping track of whether some object has been used on some device when
        // visiting current vertex
        let mut used_on = BTreeMap::new();

        let mut seq = Vec::new();

        // Contract: After this loop, all vertices appear in `value` and `input`
        for (vid, v) in g.topology_sort() {
            seq.push(vid);

            // Invariant: All allocated objects should appear in `def_at`, `def_on`

            let execution_device = execution_device(vid);
            let memory_device = v
                .node()
                .memory_device(execution_device, |c| constants_device[*c]);

            // Invariant: `used_on` records all uses before this vertex

            // Inputs of the vertex
            let v_inputs = vertex_input_of(
                vid,
                v,
                memory_device,
                &value,
                hd_info,
                &mut cloned_slices,
                objects,
                &mut def_at,
                &mut def_on,
                &mut used_on,
            );

            // Output of the vertex
            let vo = vertex_output_of(
                vid,
                v,
                memory_device,
                execution_device,
                uf_table,
                &v_inputs,
                &value,
                &mut cloned_slices,
                objects,
                &mut cpu_immortal_objects,
                &mut def_at,
                &mut def_on,
                &mut subgraph_analysis,
            );

            value.insert(vid, vo);
            input.insert(vid, v_inputs);
        }

        let def_use = Self {
            value,
            def_at,
            input,
            cloned_slices: cloned_slices.0,
            def_on,
            immortal_objects: cpu_immortal_objects,
            subgraph_analysis,
        };

        def_use.check_contracts(&seq);

        def_use
    }

    fn check_contracts(&self, seq: &[VertexId]) {
        macro_rules! check_key {
            ($mapping:expr, $key:expr, $name:expr) => {
                if !$mapping.contains_key($key) {
                    panic!("entry not found for key {:?} in {}", $key, $name)
                }
            };
        }
        for &vid in seq.iter() {
            check_key!(&self.input, &vid, "input");
            check_key!(&self.value, &vid, "value");
        }
    }

    pub fn object_is_cloned_slice(&self, object_id: ObjectId) -> Option<(ObjectId, Slice)> {
        self.cloned_slices.get_backward(&object_id).cloned()
    }

    pub fn value(&self, vid: VertexId) -> &OutputValue {
        &self.value[&vid]
    }

    pub fn input_of<'a>(
        &'a self,
        vid: VertexId,
    ) -> impl Iterator<Item = &'a (VertexId, InputValue)> + 'a {
        self.input[&vid].iter()
    }

    pub fn input_from<'a>(&'a self, vid: VertexId, from: VertexId) -> &'a InputValue {
        &self.input[&vid]
            .iter()
            .find(|(vid, _)| *vid == from)
            .unwrap()
            .1
    }

    pub fn immortal_objects(&self) -> &BTreeSet<ObjectId> {
        &self.immortal_objects
    }

    /// Checks whether the top object `object` is immortal
    pub fn is_whole_object_immortal(&self, object: ObjectId) -> bool {
        self.immortal_objects.contains(&object)
    }

    pub fn is_immortal(&self, object: &value::Object) -> bool {
        self.immortal_objects.contains(&object.id()) && object.is_whole_poly()
    }

    pub fn def_on(&self, object_id: ObjectId) -> Device {
        self.def_on[&object_id]
    }

    pub fn def_at(&self, object_id: ObjectId) -> VertexId {
        self.def_at[&object_id]
    }

    pub fn constants<'a>(
        &'a self,
        objects: &'a ObjectDefs,
    ) -> impl Iterator<Item = (value::Object, type3::Device)> + 'a {
        self.immortal_objects().iter().map(|&oid| {
            (
                value::Object::new(oid, objects[oid].with_normalized_p()),
                self.def_on(oid),
            )
        })
    }

    pub fn subgraph_analysis(&self, vid: VertexId) -> &SubgraphAnalysis {
        self.subgraph_analysis
            .get(&vid)
            .unwrap_or_else(|| panic!("object analysis not found for subgraph at {:?}", vid))
    }
}
