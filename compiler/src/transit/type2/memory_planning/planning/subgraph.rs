use super::super::prelude::*;
use allocators::smithereens_wrapper;
use object_analysis::template::SubgraphOperationOn;
use planning::machine::{AdEmitter, DummyAdHandle};
use std::ops::Deref;

pub struct Machine<'s, 'a, P, Rt: RuntimeType> {
    lcs: template::Lcs<'s, Object, Pointer<P>, ()>,
    device: Device,
    allocator: &'a mut phase1::LcsAllocator<P>,
    inputed_objects: ObjectTable<()>,
    as_is_inputed_objects: ObjectTable<()>,
    _phantom: PhantomData<Rt>,
}

struct SimpleAllocHandle<'m, P>(&'m mut Vec<ResidentalT<Object, Pointer<P>>>);

impl<'m, P> AdEmitter<P> for SimpleAllocHandle<'m, P> {
    fn allocate(&mut self, t: Object, _device: Device, pointer: Pointer<P>) {
        self.0.push(ResidentalT::new(t, pointer))
    }

    fn deallocate(&mut self, _t: Object, _device: Device, _pointer: Pointer<P>) {
        panic!("no deallocation should happen during allocating space for loads")
    }
}

struct ComputeAdHandle<'m, 's, P>(&'m mut template::Lcs<'s, Object, Pointer<P>, ()>);

impl<'m, 's, P> AdEmitter<P> for ComputeAdHandle<'m, 's, P> {
    fn allocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
        self.0.compute.emit(Ad::Allocate(t, device, pointer))
    }

    fn deallocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
        self.0.compute.emit(Ad::Deallocate(t, device, pointer))
    }
}

struct PostStoreAdHandle<'m, P>(&'m mut Vec<ResidentalT<Object, Pointer<P>>>);

impl<'m, P> AdEmitter<P> for PostStoreAdHandle<'m, P> {
    fn allocate(&mut self, _t: Object, _device: Device, _pointer: Pointer<P>) {
        panic!("no allocation expcted during post-store");
    }

    fn deallocate(&mut self, t: Object, _device: Device, pointer: Pointer<P>) {
        self.0.push(ResidentalT::new(t, pointer))
    }
}

pub struct MachineAt<'m, 's, 'a, P, Rt: RuntimeType> {
    m: &'m mut Machine<'s, 'a, P, Rt>,
    pc: Index,
}

impl<'s, 'a, P, Rt: RuntimeType> Machine<'s, 'a, P, Rt>
where
    P: UsizeId + 'static,
{
    pub fn new(
        device: Device,
        allocator: &'a mut phase1::LcsAllocator<P>,
        inputed_objects: impl Iterator<Item = Object>,
        as_is_inputed_objects: impl Iterator<Item = Object>,
    ) -> Self {
        let inputed_objects: ObjectTable<()> = inputed_objects.collect();
        let mut lcs = template::Lcs::empty(device);

        for (object, _) in inputed_objects.iter() {
            let size = object.size::<Rt>();
            let p = allocator
                .allocate(size, &object, 0, &mut DummyAdHandle::new())
                .unwrap_or_else(|e| {
                    panic!("allocate space for loading {:?} failed: {:?}", &object, e)
                });

            lcs.load.push((
                ResidentalT::new(object.clone(), p),
                ResidentalT::with_no_pointer(object.clone().whole_poly().whole(device)),
            ));
        }

        Machine {
            lcs,
            device,
            allocator,
            inputed_objects,
            as_is_inputed_objects: as_is_inputed_objects.collect(),
            _phantom: PhantomData,
        }
    }

    pub fn at<'m>(&'m mut self, pc: Index) -> MachineAt<'m, 's, 'a, P, Rt> {
        MachineAt { m: self, pc }
    }
}

impl<'s, 'a, P, Rt: RuntimeType> Machine<'s, 'a, P, Rt>
where
    P: UsizeId + 'static,
{
    /// Store `object`.
    ///
    /// If `object` is loaded, to ensure correctness of double buffering,
    /// we first clone `object` to a temporary object during computation then store the temporary object.
    pub fn store(&mut self, object: &Object, objects: &mut ObjectDefs) -> Result<(), OomError> {
        let src_pointer = self
            .allocator
            .access(object, Epsilon::Top)
            .unwrap_or_else(|| panic!("cannot find object {:?} for output on allocator", object));

        let stored_ro = if self.inputed_objects.contains(object) {
            let tmp_object = Object::new(
                objects.push(objects[object.id()].clone()),
                object.node().clone(),
            );
            let tmp_pointer = self.allocator.allocate(
                tmp_object.size::<Rt>(),
                &tmp_object,
                usize::MAX,
                &mut ComputeAdHandle(&mut self.lcs),
            )?;
            self.lcs.compute.emit(SubgraphOperation::Clone(
                ResidentalT::new(tmp_object.clone(), Some(tmp_pointer)),
                ResidentalT::new(object.clone(), Some(tmp_pointer)),
                None,
            ));

            ResidentalT::new(tmp_object, tmp_pointer)
        } else {
            ResidentalT::new(object.clone(), src_pointer)
        };

        self.lcs.store.push((
            ResidentalT::with_no_pointer(object.clone().whole_poly().whole(self.device)),
            stored_ro,
        ));

        Ok(())
    }

    pub fn write_back_intermediate(&mut self, dst: Object, src: Object) {
        let src_pointer = self
            .allocator
            .access(&src, Epsilon::Top)
            .unwrap_or_else(|| panic!("cannot find source object {:?} on allocator", src));
        let dst_pointer = self
            .allocator
            .access(&dst, Epsilon::Top)
            .unwrap_or_else(|| panic!("cannot find desitination object {:?} on allocator", src));

        self.lcs.save_itms.push((
            ResidentalT::new(dst, dst_pointer),
            ResidentalT::new(src, src_pointer),
        ));
    }
}

impl<'m, 's, 'a, P, Rt: RuntimeType> std::ops::Deref for MachineAt<'m, 's, 'a, P, Rt> {
    type Target = Machine<'s, 'a, P, Rt>;
    fn deref(&self) -> &Self::Target {
        &self.m
    }
}

impl<'m, 's, 'a, P, Rt: RuntimeType> std::ops::DerefMut for MachineAt<'m, 's, 'a, P, Rt> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.m
    }
}

impl<'m, 's, 'a, P, Rt: RuntimeType> MachineAt<'m, 's, 'a, P, Rt>
where
    P: UsizeId + 'static,
{
    pub fn emit(&mut self, r: impl Into<phase1::SubgraphOperation<'s, Pointer<P>>>) {
        self.lcs.compute.emit(r);
    }

    fn pc_usize(&self) -> usize {
        usize::from(self.pc)
    }

    /// Allocate space on device for `object`, for computation, if not already allocated.
    /// Intermediates are always already allocated.
    pub fn allocate_for_compute(&mut self, object: &Object) -> Result<Pointer<P>, OomError> {
        let size = object.size::<Rt>();
        let pc = self.pc_usize();

        if let Some(p) = self.allocator.access(object, Epsilon::Top) {
            Ok(p)
        } else {
            self.m
                .allocator
                .allocate(size, object, pc, &mut ComputeAdHandle(&mut self.m.lcs))
        }
    }

    pub fn reuse(&mut self, to_object: &Object, old_object: &Object) {
        self.allocator.reuse(to_object, old_object, Epsilon::Top);
    }

    pub fn access(&mut self, object: &Object) -> Option<Pointer<P>> {
        if self.as_is_inputed_objects.contains(object) {
            None
        } else {
            Some(
                self.allocator
                    .access(object, Epsilon::Top)
                    .unwrap_or_else(|| panic!("cannot find object {:?}", object)),
            )
        }
    }

    pub fn deallocate_for_compute(&mut self, object: &Object) {
        let pc = self.pc_usize();
        self.m.allocator.deallocate(
            object,
            pc,
            Epsilon::Top,
            &mut ComputeAdHandle(&mut self.m.lcs),
        );
    }

    pub fn deallocate_for_post_store(&mut self, object: &Object) {
        let pc = self.pc_usize();
        self.m.allocator.deallocate(
            object,
            pc,
            Epsilon::Top,
            &mut PostStoreAdHandle(&mut self.m.lcs.post_store),
        );
    }

    pub fn finish<'b>(
        mut self,
        used_objects: impl Iterator<Item = &'b Object> + 'b,
        next_uses: &mut NextUses,
    ) {
        for object in used_objects {
            next_uses.mark_use(object, self.device, self.pc);

            if next_uses.will_not_be_used_on(object, std::iter::once(self.device), self.pc) {
                if self.m.allocator.inner_contains(object) {
                    self.deallocate_for_compute(object);
                }
            }
        }
    }
}

/// Plan memory for subgraph running on single device `source.device`.
/// Here only single buffer is considered.
/// Later we must clone the `allocator` for the second buffer,
/// and realize into two versions of LCS's.
///
/// Also, if we need running multiple instances of this subgraph at the same time
/// to fully utilize multi-disk setup's throughput,
/// we can make more clones of the `allocator`.
///
/// This function returns a [`allocators::LinearAllocator`] wrapped with [`allocators::SmithereenWrapper`]
/// for prologue and body respectively.
/// The wrappers ought to be identical, containing the same copy of intermediates,
/// while the linear allocators contain objects during computation.
///
/// The intermediates (whichever real chunk of polynomial it refers to at runtime) are
/// identified by loop body's input intermediates, that is, `source.body.itms_in`.
///
/// The loads and stores of the producing LCS's point to `source.device` but not their actual
/// residence.
/// Remember to correct those fields.
pub fn plan_subgraph<'s, 'a, P, Rt: RuntimeType>(
    source: phase0::Subgraph<'s, Pointer<P>>,
    device_capacity: u64,
    objects: &mut ObjectDefs,
    hd_info: &HardwareInfo,
) -> Result<phase1::Subgraph<'s, P>, OomError>
where
    P: UsizeId + 'static,
{
    let device = source.device;

    let compute_allocator = allocators::LinearAllocator::new(device_capacity, 0, device);
    let mut itms_allocator = allocators::SmithereenWrapper::new(
        compute_allocator,
        /* We are not checking for OOM here */ 2u64.pow(40),
        0,
        smithereens_wrapper::AllAcceptFilter,
    );
    let mut itms_allocs = Vec::new();

    // Check that `prologue.itms_in` is empty
    if !source.prologue.itms_in.is_empty() {
        panic!("in-intermediates of prologue is not empty");
    }

    // Check that `prologue.itms_out` matches `body.itms_in`
    let prologue_itms_out = source.prologue.itms_out.iter().collect::<BTreeSet<_>>();
    let body_itms_in = source.body.itms_in.iter().collect::<BTreeSet<_>>();
    if prologue_itms_out != body_itms_in {
        panic!(
            "prologue out-intermediates does not match body in-intermediats, {:?} != {:?}",
            prologue_itms_out, body_itms_in
        );
    }

    source.body.itms_in.iter().for_each(|o| {
        itms_allocator
            .allocate(
                o.size::<Rt>(),
                o,
                0,
                &mut SimpleAllocHandle(&mut itms_allocs),
            )
            .unwrap();
    });

    let lcs_prologue = plan_paragraph::<_, Rt>(
        source.prologue,
        device,
        itms_allocator
            .clone()
            .with_filter(smithereens_wrapper::AllRejectFilter),
        objects,
        hd_info,
    )?;

    let lcs_body = plan_paragraph::<_, Rt>(
        source.body,
        device,
        itms_allocator
            .clone()
            .with_filter(smithereens_wrapper::AllRejectFilter),
        objects,
        hd_info,
    )?;

    let subgraph = phase1::Subgraph {
        chunker: source.chunker,
        uses: source.uses,
        on_device_uses: source.on_device_uses,
        prologue: lcs_prologue,
        body: lcs_body,
        device: source.device,
        itms: itms_allocs,
        output: source.output,
        vid: source.vid,
    };

    Ok(subgraph)
}

fn plan_paragraph<'s, 'a, P, Rt: RuntimeType>(
    source: phase0::Paragraph<'s>,
    device: Device,
    mut allocator: phase1::LcsAllocator<P>,
    objects: &mut ObjectDefs,
    hd_info: &HardwareInfo,
) -> Result<phase1::Lcs<'s, P>, OomError>
where
    P: UsizeId + 'static,
{
    let mut machine = Machine::<P, Rt>::new(
        device,
        &mut allocator,
        source.inputs.iter().cloned(),
        source.as_is_inputs.into_iter(),
    );

    // Let's use `source.ops.next_index()` to indicate the moment when store occurs
    let store_index = source.ops.next_index();

    let mut read_by = liveness::ReadBy::analyze(
        source
            .ops
            .iter()
            .map(|(idx, op)| (idx, SubgraphOperationOn(op, device))),
        hd_info,
    );
    source.outputs.iter().for_each(|object| {
        read_by.add_use(store_index, object.clone(), device);
    });

    // We also need extend liveness of loaded objects until end of paragraph
    source.inputs.iter().for_each(|object| {
        read_by.add_use(store_index, object.clone(), device);
    });

    let write_at = liveness::WriteAt::analyze(
        source
            .ops
            .iter()
            .map(|(idx, op)| (idx, SubgraphOperationOn(op, device))),
        hd_info,
    );
    let mut next_uses = liveness::NextUses::new(read_by, write_at);

    for (index, op) in source.ops.into_iter() {
        let mut machine = machine.at(index);

        use SubgraphOperation::*;

        let used_objects = SubgraphOperationOn(&op, device)
            .object_at_reads()
            .map(|(o, ..)| o.clone())
            .collect::<Vec<_>>();

        match op {
            Type2(cop) => {
                let input_objects = cop
                    .node
                    .uses_ref()
                    .map(|vi| vi.iter().map(|ira| ira.object().clone()))
                    .flatten()
                    .collect::<ObjectTable<()>>();

                let node = cop.node.relabeled(
                    |ri| {
                        ri.map(|ira| {
                            let pointer = machine.access(ira.object());
                            ira.map_v(|ra| ra.set_pointer(pointer))
                        })
                    },
                    |_| panic!("input operation not expected here"),
                );

                // Allocate outputs and temporaries.
                // We only care about objects on planning devices and unplanned devices.
                // - If the output took place of some input object, then later reuse space of the input object
                // - If the output object is the same as input, do nothing
                // - Otherwise, allocate new space
                for ora in cop.output.iter() {
                    if let Some(inplace_of) = ora.inplace_of() {
                        machine.reuse(ora.object(), inplace_of.object());
                    } else if input_objects.contains(&ora.object()) {
                        // do nothing
                    } else {
                        machine.allocate_for_compute(ora.object())?;
                    }
                }

                if let Some(t) = &cop.temporaries {
                    machine.allocate_for_compute(t.buf.object())?;
                }

                // Obtain pointer for outputs and temporaries.
                let output = cop.output.map(|ora| {
                    let pointer = machine
                        .access(ora.object())
                        .expect("to find just allocated output");
                    ora.map_v(|ra| ra.set_pointer(Some(pointer)))
                        .map_i(|ra| ra.map(|ra| ra.set_pointer(Some(pointer))))
                });

                let temps = cop.temporaries.map(|t| {
                    let pointer = machine
                        .access(t.buf.object())
                        .expect("to find just allocated temporary");

                    Temporaries {
                        buf: t.buf.set_pointer(Some(pointer)),
                        slices: t.slices,
                    }
                });

                machine.emit(ChunkedOp::new(
                    cop.vid,
                    output,
                    temps.clone(),
                    node,
                    cop.src,
                ));

                // Deallocate temporary spaces right away, for clearity
                temps.inspect(|t| {
                    machine.deallocate_for_compute(t.buf.object());
                });
            }
            Concat(dst, [former, latter]) => {
                let dst_pointer = machine.allocate_for_compute(dst.deref())?;
                let former_pointer = machine
                    .access(former.deref())
                    .expect("concat operands not as-is");
                let latter_pointer = machine
                    .access(latter.deref())
                    .expect("concat operands not as-is");
                machine.emit(Concat(
                    dst.set_pointer(dst_pointer),
                    [
                        former.set_pointer(former_pointer),
                        latter.set_pointer(latter_pointer),
                    ],
                ));
            }
            Clone(dst, src, slice) => {
                let dst_pointer = machine.allocate_for_compute(dst.deref())?;
                let src_pointer = machine.access(src.deref());
                machine.emit(Clone(
                    dst.set_pointer(Some(dst_pointer)),
                    src.set_pointer(src_pointer),
                    slice,
                ))
            }
            Ad(..) => panic!("allocation/deallocation not expected in phase0 sequence"),
        }

        machine.finish(used_objects.iter(), &mut next_uses);
    }

    for (dst, src) in source.itms_write_back {
        machine.write_back_intermediate(dst, src);
    }

    for output in source.outputs.iter() {
        machine.store(output, objects)?;
    }

    {
        let mut machine = machine.at(store_index);
        for obj in source.inputs.iter().chain(source.outputs.iter()) {
            // If it is unlucky that we must input whole polynomial for prologue,
            // we cannot free it after prologue since the polynomial is not owned
            if !obj.is_whole_poly() {
                machine.deallocate_for_post_store(obj);
            }
        }
    }

    let lcs = machine.lcs;

    Ok(lcs.map_a(|_| allocator))
}

#[derive(Clone, Hash, Debug)]
pub struct MemoryScheme {
    intermediates_size: u64,
    prologue_size: u64,
    body_size: u64,
}

#[derive(Clone, Hash, Debug)]
pub struct MemoryAllocation<P> {
    pub object: Object,
    pub pointer: Option<Pointer<P>>,
    pub scheme: MemoryScheme,
}

impl<P> MemoryAllocation<P> {
    pub fn with_pointer(self, pointer: Pointer<P>) -> Self {
        MemoryAllocation {
            pointer: Some(pointer),
            ..self
        }
    }
}

impl MemoryScheme {
    pub fn total(&self) -> u64 {
        self.intermediates_size + self.body_size + self.body_size.max(self.prologue_size)
    }

    pub fn prologue_buffer_offset(&self) -> u64 {
        0
    }

    pub fn body_buffer1_offset(&self) -> u64 {
        self.body_size.max(self.prologue_size)
    }

    pub fn body_buffer2_offset(&self) -> u64 {
        0
    }

    pub fn intermediates_offset(&self) -> u64 {
        self.body_buffer1_offset() + self.body_size
    }
}

pub struct SubgraphStatistics {
    scheme: MemoryScheme,
}

impl std::fmt::Debug for SubgraphStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::utils::human_readable_size;

        f.debug_struct("Statistics")
            .field(
                "intermediates_size",
                &human_readable_size(self.scheme.intermediates_size),
            )
            .field(
                "prologue_size",
                &human_readable_size(self.scheme.prologue_size),
            )
            .field("body_size", &human_readable_size(self.scheme.body_size))
            .field("total_size", &human_readable_size(self.scheme.total()))
            .finish()
    }
}

#[derive(Debug)]
pub struct Statistics {
    subgraphs: BTreeMap<VertexId, SubgraphStatistics>,
}

pub fn fit_subgraph<'s, 'a, P, Rt: RuntimeType>(
    source: phase0::Subgraph<'s, Pointer<P>>,
    device_capacity: u64,
    objects: &mut ObjectDefs,
    hd_info: &HardwareInfo,
) -> Result<phase1::Subgraph<'s, P>, OomError>
where
    P: UsizeId + 'static,
{
    loop {
        match plan_subgraph::<_, Rt>(source.clone(), device_capacity, objects, hd_info) {
            Ok(sg) => {
                let _scheme = MemoryScheme {
                    intermediates_size: sg.prologue.allocator.wrapper_used_bytes(),
                    prologue_size: sg.prologue.allocator.unwrap_ref().used_bytes(),
                    body_size: sg.body.allocator.unwrap_ref().used_bytes(),
                };

                return Ok(sg);
            }
            Err(..) => {
                todo!()
            }
        }
    }
}

/// Run memory planning for each subgraph.
pub fn plan_subgraphs<'s, P, Rt: RuntimeType>(
    ops: phase0::OperationSeq<'s, P>,
    objects: &mut ObjectDefs,
    device_capacity: impl Fn(Device) -> u64,
    hd_info: &HardwareInfo,
) -> Result<(phase1::OperationSeq<'s, P>, Statistics), Error<'s>>
where
    P: UsizeId + 'static,
{
    let mut st = Statistics {
        subgraphs: BTreeMap::new(),
    };

    let r = ops.try_map(|p0op| {
        use Operation::*;
        let r = match p0op {
            Type2(t2op) => Type2(t2op),
            Sub(sg) => {
                let vid = sg.vid;
                let device = sg.device;

                let sg = plan_subgraph::<_, Rt>(sg, device_capacity(device), objects, hd_info)
                    .map_err(|e| Error::subgraph_oom(e, vid))?;

                let scheme = MemoryScheme {
                    intermediates_size: sg.prologue.allocator.wrapper_used_bytes(),
                    prologue_size: sg.prologue.allocator.unwrap_ref().used_bytes(),
                    body_size: sg.body.allocator.unwrap_ref().used_bytes(),
                };

                st.subgraphs.insert(
                    vid,
                    SubgraphStatistics {
                        scheme: scheme.clone(),
                    },
                );

                let object = Object::new(
                    objects.push(TopNode::GpuBuffer(
                        scheme.total() as usize,
                        Slice::new(0, scheme.total()),
                    )),
                    ValueNode::GpuBuffer(scheme.total() as usize, Slice::new(0, scheme.total())),
                );

                Sub(phase1::ResidentalSubgraph::new(
                    sg,
                    MemoryAllocation {
                        object,
                        pointer: None,
                        scheme,
                    },
                ))
            }
            Clone(dst, src) => Clone(dst, src),
            Transfer(dst, d, src) => Transfer(dst, d, src),
            Ad(ad) => Ad(ad),
        };

        Ok(r)
    })?;

    Ok((r, st))
}
