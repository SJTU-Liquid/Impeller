use super::super::prelude::*;
use crate::utils::human_readable_size;
use type3::InstructionNode;
use zkpoly_common::heap::RoHeap;

pub type RegisterValue<P> = ResidentalValue<Object, P>;
pub type AtomRegisterValue<P> = ResidentalAtom<Object, P>;

#[derive(Debug, Clone)]
struct DepthPriorityHeap(BTreeSet<(usize, RegisterId)>);

impl Default for DepthPriorityHeap {
    fn default() -> Self {
        Self(BTreeSet::new())
    }
}

impl DepthPriorityHeap {
    pub fn insert(&mut self, reg: RegisterId, depth: usize) {
        self.0.insert((depth, reg));
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = RegisterId> + 'a {
        self.0.iter().map(|(_, x)| *x)
    }
}

#[derive(Clone, Debug)]
enum RegisterStatus<P> {
    Defined(RegisterValue<P>),
    Freed(RegisterValue<P>),
}

impl<P> RegisterStatus<P> {
    pub fn is_defined_and(&self, f: impl FnOnce(&RegisterValue<P>) -> bool) -> bool {
        match self {
            RegisterStatus::Defined(rv) => f(rv),
            _ => false,
        }
    }

    pub fn is_defined(&self) -> bool {
        match self {
            RegisterStatus::Defined(..) => true,
            _ => false,
        }
    }

    pub fn inner(&self) -> &RegisterValue<P> {
        use RegisterStatus::*;
        match self {
            Defined(v) | Freed(v) => v,
        }
    }
}

/// Keep track of what registers point to
#[derive(Clone, Debug)]
pub struct RegBooking<P> {
    /// [`RegisterId`]'s are allocated from this mapping.
    values: Heap<RegisterId, RegisterStatus<P>>,
    /// Mapping from each (pointer, device) pair to registers containing the (pointer, device).
    p2reg: BTreeMap<(P, Device), DepthPriorityHeap>,
}

impl<P> RegBooking<P> {
    pub fn empty() -> Self {
        Self {
            values: Heap::new(),
            p2reg: BTreeMap::new(),
        }
    }
}

impl<P> RegBooking<P>
where
    P: Ord + Eq + Copy + std::fmt::Debug,
{
    /// Define a register that has the designated [`RegisterValue`]
    pub fn new_reg(&mut self, rv: RegisterValue<P>) -> RegisterId {
        let depth = rv.depth();
        let reg = self.values.push(RegisterStatus::Defined(rv.clone()));
        rv.iter().for_each(|ra| {
            self.p2reg
                .entry((*ra.pointer(), ra.device()))
                .or_default()
                .insert(reg, depth);
        });
        reg
    }

    /// Define a register that has the designated [`RegisterValue`]
    pub fn new_atom_reg(&mut self, ra: AtomRegisterValue<P>) -> RegisterId {
        self.new_reg(Tree::Single(ra))
    }

    /// Mark an register as freed. Panics if the register is already freed.
    pub fn free_reg(&mut self, reg: RegisterId) {
        match &self.values[reg] {
            RegisterStatus::Defined(rv) => {
                self.values[reg] = RegisterStatus::Freed(rv.clone());
            }
            RegisterStatus::Freed(..) => panic!("register {:?} is already freed", reg),
        }
    }

    /// Look for a register that has the designated [`RegisterValue`]
    pub fn lookup_single_reg(&self, ra: &AtomRegisterValue<P>) -> Option<RegisterId> {
        self.p2reg
            .get(&(*ra.pointer(), ra.device()))
            .map(|r| {
                r.iter().find(|reg| {
                    self.values[*reg].is_defined_and(|rv| rv.is_single_and(|reg_ra| reg_ra == ra))
                })
            })
            .flatten()
    }

    /// Look for a register that points to `p` on `device`
    pub fn regs_pointing_to<'a>(
        &'a self,
        object: &'a Object,
        p: P,
        device: Device,
    ) -> impl Iterator<Item = RegisterId> + 'a {
        self.p2reg
            .get(&(p, device))
            .map(move |r| {
                r.iter().filter(move |reg| {
                    self.values[*reg].is_defined_and(move |rv| {
                        rv.is_single_and(move |reg_ra| {
                            reg_ra.pointer() == &p
                                && reg_ra.device() == device
                                && reg_ra.object() == object
                        })
                    })
                })
            })
            .into_iter()
            .flatten()
    }

    /// Look for a defined register that contains an atom points to `p` on `device`
    pub fn regs_containing_pointing_to<'a>(
        &'a self,
        p: P,
        device: Device,
    ) -> impl Iterator<Item = RegisterId> + 'a {
        self.p2reg
            .get(&(p, device))
            .map(|h| h.iter())
            .into_iter()
            .flatten()
            .filter(|r| self.values[*r].is_defined())
    }

    /// Get the value of a register. Panics if the register is freed.
    pub fn reg_value(&self, reg: RegisterId) -> &RegisterValue<P> {
        match &self.values[reg] {
            RegisterStatus::Defined(rv) => rv,
            _ => panic!("register {:?} is already freed", reg),
        }
    }

    pub fn export_reg_values(&self) -> RoHeap<RegisterId, RegisterValue<P>> {
        self.values
            .map_by_ref(&mut |_, v| v.inner().clone())
            .freeze()
            .0
    }

    pub fn reg_id_allocator(self) -> IdAllocator<RegisterId> {
        self.values.freeze().1
    }

    /// Mark all registers pointing to `object`, `pointer`, `device` as free,
    /// but does not emit any instruction.
    pub fn free_reg_pointing_to(&mut self, object: &Object, pointer: P, device: Device) {
        let regs = self
            .regs_pointing_to(object, pointer, device)
            .collect::<Vec<_>>();

        regs.into_iter().for_each(|r| self.free_reg(r));
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferFrags {
    /// Mapping from transferred size to counts
    count: BTreeMap<u64, usize>,
}

impl TransferFrags {
    pub fn new() -> Self {
        Self {
            count: BTreeMap::new(),
        }
    }

    pub fn add(&mut self, bytes: u64) {
        *self.count.entry(bytes).or_default() += 1;
    }

    pub fn get(&self, bytes: u64) -> usize {
        self.count.get(&bytes).cloned().unwrap_or_default()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (u64, usize)> + 'a {
        self.count.iter().map(|(s, c)| (*s, *c))
    }

    pub fn merge(&mut self, other: &TransferFrags) {
        for (s, c) in other.iter() {
            *self.count.entry(s).or_default() += c;
        }
    }

    pub fn mult(&mut self, n: usize) {
        for (_, c) in self.count.iter_mut() {
            *c *= n;
        }
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    /// The number of bytes transferred from device to device
    transfer_sizes: DeviceSpecific<DeviceSpecific<u64>>,
    transfered_frags: DeviceSpecific<DeviceSpecific<TransferFrags>>,
}

impl Statistics {
    pub fn transferred_bytes(&self, from: Device, to: Device) -> u64 {
        *self.transfer_sizes.get(from).get(to)
    }

    pub fn transferred_frag(&self, from: Device, to: Device) -> &TransferFrags {
        self.transfered_frags.get(from).get(to)
    }

    pub fn add_transfer(&mut self, from: Device, to: Device, bytes: u64) {
        *self.transfer_sizes.get_mut(from).get_mut(to) += bytes;
        self.transfered_frags.get_mut(from).get_mut(to).add(bytes);
    }

    pub fn merge(&mut self, other: &Statistics) {
        for (from, inner) in other.transfer_sizes.iter() {
            for (to, size) in inner.iter() {
                *self.transfer_sizes.get_mut(from).get_mut(to) += *size;
            }
        }

        for (from, inner) in other.transfered_frags.iter() {
            for (to, size) in inner.iter() {
                self.transfered_frags.get_mut(from).get_mut(to).merge(size);
            }
        }
    }

    pub fn multiple(mut self, n: usize) -> Self {
        for (_, inner) in self.transfer_sizes.iter_mut() {
            for (_, s) in inner.iter_mut() {
                *s *= n as u64;
            }
        }

        for (_, inner) in self.transfered_frags.iter_mut() {
            for (_, s) in inner.iter_mut() {
                s.mult(n);
            }
        }

        self
    }
}

impl Statistics {
    pub fn zero(n_gpus: usize) -> Self {
        Self {
            transfer_sizes: DeviceSpecific::new(n_gpus, |_| DeviceSpecific::new(n_gpus, |_| 0)),
            transfered_frags: DeviceSpecific::new(n_gpus, |_| {
                DeviceSpecific::new(n_gpus, |_| TransferFrags::new())
            }),
        }
    }

    pub fn pretty_print(
        &self,
        writer: &mut impl std::fmt::Write,
        indent: usize,
    ) -> std::fmt::Result {
        for (from, from_sizes) in self.transfer_sizes.iter() {
            for (to, size) in from_sizes.iter() {
                writeln!(
                    writer,
                    "{}{:?} -> {:?}: {}",
                    " ".repeat(indent),
                    from,
                    to,
                    human_readable_size(*size)
                )?;
            }
        }

        for (from, from_sizes) in self.transfered_frags.iter() {
            for (to, size) in from_sizes.iter() {
                writeln!(
                    writer,
                    "{}{:?} -> {:?}: {:?}",
                    " ".repeat(indent),
                    from,
                    to,
                    &size.count
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug, Clone)]
struct InstructionGroup<'s> {
    pre: Vec<type3::RegisterManipulation>,
    op: Option<(type3::InstructionNode<'s>, type3::Begin)>,
    post: Vec<type3::RegisterManipulation>,
}

impl<'s> InstructionGroup<'s> {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.pre.is_empty() && self.op.is_none() && self.post.is_empty()
    }
}

pub struct RegGrabber<F, P> {
    manips: F,
    _phantom: PhantomData<P>,
}

impl<F, P> RegGrabber<F, P>
where
    F: FnMut(type3::RegisterManipulation),
    P: Ord + Eq + Copy + std::fmt::Debug,
{
    pub fn new(f: F) -> Self {
        Self {
            manips: f,
            _phantom: PhantomData,
        }
    }

    fn unpack_tuple_for_pointing_to(
        &mut self,
        ra: &AtomRegisterValue<P>,
        reg_booking: &mut RegBooking<P>,
    ) -> Option<RegisterId> {
        let candidates = reg_booking.regs_containing_pointing_to(*ra.pointer(), ra.device());

        let (tuple_reg, indices) = candidates
            .filter_map(|reg| {
                let rv = reg_booking.reg_value(reg);
                rv.iter_with_indices()
                    .filter(|(_, reg_ra)| {
                        reg_ra.pointer() == ra.pointer()
                            && reg_ra.device() == ra.device()
                            && reg_ra.object() == ra.object()
                    })
                    .next()
                    .map(|(indices, _)| (reg, indices))
            })
            .next()?;

        let reg = indices.into_iter().fold(tuple_reg, |tuple_reg, index| {
            let rv = reg_booking.reg_value(tuple_reg);
            let unpacked_rv = rv.unwrap_tuple().get(index).unwrap();
            let unpacked_reg = reg_booking.new_reg(unpacked_rv.clone());
            (self.manips)(type3::RegisterManipulation::TupleGet {
                id: unpacked_reg,
                target: tuple_reg,
                index,
            });
            unpacked_reg
        });

        Some(reg)
    }

    pub fn reg_for_atom(
        &mut self,
        ra: &AtomRegisterValue<P>,
        reg_booking: &mut RegBooking<P>,
    ) -> RegisterId {
        if let Some(reg) = reg_booking.lookup_single_reg(ra) {
            reg
        } else {
            let reg = reg_booking
                .regs_pointing_to(ra.object(), *ra.pointer(), ra.device())
                .next();
            let reg = reg.unwrap_or_else(|| {
                self.unpack_tuple_for_pointing_to(ra, reg_booking)
                    .unwrap_or_else(|| panic!("cannot obtain register for atom {:?}", ra,))
            });

            let reg_ra = reg_booking.reg_value(reg).unwrap_leaf();

            let (deg, slice) = ra.node().unwrap_poly();
            let (reg_deg, reg_slice) = reg_ra.node().unwrap_poly();
            if deg != reg_deg {
                panic!(
                    "degree of register atom does not match provided : provided {:?}, register has {:?}",
                    ra, reg_ra
                );
            }

            if slice == reg_slice {
                reg
            } else {
                let new_reg = reg_booking.new_atom_reg(ra.clone());

                (self.manips)(type3::RegisterManipulation::SetPolyMeta {
                    id: new_reg,
                    from: reg,
                    offset: slice.begin() as usize,
                    len: slice.len() as usize,
                });

                new_reg
            }
        }
    }

    /// Look for a register that has the designated [`RegisterValue`] and is defined.
    ///
    /// If the value is a [`Tree::Leaf`] and no such register is found, this function attempt to make one by slicing an existing one;
    /// If the value is a [`Tree::Leaf`] and no register pointing to the object is found, this function panics;
    /// If the value is a [`Tree::Tuple`] this function assembles one by recursively obtaining its components;
    pub fn reg_for(
        &mut self,
        rv: &RegisterValue<P>,
        reg_booking: &mut RegBooking<P>,
    ) -> RegisterId {
        match rv {
            Tree::Single(ra) => self.reg_for_atom(ra, reg_booking),
            Tree::Tuple(components) => {
                let regs = components
                    .iter()
                    .map(|rv| self.reg_for(rv, reg_booking))
                    .collect();

                let new_reg = reg_booking.new_reg(rv.clone());
                (self.manips)(type3::RegisterManipulation::Tuple {
                    id: new_reg,
                    oprands: regs,
                });
                new_reg
            }
        }
    }
}

/// # Invariant
/// Each register is defined to point to a object only once.
/// That is, an register cannot be pointed to another object after it is allocated.
pub struct Machine<'r, X, P, T, Rt: RuntimeType> {
    instructions: type3::Seq<X, T>,
    reg_booking: &'r mut RegBooking<P>,
    statistics: Statistics,
    inst_indices_for_op: Heap<Index, Vec<X>>,
    _phantom: PhantomData<Rt>,
}

impl<'r, X, P, T, Rt: RuntimeType> Machine<'r, X, P, T, Rt> {
    pub fn new(n_gpus: usize, reg_booking: &'r mut RegBooking<P>) -> Self {
        Self {
            instructions: type3::Seq::new(),
            reg_booking,
            inst_indices_for_op: Heap::new(),
            statistics: Statistics::zero(n_gpus),
            _phantom: PhantomData,
        }
    }

    pub fn at<'s>(self, op_idx: Index) -> MachineAt<'s, 'r, X, P, T, Rt> {
        MachineAt {
            m: self,
            op: op_idx,
            group: vec![InstructionGroup::new()],
        }
    }

    pub fn at_append<'s>(self) -> MachineAt<'s, 'r, X, P, T, Rt> {
        let next_idx = self.inst_indices_for_op.next_index();
        self.at(next_idx)
    }

    pub fn reg_booking(&self) -> &RegBooking<P> {
        &self.reg_booking
    }

    pub fn reg_booking_mut(&mut self) -> &mut RegBooking<P> {
        self.reg_booking
    }

    /// For now, each operation maps to exactly one instruction
    fn inst_indices_for_op<'a>(&'a self, op_idx: Index) -> impl Iterator<Item = X> + 'a
    where
        X: Clone,
    {
        self.inst_indices_for_op[op_idx].iter().cloned()
    }

    fn set_inst_indices_for_op(&mut self, op_idx: Index, xs: Vec<X>) {
        if op_idx == self.inst_indices_for_op.next_index() {
            self.inst_indices_for_op.push(xs);
        } else {
            panic!("cannot set instruction index for operation {:?} when we have realized {} instructions", op_idx, self.inst_indices_for_op.len());
        }
    }

    pub fn finish(self) -> (type3::Seq<X, T>, Statistics) {
        (self.instructions, self.statistics)
    }
}

pub struct MachineAt<'s, 'r, X, P, T, Rt: RuntimeType> {
    m: Machine<'r, X, P, T, Rt>,
    op: Index,
    group: Vec<InstructionGroup<'s>>,
}

impl<'s, 'r, X, P, T, Rt: RuntimeType> MachineAt<'s, 'r, X, P, T, Rt>
where
    P: Ord + Eq + Copy + std::fmt::Debug,
    X: UsizeId,
{
    fn emit_post(&mut self, inst: type3::RegisterManipulation) {
        self.group.last_mut().unwrap().post.push(inst)
    }

    pub fn emit_pre(&mut self, inst: type3::RegisterManipulation) {
        self.group.last_mut().unwrap().pre.push(inst)
    }

    pub fn emit(&mut self, node: type3::InstructionNode<'s>) {
        self.emit_with_slice_begin(node, None)
    }

    /// Emit an instruction
    pub fn emit_with_slice_begin(&mut self, node: type3::InstructionNode<'s>, begin: type3::Begin) {
        if self.group.last().unwrap().op.is_some() {
            self.group.push(InstructionGroup::new());
        }

        // Log transfer for MSM and Subgraph
        match &node {
            type3::InstructionNode::Type2 {
                vertex:
                    type3::VertexNode::LastSliceable(type2::template::LastSliceableNode::Msm {
                        polys,
                        points,
                        ..
                    }),
                device,
                ..
            } => {
                let device = Device::for_execution_on(device.ensure_gpu());
                let intput_space: usize = polys
                    .iter()
                    .map(|r| {
                        let v = self.m.reg_booking.reg_value(*r);
                        v.unwrap_leaf().object_size::<Rt>()
                    })
                    .chain(points.iter().map(|r| {
                        let v = self.m.reg_booking.reg_value(*r);
                        v.unwrap_leaf().object_size::<Rt>()
                    }))
                    .sum();
                self.m
                    .statistics
                    .add_transfer(Device::Cpu, device, intput_space as u64);
            }
            type3::InstructionNode::Sub(sg) => {
                let device = Device::for_execution_on(sg.device.ensure_gpu());
                sg.prologue1.loads.iter().for_each(
                    |type3::Load {
                         input: type3::SubgraphInput { src, .. },
                         ..
                     }| {
                        let v = self.m.reg_booking.reg_value(*src);
                        let bytes = v.unwrap_leaf().object_size::<Rt>();
                        self.m.statistics.add_transfer(
                            v.unwrap_leaf().device(),
                            device,
                            bytes as u64,
                        );
                    },
                );
                sg.prologue1
                    .store
                    .iter()
                    .for_each(|type3::Store { dst, .. }| {
                        let v = self.m.reg_booking.reg_value(*dst);
                        let bytes = v.unwrap_leaf().object_size::<Rt>();
                        self.m.statistics.add_transfer(
                            device,
                            v.unwrap_leaf().device(),
                            bytes as u64,
                        );
                    });
            }
            _ => {}
        }

        self.group.last_mut().unwrap().op = Some((node, begin));
    }

    pub fn reg_for(&mut self, rv: &RegisterValue<P>) -> RegisterId {
        RegGrabber::new(|m| self.group.last_mut().unwrap().pre.push(m))
            .reg_for(rv, &mut self.m.reg_booking)
    }

    pub fn reg_for_atom(&mut self, ra: &AtomRegisterValue<P>) -> RegisterId {
        RegGrabber::new(|m| self.group.last_mut().unwrap().pre.push(m))
            .reg_for_atom(ra, &mut self.m.reg_booking)
    }

    pub fn transfer(&mut self, ra_to: AtomRegisterValue<P>, ra_from: AtomRegisterValue<P>) {
        let reg_to = self.reg_for_atom(&ra_to);
        let reg_from = self.reg_for_atom(&ra_from);

        self.m.statistics.add_transfer(
            ra_from.device(),
            ra_to.device(),
            ra_from.object().size::<Rt>().into(),
        );

        self.emit(type3::InstructionNode::Transfer {
            id: reg_to,
            from: reg_from,
        });
    }

    pub fn statistics(&mut self) -> &Statistics {
        &self.m.statistics
    }

    pub fn statistics_mut(&mut self) -> &mut Statistics {
        &mut self.m.statistics
    }

    pub fn new_reg(&mut self, rv: RegisterValue<P>) -> RegisterId {
        self.m.reg_booking.new_reg(rv)
    }

    pub fn new_atom_reg(&mut self, rv: AtomRegisterValue<P>) -> RegisterId {
        self.m.reg_booking.new_atom_reg(rv)
    }

    pub fn free_reg(&mut self, reg: RegisterId) {
        self.emit_post(type3::RegisterManipulation::StackFree { id: reg });
        self.m.reg_booking.free_reg(reg);
    }

    pub fn reg_booking_mut(&mut self) -> &mut RegBooking<P> {
        self.m.reg_booking
    }
}

impl<'s, 'r, P, Rt: RuntimeType>
    MachineAt<'s, 'r, type3::LcsPc, P, type3::SubgraphInstruction<'s>, Rt>
{
    pub fn finish(mut self) -> Machine<'r, type3::LcsPc, P, type3::SubgraphInstruction<'s>, Rt> {
        let xs = self
            .group
            .into_iter()
            .filter(|g| !g.is_empty())
            .map(|g| {
                let (op, begin) = g.op.unwrap();
                self.m.instructions.push(
                    type3::SubgraphInstruction {
                        inst: type3::Instruction::new(g.pre, op, g.post),
                        begin,
                    },
                    Vec::new(),
                )
            })
            .collect();
        self.m.set_inst_indices_for_op(self.op, xs);

        self.m
    }
}

impl<'s, 'r, P, Rt: RuntimeType>
    MachineAt<'s, 'r, type3::InstructionIndex, P, type3::Instruction<'s>, Rt>
{
    pub fn finish(
        mut self,
        dependcies: impl Iterator<Item = Index> + Clone,
    ) -> Machine<'r, type3::InstructionIndex, P, type3::Instruction<'s>, Rt> {
        let xs = self
            .group
            .into_iter()
            .filter(|g| !g.is_empty())
            .scan(None, |last_idx, g| {
                let (op, _) = g.op.unwrap();
                let this_idx = self.m.instructions.push(
                    type3::Instruction::new(g.pre, op, g.post),
                    dependcies
                        .clone()
                        .map(|d| self.m.inst_indices_for_op(d))
                        .flatten()
                        .chain(last_idx.as_ref().cloned().into_iter())
                        .collect(),
                );
                *last_idx = Some(this_idx);
                Some(this_idx)
            })
            .collect();
        self.m.set_inst_indices_for_op(self.op, xs);

        self.m
    }
}

pub trait AdEmitter<P> {
    fn allocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        addr: AllocMethod,
    ) -> RegisterId;

    fn new_register(&mut self, object: &Object, pointer: Pointer<P>, device: Device) -> RegisterId;

    fn deallocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        variant: AllocVariant,
    );
}

impl<'s, 'r, X, P, T, Rt: RuntimeType> AdEmitter<P> for MachineAt<'s, 'r, X, Pointer<P>, T, Rt>
where
    P: UsizeId,
    X: UsizeId,
{
    fn allocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        addr: AllocMethod,
    ) -> RegisterId {
        let reg = self.new_atom_reg(ResidentalT::new(object.clone().whole(device), pointer));
        self.emit(type3::InstructionNode::Malloc {
            id: reg,
            addr,
            device,
        });
        reg
    }

    fn new_register(&mut self, object: &Object, pointer: Pointer<P>, device: Device) -> RegisterId {
        self.new_atom_reg(ResidentalT::new(object.clone().whole(device), pointer))
    }

    fn deallocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        variant: AllocVariant,
    ) {
        let regs = self
            .m
            .reg_booking
            .regs_pointing_to(object, pointer, device)
            .collect::<Vec<_>>();

        let &reg = regs.first().unwrap_or_else(|| {
            panic!(
                "cannot find a register pointing to {:?} at {:?} on {:?}",
                object, pointer, device
            )
        });

        self.emit(InstructionNode::Free {
            id: reg,
            device,
            variant,
        });

        regs.into_iter().skip(1).for_each(|r| self.free_reg(r));
    }
}

pub struct SubgraphAdEmitter<'m, 's, 'r, X, P, T, Rt: RuntimeType>(
    pub &'m mut MachineAt<'s, 'r, X, P, T, Rt>,
    pub RegisterId,
);

impl<'m, 's, 'r, X, P, T, Rt: RuntimeType> AdEmitter<P>
    for SubgraphAdEmitter<'m, 's, 'r, X, Pointer<P>, T, Rt>
where
    P: UsizeId,
    X: UsizeId,
{
    fn allocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        addr: AllocMethod,
    ) -> RegisterId {
        let (offset, _) = addr.unwrap_offset();
        let reg = self
            .0
            .new_atom_reg(ResidentalT::new(object.clone().whole(device), pointer));
        self.0.emit(type3::InstructionNode::MallocOffseted {
            id: reg,
            base: self.1,
            offset,
        });
        reg
    }

    fn new_register(&mut self, object: &Object, pointer: Pointer<P>, device: Device) -> RegisterId {
        self.0.new_register(object, pointer, device)
    }

    fn deallocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        _variant: AllocVariant,
    ) {
        self.0
            .reg_booking_mut()
            .free_reg_pointing_to(object, pointer, device);
    }
}

pub struct IntermediatesMachine<'r, P> {
    reg_booking: &'r mut RegBooking<P>,
    insts: Vec<type3::OffstedAllocation>,
    base: RegisterId,
}

impl<'r, P> IntermediatesMachine<'r, P> {
    pub fn new(base: RegisterId, reg_booking: &'r mut RegBooking<P>) -> Self {
        Self {
            reg_booking,
            insts: Vec::new(),
            base,
        }
    }

    pub fn finish(self) -> Vec<type3::OffstedAllocation> {
        self.insts
    }
}

impl<'r, P> AdEmitter<P> for IntermediatesMachine<'r, Pointer<P>>
where
    P: UsizeId,
{
    fn allocate(
        &mut self,
        object: &Object,
        pointer: Pointer<P>,
        device: Device,
        addr: AllocMethod,
    ) -> RegisterId {
        let (offset, _) = addr.unwrap_offset();
        let reg = self
            .reg_booking
            .new_atom_reg(ResidentalT::new(object.clone().whole(device), pointer));
        self.insts.push(type3::OffstedAllocation {
            id: reg,
            base: self.base,
            offset,
        });
        reg
    }

    fn new_register(
        &mut self,
        _object: &Object,
        _pointer: Pointer<P>,
        _device: Device,
    ) -> RegisterId {
        panic!("should not call new_register on IntermediatesMachine")
    }

    fn deallocate(
        &mut self,
        _object: &Object,
        _pointer: Pointer<P>,
        _device: Device,
        _variant: AllocVariant,
    ) {
        panic!("should not call deallocate on IntermediatesMachine")
    }
}
