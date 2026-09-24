use super::super::allocators::SuperAllocator;
use super::super::prelude::*;
use super::criterion;
use std::ops::Deref;

#[derive(Debug)]
pub struct EjectionStatistics {
    eject_from: DeviceSpecific<u64>,
    reclaim_to: DeviceSpecific<u64>,
}

impl EjectionStatistics {
    pub fn new(n_gpus: usize) -> Self {
        Self {
            eject_from: DeviceSpecific::new(n_gpus, |_| 0),
            reclaim_to: DeviceSpecific::new(n_gpus, |_| 0),
        }
    }

    pub fn record_eject(&mut self, device: Device, size: u64) {
        *self.eject_from.get_mut(device) += size;
    }

    pub fn record_reclaim(&mut self, device: Device, size: u64) {
        *self.reclaim_to.get_mut(device) += size;
    }
}

pub struct Machine<'a, R, P, M, Rt: RuntimeType>
where
    M: criterion::Manager,
{
    ops: Seq<R>,
    current_stage: DevicesStage,
    allocators: AllocatorCollection<'a, P, M::C, M::Cf>,
    unused_allocators: AllocatorCollection<'a, P, M::C, M::Cf>,
    criterions: M,
    ejection_statistics: EjectionStatistics,
    _phantom: PhantomData<Rt>,
}

impl<'a, R, P, M, Rt: RuntimeType> Machine<'a, R, P, M, Rt>
where
    M: criterion::Manager,
    R: Constructor<Object, Pointer<P>>,
    P: UsizeId + 'static,
{
    pub fn new(
        current_stage: DevicesStage,
        mut allocators: AllocatorCollection<'a, P, M::C, M::Cf>,
        next_uses: NextUses,
        hd_info: &HardwareInfo,
    ) -> Self {
        use Stage::*;
        let current_stage_allocators = AllocatorCollection::new_owned(
            hd_info.n_gpus(),
            |device| match current_stage[device] {
                Planning => Some(
                    allocators
                        .take(device)
                        .expect("all devices' allocators to be present"),
                ),
                _ => None,
            },
            |device| match current_stage[device] {
                Unplanned => Some(Box::new(SuperAllocator::for_unplanned(device))),
                _ => None,
            },
        );

        Machine {
            criterions: M::new(next_uses, hd_info.n_gpus()),
            ops: Seq::empty(),
            current_stage,
            unused_allocators: allocators,
            allocators: current_stage_allocators,
            ejection_statistics: EjectionStatistics::new(hd_info.n_gpus()),
            _phantom: PhantomData,
        }
    }

    pub fn emit(&mut self, op: impl Into<R>)
    where
        R: std::fmt::Debug,
    {
        let r: R = op.into();
        self.ops.emit(r);
    }

    fn update_criterions(&mut self, pc: Index) {
        for (o, d, c) in self.criterions.scheduled_updates(pc) {
            if self.allocators[d].contains(&o) {
                self.allocators[d].unlock(&o, c);
            }
        }
    }

    pub fn at<'m>(&'m mut self, pc: Index) -> MachineAt<'m, 'a, R, P, M, Rt> {
        self.update_criterions(pc);

        MachineAt {
            m: self,
            pc,
            unlock_queue: BTreeSet::new(),
        }
    }

    pub fn finish(
        self,
    ) -> (
        Seq<R>,
        AllocatorCollection<'a, P, M::C, M::Cf>,
        EjectionStatistics,
    ) {
        let mut allocators = self.unused_allocators;
        let mut current_stage_allocators = self.allocators;

        for d in self.current_stage.planning_devices() {
            allocators.set(d, current_stage_allocators.take(d).unwrap());
        }

        (self.ops, allocators, self.ejection_statistics)
    }
}

pub struct MachineAt<'m, 'a, R, P, M, Rt: RuntimeType>
where
    M: criterion::Manager,
{
    m: &'m mut Machine<'a, R, P, M, Rt>,
    pc: Index,
    unlock_queue: BTreeSet<(Object, Device)>,
}

impl<'m, 'a, R, P, M, Rt: RuntimeType> std::ops::Deref for MachineAt<'m, 'a, R, P, M, Rt>
where
    M: criterion::Manager,
{
    type Target = Machine<'a, R, P, M, Rt>;
    fn deref(&self) -> &Self::Target {
        &self.m
    }
}

impl<'m, 'a, R, P, M, Rt: RuntimeType> std::ops::DerefMut for MachineAt<'m, 'a, R, P, M, Rt>
where
    M: criterion::Manager,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.m
    }
}

impl<'m, 'a, R, P, M, Rt: RuntimeType> MachineAt<'m, 'a, R, P, M, Rt>
where
    M: criterion::Manager,
    R: Constructor<Object, Pointer<P>> + std::fmt::Debug,
    P: UsizeId + 'static,
{
    /// Unlock `object` on `device` during `self.pc`.
    /// Calling this method will not cause next-use of the object to be updated to points later than `self.pc`,
    /// if the object is used during `self.pc`.
    fn unlock(&mut self, object: &Object, device: Device) {
        let pc = self.pc;
        let vc = self.criterions.victim_criterion(object, device, pc);
        self.allocators[device].unlock(object, vc);
    }

    fn pc_usize(&self) -> usize {
        usize::from(self.pc)
    }

    pub fn current_stage(&self) -> &DevicesStage {
        &self.m.current_stage
    }

    /// Allocate space for `object` on `device`.
    ///
    /// # Contracts
    /// - `device` must be planning or unplanned
    pub fn allocate_and_queue_unlock(
        &mut self,
        device: Device,
        object: &Object,
    ) -> Result<Pointer<P>, OomError> {
        let p = self.allocate(device, object)?;

        self.unlock_queue.insert((object.clone(), device));
        Ok(p)
    }

    /// Deallocate space for `object` on `device`
    /// # Contracts
    /// - `device` must be planning or unplanned
    pub fn deallocate(&mut self, device: Device, object: &Object) {
        let pc_us = self.pc_usize();
        let pc = self.pc;
        self.m.allocators[device].deallocate(
            object,
            pc_us,
            self.m.criterions.free_criterion(object, device, pc),
            &mut MachineHandle::new(&mut self.m.ops),
        );

        self.unlock_queue.remove(&(object.clone(), device));
    }

    /// Allocate space for `object` on `device`.
    /// The newly allocated object would be locked, that is, cannot be automatically ejected.
    /// CAUTION do not forget to unlock the allocated object afterwards.
    ///
    /// # Contracts
    /// - `device` must be planning or unplanned
    fn allocate(&mut self, device: Device, object: &Object) -> Result<Pointer<P>, OomError> {
        if self.current_stage[device].is_planned() {
            panic!("cannot allocate space on planned device {:?}", device);
        }

        let size = object.size::<Rt>();

        let pc_usize = self.pc_usize();
        let mut mh = MachineHandle::new(&mut self.m.ops);
        if let Ok(p) = self.m.allocators[device].allocate(size, object, pc_usize, &mut mh) {
            return Ok(p);
        }

        if self.current_stage[device].is_unplanned() {
            panic!("not expected to fail allocation on unplanned device");
        }

        let (commit, victims) = self.allocators[device].takeover_victims(size, object)?;
        let one_victim = victims.get(0).expect("victims to be non-empty").clone();

        for victim_object in victims {
            self.transfer_for_eject(&victim_object, device)?;
            self.unlock_queue.remove(&(victim_object, device));
        }

        let pc_us = self.pc_usize();
        let pc = self.pc;

        let pointer = self.m.allocators[device].accept_commit(
            commit,
            pc_us,
            // We are setting all free criterion of the deallocated addresses a same one
            self.m.criterions.free_criterion(&one_victim, device, pc),
            &mut MachineHandle::new(&mut self.m.ops),
        );

        Ok(pointer)
    }

    /// Send `object` from `from` to `to`.
    /// WARNING does not check whether `object` is already on `to`.
    ///
    /// # Contracts
    /// - `object` must be found on `from`
    /// - `from` and `to` must not be planned
    fn send(&mut self, object: &Object, to: Device, from: Device) -> Result<(), OomError> {
        if self.current_stage[from].is_planned() || self.current_stage[to].is_planned() {
            panic!(
                "cannot eject to or from planned device {:?} -> {:?}",
                from, to
            );
        }

        let _ = self.allocate(to, object)?;
        self.emit_transfer(object, to, from);
        self.unlock(object, to);

        Ok(())
    }

    /// Transfer `object` from `from` to its parent device, if it is not found on other devices.
    ///
    /// # Contracts
    /// - `object` must be on `from`
    /// - `from` must have parent device
    /// - `from` must be planning device
    fn transfer_for_eject(&mut self, object: &Object, from: Device) -> Result<(), OomError> {
        let dst_device = from.parent().expect("from to have parent device");
        if !self.current_stage[from].is_planning() {
            panic!(
                "should only eject from planning device, got {:?} which is {:?}",
                from, self.current_stage[from]
            )
        }
        assert!(!self.current_stage[dst_device].is_planned());

        if self
            .allocators
            .object_available_on1(
                object,
                self.current_stage
                    .planning_unplanned_devices()
                    .filter(|d| *d != from),
            )
            .is_some()
        {
            return Ok(());
        }

        self.ejection_statistics
            .record_eject(from, u64::from(object.size::<Rt>()));
        self.send(object, dst_device, from)
    }

    /// Issue an object transfer between devices.
    ///
    /// # Contracts
    /// - `object` must be already allocated on `from` and `to`
    /// - `from` and `to` must not be planned
    fn emit_transfer(&mut self, object: &Object, to: Device, from: Device) {
        use Stage::*;

        let make_rv =
            |device: Device, pointer: Pointer<P>, current_stage: &DevicesStage| match current_stage
                [device]
            {
                Planned => panic!("should not transfer to or from planned device {:?}", device),
                Unplanned => ResidentalT::with_no_pointer(object.clone()),
                Planning => ResidentalT::with_pointer(object.clone(), pointer),
            };

        if self.current_stage[to].is_unplanned() && self.current_stage[from].is_unplanned() {
            return;
        }

        self.m.ops.emit(R::transfer(
            make_rv(
                to,
                self.m.allocators[to]
                    .access(
                        object,
                        self.m.criterions.victim_criterion(object, to, self.pc),
                    )
                    .unwrap_or_else(|| panic!("expect to have allocated {:?} on {:?}", object, to)),
                &self.m.current_stage,
            ),
            to,
            make_rv(
                from,
                self.m.allocators[from]
                    .access(
                        object,
                        self.m.criterions.victim_criterion(object, from, self.pc),
                    )
                    .unwrap_or_else(|| {
                        panic!("expect to have allocated {:?} on {:?}", object, from)
                    }),
                &self.m.current_stage,
            )
            .map_v(|o| o.whole(from)),
        ));
    }

    /// Get `object` from whichever planning or unplanned device to device `to`, if not yet.
    ///
    /// # Contracts
    /// - `object` must can be found on some device
    /// - `object` must be planning or unplanned
    pub fn claim_and_queue_unlock(
        &mut self,
        object: &Object,
        to: Device,
    ) -> Result<Pointer<P>, OomError> {
        if self.current_stage[to].is_planned() {
            panic!("should not claim to planned device {:?}", to);
        }

        if self.allocators[to].contains(object) {
            self.unlock_queue.insert((object.clone(), to));

            return Ok(self.allocators[to].access(object, M::C::bottom()).unwrap());
        }

        if let Some(src_device) = self
            .allocators
            .object_available_on1(object, self.current_stage.planning_unplanned_devices())
        {
            self.ejection_statistics
                .record_reclaim(to, u64::from(object.size::<Rt>()));
            let _ = self.allocate(to, object)?;
            self.emit_transfer(object, to, src_device);
        } else {
            panic!(
                "cannot claim object {:?} since it is not found on any device",
                object
            )
        }

        self.unlock_queue.insert((object.clone(), to));

        Ok(self.allocators[to].access(object, M::C::bottom()).unwrap())
    }

    /// Reuse space of `old_object` to accomodate `to_object` on `device`
    pub fn reuse(&mut self, device: Device, to_object: &Object, old_object: &Object) {
        let pc = self.pc;
        self.m.allocators[device].reuse(
            to_object,
            old_object,
            self.m.criterions.victim_criterion(to_object, device, pc),
        );

        if self.unlock_queue.remove(&(old_object.clone(), device)) {
            self.unlock_queue.insert((to_object.clone(), device));
        }
    }

    /// Get pointer to `object` on `device`, if found
    pub fn access(&mut self, device: Device, object: &Object) -> Option<Pointer<P>> {
        let pc = self.pc;
        self.m.allocators[device].access(
            object,
            self.m.criterions.victim_criterion(object, device, pc),
        )
    }

    fn get_transfer_source(
        &mut self,
        src: ResidentalAtom<Object, Option<Pointer<P>>>,
    ) -> ResidentalAtom<Object, Option<Pointer<P>>> {
        match src.pointer() {
            Some(_) => {
                if !self.m.current_stage[src.device()].is_planned() {
                    panic!(
                        "source pointer of {:?} is determined but its device is not planned",
                        src
                    );
                }
                src
            }
            None => {
                let src_device = self
                    .allocators
                    .object_available_on1(
                        src.object(),
                        std::iter::once(src.device())
                            .chain(self.current_stage.planning_unplanned_devices()),
                    )
                    .unwrap_or_else(|| panic!("object {:?} is nowhere to be found", src.object()));

                let pointer = self.m.allocators[src_device]
                    .access(src.object(), M::C::bottom())
                    .unwrap();
                self.unlock_queue.insert((src.object().clone(), src_device));

                let src = if self.m.current_stage[src_device].is_planning() {
                    src.set_pointer(Some(pointer))
                } else {
                    src
                };

                src.map_v(|a| a.with_device(src_device))
            }
        }
    }

    /// Clone (potentially sliced) object from `src` to `dst`.
    /// The `src.device()` is not respected,
    /// But source and destination device are guaranteed to be planning or unplanned,
    /// if they are changed.
    ///
    /// # Contracts
    /// - `dst` or `src` must have pointer, if it's device is planned
    /// - `src.object()` must be found on some planning or unplanned device if it has no pointer
    pub fn clone_object(
        &mut self,
        dst: ResidentalT<Object, Option<Pointer<P>>>,
        src: ResidentalAtom<Object, Option<Pointer<P>>>,
    ) -> Result<(), OomError> {
        use Device::*;
        use Stage::*;

        let src = self.get_transfer_source(src);

        let dst_device = match src.device() {
            Disk => Cpu,
            otherwise => otherwise,
        };

        let dst = match self.current_stage[dst_device] {
            Planned => {
                if dst.pointer().is_none() {
                    panic!("expect atom {:?} on planned device to have a pointer", dst);
                }
                dst
            }
            Unplanned => {
                let _ = self.allocate(dst_device, dst.deref())?;
                dst
            }
            Planning => {
                let pointer = self.allocate_and_queue_unlock(dst_device, dst.deref())?;
                dst.set_pointer(Some(pointer))
            }
        };

        self.emit(R::transfer(dst, dst_device, src));

        Ok(())
    }

    /// Transfer (potentially sliced) object `src` to `dst` on `dst_device`.
    /// `src.object()` does not necessarily be on `src.device()`;
    /// It may come from any planning or unplanned device.
    ///
    /// # Contracts
    /// - `dst` or `src` must have pointer, if it's device is planned
    /// - `src.object()` must be found on some planning or unplanned device if it has no pointer
    pub fn transfer(
        &mut self,
        dst: ResidentalT<Object, Option<Pointer<P>>>,
        dst_device: Device,
        src: ResidentalAtom<Object, Option<Pointer<P>>>,
    ) -> Result<(), OomError> {
        use Stage::*;

        let src = self.get_transfer_source(src);

        let dst = match self.current_stage[dst_device] {
            Planned => {
                if dst.pointer().is_none() {
                    panic!("expect atom {:?} on planned device to have a pointer", dst);
                }
                dst
            }
            Unplanned => {
                let _ = self.allocate(dst_device, dst.deref())?;
                dst
            }
            Planning => {
                let pointer = self.allocate_and_queue_unlock(dst_device, dst.deref())?;
                dst.set_pointer(Some(pointer))
            }
        };

        self.emit(R::transfer(dst, dst_device, src));

        Ok(())
    }

    fn next_uses(&mut self) -> &mut NextUses {
        self.m.criterions.next_uses_mut()
    }

    pub fn finish(mut self) -> Result<(), OomError> {
        // Unlock objects allocated or claimed during `self.pc`
        let queue = std::mem::take(&mut self.unlock_queue);

        let pc = self.pc;
        let planning_devices = self.current_stage.planning_devices().collect::<Vec<_>>();
        let unplanned_devices = self.current_stage.unplanned_devices().collect::<Vec<_>>();

        queue.iter().for_each(|(object, device)| {
            self.m.allocators[*device]
                .unlock(object, self.m.criterions.mark_use(object, *device, pc))
        });

        // For objects that will not be used on any planning devices, eject them
        // from all planning devices
        for (object, device) in queue.iter() {
            if self.m.current_stage[*device].is_planning() {
                if self.next_uses().will_not_be_used_on(
                    object,
                    planning_devices.iter().cloned(),
                    pc,
                ) {
                    if !self.next_uses().will_not_be_used_on(
                        object,
                        unplanned_devices.iter().cloned(),
                        pc,
                    ) {
                        self.transfer_for_eject(object, *device)?;
                    }

                    for d in planning_devices.iter().cloned() {
                        if self.m.allocators[d].contains(object) {
                            self.deallocate(d, object);
                        }
                    }
                }
            }
        }

        // self.allocators[Device::Gpu(0)].finish(self.pc_usize());

        Ok(())
    }

    /// Lookup an object, starting from `root_device` and try all planning devices then unplanned devices
    ///
    /// # Contracts
    /// - `root_device` must be planning.
    pub fn lookup_object(&mut self, object: &Object) -> Option<(Device, Pointer<P>)> {
        let pc = self.pc;
        for device in self.m.current_stage.planning_unplanned_devices() {
            if let Some(p) = self.m.allocators[device].access(
                object,
                self.m.criterions.victim_criterion(object, device, pc),
            ) {
                self.unlock_queue.insert((object.clone(), device));
                return Some((device, p));
            }
        }

        return None;
    }

    /// Allocate `object` to `device`, and its parent, grandparent, etc. if space is not sufficient.
    /// Here insufficient means no space left even after evicting victims.
    ///
    /// # Contracts
    /// - `device` must be planning
    pub fn allocate_where_you_can(
        &mut self,
        object: &Object,
        device: Device,
    ) -> (Device, Pointer<P>) {
        if self.current_stage[device].is_planned() {
            panic!("cannot allocate space on planned device {:?}", device);
        }

        if let Ok(p) = self.allocate_and_queue_unlock(device, object) {
            return (device, p);
        }

        if let Some(parent_device) = device.parent() {
            self.allocate_where_you_can(object, parent_device)
        } else {
            panic!("allocate_where_you_can should not fail since top level device is expected to have infinite capacity")
        }
    }

    pub fn claim_where_you_can(&mut self, object: &Object, to: Device) -> (Device, Pointer<P>) {
        if self.current_stage[to].is_planned() {
            panic!("cannot claim object to planned device {:?}", to);
        }

        if let Ok(p) = self.claim_and_queue_unlock(object, to) {
            return (to, p);
        }

        if let Some(parent_device) = to.parent() {
            self.claim_where_you_can(object, parent_device)
        } else {
            panic!("claim_where_you_can should not fail since top level device is expected to have infinite capacity")
        }
    }
}

pub trait AdEmitter<P> {
    fn allocate(&mut self, t: Object, device: Device, pointer: Pointer<P>);
    fn deallocate(&mut self, t: Object, device: Device, pointer: Pointer<P>);
}

pub struct DummyAdHandle<P>(PhantomData<P>);

impl<P> DummyAdHandle<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> AdEmitter<P> for DummyAdHandle<P> {
    fn allocate(&mut self, _t: Object, _device: Device, _pointer: Pointer<P>) {}
    fn deallocate(&mut self, _t: Object, _device: Device, _pointer: Pointer<P>) {}
}

pub struct MachineHandle<'m, R, P> {
    m: &'m mut Seq<R>,
    _phantom: PhantomData<P>,
}

impl<'m, R, P> AdEmitter<P> for MachineHandle<'m, R, P>
where
    R: Constructor<Object, Pointer<P>> + std::fmt::Debug,
    P: Clone + std::fmt::Debug,
{
    fn allocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
        self.m.emit(R::ad(Ad::Allocate(t, device, pointer)))
    }

    fn deallocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
        self.m.emit(R::ad(Ad::Deallocate(t, device, pointer)))
    }
}

impl<'m, R, P> MachineHandle<'m, R, P>
where
    R: Constructor<Object, Pointer<P>>,
{
    pub(super) fn new(ops: &'m mut Seq<R>) -> Self {
        Self {
            m: ops,
            _phantom: PhantomData,
        }
    }
}
