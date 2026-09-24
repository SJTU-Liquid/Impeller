use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

pub struct SuperAllocator<P> {
    /// All objects that have lived on this allocator
    mapping: ObjectTable<(P, Size)>,
    /// Currently living objects
    living: ObjectTable<()>,
    peak_memory_usage: u64,
    current_memory_usage: u64,
    device: Device,
    p_allocator: IdAllocator<P>,
    /// If `physical` is false, then this allocator will not instruct machine for allocation or deallocation.
    /// That is used in case when the underlying device is unplanned.
    physical: bool,
}

impl<P> SuperAllocator<P> {
    pub fn for_unplanned(device: Device) -> Self {
        Self {
            mapping: ObjectTable::new(),
            living: ObjectTable::new(),
            device,
            p_allocator: IdAllocator::new(),
            peak_memory_usage: 0,
            current_memory_usage: 0,
            physical: false,
        }
    }

    pub fn new(device: Device) -> Self {
        Self {
            mapping: ObjectTable::new(),
            living: ObjectTable::new(),
            device,
            p_allocator: IdAllocator::new(),
            peak_memory_usage: 0,
            current_memory_usage: 0,
            physical: true,
        }
    }

    pub fn record_allocated(&mut self, size: Size) {
        self.current_memory_usage += u64::from(size);
        self.peak_memory_usage = self.peak_memory_usage.max(self.current_memory_usage);
    }

    pub fn record_deallocated(&mut self, size: Size) {
        self.current_memory_usage -= u64::from(size);
    }

    pub fn peak_memory_usage(&self) -> u64 {
        self.peak_memory_usage
    }
}

impl<P, C, Cf> Allocator<P, C, Cf> for SuperAllocator<P>
where
    C: Criterion + 'static,
    Cf: Criterion + 'static,
    P: UsizeId + 'static,
{
    fn device(&self) -> Device {
        self.device
    }

    fn layer(&self) -> u32 {
        0
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        _now: usize,
        machine: &mut dyn AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        if self.living.contains(t) {
            panic!("{:?} already allocated on {:?}", t, self.device);
        }

        let p = self.p_allocator.alloc();
        self.mapping.insert(t.clone(), (p, size));
        self.living.insert(t.clone(), ());

        let pointer = Pointer::new::<_, C, Cf>(self, p);

        if self.physical {
            machine.allocate(t.clone(), self.device, pointer);
        }

        self.record_allocated(size);

        Ok(pointer)
    }

    fn takeover_victims(
        &self,
        _size: Size,
        _t: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        panic!("SuperAllocator does not support and does not need ejecting victims")
    }

    fn accept_commit(
        &mut self,
        _commit: allocator::CommitVictims,
        _now: usize,
        _cf: Cf,
        _machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        panic!("SuperAllocator does not support and does not need ejecting victims")
    }

    fn deallocate(&mut self, t: &Object, _now: usize, _cf: Cf, machine: &mut dyn AdEmitter<P>) {
        self.living.remove(t).unwrap_or_else(|| {
            panic!(
                "deallocating object {:?} which is not recorded to be allocated",
                t
            )
        });
        let (p, size) = self.mapping[t];

        let pointer = Pointer::new::<_, C, Cf>(self, p);

        if self.physical {
            machine.deallocate(t.clone(), self.device, pointer);
        }

        self.record_deallocated(size);
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, _c: C) {
        self.living.remove(&old_object).unwrap_or_else(|| {
            panic!(
                "reusing object {:?} which is not recorded to be allocated",
                old_object
            )
        });

        let (p, size) = self.mapping[old_object];
        self.living.insert(new_object.clone(), ());
        self.mapping.insert(new_object.clone(), (p, size));
    }

    fn access(&mut self, t: &Object, _c: C) -> Option<Pointer<P>> {
        self.living.get(t)?;

        Some(
            self.mapping
                .get(t)
                .map(|(p, _)| Pointer::new::<_, C, Cf>(self, *p))
                .expect("to find the object in mapping since it is alive"),
        )
    }

    fn unlock(&mut self, _t: &Object, _c: C) {}

    fn contains(&self, t: &Object) -> bool {
        self.living.contains(t)
    }
}

impl<'a, P, C, Cf> Realizer<P> for AsRealizer<'a, SuperAllocator<P>, P, C, Cf>
where
    P: UsizeId,
{
    fn device(&self) -> Device {
        self.allocator().device
    }

    fn allocated(&self, t: &Object, pointer: &Pointer<P>) -> AllocMethod {
        if let Some(..) = pointer.here(0) {
            let (_, size) = self
                .allocator()
                .mapping
                .get(t)
                .unwrap_or_else(|| panic!("{:?} is not allocated on {:?}", t, self.device()));
            AllocMethod::Dynamic((*size).into())
        } else {
            panic!("pointer {:?} is not of layer 0", pointer)
        }
    }

    fn deallocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocVariant {
        if let Some(..) = pointer.here(0) {
            AllocVariant::Dynamic
        } else {
            panic!("pointer {:?} is not of layer 0", pointer)
        }
    }
}
