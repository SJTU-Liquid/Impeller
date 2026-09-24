use super::smithereens_pool;
use crate::{transit::type2::memory_planning::prelude::*, utils::human_readable_size};

#[derive(Debug, Clone)]
pub struct LinearAllocator<P, C, Cf> {
    pub(super) sa: smithereens_pool::Pool,
    /// Only tracks smithereen objects
    pub(super) object_residence: ObjectBijection<P>,
    /// Only tracks pointers to smithereen objects
    pub(super) mapping: Heap<P, (Addr, Size)>,
    pub(super) offset: u64,
    pub(super) device: Device,
    _phantom: PhantomData<(C, Cf)>,
}

impl<P, C, Cf> LinearAllocator<P, C, Cf> {
    pub fn new(capacity: u64, offset: u64, device: Device) -> Self {
        Self {
            sa: smithereens_pool::Pool::new(capacity),
            object_residence: ObjectBijection::new(),
            mapping: Heap::new(),
            device,
            offset,
            _phantom: PhantomData,
        }
    }

    pub fn used_bytes(&self) -> u64 {
        self.sa.used_bytes()
    }

    pub fn allocated_bytes(&self) -> u64
    where
        P: UsizeId,
    {
        self.object_residence
            .iter()
            .map(|(_, p)| u64::from(self.mapping[*p].1))
            .sum()
    }

    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }
}

impl<P, C, Cf> Allocator<P, C, Cf> for LinearAllocator<P, C, Cf>
where
    P: UsizeId + 'static,
    C: 'static,
    Cf: 'static,
{
    fn layer(&self) -> u32 {
        0
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        _now: usize,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        let addr = self
            .sa
            .allocate(SmithereenSize(size.into()))
            .ok_or_else(|| OomError::for_alloc::<Self>(self.device(), size))?;
        let p = self.mapping.push((addr, size));
        self.object_residence.insert(t.clone(), p);

        let pointer = Pointer::new::<_, C, Cf>(self, p);
        machine.allocate(t.clone(), self.device, pointer);

        Ok(pointer)
    }

    fn takeover_victims(
        &self,
        _size: Size,
        _t: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        panic!("LinearAllocator does not support takingover victims")
    }

    fn accept_commit(
        &mut self,
        _commit: allocator::CommitVictims,
        _now: usize,
        _cf: Cf,
        _machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        panic!("LinearAllocator does not support takingover victims")
    }

    fn deallocate(
        &mut self,
        t: &Object,
        _now: usize,
        _cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) {
        let p = self
            .object_residence
            .remove_forward(t)
            .unwrap_or_else(|| panic!("object {:?} is not allocated on {:?}", t, self.device));
        self.sa.deallocate(self.mapping[p].0);

        machine.deallocate(t.clone(), self.device, Pointer::new::<_, C, Cf>(self, p));
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, _c: C) {
        let p = self
            .object_residence
            .remove_forward(old_object)
            .unwrap_or_else(|| {
                panic!(
                    "cannot reuse object {:?} since it is not allcoated",
                    old_object
                )
            });
        self.object_residence.insert(new_object.clone(), p);
    }

    fn unlock(&mut self, _t: &Object, _c: C) {}

    fn access(&mut self, t: &Object, _c: C) -> Option<Pointer<P>> {
        self.object_residence
            .get_forward(t)
            .map(|p| Pointer::new::<_, C, Cf>(self, *p))
    }

    fn contains(&self, t: &Object) -> bool {
        self.object_residence.get_forward(t).is_some()
    }

    fn device(&self) -> Device {
        self.device
    }
}

impl<'a, P, C, Cf> Realizer<P> for AsRealizer<'a, LinearAllocator<P, C, Cf>, P, C, Cf>
where
    P: UsizeId,
{
    fn device(&self) -> Device {
        self.allocator().device
    }

    fn allocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocMethod {
        if let Some(p) = pointer.here(0) {
            let (addr, size) = self.allocator().mapping[p];
            AllocMethod::Offset(addr.offset(self.allocator().offset).get(), size.into())
        } else {
            panic!("pointer {:?} is not layer 0", pointer);
        }
    }

    fn deallocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocVariant {
        if let Some(..) = pointer.here(0) {
            AllocVariant::Offset
        } else {
            panic!("pointer {:?} is not layer 0", pointer);
        }
    }
}
