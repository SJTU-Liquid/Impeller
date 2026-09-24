use super::{normalize_size, smithereens_pool};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

/// The ability to choose whether to route the allocation to current [`Wrapper`]
/// or the inner allocator.
pub trait Filter {
    /// The wrapper accepts the allocation if this function returns `true`
    fn filter(&self, size: Size, object: &Object) -> bool;
    fn clone_to_dyn(&self) -> Box<dyn Filter>;
}

/// Routes smithereen objects as classified by [`normalize_size`] to the wrapper
pub struct SmithereenFilter;

impl Filter for SmithereenFilter {
    fn filter(&self, size: Size, _object: &Object) -> bool {
        match normalize_size(size) {
            Size::Integral(..) => false,
            Size::Smithereen(..) => true,
        }
    }
    fn clone_to_dyn(&self) -> Box<dyn Filter> {
        Box::new(Self)
    }
}

/// Routes all objects to the wrapper
pub struct AllAcceptFilter;

/// Routes no object to the wrapper
pub struct AllRejectFilter;

impl Filter for AllAcceptFilter {
    fn filter(&self, _size: Size, _object: &Object) -> bool {
        true
    }

    fn clone_to_dyn(&self) -> Box<dyn Filter> {
        Box::new(Self)
    }
}

impl Filter for AllRejectFilter {
    fn filter(&self, _size: Size, _object: &Object) -> bool {
        false
    }

    fn clone_to_dyn(&self) -> Box<dyn Filter> {
        Box::new(Self)
    }
}

/// A wrapper that adds a pool for small objects to the underlying allocator.
/// The smithereen pool will not eject objects, and errors when space is insufficient.
pub struct Wrapper<A, P> {
    pub(super) inner: A,
    pub(super) sa: smithereens_pool::Pool,
    /// Only tracks smithereen objects
    pub(super) object_residence: ObjectBijection<P>,
    /// Only tracks pointers to smithereen objects
    pub(super) mapping: Heap<P, (Addr, Size)>,
    pub(super) offset: u64,
    pub(super) filter: Box<dyn Filter>,
}

impl<A, P> Clone for Wrapper<A, P>
where
    A: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            sa: self.sa.clone(),
            object_residence: self.object_residence.clone(),
            mapping: self.mapping.clone(),
            offset: self.offset,
            filter: self.filter.clone_to_dyn(),
        }
    }
}

impl<A, P> std::fmt::Debug for Wrapper<A, P>
where
    A: std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wrapper")
            .field("inner", &self.inner)
            .field("sa", &self.sa)
            .field("object_residence", &self.object_residence)
            .field("mapping", &self.mapping)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<A, P> Wrapper<A, P> {
    pub fn new(inner: A, capacity: u64, offset: u64, filter: impl Filter + 'static) -> Self {
        Self {
            inner,
            sa: smithereens_pool::Pool::new(capacity),
            object_residence: ObjectBijection::new(),
            mapping: Heap::new(),
            filter: Box::new(filter),
            offset,
        }
    }

    pub fn for_filter_smithereen(inner: A, capacity: u64, offset: u64) -> Self {
        Self::new(inner, capacity, offset, SmithereenFilter)
    }

    pub fn unwrap(self) -> A {
        self.inner
    }

    pub fn unwrap_ref(&self) -> &A {
        &self.inner
    }

    pub fn capacity(&self) -> u64 {
        self.sa.capacity()
    }

    pub fn unwrap_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    pub fn with_filter(mut self, filter: impl Filter + 'static) -> Self {
        self.filter = Box::new(filter);
        self
    }

    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    pub fn wrapper_used_bytes(&self) -> u64 {
        self.sa.used_bytes()
    }

    pub fn outer_contains(&self, object: &Object) -> bool
    where
        P: UsizeId,
    {
        self.object_residence.get_forward(object).is_some()
    }

    pub fn inner_contains<C, Cf>(&self, object: &Object) -> bool
    where
        P: UsizeId,
        A: Allocator<P, C, Cf>,
        C: Criterion + 'static,
        Cf: Criterion + 'static,
    {
        self.inner.contains(object)
    }
}

impl<A, P, C, Cf> Allocator<P, C, Cf> for Wrapper<A, P>
where
    A: Allocator<P, C, Cf>,
    C: Criterion + 'static,
    Cf: Criterion + 'static,
    P: 'static + UsizeId,
{
    fn layer(&self) -> u32 {
        self.inner.layer() + 1
    }

    fn device(&self) -> Device {
        self.inner.device()
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        now: usize,
        machine: &mut dyn AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        if self.filter.filter(size, t) {
            let ss = size.into();
            let addr = self
                .sa
                .allocate(ss)
                .ok_or_else(|| OomError::for_alloc::<Self>(self.device(), size))?;
            let p = self.mapping.push((addr, size));
            self.object_residence.insert(t.clone(), p);

            let pointer = Pointer::new::<_, C, Cf>(self, p);
            machine.allocate(t.clone(), self.inner.device(), pointer);

            Ok(pointer)
        } else {
            self.inner.allocate(size, t, now, machine)
        }
    }

    fn takeover_victims(
        &self,
        size: Size,
        t: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        if self.filter.filter(size, t) {
            panic!("SmithereenWrapper does not support ejecting victims")
        } else {
            self.inner.takeover_victims(size, t)
        }
    }

    fn accept_commit(
        &mut self,
        commit: allocator::CommitVictims,
        now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        if commit.layer() == self.layer() {
            panic!("SmithereenWrapper does not support ejecting victims")
        } else {
            self.inner.accept_commit(commit, now, cf, machine)
        }
    }

    fn deallocate(&mut self, t: &Object, now: usize, cf: Cf, machine: &mut dyn AdEmitter<P>) {
        if let Some(p) = self.object_residence.remove_forward(t) {
            self.sa.deallocate(self.mapping[p].0);
            machine.deallocate(
                t.clone(),
                self.inner.device(),
                Pointer::new::<_, C, Cf>(self, p),
            );
        } else {
            self.inner.deallocate(t, now, cf, machine)
        }
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, c: C) {
        if let Some(p) = self.object_residence.remove_forward(&old_object) {
            self.object_residence.insert(new_object.clone(), p);
        } else {
            self.inner.reuse(new_object, old_object, c);
        }
    }

    fn access(&mut self, t: &Object, c: C) -> Option<Pointer<P>> {
        if let Some(p) = self.object_residence.get_forward(t) {
            Some(Pointer::new::<_, C, Cf>(self, *p))
        } else {
            self.inner.access(t, c)
        }
    }

    fn unlock(&mut self, t: &Object, c: C) {
        if self.object_residence.get_forward(t).is_none() {
            self.inner.unlock(t, c);
        }
    }

    fn contains(&self, t: &Object) -> bool {
        self.object_residence.get_forward(t).is_some() || self.inner.contains(t)
    }
}

impl<'a, A, P, C, Cf> Realizer<P> for AsRealizer<'a, Wrapper<A, P>, P, C, Cf>
where
    A: Allocator<P, C, Cf>,
    C: Criterion + 'static,
    Cf: Criterion + 'static,
    P: 'static + UsizeId,
    for<'b> AsRealizer<'b, A, P, C, Cf>: Realizer<P>,
{
    fn device(&self) -> Device {
        self.allocator().device()
    }

    fn allocated(&self, t: &Object, pointer: &Pointer<P>) -> AllocMethod {
        if let Some(p) = pointer.here(self.allocator().layer()) {
            let (addr, size) = self.allocator().mapping[p];
            AllocMethod::Offset(addr.offset(self.allocator().offset).get(), size.into())
        } else {
            AsRealizer::new(&self.allocator().inner).allocated(t, pointer)
        }
    }

    fn deallocated(&self, t: &Object, pointer: &Pointer<P>) -> AllocVariant {
        if let Some(..) = pointer.here(self.allocator().layer()) {
            AllocVariant::Offset
        } else {
            AsRealizer::new(&self.allocator().inner).deallocated(t, pointer)
        }
    }
}
