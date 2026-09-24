use super::super::prelude::*;
use planning::machine::*;

/// Create an allocator that wraps around `inner`, adding a constant pool to it.
///
/// When allocating an object that is a constant object in this pool,
/// allocation and deallocation is intercepted and skipped,
/// while accesses to constant objects always return a valid pointer.
pub struct Wrapper<A, P> {
    inner: A,
    objects: ObjectTable<P>,
}

impl<A, P> Wrapper<A, P>
where
    P: UsizeId,
{
    pub fn new(a: A, objects: impl Iterator<Item = Object>) -> Self {
        let mut p_allocator = IdAllocator::new();
        Self {
            inner: a,
            objects: objects.map(|o| (o, p_allocator.alloc())).collect(),
        }
    }

    pub fn unwrap(self) -> A {
        self.inner
    }

    pub fn unwrap_ref(&self) -> &A {
        &self.inner
    }

    pub fn unwrap_mut(&mut self) -> &mut A {
        &mut self.inner
    }
}

impl<A, P, C, Cf> Allocator<P, C, Cf> for Wrapper<A, P>
where
    A: Allocator<P, C, Cf>,
    C: Criterion + 'static,
    Cf: Criterion + 'static,
    P: UsizeId + 'static,
{
    fn access(&mut self, t: &Object, c: C) -> Option<Pointer<P>> {
        if let Some(p) = self.objects.get(t) {
            Some(Pointer::new::<_, C, Cf>(self, *p))
        } else {
            self.inner.access(t, c)
        }
    }

    fn unlock(&mut self, t: &Object, c: C) {
        if !self.objects.contains(t) {
            self.inner.unlock(t, c);
        }
    }

    fn contains(&self, t: &Object) -> bool {
        self.objects.contains(t) || self.inner.contains(t)
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        now: usize,
        machine: &mut dyn AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        if let Some(p) = self.objects.get(t) {
            let pointer = Pointer::new::<_, C, Cf>(self, *p);
            machine.allocate(t.clone(), self.device(), pointer.clone());
            Ok(pointer)
        } else {
            self.inner.allocate(size, t, now, machine)
        }
    }

    fn deallocate<'m>(&mut self, t: &Object, now: usize, cf: Cf, machine: &mut dyn AdEmitter<P>) {
        if !self.objects.contains(t) {
            self.inner.deallocate(t, now, cf, machine)
        }
    }

    fn device(&self) -> Device {
        self.inner.device()
    }

    fn layer(&self) -> u32 {
        self.inner.layer() + 1
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, c: C) {
        if self.objects.contains(old_object) {
            panic!(
                "cannot reuse space of object {:?} which is constant",
                old_object
            );
        }

        self.inner.reuse(new_object, old_object, c);
    }

    fn takeover_victims(
        &self,
        size: Size,
        t: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        self.inner.takeover_victims(size, t)
    }

    fn accept_commit(
        &mut self,
        commit: allocator::CommitVictims,
        now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        if commit.layer() == self.layer() {
            panic!("ConstantWrapper does not support ejecting victims")
        } else {
            self.inner.accept_commit(commit, now, cf, machine)
        }
    }
}

impl<'a, A, P, C, Cf> Realizer<P> for AsRealizer<'a, Wrapper<A, P>, P, C, Cf>
where
    A: Allocator<P, C, Cf>,
    C: Criterion + 'static,
    Cf: Criterion + 'static,
    P: UsizeId + 'static,
    for<'b> AsRealizer<'b, A, P, C, Cf>: Realizer<P>,
{
    fn device(&self) -> Device {
        self.allocator().device()
    }

    fn allocated(&self, _t: &Object, _pointer: &Pointer<P>) -> AllocMethod {
        panic!("ConstantWrapper does not allocate anything")
    }

    fn deallocated(&self, _t: &Object, _pointer: &Pointer<P>) -> AllocVariant {
        panic!("ConstantWrapper does not allocate anything")
    }

    fn allocate(
        &self,
        t: &Object,
        pointer: &Pointer<P>,
        machine: &mut dyn realization::AdEmitter<P>,
    ) -> RegisterId {
        if pointer.here(self.allocator().layer()).is_none() {
            AsRealizer::new(&self.allocator().inner).allocate(t, pointer, machine)
        } else {
            machine.new_register(t, *pointer, self.device())
        }
    }

    fn deallocate(
        &self,
        t: &Object,
        pointer: &Pointer<P>,
        machine: &mut dyn realization::AdEmitter<P>,
    ) {
        if pointer.here(self.allocator().layer()).is_none() {
            AsRealizer::new(&self.allocator().inner).deallocate(t, pointer, machine);
        }
    }
}
