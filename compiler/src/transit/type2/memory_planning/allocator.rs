use super::prelude::*;
use std::any::Any;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Epsilon {
    Bottom = 0,
    Top = 1,
}

impl Criterion for Epsilon {
    fn join(self, rhs: Self) -> Self {
        use Epsilon::*;
        match (self, rhs) {
            (Top, Top) => Top,
            _ => Bottom,
        }
    }

    fn top() -> Self {
        Epsilon::Top
    }

    fn bottom() -> Self {
        Epsilon::Bottom
    }
}

pub struct CommitVictims {
    payload: Box<dyn Any>,
    layer: u32,
}

impl CommitVictims {
    pub fn new(payload: Box<dyn Any>, layer: u32) -> Self {
        Self { payload, layer }
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn payload<P>(self) -> P
    where
        P: Any,
    {
        *self.payload.downcast().unwrap()
    }
}

pub trait Criterion: Clone + Ord {
    fn join(self, rhs: Self) -> Self;
    /// Best candidate.
    fn top() -> Self;
    /// Forbiddened candidate. Must satisfy `join(bottom, whaever) = bottom`.
    fn bottom() -> Self;
}

#[derive(Debug)]
pub enum OomKind {
    Alloc,
    Victim,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct OomError {
    device: Device,
    requested_size: Size,
    allocator_type_name: &'static str,
    kind: OomKind,
}

impl OomError {
    pub fn for_alloc<A>(device: Device, requested_size: Size) -> Self {
        Self {
            device,
            requested_size,
            allocator_type_name: std::any::type_name::<A>(),
            kind: OomKind::Alloc,
        }
    }

    pub fn for_victim<A>(device: Device, requested_size: Size) -> Self {
        Self {
            device,
            requested_size,
            allocator_type_name: std::any::type_name::<A>(),
            kind: OomKind::Victim,
        }
    }
}

#[derive(Clone, Debug, Hash, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pointer<P> {
    p: P,
    layer: u32,
}

impl<P> Pointer<P>
where
    P: Copy,
{
    pub fn new<A, C, Cf>(a: &A, p: P) -> Self
    where
        A: Allocator<P, C, Cf>,
    {
        Self {
            p,
            layer: a.layer(),
        }
    }

    pub fn here(self, layer: u32) -> Option<P> {
        if self.layer == layer {
            Some(self.p)
        } else {
            None
        }
    }
}

pub trait Allocator<P, C, Cf>: std::any::Any {
    fn layer(&self) -> u32;
    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        now: usize,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError>;
    fn takeover_victims(
        &self,
        size: Size,
        t: &Object,
    ) -> Result<(CommitVictims, Vec<Object>), OomError>;
    fn accept_commit(
        &mut self,
        commit: CommitVictims,
        now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P>;
    fn deallocate(
        &mut self,
        t: &Object,
        now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    );
    fn reuse(&mut self, new_object: &Object, old_object: &Object, c: C);
    /// When victim criterion of an object is bottom, it can only be set to another value through `unlock`.
    /// That is, `access` does not modify the object's victim criterion.
    fn unlock(&mut self, t: &Object, c: C);
    /// When accessing an object, one must provide the updated victim criterion
    fn access(&mut self, t: &Object, c: C) -> Option<Pointer<P>>;
    fn contains(&self, t: &Object) -> bool;
    fn device(&self) -> Device;
}

pub struct AsRealizer<'a, A, P, C, Cf> {
    allocator: &'a A,
    _phantom: PhantomData<(P, C, Cf)>,
}

impl<'a, A, P, C, Cf> AsRealizer<'a, A, P, C, Cf> {
    pub fn new(allocator: &'a A) -> Self
    where
        A: Allocator<P, C, Cf>,
    {
        Self {
            allocator,
            _phantom: PhantomData,
        }
    }

    pub fn allocator(&self) -> &'a A {
        self.allocator
    }
}

pub trait Realizer<P> {
    fn allocated(&self, t: &Object, pointer: &Pointer<P>) -> AllocMethod;

    fn deallocated(&self, t: &Object, pointer: &Pointer<P>) -> AllocVariant;

    fn device(&self) -> Device;

    /// Instruct the machine to allocate space at pointer.
    /// Relative registers should be made available after this.
    fn allocate(
        &self,
        t: &Object,
        pointer: &Pointer<P>,
        machine: &mut dyn realization::AdEmitter<P>,
    ) -> RegisterId
    where
        P: Copy,
    {
        machine.allocate(t, *pointer, self.device(), self.allocated(t, pointer))
    }

    /// Instruct the machine to deallocate space at pointer
    fn deallocate(
        &self,
        t: &Object,
        pointer: &Pointer<P>,
        machine: &mut dyn realization::AdEmitter<P>,
    ) where
        P: Copy,
    {
        machine.deallocate(t, *pointer, self.device(), self.deallocated(t, pointer))
    }
}

pub struct AllocatorCollection<'a, P, C, Cf> {
    borrowed: DeviceSpecific<Option<&'a mut dyn Allocator<P, C, Cf>>>,
    owned: DeviceSpecific<Option<Box<dyn Allocator<P, C, Cf>>>>,
}

impl<'a, P, C, Cf> std::fmt::Debug for AllocatorCollection<'a, P, C, Cf> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let borrowed = self
            .borrowed
            .iter()
            .filter_map(|(d, a)| a.as_ref().map(|_| d))
            .collect::<Vec<_>>();
        let owned = self
            .owned
            .iter()
            .filter_map(|(d, a)| a.as_ref().map(|_| d))
            .collect::<Vec<_>>();

        f.debug_struct("AllocatorCollection")
            .field("borrowed", &borrowed)
            .field("owned", &owned)
            .finish()
    }
}

impl<'a, P, C, Cf> Default for AllocatorCollection<'a, P, C, Cf> {
    fn default() -> Self {
        Self {
            borrowed: DeviceSpecific::new(0, |_| None),
            owned: DeviceSpecific::new(0, |_| None),
        }
    }
}

impl<'a, P, C, Cf> std::ops::Index<Device> for AllocatorCollection<'a, P, C, Cf> {
    type Output = dyn Allocator<P, C, Cf>;
    fn index(&self, index: Device) -> &Self::Output {
        if let Some(x) = self.borrowed.get(index).as_ref() {
            *x
        } else {
            let x = self.owned.get(index).as_ref().unwrap().as_ref();
            x
        }
    }
}

impl<'a, P, C, Cf> std::ops::IndexMut<Device> for AllocatorCollection<'a, P, C, Cf> {
    fn index_mut(&mut self, index: Device) -> &mut Self::Output {
        if let Some(x) = self.borrowed.get_mut(index).as_mut() {
            *x
        } else {
            let x = self.owned.get_mut(index).as_mut().unwrap().as_mut();
            x
        }
    }
}

impl<'a, P, C, Cf> AllocatorCollection<'a, P, C, Cf>
where
    Cf: 'static,
    C: 'static,
    P: 'static,
{
    pub fn object_available_on(
        &self,
        object: &Object,
        devices: impl Iterator<Item = Device>,
    ) -> Vec<Device> {
        devices.filter(|d| self[*d].contains(object)).collect()
    }

    pub fn object_available_on1(
        &self,
        object: &Object,
        devices: impl Iterator<Item = Device>,
    ) -> Option<Device> {
        devices.filter(|d| self[*d].contains(object)).next()
    }

    pub fn new_owned(
        n_gpus: usize,
        borrowed: impl FnMut(Device) -> Option<&'a mut dyn Allocator<P, C, Cf>>,
        owned: impl FnMut(Device) -> Option<Box<dyn Allocator<P, C, Cf>>>,
    ) -> Self {
        Self {
            borrowed: DeviceSpecific::new(n_gpus, borrowed),
            owned: DeviceSpecific::new(n_gpus, owned),
        }
    }

    pub fn new_all(borrowed: DeviceSpecific<Option<&'a mut dyn Allocator<P, C, Cf>>>) -> Self {
        let n_gpus = borrowed.gpu.len();
        Self {
            borrowed,
            owned: DeviceSpecific::new(n_gpus, |_| None),
        }
    }

    pub fn take(&mut self, device: Device) -> Option<&'a mut dyn Allocator<P, C, Cf>> {
        self.borrowed.get_mut(device).take()
    }

    pub fn set(&mut self, device: Device, a: &'a mut dyn Allocator<P, C, Cf>) {
        *self.borrowed.get_mut(device) = Some(a)
    }
}
