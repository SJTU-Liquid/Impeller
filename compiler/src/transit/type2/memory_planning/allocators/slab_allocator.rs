use super::{normalize_size, slab_pool};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

#[derive(Clone)]
pub struct SlabAllocator<P, C, Cf> {
    mapping: slab_pool::Pointers<P>,
    /// Associate each object with its residence.
    /// Don't forget to update this mapping.
    object_residence: ObjectBijection<P>,
    offset: u64,
    ia: slab_pool::Pool<C, Cf, P>,
    device: Device,
}

impl<P, C, Cf> SlabAllocator<P, C, Cf> {
    pub fn new(capacity: u64, offset: u64, max_bs: IntegralSize, device: Device) -> Self {
        Self {
            mapping: Heap::new(),
            object_residence: ObjectBijection::new(),
            offset,
            ia: slab_pool::Pool::new(capacity, max_bs),
            device,
        }
    }

    pub fn peak_memory_usage(&self) -> u64 {
        self.ia.occupied_size()
    }
}

struct CommitPayload<P> {
    victims: slab_pool::Victims<P>,
    object: Object,
}

impl<P, C, Cf> Allocator<P, C, Cf> for SlabAllocator<P, C, Cf>
where
    P: UsizeId + 'static,
    C: Criterion + std::fmt::Debug + 'static,
    Cf: Criterion + std::fmt::Debug + 'static,
{
    fn layer(&self) -> u32 {
        0
    }

    fn device(&self) -> Device {
        self.device
    }

    fn access(&mut self, t: &Object, c: C) -> Option<Pointer<P>> {
        let p = self.object_residence.get_forward(t).copied()?;
        let (size, addr) = self.mapping[p];
        self.ia.update_criterion(size, addr, c, false);

        Some(Pointer::new::<_, C, Cf>(self, p))
    }

    fn unlock(&mut self, t: &Object, c: C) {
        let p = self
            .object_residence
            .get_forward(t)
            .copied()
            .unwrap_or_else(|| panic!("cannot unlock object {:?} since it is not recorded", t));

        let (size, addr) = self.mapping[p];
        self.ia.update_criterion(size, addr, c, true);
    }

    fn contains(&self, t: &Object) -> bool {
        self.object_residence.get_forward(t).is_some()
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        now: usize,
        machine: &mut dyn AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        if self.object_residence.get_forward(t).is_some() {
            panic!("object {:?} already allocated on {:?}", t, &self.device);
        }

        let size = normalize_size(size);

        let (_size, p) = match size {
            Size::Smithereen(..) => {
                panic!("this allocator is not responsible for smithereen objects")
            }
            Size::Integral(is) => (
                is,
                self.ia
                    .allocate(is, C::bottom(), now, &mut self.mapping)
                    .ok_or_else(|| OomError::for_alloc::<Self>(self.device, size))?,
            ),
        };

        self.object_residence.insert(t.clone(), p);

        let pointer = Pointer::new::<_, C, Cf>(self, p);
        machine.allocate(t.clone(), self.device, pointer);

        Ok(pointer)
    }

    fn deallocate(&mut self, t: &Object, now: usize, cf: Cf, machine: &mut dyn AdEmitter<P>) {
        let p = self.object_residence.remove_forward(t).unwrap_or_else(|| {
            panic!(
                "object {:?} is not recorded on device {:?}",
                t, &self.device
            )
        });

        self.ia.deallocate(p, now, cf, &self.mapping);
        machine.deallocate(t.clone(), self.device, Pointer::new::<_, C, Cf>(self, p));
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, c: C) {
        let p = self
            .object_residence
            .remove_forward(old_object)
            .unwrap_or_else(|| panic!("{:?} not found on {:?}", old_object, &self.device));
        let (size, addr) = self.mapping[p];

        self.ia.update_criterion(size, addr, c, false);
        self.object_residence.insert_checked(new_object.clone(), p);
    }

    fn takeover_victims(
        &self,
        size: Size,
        object: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        let size = normalize_size(size);

        let victims = match size {
            Size::Smithereen(..) => {
                panic!("this allocator is not responsible for smithereen objects")
            }
            Size::Integral(is) => self
                .ia
                .victims(is)
                .ok_or_else(|| OomError::for_victim::<Self>(self.device, size))?,
        };

        let victim_objects = victims
            .victims
            .iter()
            .map(|p| {
                self.object_residence
                    .get_backward(&p)
                    .unwrap_or_else(|| panic!("victim pointer {:?} is not recorded", p))
                    .clone()
            })
            .collect::<Vec<_>>();

        let commit = allocator::CommitVictims::new(
            Box::new(CommitPayload {
                victims,
                object: object.clone(),
            }),
            self.layer(),
        );

        Ok((commit, victim_objects))
    }

    fn accept_commit(
        &mut self,
        commit: allocator::CommitVictims,
        now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        let payload: CommitPayload<P> = commit.payload();

        let p = self
            .ia
            .commit_victims(&payload.victims, C::bottom(), cf, now, &mut self.mapping);

        for victim_p in payload.victims.victims {
            let object = self
                .object_residence
                .remove_backward(&victim_p)
                .unwrap_or_else(|| {
                    panic!(
                        "victim at {:?} is not found to correspond to any object",
                        victim_p
                    )
                });
            machine.deallocate(
                object,
                self.device,
                Pointer::new::<_, C, Cf>(self, victim_p),
            );
        }

        self.object_residence.insert(payload.object.clone(), p);
        let pointer = Pointer::new::<_, C, Cf>(self, p);

        machine.allocate(payload.object, self.device, pointer);
        pointer
    }
}

impl<'a, P, C, Cf> Realizer<P> for AsRealizer<'a, SlabAllocator<P, C, Cf>, P, C, Cf>
where
    P: UsizeId,
{
    fn device(&self) -> Device {
        self.allocator().device
    }

    fn allocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocMethod {
        if let Some(p) = pointer.here(0) {
            let (size, addr) = self.allocator().mapping[p];
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
