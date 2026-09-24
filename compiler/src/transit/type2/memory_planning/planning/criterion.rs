use super::super::prelude::*;
pub use allocator::Epsilon;
use std::collections::VecDeque;

struct ObjectDeviceTable<T>(
    ObjectTable<DeviceSpecific<Option<T>>>,
    /* number of GPUs */ usize,
);

impl<T> ObjectDeviceTable<T>
where
    T: Default,
{
    pub fn new(n_gpus: usize) -> Self {
        Self(ObjectTable::new(), n_gpus)
    }

    pub fn set(&mut self, object: &Object, device: Device, t: T) {
        if !self.0.contains(object) {
            self.0
                .insert(object.clone(), DeviceSpecific::default(self.1));
        }

        *self.0.get_mut(object).unwrap().get_mut(device) = Some(t);
    }

    pub fn get(&self, object: &Object, device: Device) -> Option<&T> {
        let ds = self.0.get(object)?;
        ds.get(device).as_ref()
    }
}

pub trait Manager {
    /// Scheduled updates
    type Su<'c>: Iterator<Item = (Object, Device, Self::C)>
    where
        Self: 'c;
    type C: Criterion + 'static;
    type Cf: Criterion + 'static;

    fn new(next_uses: NextUses, n_gpus: usize) -> Self;
    fn victim_criterion(&mut self, object: &Object, device: Device, pc: Index) -> Self::C;
    fn mark_use(&mut self, object: &Object, device: Device, pc: Index) -> Self::C;
    fn scheduled_updates<'c>(&'c mut self, pc: Index) -> Self::Su<'c>;
    fn free_criterion(&mut self, object: &Object, device: Device, pc: Index) -> Self::Cf;
    fn next_uses_mut(&mut self) -> &mut NextUses;
}

/// Calculated by
///   min(now - t_last_use, N), t_next_use
/// Ordered by dictionary-order
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct VictimCriterion(i32, usize);

impl VictimCriterion {
    fn new(now: Index, last_use: Option<Index>, next_use: Option<Index>, n: usize) -> Self {
        let (now, last_use, next_use) = (
            usize::from(now) as isize,
            last_use
                .map(|x| usize::from(x) as isize)
                .unwrap_or(-isize::try_from(n).unwrap()),
            next_use.map(|x| usize::from(x)).unwrap_or(usize::MAX),
        );
        Self(
            (now - last_use).min(n as isize).try_into().unwrap(),
            next_use,
        )
    }
}

impl Criterion for VictimCriterion {
    fn join(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    fn top() -> Self {
        Self(i32::MAX, usize::MAX)
    }
    fn bottom() -> Self {
        Self(i32::MIN, 0)
    }
}

/// Calculated by
///   - t_free
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FreeCriterion(isize);

impl FreeCriterion {
    fn new(t_free: Option<Index>) -> Self {
        Self(
            -(t_free
                .map(|x| usize::from(x) as isize)
                .unwrap_or(isize::MIN)),
        )
    }

    fn freed_at(t_free: Index) -> Self {
        Self::new(Some(t_free))
    }
}

impl Criterion for FreeCriterion {
    fn join(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    fn top() -> Self {
        Self(isize::MAX)
    }
    fn bottom() -> Self {
        Self(isize::MIN)
    }
}

/// Calculated by
///   - t_used
/// where t_used is the time step the object is last used
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LruCriterion(isize);

impl LruCriterion {
    fn new(t_used: Option<Index>) -> Self {
        Self(
            -(t_used
                .map(|x| usize::from(x) as isize)
                .unwrap_or(isize::MIN)),
        )
    }
}

impl Criterion for LruCriterion {
    fn join(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    fn top() -> Self {
        Self(isize::MAX)
    }
    fn bottom() -> Self {
        Self(isize::MIN)
    }
}

pub struct CriterionManager {
    last_use: ObjectDeviceTable<Index>,
    next_uses: NextUses,
    scheduled_updates: VecDeque<(Object, Device)>,
    n: usize,
}

impl Manager for CriterionManager {
    type Su<'c> = ScheduledUpdates<'c>;
    type C = VictimCriterion;
    type Cf = FreeCriterion;

    fn new(next_uses: NextUses, n_gpus: usize) -> Self {
        Self {
            next_uses,
            last_use: ObjectDeviceTable::new(n_gpus),
            scheduled_updates: VecDeque::new(),
            n: 3,
        }
    }

    fn victim_criterion(&mut self, object: &Object, device: Device, pc: Index) -> VictimCriterion {
        VictimCriterion::new(
            pc,
            self.last_use.get(object, device).copied(),
            self.next_uses.query_next_use(object, device, pc),
            self.n,
        )
    }

    fn mark_use(&mut self, object: &Object, device: Device, pc: Index) -> VictimCriterion {
        let r = VictimCriterion::new(
            pc,
            self.last_use.get(object, device).copied(),
            self.next_uses.mark_use(object, device, pc),
            self.n,
        );

        self.last_use.set(object, device, pc);
        self.scheduled_updates.push_back((object.clone(), device));

        r
    }

    fn scheduled_updates<'a>(&'a mut self, pc: Index) -> ScheduledUpdates<'a> {
        ScheduledUpdates {
            man: self,
            retains: VecDeque::new(),
            pc,
        }
    }

    fn free_criterion(&mut self, _object: &Object, _device: Device, pc: Index) -> FreeCriterion {
        FreeCriterion::freed_at(pc)
    }

    fn next_uses_mut(&mut self) -> &mut NextUses {
        &mut self.next_uses
    }
}

pub struct ScheduledUpdates<'c> {
    man: &'c mut CriterionManager,
    retains: VecDeque<(Object, Device)>,
    pc: Index,
}

impl<'c> Iterator for ScheduledUpdates<'c> {
    type Item = (Object, Device, VictimCriterion);
    fn next(&mut self) -> Option<Self::Item> {
        let pc = self.pc;

        if let Some((object, device)) = self.man.scheduled_updates.pop_front() {
            let last_use = *self.man.last_use.get(&object, device).expect(
                "to have set its last_use before pushing (object, device) into queue scheduled_updates",
            );

            if usize::from(pc) - usize::from(last_use) < self.man.n {
                self.retains.push_back((object.clone(), device));
            }

            let c = VictimCriterion::new(
                pc,
                Some(last_use),
                self.man.next_uses.query_next_use(&object, device, pc),
                self.man.n,
            );

            Some((object, device, c))
        } else {
            self.man
                .scheduled_updates
                .extend(std::mem::take(&mut self.retains).into_iter());
            None
        }
    }
}

pub struct LruManager {
    next_use: NextUses,
    last_use: ObjectDeviceTable<Index>,
}

impl Manager for LruManager {
    type Su<'c>
        = std::iter::Empty<(Object, Device, Self::C)>
    where
        Self: 'c;
    type C = LruCriterion;
    type Cf = FreeCriterion;

    fn new(next_use: NextUses, n_gpus: usize) -> Self {
        Self {
            next_use,
            last_use: ObjectDeviceTable::new(n_gpus),
        }
    }

    fn free_criterion(&mut self, object: &Object, device: Device, pc: Index) -> Self::Cf {
        FreeCriterion::freed_at(pc)
    }

    fn mark_use(&mut self, object: &Object, device: Device, pc: Index) -> Self::C {
        self.last_use.set(object, device, pc);
        LruCriterion::new(Some(pc))
    }

    fn scheduled_updates<'c>(&'c mut self, pc: Index) -> Self::Su<'c> {
        std::iter::empty()
    }

    fn victim_criterion(&mut self, object: &Object, device: Device, pc: Index) -> Self::C {
        LruCriterion::new(Some(pc))
    }

    fn next_uses_mut(&mut self) -> &mut NextUses {
        &mut self.next_use
    }
}

pub struct EpsilonManager {
    next_use: NextUses,
}

impl Manager for EpsilonManager {
    type Su<'c>
        = std::iter::Empty<(Object, Device, Self::C)>
    where
        Self: 'c;
    type C = Epsilon;
    type Cf = Epsilon;

    fn new(next_use: NextUses, _n_gpus: usize) -> Self {
        EpsilonManager { next_use }
    }

    fn victim_criterion(&mut self, _object: &Object, _device: Device, _pc: Index) -> Self::C {
        Epsilon::Top
    }

    fn mark_use(&mut self, object: &Object, device: Device, pc: Index) -> Self::C {
        self.next_use.mark_use(object, device, pc);
        Epsilon::Top
    }

    fn scheduled_updates<'c>(&'c mut self, _pc: Index) -> Self::Su<'c> {
        std::iter::empty()
    }

    fn free_criterion(&mut self, _object: &Object, _device: Device, _pc: Index) -> Self::Cf {
        Epsilon::Top
    }

    fn next_uses_mut(&mut self) -> &mut NextUses {
        &mut self.next_use
    }
}
