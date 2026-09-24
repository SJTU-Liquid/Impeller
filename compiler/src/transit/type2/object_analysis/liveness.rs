use std::{borrow::Borrow, collections::VecDeque};

use crate::{
    driver,
    transit::type3::{Device, DeviceSpecific},
};

use super::value::{Object, ObjectTable};
use super::{template::ObjectRw, Index};

/// For each object on each device, a list of operation indices where it is used.
///
/// If an object is never used, the currespounding entry in the map may be empty;
/// If an object is never used on some device, the currespounding [`Vec`] can also be empty.
///
/// The vector of indices for each object and device is guaranteed to be sorted.
#[derive(Debug, Clone)]
pub struct ReadBy(ObjectTable<DeviceSpecific<Vec<Index>>>, usize);

impl ReadBy {
    pub fn add_use(&mut self, index: Index, object: Object, device: Device) {
        if !self.0.contains(&object) {
            self.0
                .insert(object.clone(), DeviceSpecific::default(self.1));
        }

        self.0.get_mut(&object).unwrap().get_mut(device).push(index);
    }

    /// Caution that uses provided here are not necessarily the real uses during memory planning.
    /// For example, `ReclaimObject`'s device of reclaiming from is only an advice,
    /// but in reality, the reclaiming object may have been popped and we'll turn to its parent device.
    ///
    /// However, it can be guaranteed that the real uses are subset of the uses inferred here,
    /// so it's safe to deallocate object after last inferred use.
    pub fn analyze<'s, R, U>(
        ops: impl Iterator<Item = (Index, U)>,
        hd_info: &driver::HardwareInfo,
    ) -> Self
    where
        R: ObjectRw<Object>,
        U: Borrow<R>,
    {
        let mut used_by = Self(ObjectTable::new(), hd_info.n_gpus());

        for (index, op) in ops {
            op.borrow().object_reads().for_each(|(object, device)| {
                used_by.add_use(index, object.clone(), device);
            });
        }

        used_by
    }
}

/// For each object on each device, we store the indices where it is defined.
pub struct WriteAt(ObjectTable<DeviceSpecific<Vec<Index>>>, usize);

impl WriteAt {
    fn add_def(&mut self, index: Index, object: Object, device: Device) {
        if !self.0.contains(&object) {
            self.0
                .insert(object.clone(), DeviceSpecific::default(self.1));
        }

        self.0.get_mut(&object).unwrap().get_mut(device).push(index);
    }

    pub fn analyze<'s, R, U>(
        ops: impl Iterator<Item = (Index, U)>,
        hd_info: &driver::HardwareInfo,
    ) -> Self
    where
        R: ObjectRw<Object>,
        U: Borrow<R>,
        R: std::fmt::Debug,
    {
        let mut def_at = Self(ObjectTable::new(), hd_info.n_gpus());

        for (index, op) in ops {
            op.borrow().object_writes().for_each(|(object, device)| {
                def_at.add_def(index, object.clone(), device);
            });
        }

        def_at
    }
}

#[derive(Debug, Clone)]
pub struct NextUses(ObjectTable<DeviceSpecific<VecDeque<Index>>>, usize);

impl NextUses {
    pub fn new(reads: ReadBy, writes: WriteAt) -> Self {
        let mut uses: ObjectTable<DeviceSpecific<VecDeque<Index>>> =
            reads.0.map(|uses| uses.map(|v| v.into()));

        writes.0.iter().for_each(|(object, writes_at)| {
            writes_at.iter().for_each(|(d, write_at)| {
                for write_at in write_at {
                    if !uses.contains(&object) {
                        uses.insert(object.clone(), DeviceSpecific::default(writes.1));
                    }
                    uses.get_mut(&object)
                        .unwrap()
                        .get_mut(d)
                        .push_front(*write_at);
                }
            });
        });
        NextUses(uses, reads.1)
    }

    fn next_uses_queue_of(&mut self, object: &Object, device: Device) -> &mut VecDeque<Index> {
        if !self.0.contains(&object) {
            self.0
                .insert(object.clone(), DeviceSpecific::default(self.1));
        }

        self.0.get_mut(&object).unwrap().get_mut(device)
    }

    fn pop_next_use_until(
        &mut self,
        object: &Object,
        device: Device,
        criterion: impl Fn(Index) -> bool,
    ) -> Option<Index> {
        while let Some(hd) = self.next_uses_queue_of(object, device).pop_front() {
            if criterion(hd) {
                self.next_uses_queue_of(object, device).push_front(hd);
                return Some(hd);
            }
        }
        None
    }

    /// Returns the next use of `object` on `device` after (and including) current pc.
    pub fn query_next_use(&mut self, object: &Object, device: Device, pc: Index) -> Option<Index> {
        let next_use = self.pop_next_use_until(object, device, |hd| hd >= pc)?;

        Some(next_use)
    }

    /// Mark that we are using the obejct on the device now, so all uses before (and including) now
    /// are popped from the queue.
    /// We'll use the first index after the current pc as the next use, if any.
    pub fn mark_use(&mut self, object: &Object, device: Device, pc: Index) -> Option<Index> {
        self.pop_next_use_until(object, device, |hd| hd > pc)
    }

    /// Check if `object` will be not used on some devices
    pub fn will_not_be_used_on(
        &mut self,
        object: &Object,
        devices: impl Iterator<Item = Device>,
        pc: Index,
    ) -> bool {
        devices
            .into_iter()
            .all(|device| self.query_next_use(object, device, pc).is_none())
    }

    pub fn next_used_device(
        &mut self,
        object: &Object,
        mut devices: impl Iterator<Item = Device>,
        pc: Index,
    ) -> Option<Device> {
        devices.find(|device| self.query_next_use(object, *device, pc).is_some())
    }
}
