use super::super::prelude::*;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Stage {
    Planning,
    Unplanned,
    Planned,
}

impl Stage {
    pub fn is_planning(&self) -> bool {
        matches!(self, Self::Planning)
    }
    pub fn is_planning_or_unplanned(&self) -> bool {
        matches!(self, Self::Planning | Self::Unplanned)
    }
    pub fn is_unplanned(&self) -> bool {
        matches!(self, Self::Unplanned)
    }
    pub fn is_planned(&self) -> bool {
        matches!(self, Self::Planned)
    }
}

#[derive(Clone, Debug)]
pub struct DevicesStage {
    devices: DeviceSpecific<Stage>,
}

impl DevicesStage {
    fn devices_for_stage<'a>(&'a self, stage: Stage) -> impl Iterator<Item = Device> + 'a {
        self.devices
            .iter()
            .filter(move |(_, s)| **s == stage)
            .map(|(d, _)| d)
    }

    pub fn planning_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.devices_for_stage(Stage::Planning)
    }

    /// Planning then unplanned devices
    pub fn planning_unplanned_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.devices_for_stage(Stage::Planning)
            .chain(self.devices_for_stage(Stage::Unplanned))
    }
    pub fn unplanned_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.devices_for_stage(Stage::Unplanned)
    }
    pub fn planned_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.devices_for_stage(Stage::Planned)
    }

    pub fn all_unplanned(n_gpus: usize) -> Self {
        Self {
            devices: DeviceSpecific::new(n_gpus, |_| Stage::Unplanned),
        }
    }

    pub fn advance(&mut self, now_plannning: impl Iterator<Item = Device>) {
        self.devices.iter_mut().for_each(|(_, st)| {
            if matches!(*st, Stage::Planning) {
                *st = Stage::Planned;
            }
        });

        now_plannning.for_each(|d| {
            *self.devices.get_mut(d) = Stage::Planning;
        });
    }
}

impl std::ops::Index<Device> for DevicesStage {
    type Output = Stage;
    fn index(&self, index: Device) -> &Self::Output {
        self.devices.get(index)
    }
}
