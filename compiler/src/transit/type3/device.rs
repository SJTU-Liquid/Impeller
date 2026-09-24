use crate::transit::type2;

#[derive(Debug, Clone, Hash, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Gpu(usize),
    Cpu,
    Disk,
}

impl Device {
    pub fn iter(n_gpus: usize) -> impl Iterator<Item = Device> {
        (0..n_gpus)
            .map(Self::Gpu)
            .chain([Device::Cpu, Device::Disk].into_iter())
    }

    /// Returns the parent memory device of current one, if any.
    /// Cold objects are rejected to a memory device's parent device.
    pub fn parent(self) -> Option<Self> {
        use Device::*;
        match self {
            Gpu(..) => Some(Cpu),
            Cpu => Some(Disk),
            Disk => None,
        }
    }

    pub fn for_execution_on(device: type2::Device) -> Self {
        use type2::Device::*;
        match device {
            Gpu(i) => Device::Gpu(i),
            Cpu => Device::Cpu,
        }
    }

    pub fn unwrap_gpu(&self) -> usize {
        if let Self::Gpu(i) = self {
            *i
        } else {
            panic!("called unwrap_gpu on {:?}", self)
        }
    }
}

#[derive(Debug, Clone, Hash, serde::Serialize, serde::Deserialize)]
pub struct DeviceSpecific<T> {
    pub gpu: Vec<T>,
    pub cpu: T,
    pub disk: T,
}

impl<T> DeviceSpecific<T> {
    pub fn get(&self, device: Device) -> &T {
        match device {
            Device::Gpu(i) => &self.gpu[i],
            Device::Cpu => &self.cpu,
            Device::Disk => &self.disk,
        }
    }

    pub fn get_mut(&mut self, device: Device) -> &mut T {
        match device {
            Device::Gpu(i) => &mut self.gpu[i],
            Device::Cpu => &mut self.cpu,
            Device::Disk => &mut self.disk,
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DeviceSpecific<U> {
        DeviceSpecific {
            gpu: self.gpu.into_iter().map(&mut f).collect(),
            cpu: (&mut f)(self.cpu),
            disk: f(self.disk),
        }
    }

    pub fn default(n_gpus: usize) -> Self
    where
        T: Default,
    {
        DeviceSpecific {
            gpu: (0..n_gpus).map(|_| Default::default()).collect(),
            cpu: T::default(),
            disk: T::default(),
        }
    }

    pub fn new(n_gpus: usize, mut f: impl FnMut(Device) -> T) -> Self {
        use Device::*;

        DeviceSpecific {
            gpu: (0..n_gpus).map(|i| f(Gpu(i))).collect(),
            cpu: f(Cpu),
            disk: f(Disk),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Device, &T)> {
        Device::iter(self.gpu.len()).map(|d| (d, self.get(d)))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Device, &mut T)> {
        std::iter::once((Device::Cpu, &mut self.cpu))
            .chain(
                self.gpu
                    .iter_mut()
                    .enumerate()
                    .map(|(i, t)| (Device::Gpu(i), t)),
            )
            .chain(std::iter::once((Device::Disk, &mut self.disk)))
    }
}

impl std::ops::Sub<Self> for DeviceSpecific<bool> {
    type Output = Self;
    fn sub(self, rhs: Self::Output) -> Self::Output {
        DeviceSpecific {
            gpu: self
                .gpu
                .into_iter()
                .zip(rhs.gpu)
                .map(|(a, b)| a && !b)
                .collect(),
            cpu: self.cpu && !rhs.cpu,
            disk: self.disk && !rhs.disk,
        }
    }
}
