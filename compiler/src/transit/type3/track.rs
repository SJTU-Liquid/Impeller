use super::lowering;
use crate::transit::type2;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    MemoryManagement,
    CoProcess(usize),
    Gpu(usize),
    Cpu,
    ToGpu,
    FromGpu,
    GpuMemory(usize),
    ToDisk,
    FromDisk,
}

impl From<Track> for zkpoly_runtime::instructions::Track {
    fn from(track: Track) -> Self {
        use Track::*;
        match track {
            MemoryManagement => Self::MemoryManagement,
            CoProcess(i) => Self::CoProcess(i),
            Gpu(i) => Self::Gpu(i),
            Cpu => Self::Cpu,
            ToGpu => Self::ToGpu,
            FromGpu => Self::FromGpu,
            GpuMemory(i) => Self::GpuMemory(i),
            ToDisk => Self::ToDisk,
            FromDisk => Self::FromDisk,
        }
    }
}

impl Track {
    pub fn is_gpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_some()
    }

    pub fn is_cpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_none()
    }

    pub fn coprocess_to_gpu(&self) -> Track {
        match self {
            Track::CoProcess(i) => Track::Gpu(*i),
            _ => panic!("called coprocess_to_gpu on {:?}", self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackSpecific<T> {
    pub(crate) memory_management: T,
    pub(crate) co_process: Vec<T>,
    pub(crate) gpu: Vec<T>,
    pub(crate) cpu: T,
    pub(crate) to_gpu: T,
    pub(crate) from_gpu: T,
    pub(crate) gpu_memory: Vec<T>,
    pub(crate) to_disk: T,
    pub(crate) from_disk: T,
}

impl<T> TrackSpecific<T> {
    pub fn default(n_gpus: usize) -> Self
    where
        T: Default,
    {
        Self {
            memory_management: T::default(),
            co_process: (0..n_gpus).map(|_| T::default()).collect(),
            gpu: (0..n_gpus).map(|_| T::default()).collect(),
            cpu: T::default(),
            to_gpu: T::default(),
            from_gpu: T::default(),
            gpu_memory: (0..n_gpus).map(|_| T::default()).collect(),
            to_disk: T::default(),
            from_disk: T::default(),
        }
    }

    pub fn get_track(&self, track: Track) -> &T {
        use Track::*;
        match track {
            MemoryManagement => &self.memory_management,
            CoProcess(i) => &self.co_process[i],
            Gpu(i) => &self.gpu[i],
            Cpu => &self.cpu,
            ToGpu => &self.to_gpu,
            FromGpu => &self.from_gpu,
            GpuMemory(i) => &self.gpu_memory[i],
            ToDisk => &self.to_disk,
            FromDisk => &self.from_disk,
        }
    }

    pub fn get_track_mut(&mut self, track: Track) -> &mut T {
        use Track::*;
        match track {
            MemoryManagement => &mut self.memory_management,
            CoProcess(i) => &mut self.co_process[i],
            Gpu(i) => &mut self.gpu[i],
            Cpu => &mut self.cpu,
            ToGpu => &mut self.to_gpu,
            FromGpu => &mut self.from_gpu,
            GpuMemory(i) => &mut self.gpu_memory[i],
            ToDisk => &mut self.to_disk,
            FromDisk => &mut self.from_disk,
        }
    }

    pub fn iter(&self, n_gpus: usize) -> impl Iterator<Item = (Track, &T)> {
        use Track::*;
        vec![
            (MemoryManagement, &self.memory_management),
            (Cpu, &self.cpu),
            (ToGpu, &self.to_gpu),
            (FromGpu, &self.from_gpu),
            (ToDisk, &self.to_disk),
            (FromDisk, &self.from_disk),
        ]
        .into_iter()
        .chain((0..n_gpus).map(|i| (Gpu(i), &self.gpu[i])))
        .chain((0..n_gpus).map(|i| (CoProcess(i), &self.co_process[i])))
        .chain((0..n_gpus).map(|i| (GpuMemory(i), &self.gpu_memory[i])))
    }

    pub fn new(t: T, n_gpus: usize) -> Self
    where
        T: Clone,
    {
        Self {
            memory_management: t.clone(),
            co_process: vec![t.clone(); n_gpus],
            gpu: vec![t.clone(); n_gpus],
            cpu: t.clone(),
            to_gpu: t.clone(),
            from_gpu: t.clone(),
            gpu_memory: vec![t.clone(); n_gpus],
            to_disk: t.clone(),
            from_disk: t.clone(),
        }
    }
}

impl Track {
    pub fn on_device(device: type2::Device) -> Track {
        use type2::Device::*;
        match device {
            Gpu(i) => Track::Gpu(i),
            Cpu => Track::Cpu,
        }
    }
}
