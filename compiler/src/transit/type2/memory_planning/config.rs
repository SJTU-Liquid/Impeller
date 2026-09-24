use crate::driver::MemoryInfo;

use super::prelude::*;
use super::PointerId;
use allocators::{PageAllocator, SlabAllocator};

#[derive(Debug, Clone, Copy)]
pub enum GpuAllocatorChoice {
    Slab,
    Page(/* page size */ u64),
}

impl GpuAllocatorChoice {
    pub fn unwrap_page(&self) -> u64 {
        match self {
            GpuAllocatorChoice::Page(x) => *x,
            _ => panic!("called unwrap_page on Slab"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CriterionChoice {
    Epsilon,
    Belady,
    Lru,
}

#[derive(Debug, Clone)]
pub struct Config {
    gpu_smithereen_space: u64,
    cpu_smithereen_space: u64,
    gpu_allocator: GpuAllocatorChoice,
    criterion: CriterionChoice,
}

impl Config {
    pub fn gpu_integral_space(&self, mem: &MemoryInfo) -> u64 {
        mem.memory_limit() - self.gpu_smithereen_space
    }

    pub fn gpu_smithereen_space(&self) -> u64 {
        self.gpu_smithereen_space
    }

    pub fn cpu_integral_space(&self, mem: &MemoryInfo) -> u64 {
        mem.memory_limit() - self.cpu_smithereen_space
    }

    pub fn gpu_page_number(&self, mem: &MemoryInfo) -> u64 {
        self.gpu_integral_space(mem) / self.gpu_allocator.unwrap_page()
    }

    pub fn min_gpu_integral_space(&self, hd_info: &HardwareInfo) -> u64 {
        hd_info
            .gpus()
            .map(|gpu| self.gpu_integral_space(gpu))
            .min()
            .unwrap_or_else(|| panic!("system has no GPU"))
    }

    pub fn gpu_allocator(&self) -> GpuAllocatorChoice {
        self.gpu_allocator
    }

    pub fn with_gpu_smithereen_space(self, x: u64) -> Self {
        Config {
            gpu_smithereen_space: x,
            ..self
        }
    }

    pub fn with_cpu_smithereen_space(self, x: u64) -> Self {
        Config {
            cpu_smithereen_space: x,
            ..self
        }
    }

    pub fn with_smithereen_space(self, x: u64) -> Self {
        Config {
            cpu_smithereen_space: x,
            gpu_smithereen_space: x,
            ..self
        }
    }

    pub fn with_gpu_allocator(self, x: GpuAllocatorChoice) -> Self {
        Config {
            gpu_allocator: x,
            ..self
        }
    }

    pub fn with_criterion(self, x: CriterionChoice) -> Self {
        Config {
            criterion: x,
            ..self
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpu_smithereen_space: 2u64.pow(25),
            cpu_smithereen_space: 2u64.pow(25),
            gpu_allocator: GpuAllocatorChoice::Slab,
            criterion: CriterionChoice::Belady,
        }
    }
}

impl Config {
    pub fn make_planner<Rt: RuntimeType>(
        &self,
        objects: &ObjectDefs,
        hd_info: &HardwareInfo,
        analysis: &type2::object_analysis::DefUse,
    ) -> Box<dyn super::MemoryPlanners<Rt>> {
        use CriterionChoice::*;
        use GpuAllocatorChoice::*;

        match (self.criterion, self.gpu_allocator) {
            (Belady, Slab) => Box::new(super::MemoryPlanner::<
                criterion::CriterionManager,
                SlabAllocator<PointerId, criterion::VictimCriterion, criterion::FreeCriterion>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_integral_space(m),
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
            (Belady, Page(page_size)) => Box::new(super::MemoryPlanner::<
                criterion::CriterionManager,
                PageAllocator<PointerId, criterion::VictimCriterion, criterion::FreeCriterion>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_page_number(m),
                page_size,
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
            (Epsilon, Slab) => Box::new(super::MemoryPlanner::<
                criterion::EpsilonManager,
                SlabAllocator<PointerId, criterion::Epsilon, criterion::Epsilon>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_integral_space(m),
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
            (Epsilon, Page(page_size)) => Box::new(super::MemoryPlanner::<
                criterion::EpsilonManager,
                PageAllocator<PointerId, criterion::Epsilon, criterion::Epsilon>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_page_number(m),
                page_size,
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
            (Lru, Slab) => Box::new(super::MemoryPlanner::<
                criterion::LruManager,
                SlabAllocator<PointerId, criterion::LruCriterion, criterion::FreeCriterion>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_integral_space(m),
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
            (Lru, Page(page_size)) => Box::new(super::MemoryPlanner::<
                criterion::LruManager,
                PageAllocator<PointerId, criterion::LruCriterion, criterion::FreeCriterion>,
                Rt,
            >::new(
                objects,
                hd_info,
                analysis,
                |m| self.gpu_page_number(m),
                page_size,
                self.gpu_smithereen_space,
                self.cpu_integral_space(hd_info.cpu()),
                self.cpu_smithereen_space,
            )),
        }
    }
}
