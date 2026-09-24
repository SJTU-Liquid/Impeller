//! This module provides methods to drive the compiler through various intermediate representations,
//! eventually producing the [`Artifect`] which can be run by the dispatcher.

use std::time::{Duration, Instant};

use zkpoly_cuda_api::{bindings::cudaDeviceSynchronize, cuda_check};
use zkpoly_runtime::args::RuntimeType;

use crate::{
    ast,
    transit::{type2, type3},
};

pub use ast::ConstantPool;

mod hardware;
pub use hardware::{DiskMemoryInfo, HardwareInfo, MemoryInfo};
pub use zkpoly_memory_pool::buddy_disk_pool::DiskIoMode;

mod options;
pub use options::{Config, DebugOptions, Type2DebugVisualizer};

pub use type2::{
    graph_scheduling::Algorithm as GraphSchedulingAlgorithm,
    memory_planning::{
        config::CriterionChoice, config::GpuAllocatorChoice, Config as MemoryPlanningConfig,
    },
    subgraph_slicing::Config as SubgraphSlicingConfig,
};
pub use type3::lowering::ArithGraphScheduler as ArithGraphSchedulerChoice;

mod versions;
use versions::Versions;

#[derive(Debug)]
pub enum Error<'s, Rt: RuntimeType> {
    Typ(ast::lowering::Error<'s, Rt>),
    Io(std::io::Error, String),
    MemoryPlanning(type2::memory_planning::Error<'s>, String),
    NotDag(String),
}

impl<'s, Rt: RuntimeType> Error<'s, Rt> {
    pub fn io_when(err: std::io::Error, when: impl Into<String>) -> Self {
        Self::Io(err, when.into())
    }

    pub fn not_dat_after(when: impl Into<String>) -> Self {
        Self::NotDag(when.into())
    }

    pub fn memory_planning(e: type2::memory_planning::Error<'s>, when: impl Into<String>) -> Self {
        Self::MemoryPlanning(e, when.into())
    }
}

#[derive(Clone, Debug)]
struct Context<'o> {
    options: &'o DebugOptions,
    io_time: Duration,
    current_pass_name: String,
}

impl<'o> Context<'o> {
    fn new(options: &'o DebugOptions, name: impl Into<String>) -> Self {
        Self {
            options,
            io_time: Duration::ZERO,
            current_pass_name: name.into(),
        }
    }

    fn pass<'s, R, Rt: RuntimeType>(
        mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Error<'s, Rt>>,
    ) -> Result<R, Error<'s, Rt>> {
        if self.options.log {
            println!("[Compiler] Applying {}", &self.current_pass_name);
        }
        let begin = Instant::now();
        let r = f(&mut self)?;
        let elapsed = begin.elapsed();
        if self.options.log {
            println!(
                "[Compiler] Finished {} - took {:?}, where debugging took {:?} and pass took {:?}",
                &self.current_pass_name,
                elapsed,
                self.io_time,
                elapsed - self.io_time
            );
        }
        Ok(r)
    }

    fn hash<T>(&self, t: &T) -> u64
    where
        T: std::hash::Hash,
    {
        use std::hash::{DefaultHasher, Hasher};
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    fn pass_print_hash<'s, R: std::hash::Hash, Rt: RuntimeType>(
        mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Error<'s, Rt>>,
    ) -> Result<R, Error<'s, Rt>> {
        if self.options.log {
            println!("[Compiler] Applying {}", &self.current_pass_name);
        }
        let begin = Instant::now();
        let r = f(&mut self)?;
        let elapsed = begin.elapsed();
        if self.options.log {
            println!(
                "[Compiler] Finished {} - took {:?}, where debugging took {:?} and pass took {:?}",
                &self.current_pass_name,
                elapsed,
                self.io_time,
                elapsed - self.io_time
            );

            println!("[Compiler] Pass hash is {:#x}", { self.hash(&r) })
        }

        Ok(r)
    }

    fn create_debug_graph_file(&self, name: &str) -> std::io::Result<std::fs::File> {
        let path = self.options.debug_dir.join(name).with_extension("html");
        std::fs::File::create(path)
    }

    fn create_debug_text_file(&self, name: &str) -> std::io::Result<std::fs::File> {
        let path = self.options.debug_dir.join(name).with_extension("txt");
        std::fs::File::create(path)
    }

    fn create_debug_html_file(&self, name: &str) -> std::io::Result<std::fs::File> {
        std::fs::File::create(self.options.debug_dir.join(name).with_extension("html"))
    }

    fn check_type2_unsubgraphed_dag<'s, Rt: RuntimeType>(
        &self,
        g: &type2::no_subgraph::Cg<'s, Rt>,
    ) -> Result<(), Error<'static, Rt>> {
        let err = |e| {
            Error::io_when(
                e,
                format!(
                    "debugging not-dag error after pass {}",
                    &self.current_pass_name
                ),
            )
        };

        if let Some(cycle) = g.g.try_find_cycle() {
            type2::pretty::unsubgraphed_cg::debug_cycle(
                g,
                &cycle,
                self.create_debug_graph_file(&(self.name().to_string() + " Not DAG"))
                    .map_err(err)?,
            )
            .map_err(err)?;
            Err(Error::not_dat_after(&self.current_pass_name))
        } else {
            Ok(())
        }
    }

    fn check_type2_subgraphed_dag<'s, Rt: RuntimeType>(
        &self,
        g: &type2::Cg<'s, Rt>,
    ) -> Result<(), Error<'static, Rt>> {
        let err = |e| {
            Error::io_when(
                e,
                format!(
                    "debugging not-dag error after pass {}",
                    &self.current_pass_name
                ),
            )
        };

        if let Some(cycle) = g.g.try_find_cycle() {
            type2::pretty::subgraphed_cg::debug_cycle(
                g,
                &cycle,
                self.create_debug_graph_file(&(self.name().to_string() + " Not DAG"))
                    .map_err(err)?,
            )
            .map_err(err)?;
            Err(Error::not_dat_after(&self.current_pass_name))
        } else {
            Ok(())
        }
    }

    fn debug_type2_unsubgraphed<'s, Rt: RuntimeType>(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        g: &type2::no_subgraph::Cg<'s, Rt>,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                type2::pretty::unsubgraphed_cg::debug_graph(
                    g,
                    ctx.create_debug_graph_file(&ctx.current_pass_name)?,
                )?;
            }
            Ok(())
        })
    }

    fn debug_type2_subgraphed<'s, Rt: RuntimeType>(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        g: &type2::Cg<'s, Rt>,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                type2::pretty::subgraphed_cg::debug_graph(
                    g,
                    ctx.create_debug_graph_file(&ctx.current_pass_name)?,
                )?;
            }
            Ok(())
        })
    }

    fn debug_type2_subgraph_schedule<'g, 's, Rt: RuntimeType>(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        schedule: &type2::graph_scheduling::SubgraphsSchedule,
        g: &type2::CgSubgraph<'g, 's, Rt>,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                let file = ctx.create_debug_graph_file(ctx.name())?;
                type2::pretty::subgraphed_cg::debug_subgraphs_schedule(&g, &schedule, file)?;
            }
            Ok(())
        })
    }

    fn debug_text<T: std::fmt::Debug, Rt: RuntimeType>(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        t: &T,
        extra_name: &str,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                let mut file =
                    ctx.create_debug_text_file(&(ctx.name().to_string() + extra_name))?;
                use std::io::Write;
                writeln!(file, "{:#?}", t)?;
            }
            Ok(())
        })
    }

    fn indent_debug_text<
        T: zkpoly_common::indent_formatter::IndentDebug<std::fs::File>,
        Rt: RuntimeType,
    >(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        t: &T,
        extra_name: &str,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                let file = ctx.create_debug_text_file(&(ctx.name().to_string() + extra_name))?;
                let mut formatter = zkpoly_common::indent_formatter::Formatter::new(file);
                t.fmt(&mut formatter)?;
            }
            Ok(())
        })
    }

    fn display_text<T: std::fmt::Display, Rt: RuntimeType>(
        &mut self,
        option: impl Fn(&DebugOptions) -> bool,
        t: &T,
        extra_name: &str,
    ) -> Result<(), Error<'static, Rt>> {
        self.io(|ctx| {
            if option(&ctx.options) {
                let mut file =
                    ctx.create_debug_text_file(&(ctx.name().to_string() + extra_name))?;
                use std::io::Write;
                writeln!(file, "{}", t)?;
            }
            Ok(())
        })
    }

    fn io<R, Rt: RuntimeType>(
        &mut self,
        f: impl FnOnce(&mut Self) -> std::io::Result<R>,
    ) -> Result<R, Error<'static, Rt>> {
        let begin = Instant::now();
        let r = f(self).map_err(|e| Error::io_when(e, &self.current_pass_name))?;
        self.io_time += begin.elapsed();
        Ok(r)
    }

    fn name(&self) -> &str {
        &self.current_pass_name
    }
}

pub mod artifect;
pub mod fresh_type2;
pub mod fresh_type3;
pub mod processed_type2;
pub mod processed_type3;
pub mod unfused_type2;

pub use artifect::Artifect;
pub use fresh_type2::FreshType2;
pub use fresh_type3::FreshType3;
pub use processed_type2::ProcessedType2;
pub use processed_type3::ProcessedType3;
pub use unfused_type2::UnfusedType2;
