use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::RuntimeType;

use super::{processed_type2::ProcessedType2, Error};
use crate::{
    driver::{Config, DebugOptions, HardwareInfo, Versions},
    transit::type2,
};

/// The Type2 IR before kernel fusion.
pub struct UnfusedType2<'s, 'i, Rt: RuntimeType> {
    pub(super) cg: type2::no_subgraph::Cg<'s, Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
    pub(super) config: &'i Config,
    pub(super) options: &'i DebugOptions,
    pub(super) hardware_info: &'i HardwareInfo,
}

impl<'s, 'i, Rt: RuntimeType> UnfusedType2<'s, 'i, Rt> {
    /// Apply kernel fusion.
    ///
    /// `versions_cpu_memory_divisors` control different versions to compile for different CPU
    /// memory usage. For example, `[1, 2, 4]` with CPU memory size set to 64GB yields versions
    /// that run with 64GB, 32GB and 16GB.
    pub fn fuse(
        self,
        versions_cpu_memory_divisors: impl Iterator<Item = u32>,
    ) -> Result<ProcessedType2<'s, 'i, Rt>, Error<'s, Rt>> {
        let UnfusedType2 {
            cg: t2cg,
            constant_table,
            uf_table,
            mut libs,
            options,
            hardware_info,
            config,
        } = self;

        // - Arithmetic Kernel Fusion
        let t2cg = if true {
            options
                .context("Arithmetic Kernel Fusion")
                .pass_print_hash(|ctx| {
                    let fused = type2::kernel_fusion::fuse_arith(
                        t2cg.clone(),
                        hardware_info,
                        &config.memory,
                        config.sliceable_subgraph.as_ref().map(|s| s.chunk_len()),
                        ctx.options.log,
                        config.arith_graph_scheudler,
                    );
                    let (compacted, remap) = fused.compact();

                    if options.log {
                        println!("[Kernel Fusion] {} vertices left", compacted.g.order());
                    }

                    ctx.check_type2_unsubgraphed_dag(&compacted)?;
                    ctx.debug_type2_unsubgraphed(|o| o.debug_kernel_fusion, &compacted)?;
                    ctx.debug_text(|o| o.debug_kernel_fusion, &remap, " Remapping")?;
                    Ok(compacted)
                })?
        } else {
            t2cg
        };

        let t2cg_versions = Versions::build(
            hardware_info,
            versions_cpu_memory_divisors,
            |_cpu, helper| {
                let t2cg = t2cg.clone();

                // - Sliceable Subgraph Fusion
                let t2cg = if let Some(ss_config) = &config.sliceable_subgraph {
                    options
                        .context(helper.pass_name("Sliceable Subgraph Fusion"))
                        .pass(|ctx| {
                            let (fused, statistics) = type2::subgraph_slicing::fuse(
                                t2cg,
                                ss_config,
                                hardware_info,
                                &config.memory,
                            );
                            ctx.check_type2_subgraphed_dag(&fused)?;
                            ctx.debug_type2_subgraphed(|o| o.debug_sliceable_subgraph, &fused)?;

                            if ctx.options.log {
                                println!("[Sliceable Subgraph] Statistics = {:#?}", &statistics);
                            }

                            ctx.debug_text(
                                |o| o.debug_sliceable_subgraph,
                                &statistics,
                                " Statistics",
                            )?;
                            Ok(fused)
                        })?
                } else {
                    type2::subgraph_slicing::id(t2cg)
                };

                if options.log {
                    let (n_ops, io_size) = t2cg.subgraphs_intensity();
                    println!(
                        "Subgraph intensity is {} / {} = {}",
                        n_ops,
                        io_size,
                        n_ops as f64 / io_size as f64
                    );
                    println!(
                        "Number of arithmetic ops in subgraphs is {}",
                        t2cg.n_subgraph_arith_ops()
                    );
                }

                // - Subgraphs Scheduling
                let schedule = options
                    .context(helper.pass_name("Subgraphs Scheduling"))
                    .pass_print_hash(|ctx| {
                        let g = t2cg.connected_subgraph();
                        let schedule = type2::graph_scheduling::SubgraphsSchedule::on(
                            &g,
                            self.config.scheduler_alg,
                        );
                        ctx.debug_type2_subgraph_schedule(
                            |o| o.debug_graph_scheduling,
                            &schedule,
                            &g,
                        )?;
                        Ok(schedule)
                    })?;

                // - Subgraph splitting
                let t2cg = options
                    .context(helper.pass_name("Sliceable Subgraph Splitting"))
                    .pass_print_hash(|ctx| {
                        let (t2cg, statistics) = type2::subgraph_splitting::tighten_chunk_len(
                            t2cg,
                            hardware_info,
                            &config.memory,
                            &schedule,
                            &mut libs,
                            config.arith_graph_scheudler,
                        );

                        if ctx.options.log {
                            println!("[Sliceable Subgraph] New Chunk Length = {:#?}", &statistics)
                        }

                        ctx.debug_text(
                            |o| o.debug_sliceable_subgraph,
                            &statistics,
                            " New Chunk Length",
                        )?;

                        Ok(t2cg)
                    })?;

                Ok((t2cg, schedule))
            },
        )?;

        Ok(ProcessedType2 {
            cg: t2cg_versions,
            constant_table,
            uf_table,
            libs,
            hardware_info,
            options,
            config: self.config,
        })
    }
}
