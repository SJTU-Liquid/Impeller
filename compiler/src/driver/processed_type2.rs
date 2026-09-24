use crate::{
    driver::{fresh_type3, Config},
    transit::type3,
    utils,
};

use super::{fresh_type3::FreshType3, ConstantPool};
use zkpoly_common::{devices::DeviceType, load_dynamic::Libs};
use zkpoly_runtime::args::{self, RuntimeType};

use super::{type2, DebugOptions, Error, HardwareInfo, MemoryInfo, Versions};
use crate::transit::type2::graph_scheduling::SubgraphsSchedule;

/// The intermediate result after applying various passes on Type2.
pub struct ProcessedType2<'s, 'i, Rt: RuntimeType> {
    pub(super) cg: Versions<(type2::Cg<'s, Rt>, SubgraphsSchedule)>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
    pub(super) options: &'i DebugOptions,
    pub(super) hardware_info: &'i HardwareInfo,
    pub(super) config: &'i Config,
}

impl<'s, 'i, Rt: RuntimeType> ProcessedType2<'s, 'i, Rt> {
    /// Lower to Type3 IR by through memory planning and some other passes.
    pub fn to_type3(
        self,
        constant_pool: &mut ConstantPool,
    ) -> Result<FreshType3<'s, 'i, Rt>, Error<'s, Rt>> {
        let Self {
            cg: t2cg_versions,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            mut libs,
            options,
            hardware_info,
            config,
        } = self;

        if hardware_info.disks_available() != constant_pool.disk.as_ref().map_or(0, |x| x.len()) {
            panic!("inconsistent number of disks in hardware info and constant pool");
        }

        // - Decide constants to be on disk or CPU:
        //   If the constants take more than 1/4 CPU space, put big constants on disk.
        let constants_device = options
            .context("Constants Placement Decision")
            .pass(|ctx| {
                let constants_device = t2const_tab.map_by_ref(&mut |_, c| {
                    if config.constants_on_disk && c.typ.can_on_disk::<Rt::Field, Rt::PointAffine>()
                    {
                        type3::Device::Disk
                    } else {
                        type3::Device::Cpu
                    }
                });

                ctx.debug_text(|o| o.debug_memory_planning, &constants_device, "")?;

                Ok(constants_device)
            })?;
        let chunk_versions = t2cg_versions.map_i(|cpu, (t2cg, subgraphs_schedule), helper| {
            let hardware_info = hardware_info.clone().with_cpu(cpu.clone());
            let hardware_info = &hardware_info;

            let g = t2cg.connected_subgraph();

            // - Decide Device
            let devices = type2::decide_device::decide(&g);

            // - Object Analysis
            let (analysis, mut objects) = options
                .context(helper.pass_name("Object Analysis"))
                .pass_print_hash(|ctx| {
                    let mut objects = type2::object_analysis::ObjectDefs::new();

                    let analysis = type2::object_analysis::DefUse::analyze(
                        &g,
                        &t2uf_tab,
                        |vid| &devices[&vid],
                        hardware_info,
                        &constants_device,
                        &mut objects,
                    );

                    ctx.io(|ctx| {
                        if options.debug_obj_analysis {
                            type2::pretty::subgraphed_cg::debug_object_analysis(
                                &g,
                                &analysis,
                                &devices,
                                ctx.create_debug_graph_file(ctx.name())?,
                            )?;
                        }
                        Ok(())
                    })?;

                    ctx.debug_text(|o| o.debug_obj_analysis, &objects, " Objects")?;

                    ctx.debug_text(|o| o.debug_obj_analysis, &analysis, " DefUse")?;

                    Ok((analysis, objects))
                })?;

            // - Graph Scheduling
            let schedule = options.context("Graph Scheduling").pass_print_hash(|ctx| {
                let cg_schedule = type2::graph_scheduling::Schedule::<type2::VertexId>::on(
                    &g,
                    &analysis,
                    &objects,
                    config.scheduler_alg,
                );
                let schedule = type2::graph_scheduling::WholeSchedule {
                    cg_schedule,
                    subgraphs_schedule,
                };

                if options.log {
                    println!(
                        "[Graph Scheduling] Estimated memory required is {}",
                        utils::human_readable_size(schedule.cg_schedule.max_occupied_size())
                    )
                }

                ctx.io(|ctx| {
                    if ctx.options.debug_graph_scheduling {
                        type2::pretty::subgraphed_cg::debug_schedule(
                            &g,
                            &schedule,
                            ctx.create_debug_graph_file(ctx.name())?,
                        )?;
                    }
                    Ok(())
                })?;

                Ok(schedule)
            })?;

            // To Phase0 MOps
            let phase0_mops = options
                .context(helper.pass_name("Phase0 MOps"))
                .pass_print_hash(|ctx| {
                    let object_last_use =
                        type2::object_analysis::construct_ops::ObjectLastUse::analyze(
                            &g, &analysis, &schedule,
                        );

                    ctx.debug_text(
                        |o| o.debug_memory_planning,
                        &object_last_use,
                        " Object Last Use",
                    )?;

                    let mops = type2::memory_planning::phase0::OperationSeq::construct(
                        &g,
                        &schedule,
                        &analysis,
                        |vid| &devices[&vid],
                        hardware_info,
                        &mut objects,
                        &mut libs,
                        &object_last_use,
                        config.arith_graph_scheudler,
                    );

                    ctx.indent_debug_text(|o| o.debug_memory_planning, &mops, "")?;

                    Ok(mops)
                })?;

            // To Phase1 MOps
            let mut phase1_mops = options
                .context(helper.pass_name("Phase1 MOps"))
                .pass_print_hash(|ctx| {
                    let (mops, statistics) = type2::memory_planning::plan_subgraphs::<Rt>(
                        phase0_mops,
                        &mut objects,
                        &config.memory,
                        hardware_info,
                    )
                    .map_err(|e| Error::memory_planning(e, ctx.name()))?;

                    if ctx.options.log {
                        println!("[Phase0 MOps] Statistics = {:#?}", &statistics);
                    }

                    ctx.debug_text(|o| o.debug_memory_planning, &statistics, " Statistics")?;
                    ctx.indent_debug_text(|o| o.debug_memory_planning, &mops, "")?;
                    Ok(mops)
                })?;

            // Planning
            let mut memory_planner = config
                .memory
                .make_planner(&objects, hardware_info, &analysis);

            let mut plan_driver = memory_planner.plan_driver(hardware_info);

            for i in 0.. {
                let (transformed_ops, finished) = options
                    .context(helper.pass_name(&format!("Planning {i}")))
                    .pass_print_hash(|ctx| {
                        let next_uses = plan_driver.analyze_next_uses(&phase1_mops);

                        // This is actually Next Uses Analysis of previous iteration
                        ctx.debug_text(|o| o.debug_memory_planning, &next_uses, " Next Uses")?;

                        let (mops, finished, ejection_statistics) = plan_driver
                            .advance(phase1_mops, next_uses)
                            .map_err(|e| Error::memory_planning(e, ctx.name()))?;

                        if ctx.options.log {
                            println!(
                                "[Planning {}] Ejection Statistics: {:?}",
                                i, ejection_statistics
                            );
                        }
                        ctx.indent_debug_text(|o| o.debug_memory_planning, &mops, "")?;

                        Ok((mops, finished))
                    })?;

                phase1_mops = transformed_ops;

                if finished {
                    break;
                }
            }

            drop(plan_driver);

            // Dependency Analysis
            let dependencies = options
                .context(helper.pass_name("Dependency Analysis"))
                .pass(|ctx| {
                    let analyzer =
                        type2::memory_planning::dependency_analysis::Analyzer::prepare_for(
                            phase1_mops.iter(),
                            hardware_info,
                        );

                    ctx.debug_text(|o| o.debug_memory_planning, &analyzer, " AWR")?;

                    let dependencies = analyzer.analyze(&phase1_mops);

                    ctx.debug_text(|o| o.debug_memory_planning, &dependencies, "")?;

                    Ok(dependencies)
                })?;

            // Realization
            let (t3chunk, statistics) = options
                .context(helper.pass_name("Realization"))
                .pass_print_hash(|ctx| {
                    let (chunk, st) = memory_planner
                        .realize(
                            phase1_mops,
                            hardware_info,
                            Box::new(|d| devices[&d].clone()),
                            &dependencies,
                        )
                        .map_err(|e| Error::memory_planning(e, ctx.name()))?;

                    if options.log {
                        println!("[Realization] Statistics: {}", &st);
                    }

                    ctx.indent_debug_text(|o| o.debug_memory_planning, &chunk.instructions, "")?;
                    ctx.debug_text(
                        |o| o.debug_memory_planning,
                        &chunk.register_value,
                        " Register Values",
                    )?;
                    ctx.display_text(|o| o.debug_realization_statistics, &st, " Statistics")?;

                    Ok((chunk, st))
                })?;

            let cpu = MemoryInfo::new(statistics.cpu_peak_usage);

            Ok((
                cpu,
                fresh_type3::Version {
                    chunk: t3chunk,
                    memory_statistics: statistics,
                },
            ))
        })?;

        Ok(FreshType3 {
            versions: chunk_versions,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            constants_device,
            libs,
            options,
            hardware_info,
            config,
        })
    }

    pub fn dump(
        &self,
        dir: impl AsRef<std::path::Path>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<()>
    where
        Rt: for<'de> serde::Deserialize<'de> + serde::Serialize,
    {
        std::fs::create_dir_all(dir.as_ref())?;

        self.cg.dump(&dir, |f, cpu, cg| {
            serde_json::to_writer_pretty(f, &(cpu, cg))?;
            Ok(())
        })?;

        let mut ct_header_f = std::fs::File::create(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header =
            args::serialization::Header::build(&self.constant_table, |_| DeviceType::CPU);
        serde_json::to_writer_pretty(&mut ct_header_f, &ct_header)?;

        let mut ct_f = std::fs::File::create(dir.as_ref().join("constants.bin"))?;
        ct_header.dump_entries_data(&self.constant_table, &mut ct_f, &mut constant_pool.cpu)?;

        Ok(())
    }
}
