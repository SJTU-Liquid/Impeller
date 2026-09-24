use crate::driver::Config;

use super::artifect::SemiArtifect;
use std::path::PathBuf;
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use super::{
    artifect, cudaDeviceSynchronize, cuda_check, fresh_type3, type2, type3, DebugOptions, Error,
    HardwareInfo, Versions,
};

/// The Type3 IR after applying various Type3 passes.
pub struct ProcessedType3<'s, 'i, Rt: RuntimeType> {
    pub(super) versions: Versions<fresh_type3::Version<'s, Rt>>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) constants_device: Heap<ConstantId, type3::Device>,
    pub(super) libs: Libs,
    pub(super) options: &'i DebugOptions,
    pub(super) hardware_info: &'i HardwareInfo,
    pub(super) config: &'i Config,
}

impl<'s, 'i, Rt: RuntimeType> ProcessedType3<'s, 'i, Rt> {
    /// Run the final kernel generation and other passes to obtain [`SemiArtifect`].
    /// `kernel_dir` controls in which directory the compiler puts temporary source files
    /// and the compiled libraries.
    pub fn to_artifect(self, kernel_dir: PathBuf) -> Result<SemiArtifect<'i, Rt>, Error<'s, Rt>> {
        let Self {
            versions,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            constants_device,
            mut libs,
            options,
            hardware_info,
            config,
        } = self;

        let versions = versions.map(|cpu, version, helper| {
            let hardware_info = hardware_info.clone().with_cpu(cpu.clone());
            let hardware_info = &hardware_info;

            // - Track Splitting
            let track_tasks = options
                .context(helper.pass_name("Track Splitting"))
                .pass_print_hash(|ctx| {
                    let track_tasks = type3::track_splitting::split(&version.chunk, hardware_info);
                    ctx.debug_text(|o| o.debug_track_splitting, &track_tasks, "")?;
                    Ok(track_tasks)
                })?;

            // To Runtime Instructions

            // - Emitting Multithread Chunk
            let (mt_chunk, f_table, event_table, stream2variable_id, variable_id_allocator, lbss) =
                options
                    .context(helper.pass_name("Multithread Chunk"))
                    .pass(|ctx| {
                        let kernel_dir = ctx.io(|_| {
                            let dir = kernel_dir.join(helper.dirname("version"));
                            std::fs::create_dir_all(&dir)?;
                            dir.canonicalize()
                        })?;

                        if ctx.options.log {
                            println!("[Compiler] Kernel Directory set to {:?}", &kernel_dir);
                        }

                        let r = type3::lowering::emit_multithread_instructions(
                            &track_tasks,
                            version.chunk,
                            t2uf_tab.clone(),
                            &mut libs,
                            kernel_dir,
                            config.arith_graph_scheudler,
                        );

                        ctx.io(|ctx| {
                            if ctx.options.debug_multithread_instructions {
                                let mut f = ctx.create_debug_html_file(ctx.name())?;
                                type3::lowering::pretty_print::print(&r.0, &r.3, &r.1, &mut f)?;
                            }
                            Ok(())
                        })?;

                        if ctx.options.log {
                            println!("[Compiler] MultithreadChunk has hash {:#x}", ctx.hash(&r.0));
                        }

                        Ok(r)
                    })?;

            // - Serialize Multithread Chunk
            let rt_chunk = options
                .context(helper.pass_name("Runtime Instructions"))
                .pass_print_hash(|ctx| {
                    let chunk = type3::lowering::lower(
                        mt_chunk,
                        f_table,
                        event_table,
                        stream2variable_id,
                        variable_id_allocator,
                        lbss,
                    );

                    ctx.io(|ctx| {
                        if ctx.options.debug_instructions {
                            let mut f = ctx.create_debug_text_file(ctx.name())?;
                            zkpoly_runtime::instructions::print_instructions(
                                &chunk.instructions,
                                &mut f,
                            )?;
                        }
                        Ok(())
                    })?;

                    Ok(chunk)
                })?;

            Ok(artifect::Version {
                chunk: rt_chunk,
                statistics: version.memory_statistics,
            })
        })?;

        unsafe {
            cuda_check!(cudaDeviceSynchronize());
        }

        Ok(SemiArtifect {
            versions,
            constant_table: t2const_tab,
            constant_devices: constants_device,
            libs,
            options: self.options,
        })
    }
}
