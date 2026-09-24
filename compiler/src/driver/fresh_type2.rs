use std::{io::Write, path::PathBuf};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::{self, RuntimeType};

use crate::driver::Config;

use super::{
    artifect::{self, Artifect, SemiArtifect},
    ast,
    processed_type2::ProcessedType2,
    type2::{self, graph_scheduling::SubgraphsSchedule},
    type3,
    unfused_type2::UnfusedType2,
    ConstantPool, DebugOptions, Error, HardwareInfo, MemoryInfo, Versions,
};

/// Type2 intermediate representation built from AST.
pub struct FreshType2<'s, 'i, Rt: RuntimeType> {
    prog: type2::Program<'s, Rt>,
    options: &'i DebugOptions,
}

impl<'s, 'i, Rt: RuntimeType> FreshType2<'s, 'i, Rt> {
    /// Build from AST. Runs type inference here.
    pub fn from_ast(
        ast: impl ast::TypeEraseable<Rt>,
        options: &'i DebugOptions,
    ) -> Result<Self, Error<'s, Rt>> {
        let (ast_cg, output_vid) = options.context("Lowering AST to Type2").pass(|ctx| {
            let (ast_cg, output_vid) = ast::lowering::Cg::new(ast);
            ctx.io(|ctx| {
                if ctx.options.debug_fresh_type2 {
                    type2::pretty::partial_typed::debug_graph(
                        &ast_cg.g,
                        ctx.create_debug_graph_file("Type2 Fresh")?,
                    )?;
                }

                if options.debug_user_function_table {
                    let mut f = ctx.create_debug_text_file("Type2 User Function Table")?;
                    write!(f, "{:#?}", ast_cg.user_function_table)?;
                }

                Ok(())
            })?;
            Ok((ast_cg, output_vid))
        })?;

        // Type inference
        let t2prog = options.context("Type Inference").pass_print_hash(|ctx| {
            match ast_cg.lower(output_vid) {
                Ok(t2prog) => {
                    ctx.debug_type2_unsubgraphed(|o| o.debug_type_inference, &t2prog.cg)?;
                    ctx.check_type2_unsubgraphed_dag(&t2prog.cg)?;
                    ctx.debug_text(
                        |o| o.debug_type_inference,
                        &t2prog.user_function_table,
                        " User Functions",
                    )?;
                    Ok(t2prog)
                }
                Err((e, g)) => {
                    ctx.io(|ctx| {
                        type2::pretty::type_error::debug_graph(
                            &g,
                            e.vid,
                            ctx.create_debug_graph_file("Type2 Type Error")?,
                        )
                    })?;

                    Err(Error::Typ(e))
                }
            }
        })?;

        if options.log {
            println!(
                "[Type Inference] Graph SingleArith IO size is {} elements",
                t2prog.cg.single_arith_io_size()
            );
        }

        Ok(Self {
            prog: t2prog,
            options,
        })
    }

    /// Apply passes to obtain [`UnfusedType2`].
    ///
    /// This method needs mutable access to `constant_pool`, because due to precomputation for MSM and NTT
    /// new constants may be added.
    pub fn apply_passes(
        self,
        hardware_info: &'i HardwareInfo,
        constant_pool: &mut ConstantPool,
        config: &'i Config,
    ) -> Result<UnfusedType2<'s, 'i, Rt>, Error<'static, Rt>> {
        let type2::Program {
            cg: t2cg,
            consant_table: mut t2const_tab,
            user_function_table: t2uf_tab,
        } = self.prog;
        let options = self.options;

        let mut libs = Libs::new();

        // Apply Type2 passes
        // - INTT Mending: Append division by n to end of each INTT
        let t2cg = options.context("INTT Mending").pass(|ctx| {
            let t2cg = type2::intt_mending::mend(t2cg, &mut t2const_tab);
            ctx.check_type2_unsubgraphed_dag(&t2cg)?;
            ctx.debug_type2_unsubgraphed(|o| o.debug_intt_mending, &t2cg)?;
            Ok(t2cg)
        })?;

        // - Precompute NTT and MSM constants
        let t2cg = options.context("NTT and MSM Precomputation").pass(|ctx| {
            let t2cg = type2::precompute::precompute(
                t2cg,
                config.memory.min_gpu_integral_space(hardware_info) as usize,
                &mut libs,
                &mut constant_pool.cpu,
                constant_pool.disk.as_mut(),
                &mut t2const_tab,
            );
            ctx.check_type2_unsubgraphed_dag(&t2cg)?;
            ctx.debug_type2_unsubgraphed(|o| o.debug_precompute, &t2cg)?;
            Ok(t2cg)
        })?;

        let cg_statistics = t2cg.graph_statistics();
        let f = std::fs::File::create(options.debug_dir.join("graph_statistics.json")).unwrap();
        serde_json::to_writer_pretty(f, &cg_statistics).unwrap();

        // - Manage Inversions: Rewrite inversions of scalars and polynomials to dedicated operators
        let t2cg = options.context("Inversion Correction").pass(|ctx| {
            let t2cg = type2::manage_inverse::manage_inverse(t2cg);
            ctx.check_type2_unsubgraphed_dag(&t2cg)?;
            ctx.debug_type2_unsubgraphed(|o| o.debug_manage_invers, &t2cg)?;
            Ok(t2cg)
        })?;

        // - Common Subexpression Elimination
        let t2cg = options
            .context("Common Subexpression Elimination")
            .pass_print_hash(|ctx| {
                let t2cg = type2::common_subexpression_elimination::cse(t2cg, &t2uf_tab);
                ctx.check_type2_unsubgraphed_dag(&t2cg)?;
                ctx.debug_type2_unsubgraphed(|o| o.debug_cse, &t2cg)?;
                Ok(t2cg)
            })?;

        Ok(UnfusedType2 {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
            options,
            hardware_info,
            config,
        })
    }

    /// Base on fresh Type2, directly load [`Artifect`] from file system without need for further passes.
    /// During loading, stored constants are written to `constant_pool`.
    pub fn load_artifect(
        mut self,
        dir: impl AsRef<std::path::Path>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<Artifect<Rt>> {
        let mut libs = Libs::new();

        let versions = Versions::load(&dir, |f| {
            let (cpu, chunk, statistics): (
                MemoryInfo,
                type3::lowering::serialization::ChunkDeserializer,
                type2::memory_planning::Statistics,
            ) = serde_json::from_reader(f)?;
            let chunk =
                chunk.deserialize_into_chunk(self.prog.user_function_table.clone(), &mut libs);
            Ok((cpu, artifect::Version { chunk, statistics }))
        })?;

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut constant_pool.cpu,
            constant_pool.disk.as_mut(),
        )?;

        Ok(Artifect {
            versions,
            constant_table: self.prog.consant_table,
            libs,
        })
    }

    pub fn load_processed_type2<'de>(
        mut self,
        dir: impl AsRef<std::path::Path>,
        buffers: &'de mut Vec<String>,
        constant_pool: &mut ConstantPool,
        config: &'i Config,
        hardware_info: &'i HardwareInfo,
        options: &'i DebugOptions,
    ) -> std::io::Result<ProcessedType2<'de, 'i, Rt>>
    where
        Rt: serde::Serialize + serde::Deserialize<'de>,
    {
        let cg = Versions::load_buffered(&dir, buffers, |buf| {
            let (cpu, cg): (MemoryInfo, (type2::Cg<'de, Rt>, SubgraphsSchedule)) =
                serde_json::from_str(buf)?;
            Ok((cpu, cg))
        })?;

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut constant_pool.cpu,
            constant_pool.disk.as_mut(),
        )?;

        Ok(ProcessedType2 {
            cg,
            constant_table: self.prog.consant_table,
            uf_table: self.prog.user_function_table,
            libs: Libs::new(),
            options,
            hardware_info,
            config,
        })
    }

    /// Apply all passes to obtain [`Artifect`].
    ///
    /// For meanings of each argument, refer to lowering methods on [`FreshType2`], [`UnfusedType2`], etc.
    pub fn to_artifect(
        self,
        hardware_info: &HardwareInfo,
        constant_pool: &mut ConstantPool,
        config: &Config,
        versions_cpu_memory_divisors: impl Iterator<Item = u32>,
        kernel_dir: PathBuf,
    ) -> Result<Artifect<Rt>, Error<'s, Rt>> {
        Ok(self
            .to_semi_artifect(
                hardware_info,
                constant_pool,
                config,
                versions_cpu_memory_divisors,
                kernel_dir,
            )?
            .finish(constant_pool))
    }

    /// Apply all passes to obtain [`SemiArtifect`].
    ///
    /// For meanings of each argument, refer to lowering methods on [`FreshType2`], [`UnfusedType2`], etc.
    pub fn to_semi_artifect(
        self,
        hardware_info: &'i HardwareInfo,
        constant_pool: &mut ConstantPool,
        config: &'i Config,
        versions_cpu_memory_divisors: impl Iterator<Item = u32>,
        kernel_dir: PathBuf,
    ) -> Result<SemiArtifect<'i, Rt>, Error<'s, Rt>> {
        self.apply_passes(hardware_info, constant_pool, config)?
            .fuse(versions_cpu_memory_divisors)?
            .to_type3(constant_pool)?
            .apply_passes()?
            .to_artifect(kernel_dir)
    }
}
