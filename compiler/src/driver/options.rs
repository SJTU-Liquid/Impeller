use std::path::PathBuf;

use crate::transit::{type2, type3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type2DebugVisualizer {
    Cytoscape,
    Graphviz,
}

/// Controls whether and where debugging files are written to.
#[derive(Debug, Clone)]
pub struct DebugOptions {
    pub(super) debug_dir: PathBuf,
    pub(super) debug_user_function_table: bool,
    pub(super) debug_fresh_type2: bool,
    pub(super) debug_type_inference: bool,
    pub(super) debug_intt_mending: bool,
    pub(super) debug_precompute: bool,
    pub(super) debug_manage_invers: bool,
    pub(super) debug_cse: bool,
    pub(super) debug_kernel_fusion: bool,
    pub(super) debug_sliceable_subgraph: bool,
    pub(super) debug_graph_scheduling: bool,
    pub(super) debug_obj_analysis: bool,
    pub(super) debug_memory_planning: bool,
    pub(super) debug_realization_statistics: bool,
    pub(super) debug_fresh_type3: bool,
    pub(super) debug_track_splitting: bool,
    pub(super) debug_multithread_instructions: bool,
    pub(super) debug_instructions: bool,
    pub(super) type2_visualizer: Type2DebugVisualizer,
    pub(super) log: bool,
}

impl DebugOptions {
    pub fn prepare_dir(&self) {
        std::fs::create_dir_all(&self.debug_dir).unwrap();

        if self.log {
            println!(
                "[Compiler] Debug directory set to {:?}",
                self.debug_dir.canonicalize().unwrap()
            );
        }
    }

    pub fn debug_dir(&self) -> &PathBuf {
        &self.debug_dir
    }

    /// Enable all debugging files, written to directory `debug_dir`.
    pub fn all(debug_dir: PathBuf) -> Self {
        Self {
            debug_dir,
            debug_user_function_table: true,
            debug_fresh_type2: true,
            debug_intt_mending: true,
            debug_type_inference: true,
            debug_precompute: true,
            debug_manage_invers: true,
            debug_kernel_fusion: true,
            debug_graph_scheduling: true,
            debug_obj_analysis: true,
            debug_cse: true,
            debug_fresh_type3: true,
            debug_sliceable_subgraph: true,
            debug_memory_planning: true,
            debug_realization_statistics: true,
            debug_track_splitting: true,
            debug_multithread_instructions: true,
            debug_instructions: true,
            type2_visualizer: Type2DebugVisualizer::Graphviz,
            log: false,
        }
    }

    /// Enable no debugging files, written to directory `debug_dir`.
    pub fn none(debug_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&debug_dir).unwrap();
        Self {
            debug_dir,
            debug_user_function_table: true,
            debug_fresh_type2: false,
            debug_cse: false,
            debug_intt_mending: false,
            debug_type_inference: false,
            debug_precompute: false,
            debug_manage_invers: false,
            debug_kernel_fusion: false,
            debug_graph_scheduling: false,
            debug_obj_analysis: false,
            debug_fresh_type3: false,
            debug_sliceable_subgraph: false,
            debug_memory_planning: false,
            debug_realization_statistics: false,
            debug_track_splitting: false,
            debug_multithread_instructions: false,
            debug_instructions: false,
            type2_visualizer: Type2DebugVisualizer::Graphviz,
            log: false,
        }
    }

    /// Enable only critical debugging files, written to directory `debug_dir`.
    pub fn minimal(debug_dir: PathBuf) -> Self {
        let r = Self::none(debug_dir);
        Self {
            debug_user_function_table: true,
            debug_obj_analysis: true,
            debug_memory_planning: true,
            debug_fresh_type3: true,
            debug_realization_statistics: true,
            debug_instructions: true,
            ..r
        }
    }

    pub fn statistics_only(debug_dir: PathBuf) -> Self {
        let r = Self::none(debug_dir);
        Self {
            debug_user_function_table: true,
            debug_realization_statistics: true,
            ..r
        }
    }

    pub fn with_memory_planning(mut self, switch: bool) -> Self {
        self.debug_memory_planning = switch;
        self.debug_obj_analysis = switch;
        self
    }

    /// Configure whether to debug Type3 intermediate results.
    pub fn with_type3(mut self, switch: bool) -> Self {
        self.debug_fresh_type3 = switch;
        self
    }

    /// Configure whether to debug emitted instructions.
    pub fn with_inst(mut self, switch: bool) -> Self {
        self.debug_multithread_instructions = switch;
        self.debug_instructions = switch;
        self
    }

    /// Configure the Type2 visualizer to use.
    pub fn with_type2_visualizer(mut self, type2_visualizer: Type2DebugVisualizer) -> Self {
        self.type2_visualizer = type2_visualizer;
        self
    }

    /// Configure whether to print the pass being applied during compilation.
    pub fn with_log(mut self, log: bool) -> Self {
        self.log = log;
        self
    }

    pub fn with_debug_dir(self, debug_dir: PathBuf) -> Self {
        Self { debug_dir, ..self }
    }

    pub(super) fn context<'o>(&'o self, name: impl Into<String>) -> super::Context<'o> {
        super::Context::new(self, name)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub(super) sliceable_subgraph: Option<type2::subgraph_slicing::Config>,
    pub(super) scheduler_alg: type2::graph_scheduling::Algorithm,
    pub(super) memory: type2::memory_planning::Config,
    pub(super) arith_graph_scheudler: type3::lowering::ArithGraphScheduler,
    pub(super) constants_on_disk: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sliceable_subgraph: Some(type2::subgraph_slicing::Config::default()),
            scheduler_alg: type2::graph_scheduling::Algorithm::SethiUllman,
            memory: type2::memory_planning::Config::default(),
            arith_graph_scheudler: type3::lowering::ArithGraphScheduler::Heuristic,
            constants_on_disk: false,
        }
    }
}

impl Config {
    pub fn memory(&self) -> &type2::memory_planning::Config {
        &self.memory
    }

    pub fn with_sliceable_subgraph(self, sc: Option<type2::subgraph_slicing::Config>) -> Self {
        Config {
            sliceable_subgraph: sc,
            ..self
        }
    }

    pub fn with_sliceable_subgraph_on(self, sc: type2::subgraph_slicing::Config) -> Self {
        self.with_sliceable_subgraph(Some(sc))
    }

    pub fn with_scheduler_alg(self, alg: type2::graph_scheduling::Algorithm) -> Self {
        Config {
            scheduler_alg: alg,
            ..self
        }
    }

    pub fn with_memory_planning(self, config: type2::memory_planning::Config) -> Self {
        Config {
            memory: config,
            ..self
        }
    }

    pub fn with_arith_graph_scheduler(self, s: type3::lowering::ArithGraphScheduler) -> Self {
        Config {
            arith_graph_scheudler: s,
            ..self
        }
    }

    pub fn map_memory_planning(
        self,
        config: impl FnOnce(type2::memory_planning::Config) -> type2::memory_planning::Config,
    ) -> Self {
        Config {
            memory: config(self.memory),
            ..self
        }
    }

    pub fn map_sliceable_subgraph(
        self,
        config: impl FnOnce(
            Option<type2::subgraph_slicing::Config>,
        ) -> Option<type2::subgraph_slicing::Config>,
    ) -> Self {
        Config {
            sliceable_subgraph: config(self.sliceable_subgraph),
            ..self
        }
    }

    pub fn with_constants_on_disk(self, x: bool) -> Self {
        Config {
            constants_on_disk: x,
            ..self
        }
    }
}
