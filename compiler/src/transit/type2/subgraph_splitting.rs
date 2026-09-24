use std::collections::BTreeMap;

use zkpoly_common::{
    digraph::internal::{Predecessors, SubDigraph},
    heap::UsizeId,
    load_dynamic::Libs,
};
use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;
use zkpoly_runtime::args::RuntimeType;

use crate::{
    driver::HardwareInfo,
    transit::type2::{self, graph_scheduling::SubgraphsSchedule, sliceable_subgraph},
    utils::human_readable_size,
};

pub trait HasNode<I> {
    type MutArgs;
    type Args;

    fn size(&self, vid: I) -> u64;
    fn temporary_size(&self, vid: I, mut_extra: &mut Self::MutArgs, extra: &Self::Args) -> u64;
    fn is_tuple_get(&self, vid: I) -> bool;
    fn is_tuple(&self, vid: I) -> bool;
    fn is_output(&self, vid: I) -> bool;
}

impl<'g, 's, Rt: RuntimeType> HasNode<sliceable_subgraph::VertexId>
    for sliceable_subgraph::CgSubgraph<'g, 's, Rt>
{
    type MutArgs = Libs;
    type Args = ArithGraphScheduler;

    fn size(&self, vid: sliceable_subgraph::VertexId) -> u64 {
        self.vertex_no_check_in_subgraph(vid).typ().inner().size()
    }

    fn is_tuple_get(&self, vid: sliceable_subgraph::VertexId) -> bool {
        self.vertex(vid).node().is_tuple_get()
    }

    fn is_tuple(&self, vid: sliceable_subgraph::VertexId) -> bool {
        matches!(
            self.vertex(vid).typ().inner(),
            type2::Typ::Tuple(..) | type2::Typ::Array(..)
        )
    }

    fn is_output(&self, vid: sliceable_subgraph::VertexId) -> bool {
        self.vertex(vid).node().is_output()
    }

    fn temporary_size(
        &self,
        vid: sliceable_subgraph::VertexId,
        libs: &mut Libs,
        ag_scheduler: &ArithGraphScheduler,
    ) -> u64 {
        self.vertex(vid)
            .node()
            .temporary_space_needed(
                &type2::DeviceCollection::default(1).with(type2::Device::Gpu(0)),
                libs,
                |i| self.vertex(*i).typ().inner(),
                *ag_scheduler,
            )
            .map_or(0, |t| t.sizes.iter().sum())
    }
}

struct Partitioner<'g, 'm, I, V>
where
    SubDigraph<'g, I, V>: HasNode<I>,
{
    /// Mapping from living vertex ID to bytes it occupied
    /// and its number of unexecuted successors
    living_vertices: BTreeMap<I, (u64, usize)>,
    /// Bytes occupied by living vertices
    occupied: u64,
    cg: SubDigraph<'g, I, V>,
    executed_vertices: Vec<I>,
    mut_args: &'m mut <SubDigraph<'g, I, V> as HasNode<I>>::MutArgs,
    args: <SubDigraph<'g, I, V> as HasNode<I>>::Args,
}

struct Token {
    temporary_size: u64,
}

impl<'g, 'm, I, V> Partitioner<'g, 'm, I, V>
where
    I: UsizeId,
    SubDigraph<'g, I, V>: HasNode<I>,
    V: Predecessors<I> + std::fmt::Debug,
{
    pub fn new(
        cg: SubDigraph<'g, I, V>,

        mut_args: &'m mut <SubDigraph<'g, I, V> as HasNode<I>>::MutArgs,
        args: <SubDigraph<'g, I, V> as HasNode<I>>::Args,
    ) -> Self {
        Self {
            living_vertices: BTreeMap::new(),
            occupied: 0,
            cg,
            executed_vertices: Vec::new(),
            mut_args,
            args,
        }
    }

    fn begin_vertex_(&mut self, vid: I) -> Token {
        let size = self.cg.size(vid);

        let temporary_size = self.cg.temporary_size(vid, self.mut_args, &self.args);

        let token = if self.cg.is_tuple(vid) {
            for succ in self.cg.successors_of(vid).collect::<Vec<_>>().into_iter() {
                if self.cg.is_tuple_get(succ) {
                    self.begin_vertex_(succ);
                } else {
                    panic!("expect all successors to be TupleGet")
                }
            }
            self.occupied += size + temporary_size;

            Token { temporary_size }
        } else {
            let n_successors = self.cg.deg_out(vid);
            self.living_vertices.insert(vid, (size, n_successors));
            self.occupied += size + temporary_size;

            Token { temporary_size }
        };

        self.executed_vertices.push(vid);

        token
    }

    pub fn begin_vertex(&mut self, vid: I) -> Token {
        if self.cg.is_tuple_get(vid) || self.cg.is_output(vid) {
            return Token { temporary_size: 0 };
        }

        self.begin_vertex_(vid)
    }

    fn finish_vertex_(&mut self, vid: I) {
        for pred in self.cg.predecessors_of_no_multiedge(vid) {
            let (_, n_unexecuted_scuccessors) = self
                .living_vertices
                .get_mut(&pred)
                .expect("not to finish an vertex while its predecessor is dead");
            *n_unexecuted_scuccessors -= 1;

            if *n_unexecuted_scuccessors == 0 {
                let (size, _) = self.living_vertices.remove(&pred).unwrap();
                self.occupied -= size;
            }
        }
    }

    pub fn finish_vertex(&mut self, vid: I, token: Token) {
        self.occupied -= token.temporary_size;

        if self.cg.is_tuple_get(vid) || self.cg.is_output(vid) {
            return;
        }

        self.finish_vertex_(vid);
    }

    pub fn occupied(&self) -> u64 {
        self.occupied
    }
}

fn estimate_occupation<'g, I, V>(
    cg: SubDigraph<'g, I, V>,
    schedule: &[I],
    mut_args: &mut <SubDigraph<'g, I, V> as HasNode<I>>::MutArgs,
    args: <SubDigraph<'g, I, V> as HasNode<I>>::Args,
) -> u64
where
    I: UsizeId,
    SubDigraph<'g, I, V>: HasNode<I>,
    V: Predecessors<I> + std::fmt::Debug,
{
    let mut partioner = Partitioner::new(cg, mut_args, args);
    let mut max_occupy = 0;

    for vid in schedule.iter() {
        let vid = *vid;
        let tok = partioner.begin_vertex(vid);
        max_occupy = max_occupy.max(partioner.occupied());
        partioner.finish_vertex(vid, tok);
    }

    max_occupy
}

pub struct SubgraphStatistics {
    estimated: u64,
    chunk_len: u64,
}

impl std::fmt::Debug for SubgraphStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubgraphStatistics")
            .field(
                "estimated",
                &crate::utils::human_readable_size(self.estimated),
            )
            .field("chunk_len", &self.chunk_len)
            .finish()
    }
}

#[derive(Debug)]
pub struct Statistics(BTreeMap<type2::VertexId, SubgraphStatistics>);

pub fn tighten_chunk_len<'s, Rt: RuntimeType>(
    mut cg: type2::Cg<'s, Rt>,
    hd_info: &HardwareInfo,
    memory_config: &super::memory_planning::Config,
    schedule: &SubgraphsSchedule,
    libs: &mut Libs,
    ag_scheduler: ArithGraphScheduler,
) -> (type2::Cg<'s, Rt>, Statistics) {
    use super::memory_planning::config::GpuAllocatorChoice;
    let limit = match memory_config.gpu_allocator() {
        GpuAllocatorChoice::Slab => {
            let log_limit = crate::utils::max_power_of_2_below(
                memory_config.min_gpu_integral_space(hd_info) as usize,
            );
            2u64.pow(log_limit - 2)
        }
        GpuAllocatorChoice::Page(..) => memory_config.min_gpu_integral_space(hd_info) / 4,
    };

    let mut statistics = Statistics(BTreeMap::new());

    for vid in cg.g.vertices() {
        if let type2::VertexNode::Subgraph(sg) = cg.g.vertex_mut(vid).node_mut() {
            let sg_schedule = schedule.for_subgraph(vid);

            let mut estimated;

            loop {
                let cg = SubDigraph::all(&sg.g);
                estimated = estimate_occupation(cg, sg_schedule.seq(), libs, ag_scheduler);

                let multiple = if estimated > limit {
                    -1
                } else if estimated * 2 <= limit {
                    1
                } else {
                    break;
                };

                let log_deg = sg
                    .deg
                    .checked_ilog2()
                    .expect("chunk degree must be power of 2");

                let chunk_length = 2u64.pow(
                    (sg.chunk_len
                        .checked_ilog2()
                        .expect("chunk length must be power of 2") as i32
                        + multiple)
                        .min(log_deg as i32 - 2)
                        .try_into()
                        .unwrap(),
                );

                if chunk_length == sg.chunk_len {
                    break;
                }

                sg.set_chunk_len(chunk_length);
            }

            statistics.0.insert(
                vid,
                SubgraphStatistics {
                    estimated: estimated,
                    chunk_len: sg.chunk_len,
                },
            );
        }
    }

    (cg, statistics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkpoly_common::define_usize_id;
    use zkpoly_common::digraph::internal::Digraph;
    use zkpoly_common::heap::Heap;

    define_usize_id!(VertexId);

    #[derive(Debug)]
    enum Vertex {
        Comp {
            size: u64,
            inputs: Vec<VertexId>,
            output_tuple: bool,
        },
        TupleGet {
            size: u64,
            input: VertexId,
            output_tuple: bool,
        },
        Output(u64, VertexId),
    }

    impl<'g> HasNode<VertexId> for SubDigraph<'g, VertexId, Vertex> {
        type MutArgs = ();
        type Args = ();

        fn size(&self, vid: VertexId) -> u64 {
            use Vertex::*;
            match self.vertex(vid) {
                Comp { size, .. } => *size,
                TupleGet { size, .. } => *size,
                Output(size, ..) => *size,
            }
        }

        fn is_tuple_get(&self, vid: VertexId) -> bool {
            matches!(self.vertex(vid), Vertex::TupleGet { .. })
        }

        fn is_tuple(&self, vid: VertexId) -> bool {
            match self.vertex(vid) {
                Vertex::Comp { output_tuple, .. } => *output_tuple,
                Vertex::TupleGet { output_tuple, .. } => *output_tuple,
                _ => false,
            }
        }

        fn is_output(&self, vid: VertexId) -> bool {
            match self.vertex(vid) {
                Vertex::Output(..) => true,
                _ => false,
            }
        }

        fn temporary_size(&self, _vid: VertexId, _m: &mut (), _a: &()) -> u64 {
            0
        }
    }

    impl Predecessors<VertexId> for Vertex {
        fn predecessors(&self) -> impl Iterator<Item = VertexId> {
            use Vertex::*;
            let boxed: Box<dyn Iterator<Item = _>> = match self {
                Comp { inputs, .. } => Box::new(inputs.iter().cloned()),
                Output(_, input) | TupleGet { input, .. } => Box::new(std::iter::once(*input)),
            };
            boxed
        }
    }

    fn build_cg() -> Digraph<VertexId, Vertex> {
        let mut vs = Heap::new();

        // v1 --> v2 --> v3
        //    |      |-> v4 -|
        //    |-> v5 ---------> v6
        use Vertex::*;
        let v1 = vs.push(Comp {
            size: 2,
            inputs: vec![],
            output_tuple: false,
        });
        let v2 = vs.push(Comp {
            size: 2,
            inputs: vec![v1],
            output_tuple: true,
        });
        let v3 = vs.push(TupleGet {
            size: 1,
            input: v2,
            output_tuple: false,
        });
        let v4 = vs.push(TupleGet {
            size: 1,
            input: v2,
            output_tuple: false,
        });
        let v5 = vs.push(Comp {
            size: 2,
            inputs: vec![v1],
            output_tuple: false,
        });
        let v6 = vs.push(Comp {
            size: 2,
            inputs: vec![v3, v4, v5],
            output_tuple: false,
        });
        let _ = vs.push(Output(2, v6));

        Digraph::from_vertices(vs)
    }

    #[test]
    fn test_estimation() {
        let cg = build_cg();
        let seq = cg.vertices().collect::<Vec<_>>();
        let cg = SubDigraph::new(&cg, Heap::repeat(true, cg.order()));

        let est = estimate_occupation(cg, &seq, &mut (), ());
        assert_eq!(est, 6);
    }
}
