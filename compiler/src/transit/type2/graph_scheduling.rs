mod prelude {
    pub use std::collections::{BTreeMap, BTreeSet};

    pub use super::super::object_analysis::{
        cg_def_use::DefUse,
        value::{ObjectDefs, ObjectId},
    };
    pub use super::super::VertexId;
    pub use crate::transit::type2::{self, sliceable_subgraph};
    pub use zkpoly_common::{
        digraph::internal::Predecessors, heap::UsizeId, mm_heap::with_data::MmHeap,
    };
    pub use zkpoly_runtime::args::RuntimeType;
}

use std::{collections::HashSet, hash::Hash};

use prelude::*;
mod def_use_dag;
use def_use_dag::{DefUseDag, ObjectDag, SubgraphObjectDag};

struct StatisticsCollector {
    /// Occupied of living vertices
    occupied: u64,
    max_occupied: u64,
}

impl StatisticsCollector {
    fn new() -> Self {
        Self {
            occupied: 0,
            max_occupied: 0,
        }
    }

    fn live(&mut self, size: u64) {
        self.occupied += size;
        self.max_occupied = self.occupied.max(self.max_occupied);
    }

    fn kill(&mut self, size: u64) {
        self.occupied -= size;
        self.max_occupied = self.max_occupied.max(self.occupied);
    }
}

pub mod plain_topology_sort;
pub mod sethi_ullman;

#[derive(Clone, Debug, Copy)]
pub enum Algorithm {
    PlainTopologySort,
    SethiUllman,
}

fn invert_seq<I: UsizeId>(seq: impl IntoIterator<Item = I>) -> BTreeMap<I, usize> {
    seq.into_iter()
        .enumerate()
        .fold(BTreeMap::new(), |mut map, (seq, i)| {
            map.insert(i, seq);
            map
        })
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub struct Schedule<I>
where
    I: Ord + Eq,
{
    seq: Vec<I>,
    seq_inv: BTreeMap<I, usize>,
    max_occupied_size: u64,
}

impl<I> Schedule<I>
where
    I: UsizeId,
{
    pub fn of_seq(seq: Vec<I>, max_occupied_size: u64) -> Self {
        let seq_inv = invert_seq(seq.iter().cloned());

        Self {
            seq,
            seq_inv,
            max_occupied_size,
        }
    }

    pub fn seq(&self) -> &[I] {
        &self.seq
    }

    pub fn seq_inv(&self, i: I) -> Option<usize> {
        self.seq_inv.get(&i).copied()
    }

    pub fn max_occupied_size(&self) -> u64 {
        self.max_occupied_size
    }

    pub fn on<'a, 's, Rt: RuntimeType>(
        cg: &type2::CgSubgraph<'a, 's, Rt>,
        du: &DefUse,
        objects: &ObjectDefs,
        alg: Algorithm,
    ) -> Schedule<type2::VertexId> {
        use Algorithm::*;

        let (cg_seq, mos) = match alg {
            PlainTopologySort => plain_topology_sort::Scheduler::new(
                &ObjectDag::new(du, objects, cg),
                cg.topology_sort().map(|(i, _)| i),
            )
            .run(),
            SethiUllman => sethi_ullman::Scheduler::new(
                &ObjectDag::new(du, objects, cg),
                cg.topology_sort().map(|(i, _)| i),
            )
            .run(),
        };
        Schedule::of_seq(cg_seq, mos)
    }
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubgraphsSchedule(BTreeMap<type2::VertexId, Schedule<sliceable_subgraph::VertexId>>);

impl SubgraphsSchedule {
    pub fn for_subgraph(&self, vid: type2::VertexId) -> &Schedule<sliceable_subgraph::VertexId> {
        &self.0[&vid]
    }

    pub fn remove(
        &mut self,
        vid: type2::VertexId,
    ) -> Option<Schedule<sliceable_subgraph::VertexId>> {
        self.0.remove(&vid)
    }

    pub fn insert(&mut self, vid: type2::VertexId, sche: Schedule<sliceable_subgraph::VertexId>) {
        self.0.insert(vid, sche);
    }

    pub fn on<'a, 's, Rt: RuntimeType>(cg: &type2::CgSubgraph<'a, 's, Rt>, alg: Algorithm) -> Self {
        let mut subgraphs_seq = BTreeMap::new();

        for vid in cg.vertices() {
            if let type2::VertexNode::Subgraph(subgraph) = cg.vertex(vid).node() {
                use Algorithm::*;

                let (seq, mos) = match alg {
                    PlainTopologySort => plain_topology_sort::Scheduler::new(
                        &SubgraphObjectDag::new(subgraph),
                        subgraph.g.topology_sort().map(|(i, _)| i),
                    )
                    .run(),
                    SethiUllman => sethi_ullman::Scheduler::new(
                        &SubgraphObjectDag::new(subgraph),
                        subgraph.g.topology_sort().map(|(i, _)| i),
                    )
                    .run(),
                };
                let inputs = subgraph.inputs.iter().cloned().collect::<HashSet<_>>();
                let seq = subgraph
                    .inputs
                    .iter()
                    .cloned()
                    .chain(seq.into_iter().filter(|i| !inputs.contains(i)))
                    .collect::<Vec<_>>();
                let mos = mos
                    + subgraph
                        .inputs
                        .iter()
                        .map(|i| subgraph.g.vertex(*i).typ().inner().size())
                        .sum::<u64>();
                subgraphs_seq.insert(vid, Schedule::of_seq(seq, mos));
            }
        }
        Self(subgraphs_seq)
    }
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub struct WholeSchedule {
    pub cg_schedule: Schedule<type2::VertexId>,
    pub subgraphs_schedule: SubgraphsSchedule,
}
