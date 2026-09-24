use std::collections::{BTreeMap, HashSet, VecDeque};

use zkpoly_common::{digraph::internal::Digraph, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

use crate::driver;

use super::{sliceable_subgraph, template::SliceableProperty, CgSubgraph, Typ, VertexId};

struct DenseSet(Heap<VertexId, bool>);

impl std::ops::Deref for DenseSet {
    type Target = Heap<VertexId, bool>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DenseSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::fmt::Debug for DenseSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_set();
        for (v, s) in self.0.iter_with_id() {
            if *s {
                ds.entry(&v);
            }
        }
        ds.finish()
    }
}

impl DenseSet {
    pub fn empty(order: usize) -> Self {
        Self(Heap::repeat(false, order))
    }

    pub fn add(&mut self, v: VertexId) {
        self.0[v] = true;
    }

    pub fn remove(&mut self, v: VertexId) {
        self.0[v] = false;
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = VertexId> + 'a {
        self.0.iter_with_id().filter(|(_, s)| **s).map(|(x, _)| x)
    }
}

mod cluster {
    use super::super::template::{SliceableNode, VertexNode};
    use super::*;
    use zkpoly_common::heap::Heap;

    /// Groups of vertices that must be fused together
    #[derive(Clone, Debug)]
    pub struct Clusters {
        containing: Heap<VertexId, Vec<VertexId>>,
        belonging: Heap<VertexId, Option<VertexId>>,
    }

    impl Clusters {
        pub fn of<'g, 's, Rt: RuntimeType>(g: &CgSubgraph<'g, 's, Rt>) -> Self {
            let mut clusters = Self {
                containing: g.map_vertices_ref(|vid, _| vec![vid]),
                belonging: g.map_vertices_ref(|vid, _v| Some(vid)),
            };

            for (vid, v) in g.topology_sort() {
                match v.node() {
                    VertexNode::Sliceable(SliceableNode::Arith { .. }) => {
                        let vertices = std::iter::once(vid)
                            .chain(g.successors_of(vid))
                            .collect::<Vec<_>>();
                        clusters.containing[vid] = vertices;
                        for &u in clusters.containing[vid].iter() {
                            clusters.belonging[u] = Some(vid);
                        }
                    }
                    _ => {}
                }
            }

            clusters
        }

        pub fn same_cluster_as(&self, vid: VertexId) -> impl Iterator<Item = VertexId> {
            let representitive = self.belonging[vid].expect("vid to be in subgraph");
            self.containing[representitive]
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
        }

        pub fn representative(&self, vid: VertexId) -> VertexId {
            self.belonging[vid].expect("vid to be in subgraph")
        }

        pub fn predecessors_of<'g, 's, Rt: RuntimeType>(
            &self,
            vid: VertexId,
            g: &CgSubgraph<'g, 's, Rt>,
        ) -> impl Iterator<Item = VertexId> {
            let representitive = self.belonging[vid].expect("vid to be in subgraph");
            // Now only case of cluster is arithmetic vertex and its successive tuple-gets,
            // so their predecessors are exactly those of representitive
            g.predecessors_of(representitive)
                .collect::<Vec<_>>()
                .into_iter()
        }

        pub fn successors_of<'g, 's, Rt: RuntimeType>(
            &self,
            vid: VertexId,
            g: &CgSubgraph<'g, 's, Rt>,
        ) -> impl Iterator<Item = VertexId> {
            let representitive = self.belonging[vid].expect("vid to be in subgraph");
            // Now only case of cluster is arithmetic vertex and its successive tuple-gets,
            // so their predecessors are exactly those of the tuple-gets, which are the vertices other than representitive in the cluster
            let successors: Vec<_> = if self.containing[representitive].len() == 1 {
                g.successors_of(representitive).collect()
            } else {
                self.containing[representitive]
                    .iter()
                    .filter(|&&u| u != representitive)
                    .map(|&u| g.successors_of(u))
                    .flatten()
                    .collect()
            };

            successors.into_iter()
        }
    }
}
use cluster::Clusters;

fn dfs<'s, 'g, It>(
    v: VertexId,
    next_vertices: &impl Fn(VertexId) -> It,
    reachable: &mut DenseSet,
    subgraph: &DenseSet,
) where
    It: Iterator<Item = VertexId>,
{
    if !subgraph[v] {
        reachable.add(v);
    }

    for w in next_vertices(v) {
        if !reachable[w] && !subgraph[w] {
            dfs(w, next_vertices, reachable, subgraph);
        }
    }
}

mod subgraph {
    use super::*;

    pub struct Subgraph<'g, 's, Rt: RuntimeType> {
        vertices: DenseSet,
        forward_reachable: DenseSet,
        backward_reachable: DenseSet,
        input_vertices: DenseSet,
        total_input_size: u64,
        g: CgSubgraph<'g, 's, Rt>,
        clusters: Clusters,
        order: usize,
    }

    impl<'s, 'g, Rt: RuntimeType> Subgraph<'g, 's, Rt> {
        pub fn start_with(v: VertexId, cg: super::CgSubgraph<'g, 's, Rt>) -> Self {
            let r = Self {
                vertices: DenseSet::empty(cg.full_order()),
                forward_reachable: DenseSet::empty(cg.full_order()),
                backward_reachable: DenseSet::empty(cg.full_order()),
                input_vertices: DenseSet::empty(cg.full_order()),
                total_input_size: 0,
                clusters: Clusters::of(&cg),
                g: cg,
                order: 0,
            };

            r
        }

        /// Add a vertex to the subgraph, along with all other vertices in the subgraph
        pub fn add_vertex(&mut self, v: VertexId) {
            // Remove v from inputs if it was there
            if self.input_vertices[v] {
                self.input_vertices.remove(v);
                self.total_input_size -= self.g.vertex(v).typ().size();
            }

            // Add predecessors of v that are not in subgraph to inputs
            for pred in self.g.predecessors_of(v) {
                if !self.vertices[pred] && !self.input_vertices[pred] {
                    self.input_vertices.add(pred);
                    self.total_input_size += self.g.vertex(pred).typ().size();
                }
            }

            self.vertices.add(v);
            dfs(
                v,
                &|u| self.g.predecessors_of(u),
                &mut self.backward_reachable,
                &self.vertices,
            );
            dfs(
                v,
                &|u| self.g.successors_of(u),
                &mut self.forward_reachable,
                &self.vertices,
            );
            self.order += 1;
        }

        pub fn is_forward_reachable(&self, v: VertexId) -> bool {
            self.forward_reachable[v]
        }

        pub fn is_backward_reachable(&self, v: VertexId) -> bool {
            self.backward_reachable[v]
        }

        pub fn contains(&self, v: VertexId) -> bool {
            self.vertices[v]
        }

        pub fn input_size_if_added(&self, v: VertexId) -> u64 {
            let mut new_input_size = self.total_input_size;
            // Get the cluster representative (arithmetic vertex)
            let representative = self.clusters.representative(v);

            // Remove representative from inputs if it was there
            if self.input_vertices[representative] {
                new_input_size -= self.g.vertex(representative).typ().size();
            }

            // Add predecessors of representative that are not in subgraph to inputs
            for pred in self.g.predecessors_of(representative) {
                if !self.vertices[pred] && !self.input_vertices[pred] {
                    new_input_size += self.g.vertex(pred).typ().size();
                }
            }

            new_input_size
        }

        pub fn clustered_predecessors_of(&self, vid: VertexId) -> impl Iterator<Item = VertexId> {
            self.clusters.predecessors_of(vid, &self.g)
        }

        pub fn clustered_successors_of(&self, vid: VertexId) -> impl Iterator<Item = VertexId> {
            self.clusters.successors_of(vid, &self.g)
        }

        pub fn same_cluster_as(&self, vid: VertexId) -> impl Iterator<Item = VertexId> {
            self.clusters.same_cluster_as(vid)
        }

        pub fn g(&self) -> &CgSubgraph<'g, 's, Rt> {
            &self.g
        }

        pub fn finish_grow(self) -> super::Growed<'g, 's, Rt> {
            Growed {
                subgraph: self.vertices,
                connected_cg: self.g,
                subgraph_order: self.order,
                input_size: self.total_input_size,
            }
        }
    }
}

use subgraph::Subgraph;

struct GrowingSubgraph<'g, 's, Rt: RuntimeType> {
    subgraph: Subgraph<'g, 's, Rt>,
    frontier: VecDeque<VertexId>,
}

impl<'g, 's, Rt: RuntimeType> GrowingSubgraph<'g, 's, Rt> {
    fn start_with(v: VertexId, cg: &'g super::Cg<'s, Rt>) -> Option<Self> {
        let frontier = if !matches!(
            cg.g.vertex(v).node(),
            super::VertexNode::Sliceable(super::template::SliceableNode::Arith { .. })
        ) {
            return None;
        } else {
            vec![]
        }
        .into();

        let mut r = Self {
            subgraph: Subgraph::start_with(v, cg.connected_subgraph()),
            frontier,
        };

        r.grow(v);

        Some(r)
    }

    /// Grow the subgraph by adding `v`
    fn grow(&mut self, v: VertexId) {
        for v in self.subgraph.same_cluster_as(v) {
            self.subgraph.add_vertex(v);
            self.frontier.push_back(v);
        }
    }

    /// Check whether a cycle would form after adding the vertex `v`
    fn no_cycle_after_adding(&self, v: VertexId) -> bool {
        self.subgraph
            .clustered_successors_of(v)
            .filter(|u| !self.subgraph.contains(*u))
            .all(|u| !self.subgraph.is_backward_reachable(u))
            && self
                .subgraph
                .clustered_predecessors_of(v)
                .filter(|u| !self.subgraph.contains(*u))
                .all(|u| !self.subgraph.is_forward_reachable(u))
    }

    fn sliceable_property(&self, v: VertexId) -> SliceableProperty {
        self.subgraph
            .g()
            .vertex(v)
            .sliceable_property(|u| self.subgraph.g().vertex(*u).typ())
    }

    fn check_input_size(&self, v: VertexId, maximum_input_size: Option<u64>) -> bool {
        if let Some(max_size) = maximum_input_size {
            let new_input_size = self.subgraph.input_size_if_added(v);
            new_input_size <= max_size
        } else {
            true
        }
    }

    /// Can grow the subgraph along edge `v` -> `u` with `v` in subgraph
    fn can_vertex_grow_forward(
        &self,
        v: VertexId,
        u: VertexId,
        maximum_input_size: Option<u64>,
    ) -> bool {
        use super::template::SliceableProperty::*;

        !self.subgraph.contains(u)
            && self.check_input_size(u, maximum_input_size)
            && self.no_cycle_after_adding(u)
            // Checking following conditions for any vertex in a cluster would do
            && self.sliceable_property(v) != Last
            && matches!(self.sliceable_property(u), Always | NonFirst | Last)
            && self.subgraph.g().vertex(u).typ().is_poly_pow2()
    }

    /// Can grow the subgraph along edge `u` -> `v` with `v` in subgraph
    fn can_vertex_grow_backward(
        &self,
        _v: VertexId,
        u: VertexId,
        maximum_input_size: Option<u64>,
    ) -> bool {
        use super::template::SliceableProperty::*;

        // We are allowing NonFirst here, and if we later find some NonFirst vertex becomes first,
        // we will remove it from the subgraph before building the super vertex
        !self.subgraph.contains(u)
            && self.check_input_size(u, maximum_input_size)
            && self.no_cycle_after_adding(u)
            // Checking following conditions for any vertex in a cluster would do
            && matches!(self.sliceable_property(u), Always | NonFirst)
            && self.subgraph.g().vertex(u).typ().is_poly_pow2()
    }

    fn try_grow(&mut self, maximum_input_size: Option<u64>) -> bool {
        if let Some(v) = self.frontier.pop_front() {
            for u in self
                .subgraph
                .g()
                .successors_of(v)
                .collect::<Vec<_>>()
                .into_iter()
            {
                if self.can_vertex_grow_forward(v, u, maximum_input_size) {
                    self.grow(u);
                }
            }
            for u in self
                .subgraph
                .g()
                .predecessors_of(v)
                .collect::<Vec<_>>()
                .into_iter()
            {
                if self.can_vertex_grow_backward(v, u, maximum_input_size) {
                    self.grow(u);
                }
            }
            true
        } else {
            false
        }
    }

    fn as_large_as_possible(mut self, maximum_input_size: Option<u64>) -> Growed<'g, 's, Rt> {
        while self.try_grow(maximum_input_size) {}
        self.subgraph.finish_grow()
    }
}

#[derive(Debug)]
struct Growed<'g, 's, Rt: RuntimeType> {
    subgraph: DenseSet,
    connected_cg: super::CgSubgraph<'g, 's, Rt>,
    subgraph_order: usize,
    input_size: u64,
}

#[derive(Clone)]
pub struct SubgraphStatistics {
    /// Where the subgraph began fusion
    pub begin: VertexId,
    /// Number of arithmetic subgraphs in subgraph
    pub n_arith_graphs: usize,
    /// Degree of polynomials within the subgraph
    pub deg: u64,
    /// Number of all arithmetic operations in subgraph
    pub arith_order: usize,
    /// Number of operators in subgraph
    pub count: usize,
    /// Whether the subgraph contains big arithmetic subgraphs
    pub big_arith: bool,
    /// Size of inputs, in bytes
    pub input_size: u64,
}

impl std::fmt::Debug for SubgraphStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubgraphStatistics")
            .field("begin", &self.begin)
            .field("n_arith_graphs", &self.n_arith_graphs)
            .field("deg", &self.deg)
            .field("arith_order", &self.arith_order)
            .field("count", &self.count)
            .field("big_arith", &self.big_arith)
            .field(
                "input_size",
                &crate::utils::human_readable_size(self.input_size),
            )
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Statistics {
    pub fused_arith_order: usize,
    pub fused_vertices: usize,
    pub fused_subgraphs: BTreeMap<VertexId, SubgraphStatistics>,
}

impl Statistics {
    fn new() -> Self {
        Self {
            fused_arith_order: 0,
            fused_vertices: 0,
            fused_subgraphs: BTreeMap::new(),
        }
    }

    fn add(&mut self, vid: VertexId, st: SubgraphStatistics) {
        self.fused_vertices += st.count;
        self.fused_arith_order += st.arith_order;
        self.fused_subgraphs.insert(vid, st);
    }
}

impl<'s, 'g, Rt: RuntimeType> Growed<'g, 's, Rt> {
    fn prune_non_first(mut self) -> Self {
        loop {
            let mut clean = true;
            for v in self.subgraph.ids() {
                if self.subgraph[v] {
                    if self
                        .connected_cg
                        .vertex(v)
                        .sliceable_property(|u| self.connected_cg.vertex(*u).typ())
                        == SliceableProperty::NonFirst
                        && !self
                            .connected_cg
                            .predecessors_of(v)
                            .all(|u| self.subgraph[u])
                    {
                        self.subgraph.remove(v);
                        self.subgraph_order -= 1;
                        clean = false;
                    }
                }
            }
            if clean {
                return self;
            }
        }
    }

    fn _check_dag(self) -> Self {
        let mut forward_reachable = DenseSet::empty(self.connected_cg.full_order());
        let mut backward_reachable = DenseSet::empty(self.connected_cg.full_order());

        for v in self.connected_cg.vertices().filter(|v| self.subgraph[*v]) {
            dfs(
                v,
                &|i| self.connected_cg.successors_of(i),
                &mut forward_reachable,
                &self.subgraph,
            );
            dfs(
                v,
                &|i| self.connected_cg.predecessors_of(i),
                &mut backward_reachable,
                &self.subgraph,
            );
        }

        for ((i, x), (_, y)) in forward_reachable
            .iter_with_id()
            .zip(backward_reachable.iter_with_id())
        {
            if *x && *y {
                panic!(
                    "{:?} is both forward and backward reachable for subgraph {:?}",
                    i, self.subgraph
                );
            }
        }

        self
    }

    pub fn decide_fuseable(
        &self,
        begin: VertexId,
        hd_info: &driver::HardwareInfo,
        memory_config: &super::memory_planning::Config,
        subgraph_minimum_order: usize,
        chunk_len: u64,
    ) -> Option<(u64, SubgraphStatistics)> {
        let count = self
            .subgraph
            .iter()
            .filter(|v| {
                use super::template::{SliceableNode::*, VertexNode::*};
                match self.connected_cg.vertex(*v).node() {
                    Sliceable(RotateIdx(..)) | TupleGet(..) => false,
                    _ => true,
                }
            })
            .count();

        let contains_big_subgraph = self.subgraph.iter().any(|vid| {
            if let super::VertexNode::Sliceable(super::template::SliceableNode::Arith { arith }) =
                self.connected_cg.vertex(vid).node()
            {
                arith.poly_degree.is_some_and(|deg| {
                    let estimated_gpu_space =
                        arith.g.order() * deg * std::mem::size_of::<Rt::Field>();
                    estimated_gpu_space > memory_config.min_gpu_integral_space(hd_info) as usize
                })
            } else {
                false
            }
        });

        if count < subgraph_minimum_order && !contains_big_subgraph {
            // println!(
            //     "[Sliceable Subgraph] Skipping subgraph {:?} because order {} < {} , contains_big_subgraph = {}",
            //     &self.subgraph, count, subgraph_minimum_order, contains_big_subgraph
            // );
            return None;
        }

        let mut deg = None;
        for (vid, typ) in self
            .subgraph
            .iter_with_id()
            .filter(|(_, s)| **s)
            .map(|(vid, _)| (vid, self.connected_cg.vertex(vid).typ()))
        {
            if let Some((_, d)) = typ.try_unwrap_poly() {
                if deg.is_some_and(|x| x != *d) {
                    panic!(
                        "[Sliceable Subgraph] polynomials in subgraph has inconsistent degree: {} from {:?} != {}",
                        d,
                        vid,
                        deg.unwrap()
                    );
                }
                deg = Some(*d);
            }
        }

        if deg.is_none() || deg.is_some_and(|deg| deg < chunk_len * 4) {
            // if print_debug {
            // println!(
            //     "[Sliceable Subgraph] Not fusing subgraph {:?} because degree is {:?} < {} * 8",
            //     &self.subgraph, deg, chunk_len
            // );
            // }
            return None;
        }
        let deg = deg.unwrap();

        let n_graphs = self
            .subgraph
            .iter()
            .filter(|vid| {
                matches!(
                    self.connected_cg.vertex(*vid).node(),
                    super::template::VertexNode::Sliceable(
                        super::template::SliceableNode::Arith { .. }
                    )
                )
            })
            .count();

        let arith_order: usize = self
            .subgraph
            .iter()
            .filter_map(|vid| match self.connected_cg.vertex(vid).node() {
                super::template::VertexNode::Sliceable(super::template::SliceableNode::Arith {
                    arith,
                }) => Some(arith.g.order()),
                _ => None,
            })
            .sum();

        Some((
            deg,
            SubgraphStatistics {
                begin,
                n_arith_graphs: n_graphs,
                deg,
                arith_order,
                count,
                big_arith: contains_big_subgraph,
                input_size: self.input_size,
            },
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub(super) chunk_len: u64,
    pub(super) minimum_order: usize,
    /// Maximum total size of inputs to the fused subgraph, in bytes.
    /// If set to `Some(size)`, subgraph fusion will be limited to subgraphs
    /// whose total input size does not exceed this limit.
    /// This helps control memory usage and communication overhead.
    pub(super) maximum_input_size: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_len: 2u64.pow(20),
            minimum_order: 10,
            maximum_input_size: None,
        }
    }
}

impl Config {
    pub fn with_chunk_len(self, len: u64) -> Self {
        Config {
            chunk_len: len,
            ..self
        }
    }

    pub fn chunk_len(&self) -> u64 {
        self.chunk_len
    }

    pub fn with_minimum_order(self, order: usize) -> Self {
        Config {
            minimum_order: order,
            ..self
        }
    }

    pub fn with_maximum_input_size(self, size: Option<u64>) -> Self {
        Config {
            maximum_input_size: size,
            ..self
        }
    }

    pub fn maximum_input_size(&self) -> Option<u64> {
        self.maximum_input_size
    }
}

fn fuse_sliceable_subgraphs_from<'s, Rt: RuntimeType>(
    v: VertexId,
    cg: &mut super::Cg<'s, Rt>,
    fused_vids: &mut DenseSet,
    subgraph_minimum_order: usize,
    chunk_len: u64,
    maximum_input_size: Option<u64>,
    hd_info: &driver::HardwareInfo,
    memory_config: &super::memory_planning::Config,
    statistics: &mut Statistics,
) {
    let growing = if let Some(growing) = GrowingSubgraph::start_with(v, &cg) {
        growing
    } else {
        return;
    };

    let growed = growing
        .as_large_as_possible(maximum_input_size)
        .prune_non_first();
    let (deg, sg_statistics) = if let Some(deg) =
        growed.decide_fuseable(v, hd_info, memory_config, subgraph_minimum_order, chunk_len)
    {
        deg
    } else {
        return;
    };

    let Growed {
        subgraph,
        connected_cg,
        ..
    } = growed;

    let mut sub2big: Heap<sliceable_subgraph::VertexId, VertexId> =
        subgraph
            .iter_with_id()
            .fold(Heap::new(), |mut acc, (v, selected)| {
                if *selected {
                    acc.push(v);
                }
                acc
            });
    sub2big.iter().for_each(|v| fused_vids.add(*v));

    // Inputs are predecessors of the subgraph no in subgraph.
    // The value of each map entry indicates whether the input is sliceable.
    let subgraph_inputs_vids = sub2big
        .iter_with_id()
        .map(|(_sub_v, v)| {
            connected_cg
                .predecessors_of(*v)
                .filter(|u| !subgraph[*u])
                .map(|u| (u, false))
                .chain(
                    connected_cg
                        .vertex(*v)
                        .node()
                        .sliceable_uses_ref()
                        .cloned()
                        .filter(|u| !subgraph[*u])
                        .map(|u| (u, true)),
                )
        })
        .flatten()
        .collect::<BTreeMap<_, _>>();

    let trival_rotations_of_inputs = {
        let mut stack: Vec<VertexId> = subgraph_inputs_vids.iter().map(|(v, _)| *v).collect();
        let mut r = HashSet::new();

        while let Some(vid) = stack.pop() {
            for succ_id in connected_cg.successors_of(vid) {
                let successor = connected_cg.vertex(succ_id);
                if let super::VertexNode::Sliceable(super::template::SliceableNode::RotateIdx(..)) =
                    successor.node()
                {
                    stack.push(succ_id);
                    r.insert(succ_id);
                }
            }
        }

        r
    };

    // Outputs are those in subgraph that have successors not in subgraph.
    // Also, they cannot be trival rotations of inputs.
    let subgraph_outputs = sub2big
        .iter_with_id()
        .filter_map(|(sub_v, v)| {
            if !connected_cg.successors_of(*v).all(|u| subgraph[u]) {
                Some((sub_v, *v))
            } else {
                None
            }
        })
        .filter(|(_, vid)| !trival_rotations_of_inputs.contains(&*vid))
        .collect::<Vec<_>>();

    let subgraph_inputs_sub_vids = subgraph_inputs_vids
        .iter()
        .map(|(v, _)| sub2big.push(*v))
        .collect::<Vec<_>>();
    let sub2big = sub2big;

    let big2sub = sub2big
        .iter_with_id()
        .fold(BTreeMap::new(), |mut acc, (sub_v, v)| {
            acc.insert(*v, sub_v);
            acc
        });

    // Relabel external vertices to subgraph vertices
    let mut sub_cg_vertices: Heap<sliceable_subgraph::VertexId, _> =
        sub2big.map_by_ref(&mut |_sub_v, v| {
            let outer_vertex = connected_cg.vertex(*v);
            let (subgraph_node, sliceable) = if let Some(sliceable) = subgraph_inputs_vids.get(v) {
                if *sliceable {
                    (sliceable_subgraph::VertexNode::Input(*v), true)
                } else {
                    (sliceable_subgraph::VertexNode::InputAsIs(*v), false)
                }
            } else {
                (
                    sliceable_subgraph::VertexNode::relabeled_external_node(
                        *v,
                        outer_vertex.node().clone(),
                        |u| {
                            *big2sub.get(&u).unwrap_or_else(|| {
                                panic!("no big2sub found for {:?} as predecessor of {:?}", u, v)
                            })
                        },
                        || chunk_len as usize,
                    ),
                    true,
                )
            };
            sliceable_subgraph::Vertex::new(
                subgraph_node,
                sliceable_subgraph::Typ::new(
                    outer_vertex.typ().map_p(|(pty, original)| {
                        (*pty, if sliceable { chunk_len } else { *original })
                    }),
                    outer_vertex.typ().clone(),
                ),
                outer_vertex.src().clone(),
            )
        });

    // Create output vertex in subgraph
    let subgraph_output_typ = sliceable_subgraph::Typ::new(
        Typ::Tuple(
            subgraph_outputs
                .iter()
                .map(|(_, v)| {
                    connected_cg
                        .vertex(*v)
                        .typ()
                        .map_p(|(pty, _)| (*pty, chunk_len))
                })
                .collect(),
        ),
        Typ::Tuple(
            subgraph_outputs
                .iter()
                .map(|(_, v)| connected_cg.vertex(*v).typ().clone())
                .collect(),
        ),
    );
    let subgraph_output_src = subgraph_outputs
        .first()
        .map(|(_, v)| connected_cg.vertex(*v).src().clone())
        .unwrap();

    let subgraph_return_vid = sub_cg_vertices.push(sliceable_subgraph::Vertex::new(
        sliceable_subgraph::VertexNode::Return(
            subgraph_outputs.iter().map(|(sub_v, _)| *sub_v).collect(),
        ),
        subgraph_output_typ.clone(),
        subgraph_output_src.clone(),
    ));

    // Add subgraph to big graph
    let subgraph = sliceable_subgraph::Cg::new(
        subgraph_inputs_sub_vids.iter().cloned().collect(),
        Digraph::from_vertices(sub_cg_vertices),
        subgraph_return_vid,
        deg,
        chunk_len,
    );
    let subgraph_vid = cg.g.add_vertex(super::Vertex::new(
        super::VertexNode::Subgraph(subgraph),
        subgraph_output_typ.outer().clone(),
        subgraph_output_src,
    ));

    statistics.add(subgraph_vid, sg_statistics);

    // Replace output vertices in original graph with TupleGet
    subgraph_outputs.iter().enumerate().for_each(|(i, (_, v))| {
        *cg.g.vertex_mut(*v).node_mut() = super::VertexNode::TupleGet(subgraph_vid, i);
    });
}

pub fn fuse<'s, Rt: RuntimeType>(
    cg: super::no_subgraph::Cg<'s, Rt>,
    config: &Config,
    hd_info: &driver::HardwareInfo,
    memory_config: &super::memory_planning::Config,
) -> (super::Cg<'s, Rt>, Statistics) {
    let mut cg = id(cg);
    let seq =
        cg.g.dfs_from(cg.output)
            .map(|(vid, _)| vid)
            .collect::<Vec<_>>();
    let mut fused_vids = DenseSet::empty(cg.g.order());

    let mut statistics = Statistics::new();

    for vid in seq {
        if fused_vids[vid] {
            continue;
        }

        fuse_sliceable_subgraphs_from(
            vid,
            &mut cg,
            &mut fused_vids,
            config.minimum_order,
            config.chunk_len,
            config.maximum_input_size,
            hd_info,
            memory_config,
            &mut statistics,
        );
    }

    (cg, statistics)
}

pub fn id<'s, Rt: RuntimeType>(cg: super::no_subgraph::Cg<'s, Rt>) -> super::Cg<'s, Rt> {
    cg.map_vertices(|v| {
        v.map_node(|vn| vn.map_s(|_| panic!("subgraph not expected to appear here")))
    })
}
