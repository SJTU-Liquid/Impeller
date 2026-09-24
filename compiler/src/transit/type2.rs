//! Data structures for Stage 2 Tree Transit IR.

use crate::ast;
use crate::transit::{self, PolyInit, SourceInfo};
pub use ast::lowering::{Constant, ConstantId, ConstantTable};
pub use typ::Typ;
use zkpoly_common::arith;
use zkpoly_common::digraph;
use zkpoly_common::digraph::internal::{Predecessors, SubDigraph};
use zkpoly_common::heap::{Heap, UsizeId};
pub use zkpoly_common::typ::PolyType;
pub use zkpoly_runtime::args::{RuntimeType, Variable};
pub use zkpoly_runtime::error::RuntimeError;

zkpoly_common::define_usize_id!(VertexId);

impl std::fmt::Display for VertexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", usize::from(*self))
    }
}

pub type Arith<I> = arith::ArithGraph<I, arith::ExprId>;

#[derive(
    Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum NttAlgorithm<I> {
    Precomputed(I),
    Standard { pq: I, omega: I },
    Undecieded,
}

impl<I> Default for NttAlgorithm<I> {
    fn default() -> Self {
        NttAlgorithm::Undecieded
    }
}

impl<I> NttAlgorithm<I> {
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([x].into_iter()),
            Standard { pq, omega: base } => Box::new([pq, base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }

    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([x].into_iter()),
            Standard { pq, omega: base } => Box::new([pq, base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }
}

impl<I> NttAlgorithm<I>
where
    I: Clone,
{
    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.uses_ref().cloned()
    }

    pub fn try_relabeled<I2, Err>(
        &self,
        mapping: &mut impl FnMut(I) -> Result<I2, Err>,
    ) -> Result<NttAlgorithm<I2>, Err> {
        use NttAlgorithm::*;
        Ok(match self {
            Precomputed(x) => Precomputed(mapping(x.clone())?),
            Standard { pq, omega: base } => Standard {
                pq: mapping(pq.clone())?,
                omega: mapping(base.clone())?,
            },
            Undecieded => Undecieded,
        })
    }

    pub fn relabeled<I2>(&self, mapping: &mut impl FnMut(I) -> I2) -> NttAlgorithm<I2> {
        self.try_relabeled::<_, ()>(&mut |i| Ok(mapping(i)))
            .unwrap()
    }
}

impl<I> NttAlgorithm<I>
where
    I: Copy + Default,
{
    pub fn decide_alg<Rt: RuntimeType>(
        len: usize,
        memory_limit: usize,
        recompute_len: usize,
    ) -> Self {
        let field_sz = size_of::<Rt::Field>();
        let total_sz = len * field_sz;
        let recompute_sz = recompute_len * field_sz;
        let temp_id = I::default(); // 0 for later allocate
        if total_sz * 3 < memory_limit {
            Self::Precomputed(temp_id)
        } else if total_sz * 2 + recompute_sz < memory_limit {
            Self::Standard {
                pq: temp_id,
                omega: temp_id,
            }
        } else if total_sz * 3 < memory_limit * 2 {
            // 1.5
            Self::Precomputed(temp_id)
        } else if total_sz + recompute_sz < memory_limit {
            Self::Standard {
                pq: temp_id,
                omega: temp_id,
            }
        } else {
            unimplemented!("out of core ntt is unimplemented")
        }
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu(usize),
    Cpu,
}

impl Device {
    pub fn iter(n_gpus: usize) -> impl Iterator<Item = Self> {
        (0..n_gpus)
            .map(|i| Self::Gpu(i))
            .chain(std::iter::once(Self::Cpu))
    }

    pub fn gpus(n_gpus: usize) -> impl Iterator<Item = Self> {
        (0..n_gpus).map(|i| Self::Gpu(i))
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceSpecific<T> {
    pub gpu: Vec<T>,
    pub cpu: T,
}

impl<T> DeviceSpecific<T> {
    pub fn get(&self, device: Device) -> &T {
        match device {
            Device::Gpu(i) => &self.gpu[i],
            Device::Cpu => &self.cpu,
        }
    }

    pub fn get_mut(&mut self, device: Device) -> &mut T {
        match device {
            Device::Gpu(i) => &mut self.gpu[i],
            Device::Cpu => &mut self.cpu,
        }
    }

    pub fn with(mut self, device: Device, t: T) -> Self {
        *self.get_mut(device) = t;
        self
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DeviceSpecific<U> {
        DeviceSpecific {
            gpu: self.gpu.into_iter().map(&mut f).collect(),
            cpu: (&mut f)(self.cpu),
        }
    }

    pub fn default(n_gpus: usize) -> Self
    where
        T: Default,
    {
        DeviceSpecific {
            gpu: (0..n_gpus).map(|_| Default::default()).collect(),
            cpu: T::default(),
        }
    }

    pub fn new(n_gpus: usize, mut f: impl FnMut() -> T) -> Self {
        DeviceSpecific {
            gpu: (0..n_gpus).map(|_| f()).collect(),
            cpu: f(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Device, &T)> {
        Device::iter(self.gpu.len()).map(|d| (d, self.get(d)))
    }

    pub fn iter_gpus(&self) -> impl Iterator<Item = (Device, &T)> {
        self.gpu
            .iter()
            .enumerate()
            .map(|(i, d)| (Device::Gpu(i), d))
    }
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        match self {
            Device::Gpu(_) => true,
            Device::Cpu => false,
        }
    }

    pub fn unwrap_gpu(&self) -> usize {
        match self {
            Device::Gpu(i) => *i,
            _ => panic!("called unwrap_gpu on {:?}", self),
        }
    }
}

#[derive(Clone, Hash)]
pub struct DeviceCollection(DeviceSpecific<bool>);

impl std::fmt::Debug for DeviceCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_set();
        for d in self.iter().filter(|(_, t)| **t).map(|(d, _)| d) {
            s.entry(&d);
        }
        s.finish()
    }
}

impl std::ops::Deref for DeviceCollection {
    type Target = DeviceSpecific<bool>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DeviceCollection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DeviceCollection {
    pub fn ensure_single(&self) -> Device {
        let mut devices = self
            .iter()
            .filter(|(_, t)| **t)
            .map(|(a, _)| a)
            .collect::<Vec<_>>();

        if devices.len() != 1 {
            panic!("expected device collection to have only one device");
        }
        devices.pop().unwrap()
    }

    pub fn default(n_gpus: usize) -> Self {
        Self(DeviceSpecific::default(n_gpus))
    }

    pub fn with(mut self, d: Device) -> Self {
        *self.get_mut(d) = true;
        self
    }

    pub fn ensure_gpu(&self) -> Device {
        if let Some((gpu, _)) = self.iter_gpus().find(|(_d, t)| **t) {
            gpu
        } else {
            panic!(
                "expected this vertex to be executed on some GPU, got {:?}",
                self
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    PreferGpu,
    Gpu,
    Cpu,
}

pub mod template;

pub mod sliceable_subgraph;

/// The computation graph with full type information and sliceable subgraphs,
/// but uses alternative labels.
pub mod alt_label {
    use super::*;
    pub type VertexNode<'s, I, Rt> = template::VertexNode<
        I,
        Arith<I>,
        ConstantId,
        user_function::Id,
        sliceable_subgraph::alt_label::Cg<'s, sliceable_subgraph::VertexId, I, Rt>,
    >;
    pub type Vertex<'s, I, Rt> = transit::Vertex<VertexNode<'s, I, Rt>, Typ<Rt>, SourceInfo<'s>>;
}

/// The standard computation graph with full type information and sliceable subgraphs,
/// after subgraph slicing.
pub type VertexNode<'s, Rt> = alt_label::VertexNode<'s, VertexId, Rt>;
pub type Vertex<'s, Rt> = alt_label::Vertex<'s, VertexId, Rt>;
pub type Cg<'s, Rt> = transit::Cg<VertexId, Vertex<'s, Rt>>;
pub type CgSubgraph<'g, 's, Rt> = SubDigraph<'g, VertexId, Vertex<'s, Rt>>;

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    pub fn subgraphs_intensity(&self) -> (usize, usize) {
        use template::SliceableNode::Arith;
        use template::VertexNode::*;

        let g = self.connected_subgraph();

        let n_ops = g
            .vertices()
            .map(|vid| match g.vertex(vid).node() {
                Subgraph(sg) => sg.n_ops(),
                Sliceable(Arith { arith }) => arith.n_arith_ops(),
                _ => 0,
            })
            .sum();
        let io_size = g
            .vertices()
            .map(|vid| match g.vertex(vid).node() {
                Subgraph(sg) => sg.io_size() as usize,
                Sliceable(Arith { arith }) => arith.io_size(std::mem::size_of::<Rt::Field>()),
                _ => 0,
            })
            .sum();
        (n_ops, io_size)
    }

    pub fn n_subgraph_arith_ops(&self) -> usize {
        use template::SliceableNode::Arith;
        use template::VertexNode::*;

        let g = self.connected_subgraph();

        g.vertices()
            .map(|vid| match g.vertex(vid).node() {
                Subgraph(sg) => sg.n_arith_ops(),
                Sliceable(Arith { arith }) => arith.n_arith_ops(),
                _ => 0,
            })
            .sum()
    }
}

/// The computation graph generated from AST.
/// It has only partial type information, and it does not contain any sliceable subgraph.
pub mod partial_typed {
    use super::*;
    pub type VertexNode =
        template::VertexNode<VertexId, Arith<VertexId>, ConstantId, user_function::Id, ()>;
    pub type Vertex<'s, T> = transit::Vertex<VertexNode, T, SourceInfo<'s>>;
}

/// The computation graph after type inference.
/// It has full tyep information, but still does not contain any sliceable subgraph.
pub mod no_subgraph {
    pub mod alt_label {
        use super::super::*;
        pub type VertexNode<I> =
            template::VertexNode<I, Arith<I>, ConstantId, user_function::Id, ()>;
        pub type Vertex<'s, I, T> = transit::Vertex<VertexNode<I>, T, SourceInfo<'s>>;
        pub type Cg<'s, I, T> = transit::Cg<I, Vertex<'s, I, T>>;
    }

    pub type VertexNode = alt_label::VertexNode<super::VertexId>;
    pub type Vertex<'s, Rt> = alt_label::Vertex<'s, super::VertexId, super::Typ<Rt>>;
    pub type Cg<'s, Rt> = alt_label::Cg<'s, super::VertexId, super::Typ<Rt>>;

    use super::template;
    use super::Predecessors;
    use std::collections::HashMap;

    #[derive(serde::Serialize)]
    pub struct Statistics {
        order: usize,
        in_deg_count: HashMap<usize, usize>,
        out_deg_count: HashMap<usize, usize>,
        live_range: HashMap<usize, usize>,
        operators: HashMap<OpCategory, usize>,
        rotation_count: HashMap<i32, usize>,
    }

    #[derive(serde::Serialize, Hash, PartialEq, Eq)]
    pub enum OpCategory {
        Ntt,
        Msm,
        Invert,
        KateDivision,
        EvaluatePoly,
        ScanMul,
        DistributePowers,
        NewPoly,
        Arith,
        Rotate,
        Constant,
        Transcript,
        Blind,
        ScanMul_,
        PolyPermute,
        IndexPoly,
        AssemblePoly,
        Interpolate,
        Other,
    }

    impl<'s, Rt: zkpoly_runtime::args::RuntimeType> Cg<'s, Rt> {
        pub fn single_arith_io_size(&self) -> u64 {
            use template::SliceableNode::SingleArith;
            use template::VertexNode::*;

            let g = self.connected_subgraph();

            g.vertices()
                .map(|vid| match g.vertex(vid).node() {
                    Sliceable(SingleArith(..)) => g
                        .vertex(vid)
                        .predecessors()
                        .map(|pred| g.vertex(pred).typ().size())
                        .sum(),
                    _ => 0,
                })
                .sum()
        }

        pub fn graph_statistics(&self) -> Statistics {
            let g = self.connected_subgraph();

            let mut st = Statistics {
                order: g.order(),
                in_deg_count: HashMap::new(),
                out_deg_count: HashMap::new(),
                live_range: HashMap::new(),
                operators: HashMap::new(),
                rotation_count: HashMap::new(),
            };

            let in_deg = g.degrees_in();
            let out_deg = g.degrees_out();

            let depths: HashMap<super::VertexId, usize> = g
                .topology_sort_with_depth()
                .map(|(depth, i, _)| (i, depth))
                .collect();

            for vid in g.vertices() {
                let n = in_deg[vid];
                *st.in_deg_count.entry(n).or_default() += 1;
                let n = out_deg[vid];
                *st.out_deg_count.entry(n).or_default() += 1;

                use template::LastSliceableNode::*;
                use template::SliceableNode::*;
                use template::VertexNode::*;
                use zkpoly_common::arith;
                let operator = match g.vertex(vid).node() {
                    Ntt { .. } => OpCategory::Ntt,
                    LastSliceable(Msm { .. }) => OpCategory::Msm,
                    Sliceable(SingleArith(arith::Arith::Unr(
                        arith::UnrOp::P(arith::ArithUnrOp::Inv),
                        ..,
                    ))) => OpCategory::Invert,
                    KateDivision(..) => OpCategory::KateDivision,
                    LastSliceable(EvaluatePoly { .. }) => OpCategory::EvaluatePoly,
                    ScanMul { .. } => OpCategory::ScanMul,
                    Sliceable(DistributePowers { .. }) => OpCategory::DistributePowers,
                    Sliceable(NewPoly(..)) => OpCategory::NewPoly,
                    Sliceable(SingleArith(..)) => OpCategory::Arith,
                    Sliceable(Arith { .. }) => OpCategory::Arith,
                    Sliceable(BatchedInvert { .. }) => OpCategory::Invert,
                    Sliceable(RotateIdx(..)) => OpCategory::Rotate,
                    Blind(..) => OpCategory::Blind,
                    HashTranscript { .. } | SqueezeScalar(..) => OpCategory::Transcript,
                    ScanMul { .. } => OpCategory::ScanMul_,
                    PolyPermute(..) => OpCategory::PolyPermute,
                    Constant(..) => OpCategory::Constant,
                    IndexPoly(..) => OpCategory::IndexPoly,
                    Interpolate { .. } => OpCategory::Interpolate,
                    AssmblePoly(..) => OpCategory::AssemblePoly,
                    _ => OpCategory::Other,
                };
                *st.operators.entry(operator).or_default() += 1;

                if let Sliceable(RotateIdx(_, offset)) = g.vertex(vid).node() {
                    *st.rotation_count.entry(*offset).or_default() += 1;
                }

                let live_range = g
                    .successors_of(vid)
                    .map(|uid| depths[&uid])
                    .max()
                    .unwrap_or(0)
                    - g.successors_of(vid)
                        .map(|uid| depths[&uid])
                        .min()
                        .unwrap_or(0);
                *st.live_range.entry(live_range).or_default() += 1;
            }

            st
        }
    }
}

impl<I: UsizeId, C, E, T, S, Sg> digraph::internal::Predecessors<I>
    for transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E, Sg>,
        T,
        S,
    >
where
    Sg: template::SubgraphNode<I>,
    I: 'static,
{
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.node().uses()
    }
}

impl<'s, I, C, Sg, T, S>
    transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, Sg>,
        T,
        S,
    >
where
    I: Copy + 'static,
    Sg: template::SubgraphNode<I>,
{
    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.node().uses()
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        self.node_mut().uses_mut()
    }
    pub fn device(&self) -> DevicePreference {
        self.node().device()
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            Sliceable(sn) => sn.mutable_uses(),
            LastSliceable(lsn) => lsn.mutable_uses(),
            Ntt { s, .. } => Box::new([*s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([*transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([*transcript].into_iter()),
            Blind(s, ..) => Box::new([*s].into_iter()),
            _ => Box::new([].into_iter()),
        }
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use template::VertexNode::*;
        match self.node_mut() {
            Sliceable(sn) => sn.mutable_uses_mut(),
            LastSliceable(lsn) => lsn.mutable_uses_mut(),
            Ntt { s, .. } => Box::new([s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([transcript].into_iter()),
            Blind(s, ..) => Box::new([s].into_iter()),
            _ => Box::new([].into_iter()),
        }
    }

    /// For each output, if it took the space of some input `i`, returns `Some(i)`, otherwise `None`.
    /// If no output is inplace, returns an infinite iterator of `None`.
    pub fn outputs_inplace<'a, 'b, Rt: RuntimeType>(
        &'b self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = Option<I>> + 'b> {
        use template::VertexNode::*;
        match self.node() {
            Sliceable(sn) => sn.outputs_inplace(),
            LastSliceable(lsn) => lsn.outputs_inplace(),
            Ntt { s, .. } => Box::new([Some(*s)].into_iter()),
            HashTranscript { transcript, .. } => Box::new([Some(*transcript)].into_iter()),
            SqueezeScalar(transcript) => Box::new([Some(*transcript), None].into_iter()),
            UserFunction(fid, args) => {
                let f_typ = &uf_table[*fid].typ;
                let r = f_typ
                    .ret_inplace
                    .iter()
                    .map(|&i| Some(args[i?]))
                    .collect::<Vec<_>>();
                Box::new(r.into_iter())
            }
            Blind(s, ..) => Box::new([Some(*s)].into_iter()),
            _ => Box::new(std::iter::repeat(None)),
        }
    }

    pub fn is_virtual(&self) -> bool {
        self.node().is_virtual()
    }
}

impl<'s, I, C, Sg, T, S>
    transit::Cg<
        I,
        transit::Vertex<
            template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, Sg>,
            T,
            S,
        >,
    >
where
    I: UsizeId + 'static,
    C: Clone,
    Sg: template::SubgraphNode<I, AltLabeled<I> = Sg>,
    Sg::AltLabeled<I>: template::SubgraphNode<I>,
{
    pub fn compact(self) -> (Self, Heap<I, Option<I>>) {
        let transit::Cg { output, g } = self;
        let connected = g.connected_component(output);
        let (g, map) = g.compact(&connected, |v, mapping| {
            v.map_node(|n| n.relabeled(|i| mapping[i].unwrap()))
        });

        (
            Self {
                output: map[output].unwrap(),
                g,
            },
            map,
        )
    }
}

impl<'s, I, C, Sg, S, Rt: RuntimeType>
    transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, Sg>,
        Typ<Rt>,
        S,
    >
where
    I: Copy + 'static,
    Sg: template::SubgraphNode<I>,
{
    pub fn sliceable_property<'a>(
        &self,
        mut input_types: impl FnMut(&I) -> &'a Typ<Rt>,
    ) -> template::SliceableProperty {
        use template::SliceableProperty::*;
        use template::VertexNode::*;
        match self.node() {
            Sliceable(template::SliceableNode::SingleArith(arith::Arith::Bin(_, lhs, rhs)))
                if input_types(lhs) != input_types(rhs) || input_types(rhs) != self.typ() =>
            {
                No
            }
            Sliceable(..) => Always,
            LastSliceable(..) => No,
            TupleGet(..) => NonFirst,
            _ => No,
        }
    }
}

impl<I, C, S>
    template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, S>
{
    pub fn deterministic<Rt: RuntimeType>(&self, uf_table: &user_function::Table<Rt>) -> bool {
        match self {
            template::VertexNode::Blind(..) => false,
            template::VertexNode::UserFunction(f, _) => uf_table[*f].f.deterministic,
            _ => true,
        }
    }
}

pub struct Temporaries {
    sizes: Vec<u64>,
    device: super::type3::Device,
}

pub struct Program<'s, Rt: RuntimeType> {
    pub(crate) cg: no_subgraph::Cg<'s, Rt>,
    pub(crate) user_function_table: user_function::Table<Rt>,
    pub(crate) consant_table: ConstantTable<Rt>,
}

impl<'s, Rt: RuntimeType> std::hash::Hash for Program<'s, Rt> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // For now, we are not checking constants
        self.cg.hash(state);
        self.user_function_table.hash(state);
    }
}

pub mod common_subexpression_elimination;
pub mod decide_device;
pub mod graph_scheduling;
pub mod intt_mending;
pub mod kernel_fusion;
pub mod manage_inverse;
pub mod memory_planning;
pub mod object_analysis;
pub mod precompute;
pub mod pretty;
pub mod subgraph_slicing;
pub mod subgraph_splitting;
pub mod temporary_space;
pub mod typ;
pub mod user_function;
