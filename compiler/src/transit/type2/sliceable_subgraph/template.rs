use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::{digraph::internal::Digraph, heap::UsizeId};
use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;
use zkpoly_runtime::args::RuntimeType;

use super::super::{DeviceCollection, Temporaries, Typ};
use super::transit::{self};
use super::{Arith, LastSliceableNode, OuterVertexNode, SliceableNode};

#[derive(
    Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum VertexNode<I, Ie> {
    Input(Ie),
    InputAsIs(Ie),
    Inner(SliceableNode<I, Arith<I>>),
    Last(LastSliceableNode<I>),
    TupleGet(I, usize),
    Return(Vec<I>),
}

impl<I, Ie> Default for VertexNode<I, Ie> {
    fn default() -> Self {
        Self::Inner(SliceableNode::NewPoly(
            0,
            transit::PolyInit::Zeros,
            zkpoly_common::typ::PolyType::Coef,
        ))
    }
}

impl<I, Ie> VertexNode<I, Ie> {
    pub fn unwrap_input(&self) -> &Ie {
        match self {
            VertexNode::Input(x) | VertexNode::InputAsIs(x) => x,
            _ => panic!("called unwrap_input on non-input vertex"),
        }
    }

    pub fn unwrap_input_mut(&mut self) -> &mut Ie {
        match self {
            VertexNode::Input(x) | VertexNode::InputAsIs(x) => x,
            _ => panic!("called unwrap_input_mut on non-input vertex"),
        }
    }

    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::empty()),
            Inner(sn) => sn.uses_ref(),
            Last(lsn) => lsn.uses_ref(),
            TupleGet(t, _) => Box::new(std::iter::once(t)),
            Return(r) => Box::new(r.iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::empty()),
            Inner(sn) => sn.uses_mut(),
            Last(lsn) => lsn.uses_mut(),
            TupleGet(t, _) => Box::new(std::iter::once(t)),
            Return(r) => Box::new(r.iter_mut()),
        }
    }

    pub fn sliceable_uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::empty()),
            Inner(sn) => sn.sliceable_uses_ref(),
            Last(lsn) => lsn.sliceable_uses_ref(),
            TupleGet(t, _) => Box::new(std::iter::once(t)),
            Return(r) => Box::new(r.iter()),
        }
    }

    pub fn immortal(&self) -> bool {
        false
    }

    pub fn wrap_type2<E, S, C>(self) -> super::super::template::VertexNode<I, Arith<I>, C, E, S> {
        use super::super::template;
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => panic!("input cannot map to a Type2 vertex"),
            Inner(sn) => template::VertexNode::Sliceable(sn),
            Last(lsn) => template::VertexNode::LastSliceable(lsn),
            TupleGet(..) => panic!("TupleGet not expected"),
            Return(..) => panic!("Return not expected"),
        }
    }

    pub fn is_tuple_get(&self) -> bool {
        matches!(self, VertexNode::TupleGet(..))
    }

    pub fn is_input(&self) -> bool {
        matches!(self, VertexNode::Input(..) | VertexNode::InputAsIs(..))
    }

    pub fn is_output(&self) -> bool {
        matches!(self, VertexNode::Return(..))
    }
}

impl<I, Ie> VertexNode<I, Ie>
where
    I: Clone,
{
    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.uses_ref().cloned()
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::empty()),
            Inner(sn) => sn.mutable_uses(),
            Last(lsn) => lsn.mutable_uses(),
            TupleGet(..) => Box::new(std::iter::empty()),
            Return(..) => Box::new(std::iter::empty()),
        }
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::empty()),
            Inner(sn) => sn.mutable_uses_mut(),
            Last(lsn) => lsn.mutable_uses_mut(),
            TupleGet(..) => Box::new(std::iter::empty()),
            Return(..) => Box::new(std::iter::empty()),
        }
    }

    pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a> {
        use VertexNode::*;
        match self {
            Input(..) | InputAsIs(..) => Box::new(std::iter::repeat(None)),
            Inner(sn) => sn.outputs_inplace(),
            Last(lsn) => lsn.outputs_inplace(),
            TupleGet(..) => Box::new(std::iter::repeat(None)),
            Return(..) => Box::new(std::iter::repeat(None)),
        }
    }

    pub fn unwrap_output(&self) -> &[I] {
        use VertexNode::*;
        match self {
            Return(r) => r,
            _ => panic!("called unwrap_output on non-output vertex"),
        }
    }

    /// Provided a vertex in the big graph and returns a vertex in sliceable subgraph.
    pub fn relabeled_external_node<E, Sg, C>(
        external_vid: Ie,
        node: OuterVertexNode<Ie, Arith<Ie>, C, E, Sg>,
        mut f: impl FnMut(Ie) -> I,
        arith_deg: impl FnOnce() -> usize,
    ) -> Self
    where
        I: Clone + Default + Ord + std::fmt::Debug,
        Ie: Clone + std::fmt::Debug,
    {
        use OuterVertexNode::*;
        let mut node = match node {
            Sliceable(sn) => Self::Inner(sn.relabeled(f)),
            TupleGet(t, i) => Self::TupleGet(f(t.clone()), i),
            _ => panic!(
                "some vertex {:?} other than Sliceable or TupleGet not expected",
                external_vid
            ),
        };

        if let Self::Inner(SliceableNode::Arith { arith }) = &mut node {
            arith.poly_degree = Some(arith_deg());
        }

        node
    }

    pub fn try_relabeled<Ie2, I2, Er>(
        self,
        mut f: impl FnMut(I) -> Result<I2, Er>,
        mut fe: impl FnMut(Ie) -> Result<Ie2, Er>,
    ) -> Result<VertexNode<I2, Ie2>, Er>
    where
        I: Clone,
        Ie: Clone,
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        use VertexNode::*;
        Ok(match self {
            InputAsIs(ie) => InputAsIs(fe(ie)?),
            Input(ie) => Input(fe(ie)?),
            Inner(sn) => Inner(sn.try_relabeled(&mut f)?),
            Last(lsn) => Last(lsn.try_relabeled(&mut f)?),
            TupleGet(t, i) => TupleGet(f(t)?, i),
            Return(r) => Return(r.iter().cloned().map(f).collect::<Result<Vec<_>, _>>()?),
        })
    }

    pub fn relabeled<Ie2, I2>(
        self,
        mut f: impl FnMut(I) -> I2,
        mut fe: impl FnMut(Ie) -> Ie2,
    ) -> VertexNode<I2, Ie2>
    where
        I: Clone,
        Ie: Clone,
        I2: Default + Ord + std::fmt::Debug + Clone,
        Ie2: Default + Ord + std::fmt::Debug + Clone,
    {
        self.try_relabeled(|x| Ok::<_, ()>(f(x)), |x| Ok(fe(x)))
            .unwrap()
    }
    pub fn temporary_space_needed<'a, Rt: RuntimeType>(
        &self,
        device: &DeviceCollection,
        libs: &mut Libs,
        input_types: impl FnOnce(&I) -> &'a Typ<Rt> + 'a,
        ag_scheduler: ArithGraphScheduler,
    ) -> Option<Temporaries> {
        match self {
            VertexNode::Inner(sn) => {
                sn.temporary_space_needed(device, libs, input_types, ag_scheduler)
            }
            VertexNode::Last(lsn) => lsn.temporary_space_needed(device, libs, input_types),
            _ => None,
        }
    }
}

#[derive(
    Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub struct Cg<I, V> {
    pub(crate) inputs: Vec<I>,
    pub(crate) loop_inputs: Vec<I>,
    pub(crate) g: Digraph<I, V>,
    pub(crate) output: I,
    pub(crate) loop_output: Vec<I>,
    pub(crate) deg: u64,
    pub(crate) chunk_len: u64,
}

impl<I, V> Cg<I, V> {
    pub fn new(inputs: Vec<I>, g: Digraph<I, V>, output: I, deg: u64, chunk_len: u64) -> Self {
        Self {
            inputs,
            g,
            output,
            loop_inputs: Vec::new(),
            loop_output: Vec::new(),
            deg,
            chunk_len,
        }
    }
}

impl<I, Ie, T, S> Cg<I, transit::Vertex<VertexNode<I, Ie>, T, S>> {
    pub fn try_relabeled<Ie2, Er>(
        &self,
        mut f: impl FnMut(Ie) -> Result<Ie2, Er>,
    ) -> Result<Cg<I, transit::Vertex<VertexNode<I, Ie2>, T, S>>, Er>
    where
        Ie: Clone,
        I: UsizeId,
        T: Clone,
        S: Clone,
    {
        let g =
            self.g
                .map_by_ref_result(&mut |_, v: &transit::Vertex<VertexNode<I, Ie>, T, S>| {
                    use VertexNode::*;
                    let node = match v.node() {
                        Input(ie) => Input(f(ie.clone())?),
                        InputAsIs(ie) => InputAsIs(f(ie.clone())?),
                        Inner(inner) => Inner(inner.clone()),
                        Last(last) => Last(last.clone()),
                        TupleGet(t, i) => TupleGet(t.clone(), *i),
                        Return(r) => Return(r.clone()),
                    };

                    Ok(transit::Vertex::new(node, v.typ().clone(), v.src().clone()))
                })?;

        Ok(Cg {
            inputs: self.inputs.clone(),
            output: self.output.clone(),
            g,
            loop_inputs: self.loop_inputs.clone(),
            loop_output: self.loop_output.clone(),
            deg: self.deg,
            chunk_len: self.chunk_len,
        })
    }

    pub fn nth_output(&self, i: usize) -> I
    where
        I: UsizeId,
    {
        self.g.vertex(self.output).node().unwrap_output()[i]
    }
}

impl<I, Ie, T, S> super::super::template::SubgraphNode<Ie>
    for Cg<I, transit::Vertex<VertexNode<I, Ie>, T, S>>
where
    I: UsizeId,
    Ie: Clone,
    T: Clone,
    S: Clone,
{
    type AltLabeled<Ie2> = Cg<I, transit::Vertex<VertexNode<I, Ie2>, T, S>>;

    fn inputs<'a>(&'a self) -> impl Iterator<Item = &'a Ie> + 'a
    where
        Ie: 'a,
    {
        self.inputs
            .iter()
            .map(|i| self.g.vertex(*i).node().unwrap_input())
    }

    fn inputs_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Ie>
    where
        Ie: 'a,
    {
        self.inputs
            .iter_mut()
            .map(|i| -> *mut Ie { self.g.vertex_mut(*i).node_mut().unwrap_input_mut() as *mut _ })
            .map(|p: *mut _| unsafe { p.as_mut().unwrap() })
    }

    fn try_relabeled<I2, Er>(
        &self,
        f: impl FnMut(Ie) -> Result<I2, Er>,
    ) -> Result<Self::AltLabeled<I2>, Er> {
        self.try_relabeled(f)
    }
}

mod labeling {
    use super::super::super::template::labelling::{EdgeLabel, LabelT};
    use super::VertexNode;
    use std::fmt::Display;

    impl<'a, I, Ie> Display for LabelT<'a, VertexNode<I, Ie>> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use VertexNode::*;
            match &self.0 {
                InputAsIs(..) => write!(f, "InputAsIs"),
                Input(..) => write!(f, "Input"),
                Inner(sn) => write!(f, "{}", LabelT(sn)),
                Last(lsn) => write!(f, "{}", LabelT(lsn)),
                TupleGet(_, i) => write!(f, "TupleGet({i})"),
                Return(..) => write!(f, "Return"),
            }
        }
    }

    impl<I, Ie> VertexNode<I, Ie> {
        pub fn labeled_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (I, Option<EdgeLabel>)> + 'a>
        where
            I: Clone,
        {
            use VertexNode::*;
            match self {
                Return(xs) => Box::new(
                    xs.iter()
                        .enumerate()
                        .map(|(i, x)| (x.clone(), Some(EdgeLabel::Enumerate("x", i)))),
                ),
                _ => Box::new(self.uses().map(|x| (x, None))),
            }
        }
    }
}

pub mod pretty {
    use super::super::super::template::{
        labelling::LabelT,
        pretty::{Style, StyleInner},
    };
    use super::*;
    use crate::transit;
    use cytoscape_visualizer::{self as vis, pretty::*};
    use zkpoly_common::arith;

    use std::fmt::{Debug, Display};

    type Vertex<Ii, I, T, S> = transit::Vertex<VertexNode<Ii, I>, T, S>;

    pub struct DefaultVertexStyle;

    impl<Ii, N, T, S> Style<Ii, transit::Vertex<N, T, S>> for DefaultVertexStyle
    where
        T: Debug,
        S: Debug,
        Ii: Display,
        for<'a> LabelT<'a, N>: Display,
    {
        fn style_vertex(
            &self,
            vid: Ii,
            v: &transit::Vertex<N, T, S>,
        ) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!("{}: {}", vid, LabelT(v.node()))).with_info(format!(
                "{:?}\\n@{:?}",
                v.typ(),
                v.src()
            ))
        }
    }

    impl<I, Ii, T, S, F> PrettyVertex<Ii, I, F> for transit::Vertex<VertexNode<Ii, I>, T, S>
    where
        Ii: Display + Clone + 'static,
        I: Display + Clone,
        T: Debug,
        S: Debug,
        F: StyleInner<Ii, I, transit::Vertex<VertexNode<Ii, I>, T, S>>,
    {
        fn pretty_vertex<'a>(
            &self,
            vid: Ii,
            ctx: Context<'a>,
            aux: &F,
            builder: &mut vis::Builder,
        ) {
            let id = ctx.identifier(vid.clone());

            match self.node() {
                VertexNode::Inner(SliceableNode::Arith { arith, .. }) => {
                    let mut cluster_builder = vis::ClusterBuilder::new(aux.style_vertex(vid, self));
                    pretty_vertices(arith, &(), ctx.push(&id), &mut cluster_builder);
                    builder.cluster(id, cluster_builder);
                }
                _ => {
                    builder.vertex(id, aux.style_vertex(vid, self));
                }
            }
        }

        fn labeled_predecessors<'a>(
            &'a self,
            vid: Ii,
            aux: &'a F,
        ) -> impl Iterator<Item = (Predecessor<Ii, I>, vis::EdgeStyle)> + 'a {
            let it: Box<dyn Iterator<Item = _>> = match self.node() {
                VertexNode::Inner(SliceableNode::Arith { .. }) => Box::new(std::iter::empty()),
                VertexNode::Input(i) | VertexNode::InputAsIs(i) => Box::new(std::iter::once((
                    Predecessor::External(i.clone()),
                    aux.style_from_outer_edge(i.clone()),
                ))),
                _ => Box::new(self.node().labeled_uses().map(move |(u, l)| {
                    (
                        Predecessor::Internal(u.clone()),
                        aux.style_edge(u.clone(), vid.clone(), l.map(|l| l.to_string())),
                    )
                })),
            };
            it
        }

        fn pretty_vertex_internal_edges<'a, G>(
            &self,
            vid: Ii,
            ctx: Context<'a>,
            aux: &F,
            _g: &G,
            builder: &mut vis::Builder,
        ) -> Vec<pretty::LooseEnd<Ii>>
        where
            G: PrettyGraph<Self, Ii, I>,
            Self: Sized,
        {
            let id = ctx.identifier(vid.clone());
            if let VertexNode::Inner(SliceableNode::Arith { arith, .. }) = self.node() {
                pretty_edges(
                    arith,
                    &arith::pretty_print::InputEdgeStyle(|outer: Ii| {
                        aux.style_edge(outer.clone(), vid.clone(), None)
                    }),
                    ctx.push(&id),
                    builder,
                )
            } else {
                Vec::new()
            }
        }
    }

    impl<I, Ii, T, S> PrettyGraph<Vertex<Ii, I, T, S>, Ii, I> for Cg<Ii, Vertex<Ii, I, T, S>>
    where
        Ii: UsizeId + Display,
        I: Display + Clone,
        T: Debug,
        S: Debug,
    {
        fn vertices(&self) -> impl Iterator<Item = Ii> {
            use VertexNode::*;
            self.g
                .vertices()
                .filter(|i| match self.g.vertex(*i).node() {
                    TupleGet(..) => false,
                    _ => true,
                })
        }

        fn vertex(&self, vid: Ii) -> &Vertex<Ii, I, T, S> {
            self.g.vertex(vid)
        }

        fn indirect_from<'a>(&self, vid: Ii, ctx: &Context<'a>) -> IndirectFrom<I> {
            match self.g.vertex(vid).node() {
                VertexNode::TupleGet(from, i) => match self.g.vertex(*from).node() {
                    VertexNode::Inner(SliceableNode::Arith { arith, .. }) => {
                        let id = ctx.identifier(*from);
                        let ifrom = arith.indirect_from(arith.outputs[*i], &ctx.push(&id));
                        ifrom.bind(|vid| self.indirect_from(vid, &ctx))
                    }
                    _ => panic!("tuple get only expected to get from Arith"),
                },
                _ => IndirectFrom::Internal(ctx.identifier(vid)),
            }
        }
    }
}
