use super::super::sliceable_subgraph;
use super::super::Arith;
use super::labelling::LabelT;
use super::*;
use crate::transit;
use cytoscape_visualizer::{self as vis, pretty::*};
use zkpoly_common::{
    digraph::internal::{Digraph, SubDigraph},
    heap::UsizeId,
};

use std::fmt::{Debug, Display};

pub type Vertex<I, C, E, S, T, Sr> = transit::Vertex<VertexNode<I, Arith<I>, C, E, S>, T, Sr>;

use super::super::sliceable_subgraph::template as ss_template;

type SsVertex<Ii, I, T, Sr> = transit::Vertex<ss_template::VertexNode<Ii, I>, T, Sr>;
type SsCg<Ii, I, T, Sr> = ss_template::Cg<Ii, SsVertex<Ii, I, T, Sr>>;

pub trait Style<I, V> {
    fn style_vertex(&self, vid: I, v: &V) -> vis::Vertex;
    #[allow(unused_variables)]
    fn style_edge(&self, from: I, to: I, label: Option<String>) -> vis::EdgeStyle {
        vis::EdgeStyle::default().with_optional_label(label)
    }
}

pub struct WithDefaultStyleFromOuterEdge<F>(pub F);

pub trait StyleInner<I, Ie, V>: Style<I, V> {
    fn style_from_outer_edge(&self, _from: Ie) -> vis::EdgeStyle {
        vis::EdgeStyle::default()
    }
}

impl<I, V, F> Style<I, V> for WithDefaultStyleFromOuterEdge<F>
where
    F: Style<I, V>,
{
    fn style_vertex(&self, vid: I, v: &V) -> cytoscape_visualizer::Vertex {
        self.0.style_vertex(vid, v)
    }
    fn style_edge(&self, from: I, to: I, label: Option<String>) -> cytoscape_visualizer::EdgeStyle {
        self.0.style_edge(from, to, label)
    }
}

impl<I, Ie, V, F> StyleInner<I, Ie, V> for WithDefaultStyleFromOuterEdge<F> where F: Style<I, V> {}

pub struct DefaultVertexStyle;

impl<I, N, T, Sr> Style<I, transit::Vertex<N, T, Sr>> for DefaultVertexStyle
where
    I: Display,
    Sr: Debug,
    T: Debug,
    for<'a> LabelT<'a, N>: Display,
{
    fn style_vertex(&self, vid: I, v: &transit::Vertex<N, T, Sr>) -> cytoscape_visualizer::Vertex {
        vis::Vertex::new(format!("{}: {}", vid, LabelT(v.node()))).with_info(format!(
            "{:?}\\n@{:?}",
            v.typ(),
            v.src()
        ))
    }
}

impl<I, C, E, T, Sr, F> PrettyVertex<I, (), F> for Vertex<I, C, E, (), T, Sr>
where
    I: Display + Copy + 'static,
    Sr: Debug,
    T: Debug,
    C: Debug,
    E: Debug,
    F: Style<I, Vertex<I, C, E, (), T, Sr>>,
{
    fn pretty_vertex<'a>(&self, vid: I, ctx: Context<'a>, aux: &F, builder: &mut vis::Builder) {
        let id = ctx.identifier(vid);

        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
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
        vid: I,
        aux: &'a F,
    ) -> impl Iterator<Item = (Predecessor<I, ()>, vis::EdgeStyle)> + 'a {
        let it: Box<dyn Iterator<Item = _>> = match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { .. }) => Box::new(std::iter::empty()),
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
        vid: I,
        ctx: Context<'a>,
        _aux: &F,
        _g: &G,
        builder: &mut vis::Builder,
    ) -> Vec<pretty::LooseEnd<I>>
    where
        G: PrettyGraph<Self, I, ()>,
        Self: Sized,
    {
        let id = ctx.identifier(vid);
        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => pretty_edges(
                arith,
                &arith::pretty_print::InputEdgeStyle(|outer: I| {
                    _aux.style_edge(outer.clone(), vid.clone(), None)
                }),
                ctx.push(&id),
                builder,
            ),
            _ => Vec::new(),
        }
    }
}

pub trait Style2<I, Ii, V, Vi>: Style<I, V> {
    type SubStyle: StyleInner<Ii, I, Vi>;
    fn sub(&self, vid: I) -> Self::SubStyle;
}

pub struct WithStyle2Default<F>(pub F);

impl<I, N, T, Sr, F> Style<I, transit::Vertex<N, T, Sr>> for WithStyle2Default<F>
where
    I: Display,
    Sr: Debug,
    T: Debug,
    F: Style<I, transit::Vertex<N, T, Sr>>,
{
    fn style_vertex(&self, vid: I, v: &transit::Vertex<N, T, Sr>) -> cytoscape_visualizer::Vertex {
        self.0.style_vertex(vid, v)
    }

    fn style_edge(&self, from: I, to: I, label: Option<String>) -> cytoscape_visualizer::EdgeStyle {
        self.0.style_edge(from, to, label)
    }
}

impl<Ii, I, N, Ni, T, Ti, Sr, F>
    Style2<I, Ii, transit::Vertex<N, T, Sr>, transit::Vertex<Ni, Ti, Sr>> for WithStyle2Default<F>
where
    I: Display,
    Ii: Display,
    Sr: Debug,
    T: Debug,
    Ti: Debug,
    F: Style<I, transit::Vertex<N, T, Sr>>,
    for<'a> LabelT<'a, Ni>: Display,
{
    type SubStyle = pretty::WithDefaultStyleFromOuterEdge<
        sliceable_subgraph::template::pretty::DefaultVertexStyle,
    >;
    fn sub(&self, _vid: I) -> Self::SubStyle {
        pretty::WithDefaultStyleFromOuterEdge(
            sliceable_subgraph::template::pretty::DefaultVertexStyle,
        )
    }
}

impl<I, C, E, T, Ti, Sr, Ii, F> PrettyVertex<I, (), F>
    for Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>
where
    I: Display + Copy + 'static + Debug,
    Sr: Debug + Clone,
    T: Debug + Clone,
    Ti: Debug + Clone,
    C: Debug + Clone,
    E: Debug,
    Ii: Copy + Display + UsizeId + 'static,
    F: Style2<I, Ii, Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>, SsVertex<Ii, I, Ti, Sr>>,
{
    fn pretty_vertex<'a>(&self, vid: I, ctx: Context<'a>, aux: &F, builder: &mut vis::Builder) {
        let id = ctx.identifier(vid.clone());

        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
                let mut cluster_builder = vis::ClusterBuilder::new(aux.style_vertex(vid, self));
                pretty_vertices(arith, &(), ctx.push(&id), &mut cluster_builder);
                builder.cluster(id, cluster_builder);
            }
            VertexNode::Subgraph(s) => {
                let mut cluster_builder = vis::ClusterBuilder::new(aux.style_vertex(vid, self));
                pretty_vertices(s, &aux.sub(vid), ctx.push(&id), &mut cluster_builder);
                builder.cluster(id, cluster_builder);
            }
            _ => {
                builder.vertex(id, aux.style_vertex(vid, self));
            }
        }
    }

    fn labeled_predecessors<'a>(
        &'a self,
        vid: I,
        aux: &'a F,
    ) -> impl Iterator<Item = (Predecessor<I, ()>, vis::EdgeStyle)> + 'a {
        let it: Box<dyn Iterator<Item = _>> = match self.node() {
            VertexNode::Subgraph(..) | VertexNode::Sliceable(SliceableNode::Arith { .. }) => {
                Box::new(std::iter::empty())
            }
            _ => Box::new(self.node().labeled_uses().map(move |(u, l)| {
                (
                    Predecessor::Internal(u.clone()),
                    aux.style_edge(u.clone(), vid.clone(), l.map(|x| x.to_string())),
                )
            })),
        };
        it
    }

    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        vid: I,
        ctx: Context<'a>,
        aux: &F,
        _g: &G,
        builder: &mut vis::Builder,
    ) -> Vec<pretty::LooseEnd<I>>
    where
        G: PrettyGraph<Self, I, ()>,
        Self: Sized,
    {
        let id = ctx.identifier(vid);
        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => pretty_edges(
                arith,
                &arith::pretty_print::InputEdgeStyle(|outer: I| {
                    aux.style_edge(outer.clone(), vid.clone(), None)
                }),
                ctx.push(&id),
                builder,
            ),
            VertexNode::Subgraph(s) => pretty_edges(s, &aux.sub(vid), ctx.push(&id), builder),
            _ => Vec::new(),
        }
    }
}

impl<'g, I, C, E, T, Ti, Sr, Ii> PrettyGraph<Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>, I, ()>
    for SubDigraph<'g, I, Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    Ti: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
    Ii: UsizeId + Display + 'static,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        SubDigraph::vertices(self).filter(|vid| match self.vertex(*vid).node() {
            VertexNode::TupleGet(from, ..)
                if matches!(
                    self.vertex(*from).node(),
                    VertexNode::Sliceable(SliceableNode::Arith { .. }) | VertexNode::Subgraph(..)
                ) =>
            {
                false
            }
            _ => true,
        })
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr> {
        SubDigraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> IndirectFrom<()> {
        use VertexNode::*;

        let internal_id = |i| IndirectFrom::Internal(ctx.identifier(i));

        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Sliceable(SliceableNode::Arith { arith, .. }) => {
                    let from_id = ctx.identifier(*from);
                    arith
                        .indirect_from(arith.outputs[*i], &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                Subgraph(sg) => {
                    let from_id = ctx.identifier(*from);
                    sg.indirect_from(sg.nth_output(*i), &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                _ => internal_id(vid),
            },
            Subgraph(sg) => {
                let id = ctx.identifier(vid);
                sg.indirect_from(sg.output, &ctx.push(&id))
                    .bind(|vid| self.indirect_from(vid, ctx))
            }
            _ => internal_id(vid),
        }
    }
}

impl<I, C, E, T, Ti, Sr, Ii> PrettyGraph<Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>, I, ()>
    for Digraph<I, Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    Ti: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
    Ii: UsizeId + Display + 'static,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        Digraph::vertices(self).filter(|vid| match self.vertex(*vid).node() {
            VertexNode::TupleGet(from, ..)
                if matches!(
                    self.vertex(*from).node(),
                    VertexNode::Sliceable(SliceableNode::Arith { .. }) | VertexNode::Subgraph(..)
                ) =>
            {
                false
            }
            _ => true,
        })
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, SsCg<Ii, I, Ti, Sr>, T, Sr> {
        Digraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> IndirectFrom<()> {
        use VertexNode::*;

        let internal_id = |i| IndirectFrom::Internal(ctx.identifier(i));

        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Sliceable(SliceableNode::Arith { arith, .. }) => {
                    let from_id = ctx.identifier(*from);
                    arith
                        .indirect_from(arith.outputs[*i], &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                Subgraph(sg) => {
                    let from_id = ctx.identifier(*from);
                    sg.indirect_from(sg.nth_output(*i), &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                _ => internal_id(vid),
            },
            Subgraph(sg) => {
                let id = ctx.identifier(vid);
                sg.indirect_from(sg.output, &ctx.push(&id))
                    .bind(|vid| self.indirect_from(vid, ctx))
            }
            _ => internal_id(vid),
        }
    }
}

impl<'g, I, C, E, T, Sr> PrettyGraph<Vertex<I, C, E, (), T, Sr>, I, ()>
    for SubDigraph<'g, I, Vertex<I, C, E, (), T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        SubDigraph::vertices(self).filter(|vid| match self.vertex(*vid).node() {
            VertexNode::TupleGet(from, ..)
                if matches!(
                    self.vertex(*from).node(),
                    VertexNode::Sliceable(SliceableNode::Arith { .. })
                ) =>
            {
                false
            }
            _ => true,
        })
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, (), T, Sr> {
        SubDigraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> IndirectFrom<()> {
        let internal_id = |i: I| IndirectFrom::Internal(ctx.identifier(i));

        use VertexNode::*;
        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Sliceable(SliceableNode::Arith { arith, .. }) => {
                    let from_id = ctx.identifier(*from);
                    arith
                        .indirect_from(arith.outputs[*i], &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                Subgraph(..) => panic!("this variant is not expected"),
                _ => internal_id(vid),
            },
            _ => internal_id(vid),
        }
    }
}

impl<I, C, E, T, Sr> PrettyGraph<Vertex<I, C, E, (), T, Sr>, I, ()>
    for Digraph<I, Vertex<I, C, E, (), T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        Digraph::vertices(self).filter(|vid| match self.vertex(*vid).node() {
            VertexNode::TupleGet(from, ..)
                if matches!(
                    self.vertex(*from).node(),
                    VertexNode::Sliceable(SliceableNode::Arith { .. })
                ) =>
            {
                false
            }
            _ => true,
        })
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, (), T, Sr> {
        Digraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> IndirectFrom<()> {
        let internal_id = |i: I| IndirectFrom::Internal(ctx.identifier(i));

        use VertexNode::*;
        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Sliceable(SliceableNode::Arith { arith, .. }) => {
                    let from_id = ctx.identifier(*from);
                    arith
                        .indirect_from(arith.outputs[*i], &ctx.push(&from_id))
                        .bind(|vid| self.indirect_from(vid, &ctx))
                }
                Subgraph(..) => panic!("this variant is not expected"),
                _ => internal_id(vid),
            },
            _ => internal_id(vid),
        }
    }
}
