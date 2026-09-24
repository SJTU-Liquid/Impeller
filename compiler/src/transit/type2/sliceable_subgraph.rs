use zkpoly_common::digraph;
use zkpoly_common::digraph::internal::SubDigraph;
use zkpoly_runtime::args::RuntimeType;

use super::template::{LastSliceableNode, SliceableNode, VertexNode as OuterVertexNode};
use super::transit::{self, SourceInfo};
use super::Arith;

zkpoly_common::define_usize_id!(VertexId);

impl std::fmt::Display for VertexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", usize::from(*self))
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct Typ<Rt: RuntimeType> {
    /// The type of the vertex in the subgraph
    inner: super::Typ<Rt>,
    /// The type of the vertex before it is fused into the subgraph
    outer: super::Typ<Rt>,
}

impl<Rt: RuntimeType> Typ<Rt> {
    pub fn new(inner: super::Typ<Rt>, outer: super::Typ<Rt>) -> Self {
        Self { inner, outer }
    }

    pub fn inner(&self) -> &super::Typ<Rt> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut super::Typ<Rt> {
        &mut self.inner
    }

    pub fn outer(&self) -> &super::Typ<Rt> {
        &self.outer
    }

    pub fn tuple(v: Vec<Self>) -> Self {
        let (inner, outer): (Vec<_>, Vec<_>) = v
            .into_iter()
            .map(|Typ { inner, outer }| (inner, outer))
            .unzip();

        Self {
            inner: super::Typ::Tuple(inner),
            outer: super::Typ::Tuple(outer),
        }
    }
}

pub mod template;

pub mod alt_label {
    use zkpoly_common::heap::UsizeId;

    use super::*;

    pub type VertexNode<I, Ie> = template::VertexNode<I, Ie>;
    pub type Vertex<'s, I, Ie, T> = transit::Vertex<VertexNode<I, Ie>, T, SourceInfo<'s>>;
    pub type Cg<'s, I, Ie, Rt> = template::Cg<I, Vertex<'s, I, Ie, super::Typ<Rt>>>;
    pub type CgPartilTyped<'s, I, Ie, T> = template::Cg<I, Vertex<'s, I, Ie, T>>;

    impl<'s, I, Ie, Rt: RuntimeType> Cg<'s, I, Ie, Rt>
    where
        I: UsizeId,
    {
        pub fn set_chunk_len(&mut self, chunk_length: u64) {
            self.chunk_len = chunk_length;

            for vid in self.g.vertices() {
                let v = self.g.vertex_mut(vid);
                let is_input_as_is = match v.node() {
                    template::VertexNode::InputAsIs(..) => true,
                    _ => false,
                };

                if !is_input_as_is {
                    let typ = v.typ_mut().inner_mut();
                    *typ = typ.map_p(|(p, _)| (*p, chunk_length));
                }

                if let VertexNode::Inner(SliceableNode::Arith { arith }) = v.node_mut() {
                    arith.poly_degree = Some(chunk_length as usize);
                }
            }
        }

        pub fn n_ops(&self) -> usize {
            use template::VertexNode::*;
            use SliceableNode::*;
            self.g
                .vertices()
                .map(|vid| match self.g.vertex(vid).node() {
                    Inner(Arith { arith }) => arith.n_arith_ops(),
                    Inner(RotateIdx(..)) => 0,
                    _ => 1,
                })
                .sum()
        }

        pub fn n_arith_ops(&self) -> usize {
            use template::VertexNode::*;
            use SliceableNode::*;
            self.g
                .vertices()
                .map(|vid| match self.g.vertex(vid).node() {
                    Inner(Arith { arith }) => arith.n_arith_ops(),
                    _ => 0,
                })
                .sum()
        }

        pub fn io_size(&self) -> u64 {
            self.inputs
                .iter()
                .map(|i| self.g.vertex(*i).typ().outer().size())
                .sum::<u64>()
                + self.g.vertex(self.output).typ().outer().size()
        }
    }
}

pub type VertexNode = alt_label::VertexNode<VertexId, super::VertexId>;
pub type Vertex<'s, Rt> = alt_label::Vertex<'s, VertexId, super::VertexId, Typ<Rt>>;
pub type Cg<'s, Rt> = alt_label::Cg<'s, VertexId, super::VertexId, Rt>;
pub type CgSubgraph<'g, 's, Rt> = SubDigraph<'g, VertexId, Vertex<'s, Rt>>;

pub type CgPartialTyped<'s, T> =
    template::Cg<VertexId, alt_label::Vertex<'s, VertexId, super::VertexId, T>>;

impl<I, Ie, T, S> digraph::internal::Predecessors<I>
    for transit::Vertex<template::VertexNode<I, Ie>, T, S>
where
    I: Clone,
{
    fn predecessors(&self) -> impl Iterator<Item = I> {
        self.node().uses()
    }
}
