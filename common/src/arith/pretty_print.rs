use std::fmt::Display;

use crate::{arith::ArithVertex, heap::UsizeId};
use cytoscape_visualizer::{
    self as vis,
    pretty::{Context, IndirectFrom, LooseEnd, Predecessor, PrettyGraph, PrettyVertex},
};

use super::{Arith, ArithGraph, Operation};

impl<I, Ii> PrettyVertex<Ii, I, ()> for ArithVertex<I, Ii>
where
    Ii: std::fmt::Display + Copy + 'static,
    I: Clone + 'static,
{
    fn pretty_vertex<'a>(
        &self,
        vid: Ii,
        ctx: Context<'a>,
        _aux: &(),
        builder: &mut cytoscape_visualizer::Builder,
    ) {
        match &self.op {
            _ => {
                builder.vertex(
                    ctx.identifier(vid),
                    vis::Vertex::new(format_node_label(&self.op)),
                );
            }
        }
    }

    fn labeled_predecessors<'a>(
        &'a self,
        _vid: Ii,
        _aux: &'a (),
    ) -> impl Iterator<Item = (Predecessor<Ii, I>, vis::EdgeStyle)> + 'a {
        let op = &self.op;
        let it: Box<dyn Iterator<Item = _>> = match &self.op {
            Operation::Input { outer_id, .. } => Box::new(std::iter::once((
                Predecessor::External(outer_id.clone()),
                vis::EdgeStyle::default(),
            ))),
            _ => Box::new(labeled_uses(op).into_iter().map(|(u, label)| {
                (
                    Predecessor::Internal(u),
                    vis::EdgeStyle::default().with_label(label),
                )
            })),
        };
        it
    }

    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        _vid: Ii,
        _context: Context<'a>,
        _aux: &(),
        _g: &G,
        _builder: &mut cytoscape_visualizer::Builder,
    ) -> Vec<LooseEnd<Ii>>
    where
        G: PrettyGraph<Self, Ii, I>,
        Self: Sized,
    {
        // no arith vertex has internal edges
        Vec::new()
    }
}

pub struct InputEdgeStyle<F>(pub F);

impl<I, Ii, F> PrettyVertex<Ii, I, InputEdgeStyle<F>> for ArithVertex<I, Ii>
where
    Ii: std::fmt::Display + Copy + 'static,
    I: Clone + 'static,
    F: Fn(I) -> vis::EdgeStyle,
{
    fn pretty_vertex<'a>(
        &self,
        vid: Ii,
        ctx: Context<'a>,
        _aux: &InputEdgeStyle<F>,
        builder: &mut cytoscape_visualizer::Builder,
    ) {
        match &self.op {
            _ => {
                builder.vertex(
                    ctx.identifier(vid),
                    vis::Vertex::new(format_node_label(&self.op)),
                );
            }
        }
    }

    fn labeled_predecessors<'a>(
        &'a self,
        _vid: Ii,
        aux: &'a InputEdgeStyle<F>,
    ) -> impl Iterator<Item = (Predecessor<Ii, I>, vis::EdgeStyle)> + 'a {
        let op = &self.op;
        let it: Box<dyn Iterator<Item = _>> = match &self.op {
            Operation::Input { outer_id, .. } => Box::new(std::iter::once((
                Predecessor::External(outer_id.clone()),
                (aux.0)(outer_id.clone()),
            ))),
            _ => Box::new(labeled_uses(op).into_iter().map(|(u, label)| {
                (
                    Predecessor::Internal(u),
                    vis::EdgeStyle::default().with_label(label),
                )
            })),
        };
        it
    }

    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        _vid: Ii,
        _context: Context<'a>,
        _aux: &InputEdgeStyle<F>,
        _g: &G,
        _builder: &mut cytoscape_visualizer::Builder,
    ) -> Vec<LooseEnd<Ii>>
    where
        G: PrettyGraph<Self, Ii, I>,
        Self: Sized,
    {
        // no arith vertex has internal edges
        Vec::new()
    }
}

impl<I, Ii> PrettyGraph<ArithVertex<I, Ii>, Ii, I> for ArithGraph<I, Ii>
where
    Ii: UsizeId + std::fmt::Display + 'static,
    I: Clone + Display,
{
    fn vertices(&self) -> impl Iterator<Item = Ii> {
        self.g.vertices()
    }

    fn vertex(&self, vid: Ii) -> &ArithVertex<I, Ii> {
        self.g.vertex(vid)
    }

    fn indirect_from<'a>(&self, vid: Ii, ctx: &Context<'a>) -> IndirectFrom<I> {
        IndirectFrom::Internal(ctx.identifier(vid))
    }
}

pub(crate) fn format_node_label<I, Ii>(op: &Operation<I, Ii>) -> String {
    match op {
        Operation::Arith(Arith::Bin(op, ..)) => {
            format!("{:?}", op)
        }
        Operation::Arith(Arith::Unr(op, ..)) => {
            format!("{:?}", op)
        }
        Operation::Input {
            typ, mutability, ..
        } => {
            format!("Input({:?}, {:?})", typ, mutability)
        }
        Operation::Output { typ, .. } => {
            format!("Output({:?})", typ)
        }
        Operation::Todo => {
            format!("Todo")
        }
    }
}

pub(crate) fn labeled_uses<I, Ii: Copy>(op: &Operation<I, Ii>) -> Vec<(Ii, &'static str)> {
    match op {
        Operation::Arith(Arith::Bin(_, lhs, rhs)) => vec![(*lhs, "lhs"), (*rhs, "rhs")],
        Operation::Arith(Arith::Unr(_, expr)) => vec![(*expr, "")],
        Operation::Input { .. } => vec![],
        Operation::Output {
            store_node,
            in_node,
            ..
        } => in_node
            .iter()
            .map(|e| (*e, "in"))
            .chain(std::iter::once((*store_node, "store")))
            .collect(),
        Operation::Todo => vec![],
    }
}
