use std::fmt::Display;

use crate as vis;

pub use vis::Id;

#[derive(Clone)]
pub struct Context<'a> {
    prefixes: Vec<&'a str>,
}

impl<'a> Context<'a> {
    pub fn pop(&self) -> Self {
        let mut r = self.clone();
        r.prefixes.pop().expect("empty context");
        r
    }

    pub fn push(&self, s: &'a str) -> Self {
        let mut r = self.clone();
        r.prefixes.push(s);
        r
    }

    pub fn new() -> Self {
        Self {
            prefixes: Vec::new(),
        }
    }

    pub fn identifier<I>(&self, i: I) -> vis::Id
    where
        I: Display,
    {
        format!("{}_{}", self, &i)
    }
}

impl<'a> std::fmt::Display for Context<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.prefixes.last().unwrap_or(&""))
    }
}

pub trait PrettyGraph<V, I, Ie> {
    fn vertices(&self) -> impl Iterator<Item = I>;
    fn vertex(&self, vid: I) -> &V;
    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> IndirectFrom<Ie>;
}

impl<I, Ie, V> PrettyGraph<V, I, Ie> for () {
    fn vertices(&self) -> impl Iterator<Item = I> {
        std::iter::empty()
    }
    fn vertex(&self, _vid: I) -> &V {
        panic!("this method is not expected to be called because unit PrettyGraph is used in case when there is no outside graph of a graph")
    }
    fn indirect_from<'a>(&self, _vid: I, _ctx: &Context<'a>) -> IndirectFrom<Ie> {
        panic!("this method is not expected to be called because unit PrettyGraph is used in case when there is no outside graph of a graph")
    }
}

#[derive(Debug, Clone)]
pub enum IndirectFrom<Ie> {
    External(Ie),
    Internal(Id),
}

impl<Ie> IndirectFrom<Ie> {
    pub fn bind<I>(self, f: impl FnOnce(Ie) -> IndirectFrom<I>) -> IndirectFrom<I> {
        use IndirectFrom::*;
        match self {
            External(x) => f(x),
            Internal(x) => Internal(x),
        }
    }
}

pub enum Predecessor<I, Ie> {
    External(Ie),
    Internal(I),
}

pub struct LooseEnd<I> {
    pub from: I,
    pub to: Id,
    pub edge_style: super::EdgeStyle,
}

pub trait PrettyVertex<I, Ie, A> {
    fn pretty_vertex<'a>(&self, vid: I, ctx: Context<'a>, aux: &A, builder: &mut vis::Builder);
    fn labeled_predecessors<'a>(
        &'a self,
        vid: I,
        aux: &'a A,
    ) -> impl Iterator<Item = (Predecessor<I, Ie>, vis::EdgeStyle)> + 'a;
    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        vid: I,
        ctx: Context<'a>,
        aux: &A,
        g: &G,
        builder: &mut vis::Builder,
    ) -> Vec<LooseEnd<I>>
    where
        G: PrettyGraph<Self, I, Ie>,
        Self: Sized;
}

pub fn pretty_vertices<'a, G, I, Ie, V, A>(
    g: &G,
    aux: &A,
    ctx: Context<'a>,
    builder: &mut vis::Builder,
) where
    V: PrettyVertex<I, Ie, A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I, Ie>,
{
    for vid in g.vertices() {
        let vertex = g.vertex(vid);
        vertex.pretty_vertex(vid, ctx.clone(), aux, builder);
    }
}

pub fn pretty_edges<'a, G, I, Ie, V, A>(
    g: &G,
    aux: &A,
    ctx: Context<'a>,
    builder: &mut vis::Builder,
) -> Vec<LooseEnd<Ie>>
where
    V: PrettyVertex<I, Ie, A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I, Ie>,
{
    let inner_loose_ends = g
        .vertices()
        .map(|vid| {
            let vertex = g.vertex(vid);
            vertex
                .pretty_vertex_internal_edges(vid, ctx.clone(), aux, g, builder)
                .into_iter()
        })
        .flatten()
        .collect::<Vec<_>>();

    let mut loose_ends = Vec::new();

    for LooseEnd {
        from,
        to,
        edge_style,
    } in inner_loose_ends.into_iter()
    {
        match g.indirect_from(from, &ctx) {
            IndirectFrom::Internal(id) => {
                builder.edge(vis::Edge::new(id, to).with_style(edge_style));
            }
            IndirectFrom::External(vid) => loose_ends.push(LooseEnd {
                from: vid,
                to,
                edge_style,
            }),
        }
    }

    for vid in g.vertices() {
        let vertex = g.vertex(vid);
        for (pred, es) in vertex.labeled_predecessors(vid, aux) {
            match pred {
                Predecessor::Internal(pred) => {
                    let ifrom = g.indirect_from(pred, &ctx);
                    match ifrom {
                        IndirectFrom::Internal(id) => {
                            builder.edge(vis::Edge::new(id, ctx.identifier(vid)).with_style(es));
                        }
                        IndirectFrom::External(evid) => loose_ends.push(LooseEnd {
                            from: evid,
                            to: ctx.identifier(vid),
                            edge_style: es,
                        }),
                    }
                }
                Predecessor::External(evid) => loose_ends.push(LooseEnd {
                    from: evid,
                    to: ctx.identifier(vid),
                    edge_style: es,
                }),
            }
        }
    }

    loose_ends
}

pub fn pretty<'a, G, I, V, A>(g: &G, aux: &A, builder: &mut vis::Builder)
where
    V: PrettyVertex<I, (), A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I, ()>,
{
    let ctx = Context::new();
    pretty_vertices(g, aux, ctx.clone(), builder);
    pretty_edges::<G, I, (), V, A>(g, aux, ctx, builder);
}

pub fn pretty_with_first_shown<'a, G, I, V, A>(
    g: &G,
    aux: &A,
    builder: &mut vis::DigraphBuilder,
    first_shown: impl Iterator<Item = I>,
) where
    V: PrettyVertex<I, (), A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I, ()>,
{
    let ctx = Context::new();
    pretty_vertices(g, aux, ctx.clone(), builder);
    pretty_edges::<G, I, (), V, A>(g, aux, ctx.clone(), builder);
    builder.first_show(first_shown.map(|v| ctx.identifier(v)));
}
