use std::{
    fmt::{Debug, Display},
    io::Write,
};

use super::template::*;
use zkpoly_common::heap::UsizeId;
use zkpoly_common::indent_formatter::*;

impl<W, X, T> IndentDebug<W> for Seq<X, T>
where
    W: Write,
    X: Display + Debug + UsizeId,
    T: IndentDebug<W>,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        let mut dl = f.debug_lines();

        for (x, t) in self.iter() {
            let deps = self.dependencies(x).collect::<Vec<_>>();

            dl.entry_with(|f| {
                f.debug_struct()
                    .entry_with(&format!("{}", x), |f| {
                        f.debug_struct()
                            .entry("Item", t)
                            .entry_debug("Dependencies", &deps)
                            .finish()
                    })
                    .finish()
            });
        }

        dl.finish()
    }
}

impl<'s, W, I, V> IndentDebug<W> for SubgraphInstruction<'s, I, V>
where
    W: Write,
    I: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        f.debug_struct()
            .entry("Inst", &self.inst)
            .entry_debug("Begin", &self.begin)
            .finish()
    }
}

impl<'s, W, I, V> IndentDebug<W> for Lcs<'s, I, V>
where
    W: Write,
    I: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        f.debug_struct()
            .entry("Loads", &SliceOfDebug(&self.loads))
            .entry("Compute", &self.compute)
            .entry("Pre Store", &SliceOfDebug(&self.pre_store))
            .entry("Store", &SliceOfDebug(&self.store))
            .finish()
    }
}

impl<'s, W, I, V> IndentDebug<W> for Subgraph<'s, I, V>
where
    W: Write,
    I: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        let mut ds = f.debug_struct();
        ds.entry("Prologue", &self.prologue1);

        for (i, body) in self.bodies.iter().enumerate() {
            ds.entry(&format!("Body {i}"), body);
        }

        ds.entry(
            "Intermediates Allocations",
            &SliceOfDebug(&self.itms_allocations),
        );
        ds.entry(
            "Intermediates Deallocations",
            &SliceOfDebug(&self.itms_deallocation),
        );

        ds.finish()
    }
}

impl<'s, W, I, V> IndentDebug<W> for InstructionNode<'s, I, V>
where
    W: Write,
    I: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        match &self {
            InstructionNode::Sub(subgraph) => f
                .debug_struct()
                .entry(
                    &format!(
                        "Subgraph on {:?}, buffer = {:?}, chunker = {:?}, vid = {:?}",
                        subgraph.io_devices, subgraph.buffer, subgraph.chunker, subgraph.vid
                    ),
                    subgraph,
                )
                .finish(),
            _ => f.entry_with(|f| write!(f, "{:?}", self)),
        }
    }
}

impl<'s, W, I, V> IndentDebug<W> for Instruction<'s, I, V>
where
    W: Write,
    I: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        let mut ds = f.debug_struct();

        if !self.pre.is_empty() {
            ds.entry("Pre", &SliceOfDebug(&self.pre));
        }
        ds.entry("Op", &self.op);

        if !self.post.is_empty() {
            ds.entry("Post", &SliceOfDebug(&self.post));
        }

        ds.finish()
    }
}
