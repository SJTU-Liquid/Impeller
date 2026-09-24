use super::*;
use std::fmt::Debug;
use std::io::Write;
use zkpoly_common::indent_formatter::*;

impl<'s, W, O, P, S> IndentDebug<W> for Operation<'s, O, P, S>
where
    W: Write,
    O: Debug,
    P: Debug,
    S: IndentDebug<W> + Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        use Operation::*;

        match self {
            Sub(s) => f.debug_struct().entry("Sub", s).finish(),
            _ => f.entry_with(|f| write!(f, "{:?}", self)),
        }
    }
}

impl<'s, W, O, P> IndentDebug<W> for SubgraphOperation<'s, O, P>
where
    O: Debug,
    P: Debug,
    W: Write,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        f.entry_with(|f| write!(f, "{:?}", self))
    }
}

impl<W, O, P, B, I> IndentDebug<W> for Subgraph<O, P, B, I>
where
    O: Debug,
    P: Debug,
    B: IndentDebug<W>,
    I: Debug,
    W: Write,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        f.debug_struct()
            .entry_debug("chunker", &self.chunker)
            .entry_debug("uses", &self.uses)
            .entry_debug("on_device_uses", &self.on_device_uses)
            .entry_debug("device", &self.device)
            .entry_debug("output", &self.output)
            .entry_debug("itms", &self.itms)
            .entry_debug("vid", &self.vid)
            .entry("prologue", &self.prologue)
            .entry("body", &self.body)
            .finish()
    }
}

impl<'s, W, O, P, A> IndentDebug<W> for Lcs<'s, O, P, A>
where
    O: Debug,
    P: Debug,
    W: Write,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        f.debug_struct()
            .entry_debug("device", &self.device)
            .entry("load", &SliceOfDebug(&self.load))
            .entry("compute", &self.compute)
            .entry("store", &SliceOfDebug(&self.store))
            .entry_debug("save_itms", &self.save_itms)
            .entry("post_store", &SliceOfDebug(&self.post_store))
            .finish()
    }
}

impl<'s, W, R> IndentDebug<W> for Seq<R>
where
    R: IndentDebug<W>,
    W: Write,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        self.iter()
            .fold(f.debug_lines(), |mut dl, (idx, r)| {
                dl.entry_with(|f| f.debug_struct().entry(&format!("{}", idx), r).finish());
                dl
            })
            .finish()
    }
}
