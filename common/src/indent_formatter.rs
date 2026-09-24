use std::fmt::Debug;
use std::io::Write;

pub trait IndentDebug<W> {
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()>;
}

pub struct Formatter<W> {
    writer: W,
    indent: usize,
}

impl<W> Formatter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer, indent: 0 }
    }
}

impl<W> Write for Formatter<W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl<W> Formatter<W> {
    pub fn debug_struct<'f>(&'f mut self) -> DebugStruct<'f, W> {
        DebugStruct {
            inner: Inner { f: self, r: Ok(()) },
        }
    }

    pub fn debug_lines<'f>(&'f mut self) -> DebugLines<'f, W> {
        DebugLines {
            inner: Inner { f: self, r: Ok(()) },
        }
    }

    fn write_indents(&mut self) -> std::io::Result<()>
    where
        W: Write,
    {
        write!(&mut self.writer, "{}", " ".repeat(self.indent))
    }

    pub fn entry_with(
        &mut self,
        f: impl FnOnce(&mut Formatter<W>) -> std::io::Result<()>,
    ) -> std::io::Result<()>
    where
        W: Write,
    {
        self.write_indents()?;
        f(self)?;
        write!(self, "\n")
    }
}

struct Inner<'f, W> {
    f: &'f mut Formatter<W>,
    r: std::io::Result<()>,
}

impl<'f, W> Inner<'f, W> {
    fn write_with(&mut self, f: impl FnOnce(&mut W) -> std::io::Result<()>) {
        match &mut self.r {
            Ok(_) => self.r = f(&mut self.f.writer),
            Err(..) => {}
        }
    }

    fn write_with_f(&mut self, f: impl FnOnce(&mut Formatter<W>) -> std::io::Result<()>) {
        match &mut self.r {
            Ok(_) => self.r = f(self.f),
            Err(..) => {}
        }
    }
}

pub struct DebugStruct<'f, W> {
    inner: Inner<'f, W>,
}

impl<'f, W> DebugStruct<'f, W>
where
    W: Write,
{
    pub fn entry_with(
        &mut self,
        field_name: &str,
        field: impl FnOnce(&mut Formatter<W>) -> std::io::Result<()>,
    ) -> &mut Self {
        let prefix = " ".repeat(self.inner.f.indent);
        self.inner
            .write_with(|f| writeln!(f, "{}{}:", &prefix, field_name));

        self.inner.f.indent += 2;
        self.inner.write_with_f(|f| field(f));
        self.inner.f.indent -= 2;
        self
    }

    pub fn entry(&mut self, field_name: &str, field: &impl IndentDebug<W>) -> &mut Self {
        self.entry_with(field_name, |f| field.fmt(f))
    }

    pub fn entry_debug(&mut self, field_name: &str, field: &impl Debug) -> &mut Self {
        self.entry_with(field_name, |f| {
            f.write_indents()?;
            write!(f, "{:?}\n", field)
        })
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        let mut r = Ok(());
        std::mem::swap(&mut r, &mut self.inner.r);
        r
    }
}

pub struct DebugLines<'f, W> {
    inner: Inner<'f, W>,
}

impl<'f, W> DebugLines<'f, W>
where
    W: Write,
{
    pub fn entry_with(
        &mut self,
        field: impl FnOnce(&mut Formatter<W>) -> std::io::Result<()>,
    ) -> &mut Self {
        self.inner.write_with_f(|f| field(f));
        self
    }

    pub fn entry(&mut self, field: &impl IndentDebug<W>) -> &mut Self {
        self.entry_with(|f| field.fmt(f))
    }

    pub fn entry_debug(&mut self, field: &impl Debug) -> &mut Self {
        self.entry_with(|f| {
            f.write_indents()?;
            write!(f, "{:?}\n", field)
        })
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        let mut r = Ok(());
        std::mem::swap(&mut r, &mut self.inner.r);
        r
    }
}

impl<W, T> IndentDebug<W> for &[T]
where
    W: Write,
    T: IndentDebug<W>,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        self.iter()
            .fold(f.debug_lines(), |mut dl, itm| {
                dl.entry(itm);
                dl
            })
            .finish()
    }
}

pub struct SliceOfDebug<'a, T>(pub &'a [T]);

impl<'a, W, T> IndentDebug<W> for SliceOfDebug<'a, T>
where
    W: Write,
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<W>) -> std::io::Result<()> {
        self.0
            .iter()
            .fold(f.debug_lines(), |mut dl, itm| {
                dl.entry_debug(itm);
                dl
            })
            .finish()
    }
}
