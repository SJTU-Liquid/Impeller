use super::prelude::*;
use std::hash::Hash;
use type3::Device;

pub type Atom<O> = value::Atom<O, value::ValueNode>;
pub type Value<O> = value::Tree<Atom<O>>;
pub type OutputAtom<O> = value::OutputT<Atom<O>, Atom<O>>;
pub type InputAtom<O> = value::InputT<Atom<O>>;

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResidentalT<V, P>(V, P);

impl<V, P> ResidentalT<V, P> {
    pub fn new(value: V, pointer: P) -> Self {
        ResidentalT(value, pointer)
    }

    pub fn pointer(&self) -> &P {
        &self.1
    }

    pub fn pointer_mut(&mut self) -> &mut P {
        &mut self.1
    }

    pub fn split_mut(&mut self) -> (&mut V, &mut P) {
        (&mut self.0, &mut self.1)
    }

    pub fn with_no_pointer(value: V) -> ResidentalT<V, Option<P>> {
        ResidentalT(value, None)
    }

    pub fn with_pointer(v: V, pointer: P) -> ResidentalT<V, Option<P>> {
        ResidentalT(v, Some(pointer))
    }

    pub fn set_pointer<P1>(self, pointer: P1) -> ResidentalT<V, P1> {
        ResidentalT(self.0, pointer)
    }

    pub fn value(&self) -> &V {
        &self.0
    }

    pub fn map_v<V1>(self, f: impl FnOnce(V) -> V1) -> ResidentalT<V1, P> {
        ResidentalT(f(self.0), self.1)
    }
}

impl<V, P> std::ops::Deref for ResidentalT<V, P> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V, P> From<V> for ResidentalT<V, P>
where
    P: Default,
{
    fn from(value: V) -> Self {
        ResidentalT(value, P::default())
    }
}

impl<V, P> ResidentalT<V, Option<P>> {
    pub fn assume_pointed(self) -> ResidentalT<V, P> {
        ResidentalT(self.0, self.1.unwrap())
    }
}

pub type ResidentalAtom<O, P> = ResidentalT<Atom<O>, P>;
pub type ResidentalValue<O, P> = value::Tree<ResidentalAtom<O, P>>;

pub type InputRv<O, P> = value::Tree<value::InputT<ResidentalAtom<O, P>>>;
pub type OutputRv<O, P> = value::Tree<value::OutputT<ResidentalAtom<O, P>, ResidentalAtom<O, P>>>;

pub type VertexNode<O, P> = type2::no_subgraph::alt_label::VertexNode<InputRv<O, P>>;
pub type ChunkedNode<O, P> =
    type2::sliceable_subgraph::alt_label::VertexNode<InputRv<O, P>, InputRv<O, P>>;

#[derive(Debug, Hash, Clone)]
pub struct Temporaries<O, P> {
    pub(crate) buf: ResidentalAtom<O, P>,
    pub(crate) slices: Vec<Slice>,
}

#[derive(Debug, Hash, Clone)]
pub struct Type2Op<'s, O, P> {
    pub(crate) vid: type2::VertexId,
    pub(crate) output: OutputRv<O, P>,
    pub(crate) node: VertexNode<O, P>,
    pub(crate) temporaries: Option<Temporaries<O, P>>,
    pub(crate) src: SourceInfo<'s>,
}

impl<'s, O, P> Type2Op<'s, O, P> {
    pub fn new(
        vid: type2::VertexId,
        output: OutputRv<O, P>,
        temporaries: Option<Temporaries<O, P>>,
        node: VertexNode<O, P>,
        src: SourceInfo<'s>,
    ) -> Self {
        Self {
            vid,
            output,
            temporaries,
            node,
            src,
        }
    }
}

#[derive(Debug, Clone, Hash)]
pub struct ChunkedOp<'s, O, P> {
    pub(crate) vid: type2::sliceable_subgraph::VertexId,
    pub(crate) output: OutputRv<O, P>,
    pub(crate) temporaries: Option<Temporaries<O, P>>,
    pub(crate) node: ChunkedNode<O, P>,
    pub(crate) src: SourceInfo<'s>,
}

impl<'s, O, P> ChunkedOp<'s, O, P> {
    pub fn new(
        vid: type2::sliceable_subgraph::VertexId,
        output: OutputRv<O, P>,
        temporaries: Option<Temporaries<O, P>>,
        node: ChunkedNode<O, P>,
        src: SourceInfo<'s>,
    ) -> Self {
        Self {
            vid,
            output,
            temporaries,
            node,
            src,
        }
    }
}

#[derive(Debug, Hash, Clone)]
pub struct Subgraph<O, P, B, I> {
    pub(crate) chunker: super::subgraph_elaboration::Chunker,
    /// All inputs to the subgraph
    pub(crate) uses: Vec<InputRv<O, P>>,
    /// As-is inputs to the subgraph.
    /// That is, they are not sliced and must be on device before the subgraph is executed.
    pub(crate) on_device_uses: Vec<InputRv<O, P>>,
    pub(crate) device: Device,
    pub(crate) prologue: B,
    pub(crate) body: B,
    pub(crate) output: OutputRv<O, P>,
    pub(crate) itms: I,
    pub(crate) vid: type2::VertexId,
}

impl<O, P, B> Subgraph<O, P, B, ()> {
    pub fn new(
        uses: Vec<InputRv<O, P>>,
        on_device_uses: Vec<InputRv<O, P>>,
        device: Device,
        prologue: B,
        body: B,
        output: OutputRv<O, P>,
        vid: type2::VertexId,
        chunker: super::subgraph_elaboration::Chunker,
    ) -> Self {
        Self {
            uses,
            on_device_uses,
            device,
            prologue,
            body,
            output,
            itms: (),
            vid,
            chunker,
        }
    }
}

#[derive(Debug, Hash, Clone)]
pub enum Ad<O, P> {
    /// Allocate some object on device
    Allocate(O, Device, P),

    /// Deallocate some object on device
    Deallocate(O, Device, P),
}

/// A memory relevant operation involved in executing a Type2 computation graph.
///
/// Type parameter [`T`] is the smallest unit of memory transferring, Token.
/// [`P`] means Pointer, by which the object is accessed from memory.
///
/// Only operations with a specific pointer for each operand can be lowered to Type3.
#[derive(Debug, Hash, Clone)]
pub enum Operation<'s, O, P, S> {
    /// A Type2 vertex translated to value inputs/outputs.
    ///
    /// Depending on current device of memory planning, some of the values' pointers
    /// may be unknown.
    ///
    /// All temporary spaces belong to the same object.
    Type2(Type2Op<'s, O, Option<P>>),

    /// A subgraph of Type2 CG that is executed for each chunk of output polynomials.
    /// Currently all computation in this subgraph must happen on GPU and no transfer between CPU and GPU
    /// can occur during its execution.
    Sub(S),

    /// `(dst, d, src)` Transfer (potential sliced) object `src` to `dst` on `d`.
    /// The source device specified in `src` can be arbitary when the pointer is not yet determined.
    ///
    /// Currently Disk to Disk transfer is not supported.
    Transfer(
        ResidentalT<O, Option<P>>,
        Device,
        ResidentalAtom<O, Option<P>>,
    ),

    /// `(dst, src)` Clone (potential sliced) object `src` to `dst` object.
    /// The source and destination device is not guaranteed.
    ///
    /// During planning, the source and destination device are guaranteed to only
    /// change to planning or unplanned device.
    Clone(ResidentalT<O, Option<P>>, ResidentalAtom<O, Option<P>>),

    /// Allocation and deallocation
    Ad(Ad<O, P>),
}

impl<'s, O, P, S> From<Type2Op<'s, O, Option<P>>> for Operation<'s, O, P, S> {
    fn from(value: Type2Op<'s, O, Option<P>>) -> Self {
        Self::Type2(value)
    }
}

#[derive(Debug, Hash, Clone)]
pub enum SubgraphOperation<'s, O, P> {
    /// A Type2 vertex translated to value inputs/outputs.
    ///
    /// Depending on current device of memory planning, some of the values' pointers
    /// may be unknown.
    ///
    /// All temporary spaces belong to the same object.
    Type2(ChunkedOp<'s, O, Option<P>>),

    /// [`Clone2(o1, {s_i})`] means o1 <- Clone {o2_i}, s_i,
    /// i.e., Clone slices `s_i` and concatenate them to obtain `o1`
    Concat(ResidentalT<O, P>, [ResidentalT<O, P>; 2]),

    /// Clone src to dst, with possible slice
    Clone(
        /* dst   */ ResidentalT<O, Option<P>>,
        /* src   */ ResidentalT<O, Option<P>>,
        /* slice */ Option<Slice>,
    ),

    /// Allocation and deallocation
    Ad(Ad<O, P>),
}

impl<'s, O, P> SubgraphOperation<'s, O, P> {
    pub fn maybe_unfulfilled_pointer_mut<'a>(
        &'a mut self,
    ) -> Box<dyn Iterator<Item = (&'a O, &'a mut Option<P>)> + 'a> {
        use SubgraphOperation::*;

        match self {
            Type2(t2op) => Box::new(
                t2op.node
                    .uses_mut()
                    .map(|vi| {
                        vi.iter_mut().map(|v| {
                            let (o, p) = v.split_mut();
                            (o.object(), p)
                        })
                    })
                    .flatten(),
            ),
            Clone(_, src, _) => Box::new(std::iter::once({
                let (o, p) = src.split_mut();
                (&*o, p)
            })),
            Concat(..) | Ad(..) => Box::new(std::iter::empty()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lcs<'s, O, P, A> {
    /// The loads to make by cloning a part of src object to make the dst object.
    ///
    /// Src device here default to the device running this LCS,
    /// but remember to determine actualy one.
    pub load: Vec<(
        /* dst */ ResidentalT<O, P>,
        /* src */ ResidentalAtom<O, Option<P>>,
    )>,

    /// The Type2 operations and clone/slicing/concat to carry out.
    pub compute: Seq<SubgraphOperation<'s, O, P>>,

    /// The stores to make by cloing the src object to a part of the dst object.
    ///
    /// All src objects must be those produced during running operations in `compute`,
    /// so that stores only depends on computes, and double buffer would be sufficient.
    ///
    /// Dst device here default to the device running this LCS,
    /// but remember to determine actualy one.
    pub store: Vec<(
        /* dst */ ResidentalAtom<O, Option<P>>,
        /* src */ ResidentalT<O, P>,
    )>,

    /// Write out-intermediates.
    ///
    /// We list them individually here instead of representing them using `Clone`
    /// so as not to break the SSA property of operations.
    pub save_itms: Vec<(
        /* dst */ ResidentalT<O, P>,
        /* src */ ResidentalT<O, P>,
    )>,

    /// Post store.
    ///
    /// Currently used for freeing store sources.
    pub post_store: Vec<ResidentalT<O, P>>,

    /// The storage device storing all this LCS.
    pub device: Device,

    /// The allocations for computation
    pub allocator: A,
}

impl<'s, O, P, A> Hash for Lcs<'s, O, P, A>
where
    O: Hash,
    P: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.load.hash(state);
        self.compute.hash(state);
        self.store.hash(state);
        self.save_itms.hash(state);
        self.post_store.hash(state);
        self.device.hash(state);
    }
}

impl<'s, O, P> Lcs<'s, O, P, ()> {
    pub fn empty(device: Device) -> Self {
        Self {
            load: Vec::new(),
            compute: Seq::empty(),
            save_itms: Vec::new(),
            store: Vec::new(),
            post_store: Vec::new(),
            device,
            allocator: (),
        }
    }
}

impl<'s, O, P, A> Lcs<'s, O, P, A> {
    pub fn map_a<A1>(self, f: impl FnOnce(A) -> A1) -> Lcs<'s, O, P, A1> {
        Lcs {
            load: self.load,
            compute: self.compute,
            store: self.store,
            save_itms: self.save_itms,
            device: self.device,
            post_store: self.post_store,
            allocator: f(self.allocator),
        }
    }
}

impl<'s, O, P> From<ChunkedOp<'s, O, Option<P>>> for SubgraphOperation<'s, O, P> {
    fn from(value: ChunkedOp<'s, O, Option<P>>) -> Self {
        Self::Type2(value)
    }
}

impl<'s, O, P> From<Ad<O, P>> for SubgraphOperation<'s, O, P> {
    fn from(value: Ad<O, P>) -> Self {
        SubgraphOperation::Ad(value)
    }
}

pub trait Constructor<O, P> {
    fn ad(ad: Ad<O, P>) -> Self;
    fn clone(dst: ResidentalT<O, Option<P>>, src: ResidentalAtom<O, Option<P>>) -> Self;
    fn transfer(
        dst: ResidentalT<O, Option<P>>,
        dst_device: Device,
        src: ResidentalAtom<O, Option<P>>,
    ) -> Self;
}

impl<'s, O, P, S> Constructor<O, P> for Operation<'s, O, P, S> {
    fn ad(ad: Ad<O, P>) -> Self {
        Operation::Ad(ad)
    }

    fn clone(dst: ResidentalT<O, Option<P>>, src: ResidentalAtom<O, Option<P>>) -> Self {
        Operation::Clone(dst, src)
    }

    fn transfer(
        dst: ResidentalT<O, Option<P>>,
        dst_device: Device,
        src: ResidentalAtom<O, Option<P>>,
    ) -> Self {
        Operation::Transfer(dst, dst_device, src)
    }
}

impl<'s, O, P> Constructor<O, P> for SubgraphOperation<'s, O, P> {
    fn ad(ad: Ad<O, P>) -> Self {
        ad.into()
    }

    fn clone(_dst: ResidentalT<O, Option<P>>, _src: ResidentalAtom<O, Option<P>>) -> Self {
        panic!("not expected to use this contructor")
    }

    fn transfer(
        _dst: ResidentalT<O, Option<P>>,
        _dst_device: Device,
        _src: ResidentalAtom<O, Option<P>>,
    ) -> Self {
        panic!("not expected to transfer across devices in subgraph");
    }
}

pub trait ObjectAtRw<O, P> {
    fn object_at_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, P, Device)> + 'a>
    where
        O: 'a;

    fn object_at_writes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (value::OutputT<&'a O, &'a O>, P, Device)> + 'a>
    where
        O: 'a;
    fn object_at_allocates<'a>(&'a self) -> Option<(&'a O, P, Device)>;
    fn object_at_deallocates<'a>(&'a self) -> Option<(&'a O, P, Device)>;
}

pub trait ObjectRw<O> {
    fn object_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a;

    fn object_writes<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a;
    fn object_allocates<'a>(&'a self) -> Option<(&'a O, Device)>;
    fn object_deallocates<'a>(&'a self) -> Option<(&'a O, Device)>;
}

impl<O, P> ObjectAtRw<O, P> for Ad<O, P>
where
    P: Clone,
{
    fn object_at_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, P, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(std::iter::empty())
    }

    fn object_at_writes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (value::OutputT<&'a O, &'a O>, P, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(std::iter::empty())
    }

    fn object_at_allocates<'a>(&'a self) -> Option<(&'a O, P, Device)> {
        use Ad::*;
        match self {
            Allocate(o, d, p) => Some((o, p.clone(), *d)),
            Deallocate(..) => None,
        }
    }

    fn object_at_deallocates<'a>(&'a self) -> Option<(&'a O, P, Device)> {
        use Ad::*;
        match self {
            Deallocate(o, d, p) => Some((o, p.clone(), *d)),
            Allocate(..) => None,
        }
    }
}

impl<O, P, B, I> ObjectAtRw<O, P> for Subgraph<O, P, B, I>
where
    P: Clone,
{
    fn object_at_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, P, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(
            self.uses
                .iter()
                .flat_map(|rvs| {
                    rvs.iter()
                        .map(|rv| (rv.object(), rv.pointer().clone(), rv.device().clone()))
                })
                .chain(self.on_device_uses.iter().flat_map(|rvs| {
                    rvs.iter()
                        .map(|rv| (rv.object(), rv.pointer().clone(), rv.device().clone()))
                })),
        )
    }

    fn object_at_writes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (value::OutputT<&'a O, &'a O>, P, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.output.iter().map(|v| {
            (
                v.map_both_ref(|ra| ra.object()),
                v.pointer().clone(),
                v.device(),
            )
        }))
    }

    fn object_at_allocates<'a>(&'a self) -> Option<(&'a O, P, Device)> {
        None
    }

    fn object_at_deallocates<'a>(&'a self) -> Option<(&'a O, P, Device)> {
        None
    }
}

impl<'s, O, P, S> ObjectAtRw<O, Option<P>> for Operation<'s, O, P, S>
where
    S: ObjectAtRw<O, Option<P>>,
    P: Clone,
{
    fn object_at_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Option<P>, Device)> + 'a>
    where
        O: 'a,
    {
        use Operation::*;
        match self {
            Type2(t2op) => Box::new(
                t2op.node
                    .uses_ref()
                    .map(|vi| {
                        vi.iter()
                            .map(|v| (v.object(), v.pointer().clone(), v.device()))
                    })
                    .flatten()
                    .chain(
                        t2op.temporaries
                            .as_ref()
                            .map(|t| {
                                let v = &t.buf;
                                (v.object(), v.pointer().clone(), v.device())
                            })
                            .into_iter(),
                    ),
            ),
            Sub(sg) => sg.object_at_reads(),
            Clone(_, rv) | Transfer(_, _, rv) => {
                Box::new([(rv.object(), rv.pointer().clone(), rv.device())].into_iter())
            }
            Ad(..) => Box::new(std::iter::empty()),
        }
    }

    fn object_at_writes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (value::OutputT<&'a O, &'a O>, Option<P>, Device)> + 'a>
    where
        O: 'a,
    {
        use Operation::*;
        match self {
            Type2(t2op) => Box::new(t2op.output.iter().map(|v| {
                (
                    v.map_both_ref(|ra| ra.object()),
                    v.pointer().clone(),
                    v.device(),
                )
            })),
            Sub(sg) => sg.object_at_writes(),
            Clone(ro, src) => Box::new(
                [(
                    value::OutputT::non_inplace(ro.value()),
                    ro.pointer().clone(),
                    src.device(),
                )]
                .into_iter(),
            ),
            Transfer(ro, d, _) => Box::new(
                [(
                    value::OutputT::non_inplace(ro.value()),
                    ro.pointer().clone(),
                    *d,
                )]
                .into_iter(),
            ),
            Ad(..) => Box::new(std::iter::empty()),
        }
    }

    fn object_at_allocates<'a>(&'a self) -> Option<(&'a O, Option<P>, Device)> {
        if let Operation::Ad(ad) = self {
            ad.object_at_allocates().map(|(o, p, d)| (o, Some(p), d))
        } else {
            None
        }
    }

    fn object_at_deallocates<'a>(&'a self) -> Option<(&'a O, Option<P>, Device)> {
        if let Operation::Ad(ad) = self {
            ad.object_at_deallocates().map(|(o, p, d)| (o, Some(p), d))
        } else {
            None
        }
    }
}

impl<O, P> ObjectRw<O> for Ad<O, P>
where
    P: Clone,
{
    fn object_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_reads().map(|(o, _, d)| (o, d)))
    }

    fn object_writes<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_writes().map(|(o, _, d)| (*o, d)))
    }

    fn object_allocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_allocates().map(|(o, _, d)| (o, d))
    }

    fn object_deallocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_deallocates().map(|(o, _, d)| (o, d))
    }
}

impl<O, P, B, I> ObjectRw<O> for Subgraph<O, P, B, I>
where
    P: Clone,
{
    fn object_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_reads().map(|(o, _, d)| (o, d)))
    }

    fn object_writes<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_writes().map(|(o, _, d)| (*o, d)))
    }

    fn object_allocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_allocates().map(|(o, _, d)| (o, d))
    }

    fn object_deallocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_deallocates().map(|(o, _, d)| (o, d))
    }
}

impl<'s, O, P, S> ObjectRw<O> for Operation<'s, O, P, S>
where
    P: Clone,
    S: ObjectAtRw<O, Option<P>>,
{
    fn object_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_reads().map(|(o, _, d)| (o, d)))
    }

    fn object_writes<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_writes().map(|(o, _, d)| (*o, d)))
    }

    fn object_allocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_allocates().map(|(o, _, d)| (o, d))
    }

    fn object_deallocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_deallocates().map(|(o, _, d)| (o, d))
    }
}

#[derive(Clone, Debug)]
pub struct SubgraphOperationOn<'b, 's, O, P>(pub &'b SubgraphOperation<'s, O, P>, pub Device);

impl<'b, 's, O, P> ObjectAtRw<O, Option<P>> for SubgraphOperationOn<'b, 's, O, P>
where
    P: Clone,
{
    fn object_at_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Option<P>, Device)> + 'a>
    where
        O: 'a,
    {
        use SubgraphOperation::*;

        match &self.0 {
            Type2(t2op) => Box::new(
                t2op.node
                    .uses_ref()
                    .map(|vi| {
                        vi.iter()
                            .map(|v| (v.object(), v.pointer().clone(), v.device()))
                    })
                    .flatten()
                    .chain(
                        t2op.temporaries
                            .as_ref()
                            .map(|t| {
                                let v = &t.buf;
                                (v.object(), v.pointer().clone(), v.device())
                            })
                            .into_iter(),
                    ),
            ),
            Concat(_, [src1, src2]) => Box::new(
                [
                    (src1.deref(), Some(src1.pointer().clone()), self.1),
                    (src2.deref(), Some(src2.pointer().clone()), self.1),
                ]
                .into_iter(),
            ),
            Clone(_, src, _) => Box::new(std::iter::once((
                src.deref(),
                src.pointer().clone(),
                self.1,
            ))),
            Ad(..) => Box::new(std::iter::empty()),
        }
    }

    fn object_at_writes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (value::OutputT<&'a O, &'a O>, Option<P>, Device)> + 'a>
    where
        O: 'a,
    {
        use SubgraphOperation::*;

        match &self.0 {
            Type2(t2op) => Box::new(t2op.output.iter().map(|v| {
                (
                    v.map_both_ref(|ra| ra.object()),
                    v.pointer().clone(),
                    v.device(),
                )
            })),
            Concat(dst, ..) => Box::new(std::iter::once((
                value::OutputT::non_inplace(dst.deref()),
                Some(dst.pointer().clone()),
                self.1,
            ))),
            Clone(dst, ..) => Box::new(std::iter::once((
                value::OutputT::non_inplace(dst.deref()),
                dst.pointer().clone(),
                self.1,
            ))),

            Ad(..) => Box::new(std::iter::empty()),
        }
    }

    fn object_at_allocates<'a>(&'a self) -> Option<(&'a O, Option<P>, Device)> {
        match &self.0 {
            SubgraphOperation::Ad(Ad::Allocate(o, d, p)) => Some((o, Some(p.clone()), *d)),
            _ => None,
        }
    }

    fn object_at_deallocates<'a>(&'a self) -> Option<(&'a O, Option<P>, Device)> {
        match &self.0 {
            SubgraphOperation::Ad(Ad::Deallocate(o, d, p)) => Some((o, Some(p.clone()), *d)),
            _ => None,
        }
    }
}

impl<'b, 's, O, P> ObjectRw<O> for SubgraphOperationOn<'b, 's, O, P>
where
    P: Clone,
{
    fn object_reads<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_reads().map(|(o, _, d)| (o, d)))
    }

    fn object_writes<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a O, Device)> + 'a>
    where
        O: 'a,
    {
        Box::new(self.object_at_writes().map(|(o, _, d)| (*o, d)))
    }

    fn object_allocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_allocates().map(|(o, _, d)| (o, d))
    }

    fn object_deallocates<'a>(&'a self) -> Option<(&'a O, Device)> {
        self.object_at_deallocates().map(|(o, _, d)| (o, d))
    }
}

define_usize_id!(Index);

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Index {
    pub fn inf() -> Self {
        Index::from(usize::MAX)
    }
}

/// A sequence of [`Operation`]
#[derive(Debug, Hash, Clone)]
pub struct Seq<R>(Heap<Index, R>);

impl<R> Seq<R> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn empty() -> Self {
        Seq(Heap::new())
    }

    pub fn emit(&mut self, op: impl Into<R>) {
        self.0.push(op.into());
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Index, &'a R)> + 'a {
        self.0.iter_with_id()
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Index, &'a mut R)> + 'a {
        self.0.iter_mut_with_id()
    }

    pub fn into_iter(self) -> impl Iterator<Item = (Index, R)> {
        self.0.into_iter_with_id()
    }

    pub fn from_heap(h: Heap<Index, R>) -> Self {
        Self(h)
    }

    pub fn try_map<R1, E>(self, mut f: impl FnMut(R) -> Result<R1, E>) -> Result<Seq<R1>, E> {
        Ok(Seq(self.0.try_map(&mut |_, r| f(r))?))
    }

    pub fn next_index(&self) -> Index {
        self.0.next_index()
    }
}

impl<R> std::ops::Index<Index> for Seq<R> {
    type Output = R;
    fn index(&self, index: Index) -> &Self::Output {
        &self.0[index]
    }
}

mod pretty_print;
