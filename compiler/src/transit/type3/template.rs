use super::Device;
use std::iter::{empty, once};
use zkpoly_common::heap::Heap;
use zkpoly_common::typ::{PolySlice, PolyType, Slice};
use zkpoly_common::{define_usize_id, heap::UsizeId};
use zkpoly_runtime::instructions::{AllocMethod, AllocVariant};

use crate::{
    ast::PolyInit,
    transit::{type2, SourceInfo},
};
use type2::object_analysis::Chunker;

#[derive(Debug, Hash, Clone)]
pub struct Seq<X, T> {
    instructions: Heap<X, (T, Vec<X>)>,
}

impl<X, T> Seq<X, T> {
    pub fn new() -> Self {
        Self {
            instructions: Heap::new(),
        }
    }
}

impl<X, T> Seq<X, T>
where
    X: UsizeId,
{
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (X, &'a T)> + 'a {
        self.instructions.iter_with_id().map(|(i, (t, _))| (i, t))
    }

    pub fn dependencies<'a>(&'a self, x: X) -> impl Iterator<Item = X> + 'a {
        self.instructions[x].1.iter().cloned()
    }
}

impl<X, T> std::ops::Index<X> for Seq<X, T>
where
    X: UsizeId,
{
    type Output = T;
    fn index(&self, index: X) -> &Self::Output {
        &self.instructions[index].0
    }
}

impl<X, T> Seq<X, T>
where
    X: UsizeId,
{
    pub fn push(&mut self, t: T, dependencies: Vec<X>) -> X {
        if dependencies
            .iter()
            .any(|d| *d >= self.instructions.len().into())
        {
            panic!("dependency to not-yet emitted instruction");
        }

        self.instructions.push((t, dependencies))
    }
}

#[derive(Debug, Hash, Clone)]
pub struct Instruction<'s, I, V> {
    pub(super) pre: Vec<RegisterManipulation<I>>,
    pub(super) op: InstructionNode<'s, I, V>,
    pub(super) post: Vec<RegisterManipulation<I>>,
}

pub type Begin = Option<PolySlice>;

#[derive(Debug, Hash, Clone)]
pub struct SubgraphInstruction<'s, I, V> {
    pub inst: Instruction<'s, I, V>,
    pub begin: Begin,
}

impl<'s, I, V> Instruction<'s, I, V> {
    pub fn new(
        pre: Vec<RegisterManipulation<I>>,
        op: InstructionNode<'s, I, V>,
        post: Vec<RegisterManipulation<I>>,
    ) -> Self {
        Self { pre, op, post }
    }
}

define_usize_id!(LcsPc);

impl std::fmt::Display for LcsPc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Allocate space for the input, then load it
#[derive(Debug, Hash, Clone)]
pub struct Load<I> {
    pub id: I,
    pub offset: usize,
    pub input: SubgraphInput<I>,
}

#[derive(Debug, Hash, Clone)]
pub struct Store<I> {
    pub dst: I,
    pub src: I,
    pub begin: Option<Slice>,
}

#[derive(Debug, Hash, Clone)]
pub struct Lcs<'s, I, V> {
    pub loads: Vec<Load<I>>,
    pub compute: Seq<LcsPc, SubgraphInstruction<'s, I, V>>,
    pub pre_store: Vec<RegisterManipulation<I>>,
    pub store: Vec<Store<I>>,
}

/// An input to the subgraph
#[derive(Debug, Hash, Clone)]
pub struct SubgraphInput<I> {
    /// Where to take the input from
    pub src: I,
    /// Where the first chunk calculated in this LCS is taken,
    /// if the input is a polynomail.
    pub begin: Option<Slice>,
}

#[derive(Debug, Hash, Clone)]
pub struct OffsetedAllocation<I> {
    pub id: I,
    pub base: I,
    pub offset: usize,
}

#[derive(Debug, Hash, Clone)]
pub struct Subgraph<'s, I, V> {
    pub vid: type2::VertexId,
    pub chunker: Chunker,
    /// We need three versions of LCS for double buffering:
    ///
    ///   (Lp1 for load prologue to buffer 1, 1Cp1 for compute prologue from buffer 1 to buffer 1, etc.)
    ///  
    ///    L    C   S    |       Buffers
    ///   Lp1            |   Bp1   |
    ///   Lb2 1Cp1       |   Bp1   |  Bb2  |Bi|
    ///   Lb1 2Cb2 1Sp   |  Bb1  | |  Bb2  |Bi|
    ///       1Cb1 2Sb   |  Bb1  | |  Bb2  |Bi|
    ///            1Sb   |  Bb1  |
    pub itms_allocations: Vec<OffsetedAllocation<I>>,
    pub prologue1: Lcs<'s, I, V>,
    pub bodies: Vec<Lcs<'s, I, V>>,
    pub device: type2::DeviceCollection,
    pub buffer: I,
    pub itms_deallocation: Vec<RegisterManipulation<I>>,
    /// The devices where input/output of the subgraph can be from/to
    pub io_devices: super::device::DeviceSpecific<bool>,
}

#[derive(Debug, Hash, Clone)]
pub enum RegisterManipulation<I> {
    TupleGet {
        id: I,
        target: I,
        index: usize,
    },
    Tuple {
        id: I,
        oprands: Vec<I>,
    },
    StackFree {
        id: I,
    },
    SetPolyMeta {
        id: I,
        from: I,
        offset: usize,
        len: usize,
    },
    SliceBuffer {
        id: I,
        operand: I,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone)]
pub enum Debug<'s> {
    Type2(type2::VertexId, SourceInfo<'s>),
    Subgraph(
        type2::VertexId,
        type2::sliceable_subgraph::VertexId,
        SourceInfo<'s>,
    ),
}

impl<'s> std::hash::Hash for Debug<'s> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Debug::Type2(vid, ..) => vid.hash(state),
            Debug::Subgraph(vid, ..) => vid.hash(state),
        }
    }
}

#[derive(Debug, Hash, Clone)]
pub enum InstructionNode<'s, I, V> {
    Type2 {
        /// (a, Some(b)) says that a use b inplace, so we need move b to a when lowering this to
        /// runtime instruction.
        /// (a, None) syas that a needs to be newly allocated, and there needs no move.
        ids: Vec<(I, Option<I>)>,
        temp: Vec<I>,
        vertex: V,
        debug: Debug<'s>,
        device: type2::DeviceCollection,
    },
    Sub(Subgraph<'s, I, V>),
    Malloc {
        id: I,
        device: Device,
        addr: AllocMethod,
    },
    MallocOffseted {
        id: I,
        base: I,
        offset: usize,
    },
    Free {
        id: I,
        device: Device,
        variant: AllocVariant,
    },
    Transfer {
        id: I,
        from: I,
    },
    FillPoly {
        id: I,
        deg: usize,
        init: PolyInit,
        pty: PolyType,
    },
}

impl<I> RegisterManipulation<I>
where
    I: Copy,
{
    pub fn defs<'s>(&'s self) -> Box<dyn Iterator<Item = I> + 's> {
        use RegisterManipulation::*;

        match self {
            StackFree { .. } => Box::new(empty()),
            Tuple { id, .. } => Box::new(once(*id)),
            SetPolyMeta { id, .. } => Box::new(once(*id)),
            TupleGet { id, .. } => Box::new(once(*id)),
            SliceBuffer { id, .. } => Box::new(once(*id)),
        }
    }

    pub fn uses<'s>(&'s self) -> Box<dyn Iterator<Item = I> + 's> {
        use RegisterManipulation::*;
        match self {
            StackFree { id } => Box::new(once(*id)),
            Tuple { oprands, .. } => Box::new(oprands.iter().copied()),
            SetPolyMeta { from, .. } => Box::new(once(*from)),
            SliceBuffer { operand, .. } => Box::new(once(*operand)),
            TupleGet { target, .. } => Box::new(once(*target)),
        }
    }
}

impl<'s, I, V> InstructionNode<'s, I, V>
where
    I: Copy,
{
    pub fn defs<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use InstructionNode::*;

        match self {
            Type2 { ids, .. } => Box::new(ids.iter().map(|(x, _)| *x)),
            Malloc { id, .. } => Box::new(once(*id)),
            MallocOffseted { id, .. } => Box::new(once(*id)),
            Sub(sg) => Box::new(sg.bodies[0].loads.iter().map(|l| l.id)),
            Free { .. } => Box::new(empty()),
            Transfer { id, .. } => Box::new(once(*id)),
            FillPoly { id, .. } => Box::new(once(*id)),
        }
    }

    pub fn is_alloc(&self) -> bool {
        use InstructionNode::*;
        match self {
            Malloc { .. } => true,
            _ => false,
        }
    }

    pub fn is_dealloc(&self) -> bool {
        use InstructionNode::*;
        match self {
            Free { .. } => true,
            _ => false,
        }
    }
}
