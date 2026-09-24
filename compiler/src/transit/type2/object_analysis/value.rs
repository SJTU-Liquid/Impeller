//! for `atom: Atom<Object, ValueNode>`, in case of being polynomials,
//! `atom.object().node()`'s degree is degree of the polynomial that the object belongs to,
//! while its slice is the object's position in the polynomial.
//!
//! The degree of `atom.node()` is degree of the object, that is, length of `atom.object().node().unwrap_poly().1.len()`,
//! and its slice is the atom's position in the polynomial.
//!
//! An [`Object`] is a single, immutable piece of data on some device, while an `Atom` is some sort of
//! reference to the object that can point to a slice of the object, if the object is a polynomial.
//!
//! A tree of `Atom` consitutes a `Value`, which represents the stuff stored in a runtime register.

use super::size::Size;
use crate::transit::type3::Device;
use zkpoly_common::arith::Mutability;
use zkpoly_common::typ::{template::Typ, Slice};
use zkpoly_common::{define_usize_id, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

define_usize_id!(ObjectId);

impl From<&ObjectId> for ObjectId {
    fn from(value: &ObjectId) -> Self {
        value.clone()
    }
}

pub type ValueNode = Typ<Slice>;
pub type TopNode = Typ<()>;

pub type ObjectDefs = Heap<ObjectId, TopNode>;

/// An object is an immutable piece of data on some storage device.
/// Objects that are slices belonging to the same [`ObjectId`] can be concatenated to a complete object,
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Object {
    id: ObjectId,
    node: ValueNode,
}

impl Object {
    pub fn new(object_id: ObjectId, node: ValueNode) -> Self {
        Object {
            id: object_id,
            node,
        }
    }

    pub fn id(&self) -> ObjectId {
        self.id
    }

    pub fn node(&self) -> &ValueNode {
        &self.node
    }

    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        Object {
            id: object_id,
            node: self.node.clone(),
        }
    }

    pub fn size<Rt: RuntimeType>(&self) -> Size {
        Size::Smithereen(super::size::SmithereenSize(
            self.node.size::<Rt::Field, Rt::PointAffine>() as u64,
        ))
    }

    pub fn whole(self, device: Device) -> Atom<Object, ValueNode> {
        let vn = self
            .node
            .map_scalar_array_len(|_, s| s.len() as usize)
            .map_p(|_, s| Slice::new(0, s.len()));
        Atom::new(self, device, vn)
    }

    pub fn whole_poly(mut self) -> Self {
        self.node = self.node.map_p(|deg, _s| Slice::new(0, deg as u64));
        self
    }

    pub fn is_whole_poly(&self) -> bool {
        let mut r = true;
        self.node.map_p(|deg, s| {
            r = s.begin() == 0 && s.len() == deg as u64;
        });
        r
    }

    pub fn map_node(&self, f: impl FnOnce(&ValueNode) -> ValueNode) -> Self {
        Object {
            id: self.id,
            node: f(&self.node),
        }
    }
}

mod object_table;
pub use object_table::{ObjectBijection, ObjectTable};

/// Represents what we know now about what's inside a runtime register
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Atom<O, N> {
    /// The object it points to
    object: O,
    /// On which memory device the object resides.
    /// For [`super::Operation`]'s, this is determined by execution device of the operation.
    device: Device,
    /// Variant of the value
    node: N,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tree<S> {
    Single(S),
    Tuple(Vec<Tree<S>>),
}

impl<S> Default for Tree<S> {
    fn default() -> Self {
        Tree::Tuple(Vec::new())
    }
}

pub mod tree_iter {
    use super::Tree;

    pub struct Iter<'t, S> {
        pub(super) stack: Vec<&'t Tree<S>>,
    }

    impl<'t, S> Iterator for Iter<'t, S> {
        type Item = &'t S;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let top = self.stack.pop()?;
                match top {
                    Tree::Single(x) => return Some(x),
                    Tree::Tuple(xs) => xs.iter().for_each(|x| self.stack.push(x)),
                }
            }
        }
    }

    pub struct IterWithIndices<'t, S> {
        pub(super) stack: Vec<(&'t Tree<S>, Vec<usize>)>,
    }

    impl<'t, S> Iterator for IterWithIndices<'t, S> {
        type Item = (Vec<usize>, &'t S);

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let (top, top_indices) = self.stack.pop()?;
                match top {
                    Tree::Single(x) => return Some((top_indices, x)),
                    Tree::Tuple(xs) => xs.iter().enumerate().for_each(|(i, x)| {
                        let indices = top_indices
                            .iter()
                            .cloned()
                            .chain(std::iter::once(i))
                            .collect();
                        self.stack.push((x, indices))
                    }),
                }
            }
        }
    }

    pub struct IterMut<'t, S> {
        pub(super) stack: Vec<&'t mut Tree<S>>,
    }

    impl<'t, S> Iterator for IterMut<'t, S> {
        type Item = &'t mut S;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let top = self.stack.pop()?;
                match top {
                    Tree::Single(x) => return Some(x),
                    Tree::Tuple(xs) => xs.iter_mut().for_each(|x| self.stack.push(x)),
                }
            }
        }
    }
}

impl<S> Tree<S> {
    fn map_ref_<S1>(&self, f: &mut impl FnMut(&S) -> S1) -> Tree<S1> {
        match self {
            Tree::Single(s) => Tree::Single(f(s)),
            Tree::Tuple(vs) => Tree::Tuple(vs.iter().map(|x| x.map_ref_(f)).collect()),
        }
    }

    pub fn map_ref<S1>(&self, mut f: impl FnMut(&S) -> S1) -> Tree<S1> {
        self.map_ref_(&mut f)
    }

    pub fn map_<S1>(self, f: &mut impl FnMut(S) -> S1) -> Tree<S1> {
        match self {
            Tree::Single(s) => Tree::Single(f(s)),
            Tree::Tuple(vs) => Tree::Tuple(vs.into_iter().map(|x| x.map_(f)).collect()),
        }
    }

    pub fn map<S1>(self, mut f: impl FnMut(S) -> S1) -> Tree<S1> {
        self.map_(&mut f)
    }

    pub fn is_single_and(&self, pred: impl FnOnce(&S) -> bool) -> bool {
        match self {
            Tree::Single(s) => pred(s),
            _ => false,
        }
    }

    fn _reduce_ref<T>(
        &self,
        base: &mut impl FnMut(&S) -> T,
        induce: &mut impl FnMut(Vec<T>) -> T,
    ) -> T {
        use Tree::*;
        match self {
            Single(s) => base(s),
            Tuple(xs) => {
                let ts = xs.iter().map(|x| x._reduce_ref(base, induce)).collect();
                induce(ts)
            }
        }
    }

    pub fn reduce_ref<T>(
        &self,
        mut base: impl FnMut(&S) -> T,
        mut induce: impl FnMut(Vec<T>) -> T,
    ) -> T {
        self._reduce_ref(&mut base, &mut induce)
    }

    fn try_map_<S1, E>(self, f: &mut impl FnMut(S) -> Result<S1, E>) -> Result<Tree<S1>, E> {
        Ok(match self {
            Tree::Single(s) => Tree::Single(f(s)?),
            Tree::Tuple(vs) => Tree::Tuple(
                vs.into_iter()
                    .map(|x| x.try_map_(f))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        })
    }

    pub fn try_map<S1, E>(self, mut f: impl FnMut(S) -> Result<S1, E>) -> Result<Tree<S1>, E> {
        self.try_map_(&mut f)
    }

    pub fn iter1<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Tree<S>> + 'a> {
        match self {
            Tree::Single(..) => Box::new(std::iter::once(self)),
            Tree::Tuple(xs) => Box::new(xs.iter()),
        }
    }

    pub fn into_iter1<'a>(self) -> Box<dyn Iterator<Item = Tree<S>> + 'a>
    where
        S: 'a,
    {
        match self {
            Tree::Single(..) => Box::new(std::iter::once(self)),
            Tree::Tuple(xs) => Box::new(xs.into_iter()),
        }
    }

    pub fn iter1_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Tree<S>> + 'a> {
        match self {
            Tree::Single(..) => Box::new(std::iter::once(self)),
            Tree::Tuple(xs) => Box::new(xs.iter_mut()),
        }
    }

    pub fn unwrap_leaf_mut(&mut self) -> &mut S {
        match self {
            Tree::Single(x) => x,
            _ => panic!("called unwrap_leaf_mut on Tuple node"),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Tree::Single(..) => 0,
            Tree::Tuple(xs) => xs.iter().map(|x| x.depth()).max().unwrap_or(0) + 1,
        }
    }

    pub fn unwrap_leaf(&self) -> &S {
        match self {
            Tree::Single(x) => x,
            _ => panic!("called unwrap_leaf on Tuple node"),
        }
    }

    pub fn unwrap_tuple(&self) -> &[Tree<S>] {
        match self {
            Tree::Single(..) => panic!("called unwrap_tuple on Single node"),
            Tree::Tuple(xs) => xs.as_slice(),
        }
    }

    pub fn iter<'a>(&'a self) -> tree_iter::Iter<'a, S> {
        tree_iter::Iter { stack: vec![self] }
    }

    pub fn iter_with_indices<'a>(&'a self) -> tree_iter::IterWithIndices<'a, S> {
        tree_iter::IterWithIndices {
            stack: vec![(self, Vec::new())],
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> tree_iter::IterMut<'a, S> {
        tree_iter::IterMut { stack: vec![self] }
    }
}

impl<O, N> Atom<O, N> {
    pub fn new(object: O, device: Device, node: N) -> Self {
        Atom {
            object,
            device,
            node,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn object(&self) -> &O {
        &self.object
    }
    pub fn object_mut(&mut self) -> &mut O {
        &mut self.object
    }

    pub fn with_device(&self, device: Device) -> Self
    where
        N: Clone,
        O: Clone,
    {
        Atom {
            node: self.node.clone(),
            object: self.object.clone(),
            device,
        }
    }

    pub fn with_object(&self, object: O) -> Self
    where
        N: Clone,
    {
        Atom {
            object,
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn map_object<O1>(&self, f: impl FnOnce(&O) -> O1) -> Atom<O1, N>
    where
        N: Clone,
    {
        Atom {
            object: f(&self.object),
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn map_node<N1>(&self, f: impl FnOnce(&N) -> N1) -> Atom<O, N1>
    where
        O: Clone,
    {
        Atom {
            object: self.object.clone(),
            device: self.device,
            node: f(&self.node),
        }
    }

    pub fn with_node(&self, node: N) -> Self
    where
        O: Clone,
    {
        Atom {
            object: self.object.clone(),
            device: self.device,
            node,
        }
    }

    pub fn node(&self) -> &N {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut N {
        &mut self.node
    }
}

impl<N> Atom<Object, N> {
    pub fn with_object_id(&self, object_id: ObjectId) -> Self
    where
        N: Clone,
    {
        Atom {
            object: self.object.with_object_id(object_id),
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn object_id(&self) -> ObjectId {
        self.object.id()
    }
}

impl Atom<Object, ValueNode> {
    pub fn is_whole(&self) -> bool {
        let mut r = false;
        self.node.map_p(|deg, _slice| {
            let (whole_deg, whole_slice) = self.object.node.unwrap_poly();
            assert!(whole_deg == deg);
            r = whole_slice.begin() == 0 && whole_slice.end() == whole_deg as u64;
        });
        r
    }
}

impl<O> Atom<O, ValueNode> {
    pub fn object_size<Rt: RuntimeType>(&self) -> usize {
        use std::mem::size_of;
        match &self.node {
            Typ::ScalarArray { len, .. } => len * size_of::<Rt::Field>(),
            Typ::Scalar => size_of::<Rt::Field>(),
            Typ::Transcript => size_of::<Rt::Trans>(),
            Typ::PointBase { len } => len * size_of::<Rt::PointAffine>(),
            Typ::Point => size_of::<Rt::PointAffine>(),
            Typ::Tuple => panic!("a value should never point to a tuple"),
            Typ::Stream => panic!("a value should never point to a stream"),
            Typ::GpuBuffer(len, ..) => *len,
            Typ::Any(_, len) => *len,
        }
    }

    /// Try rotate the value, which must be a [`Typ::ScalarArray`].
    /// If the value is a rotated polynomial, i.e., slice length equals data length,
    /// then it is rotated by `delta`.
    /// Otherwise, [`None`] is returned.
    ///
    /// # Panics
    /// When the value is not a [`Typ::ScalarArray`].
    pub fn rotate(&self, delta: i32) -> Option<Self>
    where
        O: Clone,
    {
        match &self.node {
            Typ::ScalarArray { len, meta } => {
                if meta.len() == *len as u64 {
                    let offset = (meta.begin() as i64 + delta as i64).rem_euclid(*len as i64);
                    Some(self.with_node(Typ::ScalarArray {
                        len: *len,
                        meta: Slice::new(offset as u64, meta.len()),
                    }))
                } else {
                    None
                }
            }
            _ => panic!("can only rotate a scalar array"),
        }
    }

    /// Slice the value, which must be a [`Typ::ScalarArray`].
    /// A slice is still continuous in memory after slicing it, so this function always success.
    ///
    /// # Panics
    /// When the value is not a [`Typ::ScalarArray`].
    pub fn slice(&self, begin: u64, end: u64) -> Self
    where
        O: Clone,
    {
        match &self.node {
            Typ::ScalarArray { len: deg, meta } => {
                let offset = (meta.begin() + begin) % *deg as u64;
                self.with_node(Typ::ScalarArray {
                    len: *deg as usize,
                    meta: Slice::new(offset, end - begin),
                })
            }
            _ => panic!("can only slice a scalar array"),
        }
    }

    /// Check if the value is a sliced polynomial that is shorter then original
    pub fn is_shorten_sliced(&self) -> bool {
        match &self.node {
            Typ::ScalarArray { len: deg, meta } => *deg as usize != meta.len() as usize,
            _ => false,
        }
    }

    /// Check if the value is a sliced polynomial that is only rotated but not shortened
    pub fn is_pure_rotated(&self) -> bool {
        match &self.node {
            Typ::ScalarArray { len: deg, meta } => {
                *deg as usize == meta.len() as usize && meta.begin() != 0
            }
            _ => false,
        }
    }
}

/// One of the values outputed by a vertex.
/// If `.1` is [`Some`], it is the index of the input value taken inplace.
/// A input value is said to be taken inplace if the underlying memory space is then used by the output.
#[derive(Debug, Clone, Hash)]
pub struct OutputT<V, I> {
    v: V,
    inplace_of: Option<I>,
}

impl<V, I> std::ops::Deref for OutputT<V, I> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V, I> std::ops::DerefMut for OutputT<V, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl<V, I> OutputT<V, I> {
    pub fn new(value: V, inplace: Option<I>) -> Self {
        OutputT {
            v: value,
            inplace_of: inplace,
        }
    }

    pub fn map_v<V1>(self, f: impl FnOnce(V) -> V1) -> OutputT<V1, I> {
        OutputT {
            v: f(self.v),
            inplace_of: self.inplace_of,
        }
    }

    pub fn map_i<I1>(self, f: impl FnOnce(Option<I>) -> Option<I1>) -> OutputT<V, I1> {
        OutputT {
            v: self.v,
            inplace_of: f(self.inplace_of),
        }
    }

    pub fn non_inplace(value: V) -> Self {
        OutputT {
            v: value,
            inplace_of: None,
        }
    }

    pub fn with_non_inplace(&self) -> Self
    where
        V: Clone,
    {
        OutputT {
            v: self.v.clone(),
            inplace_of: None,
        }
    }

    pub fn v(self) -> V {
        self.v
    }

    pub fn inplace_of(&self) -> Option<I>
    where
        I: Clone,
    {
        self.inplace_of.clone()
    }

    pub fn inplace_of_mut(&mut self) -> &mut Option<I> {
        &mut self.inplace_of
    }
}

impl<V> OutputT<V, V> {
    pub fn map_both_ref<V1>(
        &self,
        mut f: impl for<'a> FnMut(&'a V) -> &'a V1,
    ) -> OutputT<&V1, &V1> {
        OutputT {
            v: f(&self.v),
            inplace_of: self.inplace_of.as_ref().map(f),
        }
    }
}

pub type OutputValue<V, I> = Tree<OutputT<V, I>>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InputT<V> {
    v: V,
    mutability: Mutability,
}

impl<V> InputT<V> {
    pub fn new(v: V, m: Mutability) -> Self {
        Self { v, mutability: m }
    }

    pub fn map_v<V1>(self, f: impl FnOnce(V) -> V1) -> InputT<V1> {
        InputT {
            v: f(self.v),
            mutability: self.mutability,
        }
    }

    pub fn immutable(v: V) -> Self {
        Self::new(v, Mutability::Const)
    }

    pub fn mutability(&self) -> Mutability {
        self.mutability
    }

    pub fn v(self) -> V {
        self.v
    }
}

impl<V> std::ops::Deref for InputT<V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V> std::ops::DerefMut for InputT<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

pub type InputValue<V> = Tree<InputT<V>>;
