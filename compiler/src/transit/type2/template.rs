use zkpoly_common::{arith, load_dynamic::Libs, msm_config::MsmConfig};
use zkpoly_core::fused_kernels::kernel_gen::ArithGraphScheduler;
use zkpoly_runtime::args::{EntryId, RuntimeType};

use super::{
    temporary_space, transit, DeviceCollection, DevicePreference, NttAlgorithm, PolyInit, PolyType,
    Temporaries, Typ,
};
use crate::transit::type3;

#[derive(
    Debug, Clone, Eq, PartialEq, Ord, Hash, PartialOrd, serde::Serialize, serde::Deserialize,
)]
/// An operation that can be performed on a sliced polynomial,
/// and result of which can be simply concated to obtain the full result
pub enum SliceableNode<I, A> {
    NewPoly(u64, PolyInit, PolyType),
    SingleArith(arith::Arith<I>),
    ScalarInvert(I),
    Arith { arith: A },
    RotateIdx(I, i32),
    BatchedInvert(I),
    DistributePowers { poly: I, powers: I },
}

#[derive(
    Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
/// These operations can be sliced, but they only gives partial result.
/// So vertices following this node cannot group into a sliceable subgraph
pub enum LastSliceableNode<I> {
    /// Partial MSM result is the inner product between the polynomial slice
    /// and the currespounding points.
    /// To get the full result, we only need to sum those points.
    Msm {
        polys: Vec<I>,
        points: Vec<I>,
        alg: MsmConfig,
    },
    /// Partial polynomial evaluation result of EvaluatePoly([a_0, a_1, ..., a_(n - 1)], x)
    /// is y_i = a_0 + a_1 x + ... + a_(n - 1) x^(n - 1).
    /// Therefore to get the full result, we need to evaluate
    /// y = y_0 + y_1 x^n + ... + y_(m - 1) x^(n (m - 1))
    EvaluatePoly { poly: I, at: I },
}

pub trait SubgraphNode<I> {
    type AltLabeled<I2>;

    fn inputs<'a>(&'a self) -> impl Iterator<Item = &'a I> + 'a
    where
        I: 'a;

    fn inputs_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut I> + 'a
    where
        I: 'a;

    fn try_relabeled<I2, Er>(
        &self,
        f: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<Self::AltLabeled<I2>, Er>;
}

impl<I> SubgraphNode<I> for () {
    type AltLabeled<I2> = ();

    fn inputs<'a>(&'a self) -> impl Iterator<Item = &'a I>
    where
        I: 'a,
    {
        #[allow(unreachable_code)]
        std::iter::once(panic!("variant Subgraph is not expected in this type"))
    }

    fn inputs_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut I>
    where
        I: 'a,
    {
        #[allow(unreachable_code)]
        std::iter::once(panic!("variant Subgraph is not expected in this type"))
    }

    fn try_relabeled<I2, Er>(
        &self,
        _f: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<Self::AltLabeled<I2>, Er> {
        #[allow(unreachable_code)]
        Ok(panic!("variant Subgraph is not expected in this type"))
    }
}

#[derive(
    Debug, Clone, Eq, PartialEq, Ord, Hash, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum VertexNode<I, A, C, E, S> {
    Sliceable(SliceableNode<I, A>),
    LastSliceable(LastSliceableNode<I>),
    Subgraph(S),
    Extend(I, u64),
    Entry(EntryId),
    Return(I),
    /// A constant that can not be sliced, e.g. twiddle factors
    Constant(C),
    /// Convert a local from one representation to another
    Ntt {
        s: I,
        to: PolyType,
        from: PolyType,
        alg: NttAlgorithm<I>,
    },
    Slice(I, u64, u64),
    Interpolate {
        xs: Vec<I>,
        ys: Vec<I>,
    },
    Array(Vec<I>),
    AssmblePoly(u64, Vec<I>),
    Blind(I, u64, u64),
    HashTranscript {
        transcript: I,
        value: I,
        typ: transit::HashTyp,
    },
    /// Returns (transcript, scalar)
    SqueezeScalar(I),
    PolyPermute(I, I, usize),
    ArrayGet(I, usize),
    TupleGet(I, usize),
    UserFunction(E, Vec<I>),
    KateDivision(I, I),
    ScanMul {
        x0: I,
        poly: I,
    },
    IndexPoly(I, u64),
    AssertEq(I, I, Option<String>),
    Print(I, String),
}

impl<I, A, C, E, S> Default for VertexNode<I, A, C, E, S> {
    fn default() -> Self {
        Self::Sliceable(SliceableNode::NewPoly(0, PolyInit::Zeros, PolyType::Coef))
    }
}

impl<I> SliceableNode<I, super::Arith<I>> {
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use SliceableNode::*;
        match self {
            NewPoly(_, _, _) => Box::new(std::iter::empty()),
            SingleArith(arith) => Box::new(arith.uses_ref()),
            ScalarInvert(x) => Box::new([x].into_iter()),
            Arith { arith, .. } => Box::new(arith.uses_ref()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            DistributePowers { poly, powers } => Box::new([poly, powers].into_iter()),
        }
    }

    pub fn sliceable_uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use SliceableNode::*;
        match self {
            NewPoly(_, _, _) => Box::new(std::iter::empty()),
            SingleArith(arith) => Box::new(arith.sliceable_uses_ref()),
            ScalarInvert(x) => Box::new([x].into_iter()),
            Arith { arith, .. } => Box::new(arith.sliceable_uses_ref()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly].into_iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use SliceableNode::*;
        match self {
            NewPoly(_, _, _) => Box::new(std::iter::empty()),
            SingleArith(arith) => Box::new(arith.uses_mut()),
            ScalarInvert(x) => Box::new([x].into_iter()),
            Arith { arith, .. } => Box::new(arith.uses_mut()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            DistributePowers { poly, powers } => Box::new([poly, powers].into_iter()),
        }
    }
}

fn on_gpu(device: &DeviceCollection) -> type3::Device {
    type3::Device::for_execution_on(device.ensure_gpu())
}
impl<I> SliceableNode<I, super::Arith<I>>
where
    I: Clone,
{
    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use SliceableNode::*;
        match self {
            Arith { arith, .. } => arith.mutable_uses(),
            BatchedInvert(x) => Box::new([x.clone()].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly.clone()].into_iter()),
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([scalar.clone()].into_iter())
            }
            ScalarInvert(val) => Box::new([val.clone()].into_iter()),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use SliceableNode::*;
        match self {
            Arith { arith, .. } => arith.mutable_uses_mut(),
            BatchedInvert(x) => Box::new([x].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly].into_iter()),
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([scalar].into_iter())
            }
            ScalarInvert(val) => Box::new([val].into_iter()),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a> {
        use SliceableNode::*;
        match self {
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new(std::iter::once(Some(scalar.clone())))
            }
            ScalarInvert(val) => Box::new(std::iter::once(Some(val.clone()))),
            Arith { arith, .. } => arith.outputs_inplace(),
            BatchedInvert(poly) => Box::new(std::iter::once(Some(poly.clone()))),
            DistributePowers { poly, .. } => Box::new(std::iter::once(Some(poly.clone()))),
            _ => Box::new(std::iter::repeat(None)),
        }
    }

    pub fn try_relabeled<I2, Er>(
        self,
        mut mapping: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<SliceableNode<I2, super::Arith<I2>>, Er>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        use SliceableNode::*;
        Ok(match self {
            NewPoly(deg, init, typ) => NewPoly(deg, init, typ),
            SingleArith(a) => SingleArith(a.try_relabeled(&mut mapping)?),
            ScalarInvert(x) => ScalarInvert(mapping(x)?),
            Arith { arith } => Arith {
                arith: arith.try_relabeled(&mut mapping)?,
            },
            RotateIdx(x, n) => RotateIdx(mapping(x)?, n),
            BatchedInvert(poly) => BatchedInvert(mapping(poly)?),
            DistributePowers { poly, powers } => DistributePowers {
                poly: mapping(poly)?,
                powers: mapping(powers)?,
            },
        })
    }

    pub fn relabeled<I2>(
        self,
        mut mapping: impl FnMut(I) -> I2,
    ) -> SliceableNode<I2, super::Arith<I2>>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        self.try_relabeled::<I2, ()>(|x| Ok(mapping(x))).unwrap()
    }

    pub fn temporary_space_needed<'a, Rt: RuntimeType>(
        &self,
        device: &DeviceCollection,
        libs: &mut Libs,
        input_types: impl FnOnce(&I) -> &'a Typ<Rt> + 'a,
        ag_scheduler: ArithGraphScheduler,
    ) -> Option<Temporaries> {
        use SliceableNode::*;

        match self {
            Arith { arith } => Some(
                Temporaries {
                    sizes: temporary_space::arith::<Rt, I, arith::ExprId>(arith, ag_scheduler),
                    device: on_gpu(device),
                }, // We are using only one gpu for now
            ),
            ScalarInvert { .. } => None,
            SingleArith(..) => None,
            NewPoly(..) => None,
            RotateIdx(..) => None,
            BatchedInvert(poly) => {
                let (_, len) = input_types(poly).unwrap_poly();
                Some(Temporaries {
                    sizes: temporary_space::poly_invert::<Rt>(*len as usize, libs),
                    device: on_gpu(device),
                })
            }
            DistributePowers { .. } => None,
        }
    }
}

fn on_device(device: &DeviceCollection) -> type3::Device {
    device
        .iter()
        .find(|(_, t)| **t)
        .map(|(d, _)| type3::Device::for_execution_on(d))
        .expect("some device to run on")
}

impl<I> LastSliceableNode<I> {
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(polys.iter().chain(points.iter())),
            EvaluatePoly { poly, at } => Box::new([poly, at].into_iter()),
        }
    }

    pub fn sliceable_uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(polys.iter().chain(points.iter())),
            EvaluatePoly { poly, .. } => Box::new([poly].into_iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(polys.iter_mut().chain(points.iter_mut())),
            EvaluatePoly { poly, at } => Box::new([poly, at].into_iter()),
        }
    }

    pub fn temporary_space_needed<'a, Rt: RuntimeType>(
        &self,
        device: &DeviceCollection,
        libs: &mut Libs,
        input_types: impl FnOnce(&I) -> &'a Typ<Rt> + 'a,
    ) -> Option<Temporaries> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, alg, .. } => {
                let (_, len) = input_types(polys.first().unwrap()).unwrap_poly();
                Some(Temporaries {
                    sizes: temporary_space::msm::<Rt>(alg, *len as usize, libs),
                    // We are using only one gpu for now
                    device: on_gpu(device),
                })
            }
            EvaluatePoly { poly, .. } => {
                let (_, len) = input_types(poly).unwrap_poly();
                Some(Temporaries {
                    sizes: temporary_space::poly_eval::<Rt>(*len as usize, libs),
                    device: on_device(device),
                })
            }
        }
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        Box::new(std::iter::empty())
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        Box::new(std::iter::empty())
    }

    pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a>
    where
        I: Clone,
    {
        Box::new(std::iter::repeat(None))
    }

    pub fn try_relabeled<I2, Er>(
        self,
        mut mapping: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<LastSliceableNode<I2>, Er>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
        I: Clone,
    {
        use LastSliceableNode::*;
        Ok(match self {
            Msm { polys, points, alg } => Msm {
                polys: polys
                    .into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
                points: points
                    .into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
                alg: alg,
            },
            EvaluatePoly { poly, at } => EvaluatePoly {
                poly: mapping(poly)?,
                at: mapping(at)?,
            },
        })
    }

    pub fn relabeled<I2>(self, mut mapping: impl FnMut(I) -> I2) -> LastSliceableNode<I2>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
        I: Clone,
    {
        self.try_relabeled::<I2, ()>(|x| Ok(mapping(x))).unwrap()
    }
}

#[derive(PartialEq, Eq)]
pub enum SliceableProperty {
    /// This vertex is always sliceable
    Always,
    /// This vertex is sliceable if it is output vertex of the sliceable subgraph
    Last,
    /// This vertex is sliceable if at all its inputs are within the sliceable subgraph
    NonFirst,
    /// This vertex is not sliceable
    No,
}

impl<I, A, C, E, S> VertexNode<I, A, C, E, S> {
    pub fn unwrap_tuple_get_mut(&mut self) -> (&mut I, &mut usize) {
        match self {
            VertexNode::TupleGet(pred, idx) => (pred, idx),
            _ => panic!("called unwrap_tuple_get_mut on other variant"),
        }
    }

    pub fn unwrap_tuple_get(&self) -> (I, usize)
    where
        I: Clone,
    {
        match self {
            VertexNode::TupleGet(pred, idx) => (pred.clone(), *idx),
            _ => panic!("called unwrap_tuple_get_mut on other variant"),
        }
    }
}

impl<I, C, E, S> VertexNode<I, super::Arith<I>, C, E, S>
where
    S: SubgraphNode<I>,
{
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a>
    where
        I: 'a,
    {
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.uses_ref(),
            LastSliceable(lsn) => lsn.uses_ref(),
            Subgraph(s) => Box::new(s.inputs()),
            Extend(x, _) => Box::new([x].into_iter()),
            Ntt { s, alg, .. } => Box::new([s].into_iter().chain(alg.uses_ref())),
            ScanMul { x0, poly } => Box::new([x0, poly].into_iter()),
            Slice(x, ..) => Box::new([x].into_iter()),
            Interpolate { xs, ys } => Box::new(xs.iter().chain(ys.iter())),
            Array(es) => Box::new(es.iter()),
            AssmblePoly(_, es) => Box::new(es.iter()),
            Blind(x, ..) => Box::new(std::iter::once(x)),
            HashTranscript {
                transcript, value, ..
            } => Box::new([transcript, value].into_iter()),
            SqueezeScalar(x) => Box::new([x].into_iter()),
            TupleGet(x, _) => Box::new([x].into_iter()),
            ArrayGet(x, _) => Box::new([x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter()),
            KateDivision(lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Return(x) => Box::new([x].into_iter()),
            IndexPoly(x, _) => Box::new([x].into_iter()),
            AssertEq(x, y, _msg) => Box::new([x, y].into_iter()),
            Print(x, _) => Box::new([x].into_iter()),
            PolyPermute(input, table, _) => Box::new([input, table].into_iter()),
            Entry(_) | Constant(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn sliceable_uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a>
    where
        I: 'a,
    {
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.sliceable_uses_ref(),
            LastSliceable(lsn) => lsn.sliceable_uses_ref(),
            _ => self.uses_ref(),
        }
    }

    pub fn memory_device(
        &self,
        execution_device: &DeviceCollection,
        constants_device: impl FnOnce(&C) -> type3::Device,
    ) -> type3::Device {
        use VertexNode::*;

        match self {
            Constant(c) => constants_device(c),
            LastSliceable(LastSliceableNode::Msm { .. }) => type3::Device::Cpu,
            _ => type3::Device::for_execution_on(execution_device.ensure_single()),
        }
    }

    pub fn unwrap_constant(&self) -> &C {
        match self {
            Self::Constant(constant) => constant,
            _ => panic!("this is not constant"),
        }
    }
    pub fn unwrap_ntt_alg_mut(&mut self) -> &mut NttAlgorithm<I> {
        match self {
            Self::Ntt { alg, .. } => alg,
            _ => panic!("this is not ntt"),
        }
    }

    pub fn unwrap_msm(&mut self) -> (&mut Vec<I>, &mut Vec<I>, &mut MsmConfig) {
        match self {
            Self::LastSliceable(LastSliceableNode::Msm { polys, points, alg }) => {
                (polys, points, alg)
            }
            _ => panic!("this is not msm"),
        }
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.uses_mut(),
            LastSliceable(lsn) => lsn.uses_mut(),
            Subgraph(s) => Box::new(s.inputs_mut()),
            Extend(x, _) => Box::new([x].into_iter()),
            Ntt { s, alg, .. } => Box::new([s].into_iter().chain(alg.uses_mut())),
            ScanMul { x0, poly } => Box::new([x0, poly].into_iter()),
            Slice(x, ..) => Box::new([x].into_iter()),
            Interpolate { xs, ys } => Box::new(xs.iter_mut().chain(ys.iter_mut())),
            Array(es) => Box::new(es.iter_mut()),
            HashTranscript {
                transcript, value, ..
            } => Box::new([transcript, value].into_iter()),
            SqueezeScalar(x) => Box::new([x].into_iter()),
            Blind(x, ..) => Box::new(std::iter::once(x)),
            TupleGet(x, _) => Box::new([x].into_iter()),
            ArrayGet(x, _) => Box::new([x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter_mut()),
            KateDivision(lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Return(x) => Box::new([x].into_iter()),
            IndexPoly(x, _) => Box::new([x].into_iter()),
            AssertEq(x, y, _msg) => Box::new([x, y].into_iter()),
            Print(x, _) => Box::new([x].into_iter()),
            PolyPermute(input, table, _) => Box::new([input, table].into_iter()),
            AssmblePoly(_, xs) => Box::new(xs.iter_mut()),
            Entry(_) | Constant(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn device(&self) -> DevicePreference {
        use DevicePreference::*;
        use LastSliceableNode::*;
        use SliceableNode::*;
        use VertexNode::*;
        match self {
            Sliceable(sn) => match sn {
                NewPoly(..) => PreferGpu,
                SingleArith(..) => Gpu,
                ScalarInvert(..) => PreferGpu,
                Arith { .. } => Gpu,
                RotateIdx(..) => Cpu,
                BatchedInvert(..) => Gpu,
                DistributePowers { .. } => Gpu,
            },
            LastSliceable(lsn) => match lsn {
                Msm { .. } => Gpu,
                EvaluatePoly { .. } => Gpu,
            },
            Extend(..) => PreferGpu,
            IndexPoly { .. } => PreferGpu,
            Ntt { .. } => Gpu,
            KateDivision(_, _) => Gpu,
            PolyPermute(_, _, _) => Gpu,
            Subgraph(..) => Gpu,
            ScanMul { .. } => Gpu,
            _ => Cpu,
        }
    }

    pub fn no_supports_shorten_slice(&self) -> bool {
        use VertexNode::*;
        match self {
            LastSliceable(LastSliceableNode::Msm { .. }) => true,
            Subgraph(..) => true,
            Ntt { .. } => true,
            _ => false,
        }
    }

    pub fn no_supports_pure_rotation(&self) -> bool {
        use VertexNode::*;
        match self {
            LastSliceable(LastSliceableNode::Msm { .. }) => true,
            Ntt { .. } => true,
            _ => false,
        }
    }

    pub fn is_virtual(&self) -> bool {
        use VertexNode::*;
        match self {
            Array(..)
            | ArrayGet(..)
            | TupleGet(..)
            | Sliceable(SliceableNode::RotateIdx(..))
            | Slice(..) => true,
            _ => false,
        }
    }

    pub fn unexpcted_during_kernel_gen(&self) -> bool {
        use VertexNode::*;
        match self {
            Extend(..) | Sliceable(SliceableNode::NewPoly(..)) => true,
            _ => self.is_virtual(),
        }
    }

    pub fn is_return(&self) -> bool {
        use VertexNode::*;
        match self {
            Return(..) => true,
            _ => false,
        }
    }

    pub fn immortal(&self) -> bool {
        use VertexNode::*;
        match self {
            Constant(..) => true,
            Entry(..) => true,
            _ => false,
        }
    }

    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a
    where
        I: Clone,
    {
        self.uses_ref().cloned()
    }

    pub fn temporary_space_needed<'a, Rt: RuntimeType>(
        &self,
        device: &DeviceCollection,
        libs: &mut Libs,
        input_types: impl FnOnce(&I) -> &'a Typ<Rt> + 'a,
        ag_scheduler: ArithGraphScheduler,
    ) -> Option<Temporaries>
    where
        I: Clone,
    {
        use VertexNode::*;

        match self {
            Sliceable(sn) => sn.temporary_space_needed(device, libs, input_types, ag_scheduler),
            LastSliceable(lsn) => lsn.temporary_space_needed(device, libs, input_types),
            Extend(..) => None,
            Subgraph(..) => None,
            Constant(..) => None,
            Return(..) => None,
            Entry(..) => None,
            Ntt { .. } => None,
            Slice(..) => None,
            Interpolate { .. } => None,
            Array(..) => None,
            AssmblePoly(..) => None,
            HashTranscript { .. } => None,
            SqueezeScalar(..) => None,
            TupleGet(..) => None,
            ArrayGet(..) => None,
            UserFunction(..) => None,
            Blind(..) => None,
            KateDivision(lhs, _) => {
                let (_, len) = input_types(lhs).unwrap_poly();
                Some(Temporaries {
                    sizes: temporary_space::kate_division::<Rt>(*len as usize, libs),
                    device: on_gpu(device),
                })
            }
            PolyPermute(_, _, usable) => Some(Temporaries {
                sizes: temporary_space::poly_permute::<Rt>(*usable as usize, libs),
                device: on_gpu(device),
            }),
            ScanMul { poly, .. } => {
                let (_, len) = input_types(poly).unwrap_poly();
                Some(Temporaries {
                    sizes: temporary_space::poly_scan::<Rt>(*len as usize, libs),
                    device: on_gpu(device),
                })
            }
            IndexPoly(..) => None,
            AssertEq(..) => None,
            Print(..) => None,
        }
    }
}

impl<I, C, E, S> VertexNode<I, super::Arith<I>, C, E, S> {
    pub fn map_s<S1>(self, f: impl FnOnce(S) -> S1) -> VertexNode<I, super::Arith<I>, C, E, S1> {
        use VertexNode::*;
        match self {
            Sliceable(sn) => Sliceable(sn),
            LastSliceable(lsn) => LastSliceable(lsn),
            Subgraph(s) => Subgraph(f(s)),
            Constant(c) => Constant(c),
            Extend(s, deg) => Extend(s, deg),
            PolyPermute(input, table, len) => PolyPermute(input, table, len),
            Entry(idx) => Entry(idx),
            Return(x) => Return(x),
            Ntt { alg, s, to, from } => Ntt { alg, s, to, from },
            Blind(x, a, b) => Blind(x, a, b),
            ScanMul { x0, poly } => ScanMul { x0, poly },
            Slice(s, start, end) => Slice(s, start, end),
            Interpolate { xs, ys } => Interpolate { xs, ys },
            Array(es) => Array(es),
            AssmblePoly(s, es) => AssmblePoly(s, es),
            HashTranscript {
                transcript,
                value,
                typ,
            } => HashTranscript {
                transcript,
                value,
                typ,
            },
            SqueezeScalar(transcript) => SqueezeScalar(transcript),
            TupleGet(s, i) => TupleGet(s, i),
            ArrayGet(s, i) => ArrayGet(s, i),
            UserFunction(fid, args) => UserFunction(fid, args),
            KateDivision(lhs, rhs) => KateDivision(lhs, rhs),
            IndexPoly(x, idx) => IndexPoly(x, idx),
            AssertEq(x, y, msg) => AssertEq(x, y, msg),
            Print(x, s) => Print(x, s),
        }
    }
}

impl<'s, I, C, E, S> VertexNode<I, super::Arith<I>, C, E, S>
where
    I: Clone + 'static,
    C: Clone,
    E: Clone,
    S: SubgraphNode<I>,
{
    pub fn try_relabeled<I2, Er>(
        self,
        mut mapping: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<VertexNode<I2, arith::ArithGraph<I2, arith::ExprId>, C, E, S::AltLabeled<I2>>, Er>
    where
        I2: Default + Ord + std::fmt::Debug + Clone + 'static,
    {
        use VertexNode::*;

        let r = match self {
            Sliceable(sn) => Sliceable(SliceableNode::try_relabeled(sn, &mut mapping)?),
            LastSliceable(lsn) => LastSliceable(lsn.try_relabeled(mapping)?),
            Subgraph(s) => Subgraph(s.try_relabeled(&mut mapping)?),
            Constant(c) => Constant(c),
            Extend(s, deg) => Extend(mapping(s)?, deg),
            PolyPermute(input, table, len) => PolyPermute(mapping(input)?, mapping(table)?, len),
            Entry(idx) => Entry(idx),
            Return(x) => Return(mapping(x)?),
            Ntt { alg, s, to, from } => Ntt {
                alg: alg.try_relabeled(&mut mapping)?,
                s: mapping(s)?,
                to: to,
                from: from,
            },
            Slice(s, start, end) => Slice(mapping(s)?, start, end),
            Interpolate { xs, ys } => Interpolate {
                xs: xs
                    .into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
                ys: ys
                    .into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
            },
            Blind(x, begin, end) => Blind(mapping(x)?, begin, end),
            Array(es) => Array(
                es.into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
            ),
            AssmblePoly(s, es) => AssmblePoly(
                s,
                es.into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
            ),
            HashTranscript {
                transcript,
                value,
                typ,
            } => HashTranscript {
                transcript: mapping(transcript)?,
                value: mapping(value)?,
                typ: typ,
            },
            SqueezeScalar(transcript) => SqueezeScalar(mapping(transcript)?),
            TupleGet(s, i) => TupleGet(mapping(s)?, i),
            ArrayGet(s, i) => ArrayGet(mapping(s)?, i),
            UserFunction(fid, args) => UserFunction(
                fid,
                args.into_iter()
                    .map(|x| mapping(x))
                    .collect::<Result<_, _>>()?,
            ),
            ScanMul { x0, poly } => ScanMul {
                x0: mapping(x0)?,
                poly: mapping(poly)?,
            },
            KateDivision(lhs, rhs) => KateDivision(mapping(lhs)?, mapping(rhs)?),
            IndexPoly(x, idx) => IndexPoly(mapping(x)?, idx),
            AssertEq(x, y, msg) => AssertEq(mapping(x)?, mapping(y)?, msg),
            Print(x, s) => Print(mapping(x)?, s),
        };

        Ok(r)
    }

    pub fn branch_s<T, Ct>(
        self,
        ctx: &mut Ct,
        f1: impl FnOnce(VertexNode<I, super::Arith<I>, C, E, ()>, &mut Ct) -> T,
        f2: impl FnOnce(S, &mut Ct) -> T,
    ) -> T
    where
        S: Clone,
    {
        let mut s = None;
        let no_s = self.map_s(|s0| s = Some(s0));

        if let Some(s) = s {
            f2(s, ctx)
        } else {
            f1(no_s, ctx)
        }
    }

    pub fn relabeled<I2: Default + Ord + std::fmt::Debug + Clone>(
        self,
        mut mapping: impl FnMut(I) -> I2,
    ) -> VertexNode<I2, arith::ArithGraph<I2, arith::ExprId>, C, E, S::AltLabeled<I2>>
    where
        I2: 'static,
    {
        self.try_relabeled::<_, ()>(|i| Ok(mapping(i))).unwrap()
    }

    pub fn track(&self, device: &DeviceCollection) -> type3::Track {
        use super::super::type3::Track::*;
        use SliceableNode::*;
        use VertexNode::*;

        let err_no_track = |name: &'static str| -> ! { panic!("no track for {}", name) };
        let track = || {
            let gpu = device.iter_gpus().find(|(_, t)| **t).map(|(gpu, _)| gpu);
            match (gpu, device.cpu) {
                (Some(i), true) => CoProcess(i.unwrap_gpu()),
                (Some(gpu), false) => type3::Track::on_device(gpu),
                (None, true) => type3::Track::Cpu,
                _ => panic!("nothing can run on no device"),
            }
        };

        match self {
            Sliceable(sn) => match sn {
                RotateIdx(..) => err_no_track("RotateIdx"),
                _ => track(),
            },
            LastSliceable(..) => track(),
            Constant(..) => Cpu,
            Subgraph(..) => err_no_track("Subgraph"),
            Extend(..) => track(),
            PolyPermute(..) => track(),
            Entry(..) => Cpu,
            Return(..) => MemoryManagement,
            Blind(..) => Cpu,
            ScanMul { .. } => track(),
            Ntt { .. } => track(),
            Slice(..) => err_no_track("Slice"),
            Interpolate { .. } => Cpu,
            Array(..) => err_no_track("Array"),
            AssmblePoly(..) => Cpu,
            HashTranscript { .. } => Cpu,
            SqueezeScalar(..) => Cpu,
            TupleGet(..) => err_no_track("TupleGet"),
            ArrayGet(..) => err_no_track("ArrayGet"),
            UserFunction(..) => Cpu,
            KateDivision(..) => track(),
            IndexPoly(..) => track(),
            AssertEq(..) => Cpu,
            Print(..) => Cpu,
        }
    }
}

pub mod labelling;
pub mod pretty;
