use std::mem::size_of;
use zkpoly_common::typ::PolyType;
use zkpoly_runtime::args::RuntimeType;

pub mod template {
    use zkpoly_common::typ::AnyTypeId;
    use zkpoly_runtime::args::Variable;

    use super::RuntimeType;
    use std::{fmt::Debug, marker::PhantomData};

    #[derive(Debug, Clone, Hash, serde::Serialize, serde::Deserialize)]
    pub enum Typ<Rt: RuntimeType, P> {
        Poly(P),
        PointBase { log_n: u32 },
        Scalar,
        Transcript,
        Point,
        Tuple(Vec<Typ<Rt, P>>),
        Array(Box<Typ<Rt, P>>, usize),
        Any(AnyTypeId, usize),
        _Phantom(PhantomData<Rt>),
    }

    impl<Rt: RuntimeType, P> PartialEq for Typ<Rt, P>
    where
        P: Eq,
    {
        fn eq(&self, other: &Self) -> bool {
            use Typ::*;
            match (self, other) {
                (Poly(p1), Poly(p2)) => p1 == p2,
                (PointBase { log_n: l1 }, PointBase { log_n: l2 }) => l1 == l2,
                (Scalar, Scalar) => true,
                (Transcript, Transcript) => true,
                (Point, Point) => true,
                (Tuple(ts1), Tuple(ts2)) => ts1 == ts2,
                (Array(t1, l1), Array(t2, l2)) => t1 == t2 && l1 == l2,
                (Any(t1, s1), Any(t2, s2)) => t1 == t2 && s1 == s2,
                _ => false,
            }
        }
    }

    impl<Rt: RuntimeType, P> Eq for Typ<Rt, P> where P: Eq {}

    impl<Rt: RuntimeType, P: Debug> Typ<Rt, P> {
        pub fn match_arg(&self, other: &Variable<Rt>) {
            use Variable::*;
            match (self, other) {
                (Typ::Scalar, Scalar(..)) => {}
                (Typ::Poly(..), ScalarArray(..)) => {}
                (Typ::PointBase { .. }, PointArray(..)) => {}
                (Typ::Transcript, Transcript(..)) => {}
                (Typ::Point, Point(..)) => {}
                (Typ::Any(..), Any(..)) => {}
                (Typ::Tuple(ts), Tuple(vs)) => {
                    if ts.len() != vs.len() {
                        panic!("tuple length mismatch");
                    }
                    for (t, v) in ts.iter().zip(vs.iter()) {
                        t.match_arg(v);
                    }
                }
                (Typ::Array(t, len), Tuple(vs)) => {
                    if *len != vs.len() {
                        panic!("array length mismatch");
                    }
                    for v in vs.iter() {
                        t.match_arg(v);
                    }
                }
                _ => panic!("expected {:?}, got {:?}", self, other),
            }
        }
    }

    impl<Rt: RuntimeType, P> Typ<Rt, P> {
        pub fn unwrap_poly(&self) -> &P {
            use Typ::*;
            match self {
                Poly(p) => p,
                _ => panic!("called unwrap_poly on non-poly type"),
            }
        }

        pub fn try_unwrap_poly(&self) -> Option<&P> {
            use Typ::*;
            match self {
                Poly(p) => Some(p),
                _ => None,
            }
        }

        pub fn try_unwrap_point_base(&self) -> Option<u32> {
            use Typ::*;
            match self {
                PointBase { log_n } => Some(*log_n),
                _ => None,
            }
        }

        pub fn hashable(&self) -> bool {
            use Typ::*;
            match self {
                Scalar | Poly(..) | Point => true,
                _ => false,
            }
        }

        pub fn try_unwrap_tuple(&self) -> Option<&Vec<Typ<Rt, P>>> {
            use Typ::*;
            match self {
                Tuple(ts) => Some(ts),
                _ => None,
            }
        }

        pub fn try_unwrap_array(&self) -> Option<(&Typ<Rt, P>, usize)> {
            use Typ::*;
            match self {
                Array(t, l) => Some((t.as_ref(), *l)),
                _ => None,
            }
        }

        pub fn unwrap_array(&self) -> (&Typ<Rt, P>, usize) {
            use Typ::*;
            match self {
                Array(t, l) => (t.as_ref(), *l),
                _ => panic!("called unwrap_array on non-array type"),
            }
        }

        fn map_p_<P1>(&self, f: &mut impl FnMut(&P) -> P1) -> Typ<Rt, P1> {
            use Typ::*;
            match self {
                Poly(p) => Poly(f(p)),
                PointBase { log_n } => PointBase { log_n: *log_n },
                Scalar => Scalar,
                Transcript => Transcript,
                Tuple(elems) => Tuple(elems.iter().map(|e| e.map_p_(f)).collect()),
                Point => Point,
                Array(t, len) => Array(Box::new(t.map_p_(f)), *len),
                Any(id, size) => Any(*id, *size),
                _Phantom(_) => _Phantom(PhantomData),
            }
        }

        pub fn map_p<P1>(&self, mut f: impl FnMut(&P) -> P1) -> Typ<Rt, P1> {
            self.map_p_(&mut f)
        }

        pub fn iter<'t>(&'t self) -> TypIter<'t, Rt, P> {
            TypIter { stack: vec![self] }
        }
    }

    pub struct TypIter<'t, Rt: RuntimeType, P> {
        pub(super) stack: Vec<&'t Typ<Rt, P>>,
    }

    impl<'t, Rt: RuntimeType, P> Iterator for TypIter<'t, Rt, P> {
        type Item = &'t Typ<Rt, P>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let top = self.stack.pop()?;
                match top {
                    Typ::Tuple(xs) => xs.iter().for_each(|x| self.stack.push(x)),
                    Typ::Array(x, n) => (0..*n).for_each(|_| self.stack.push(x.as_ref())),
                    _ => return Some(top),
                }
            }
        }
    }
}

pub type Typ<Rt: RuntimeType> = template::Typ<Rt, (PolyType, u64)>;

impl<Rt: RuntimeType> Typ<Rt> {
    pub fn size(&self) -> u64 {
        use template::Typ::*;
        match self {
            Poly((_, deg)) => *deg * size_of::<Rt::Field>() as u64,
            PointBase { log_n } => (1 << log_n) * 2 * size_of::<Rt::Field>() as u64,
            Scalar => size_of::<Rt::Field>() as u64,
            Transcript => size_of::<Rt::Trans>() as u64,
            Point => 2 * size_of::<Rt::Field>() as u64,
            Tuple(ts) => ts.iter().map(|t| t.size()).sum(),
            Array(t, len) => t.size() * (*len as u64),
            Any(_, size) => *size as u64,
            _Phantom(_) => unreachable!(),
        }
    }

    pub fn lagrange(deg: u64) -> Self {
        Self::Poly((PolyType::Lagrange, deg))
    }

    pub fn coef(deg: u64) -> Self {
        Self::Poly((PolyType::Coef, deg))
    }

    pub fn is_poly_pow2(&self) -> bool {
        use template::Typ::*;
        match self {
            Poly((_, deg)) => deg.is_power_of_two(),
            _ => true,
        }
    }
}
