pub mod value;
pub use value::ObjectDefs;

mod prelude {
    pub use std::collections::{BTreeMap, BTreeSet};
    pub use std::ops::Deref;

    pub use crate::driver::HardwareInfo;
    pub use crate::transit::type2;
    pub use crate::transit::type3;

    pub use super::template::Seq;
    pub use super::{
        lower_typ,
        template::{
            ChunkedNode, ChunkedOp, ResidentalAtom, ResidentalT, SubgraphOperation, Temporaries,
        },
        value::{self, ObjectDefs, ObjectId, TopNode, ValueNode},
    };
    pub(super) use type2::{sliceable_subgraph, template::SliceableNode, ConstantId, SourceInfo};

    pub use zkpoly_common::{
        arith::Mutability, bijection::Bijection, define_usize_id, heap::Heap, load_dynamic::Libs,
        typ::Slice,
    };
    pub use zkpoly_runtime::args::RuntimeType;
}

use prelude::*;
mod subgraph_elaboration;
pub use subgraph_elaboration::slice_analysis::Intermediates;
pub use subgraph_elaboration::{slice_analysis::DefUse as SubgraphDefUse, Chunker, Paragraph};

/// Determine [`ValueNode`] for a freshly defined Type2 type.
/// That is, no slice, no rotation.
pub fn lower_typ<S, Rt: RuntimeType>(
    t2typ: &type2::Typ<Rt>,
    leaf_constructor: &mut impl FnMut(ValueNode) -> S,
) -> value::Tree<S>
where
    S: Clone,
{
    use type2::typ::template::Typ::*;
    let f = leaf_constructor;
    match t2typ {
        Poly((_, deg0)) => value::Tree::Single(f(ValueNode::plain_scalar_array(*deg0 as usize))),
        Scalar => value::Tree::Single(f(ValueNode::Scalar)),
        Point => value::Tree::Single(f(ValueNode::Point)),
        PointBase { log_n } => value::Tree::Single(f(ValueNode::PointBase {
            len: 2usize.pow(*log_n),
        })),
        Transcript => value::Tree::Single(f(ValueNode::Transcript)),
        Tuple(vs) => {
            let vs = vs.iter().map(|x| lower_typ(x, f)).collect();
            value::Tree::Tuple(vs)
        }
        Array(t, len) => value::Tree::Tuple((0..*len).map(|_| lower_typ(t, f)).collect()),
        Any(tid, size) => {
            value::Tree::Single(f(ValueNode::Any(tid.clone().into(), *size as usize)))
        }
        _Phantom(_) => unreachable!(),
    }
}

pub mod template;

pub use template::{Index, Operation, Seq};

pub mod cg_def_use;
pub use cg_def_use::{DefUse, SubgraphAnalysis};
pub mod size;

/// Given an iterator of sizes of buffers, and align the buffer must obey,
/// returns the total size of the merged buffer and the (offset, length)'s of each buffer slice,
/// where the length may be longer than original size.
fn merge_temporary_buffer(sizes: impl Iterator<Item = u64>, align: u64) -> (u64, Vec<Slice>) {
    let mut offset = 0;
    let mut slices = Vec::new();

    let align = |offset: u64| -> u64 { (offset + align - 1) / align * align };

    for size in sizes {
        let next_offset = align(offset + size);
        slices.push(Slice::new(offset, next_offset - offset));

        offset = next_offset;
    }

    (offset, slices)
}

pub mod construct_ops;
pub mod liveness;
