use crate::transit::type2::memory_planning::prelude::*;

pub mod constant_wrapper;
pub mod linear_allocator;
pub mod page_allocator;
pub mod slab_allocator;
pub mod slab_pool;
pub mod smithereens_pool;
pub mod smithereens_wrapper;
pub mod super_allocator;

pub use constant_wrapper::Wrapper as ConstantWrapper;
pub use linear_allocator::LinearAllocator;
pub use page_allocator::PageAllocator;
pub use slab_allocator::SlabAllocator;
pub use smithereens_wrapper::Wrapper as SmithereenWrapper;
pub use super_allocator::SuperAllocator;

const SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD: u64 = 2u64.pow(16);
const LOG_MIN_INTEGRAL_SIZE: u32 = 10;

pub fn normalize_size(size: Size) -> Size {
    match size {
        Size::Integral(is) => {
            if is.0 < LOG_MIN_INTEGRAL_SIZE {
                Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
            } else {
                Size::Integral(is)
            }
        }
        Size::Smithereen(ss) => {
            if let Ok(is) = IntegralSize::try_from(ss) {
                if is.0 < LOG_MIN_INTEGRAL_SIZE {
                    Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
                } else {
                    Size::Integral(is)
                }
            } else if ss.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD {
                Size::Integral(IntegralSize::ceiling(ss))
            } else {
                Size::Smithereen(ss)
            }
        }
    }
}

use object_analysis::value::{ObjectId, TopNode};

/// In ascending order.
pub fn collect_integral_sizes<'a, Rt: RuntimeType>(
    objects: impl Iterator<Item = (ObjectId, &'a TopNode)>,
) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeMap::<IntegralSize, ObjectId>::new();

    for (oid, node) in objects {
        if let Size::Integral(is) = normalize_size(Size::Smithereen(SmithereenSize(
            node.size::<Rt::Field, Rt::PointAffine>() as u64,
        ))) {
            integral_sizes.insert(is, oid);
        }
    }

    println!("LBSS is {:?}", &integral_sizes);

    integral_sizes.into_iter().map(|(b, _)| b).collect()
}
