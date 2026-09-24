use crate::driver::Config;

use super::{processed_type3::ProcessedType3, type3, HardwareInfo};
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use super::{type2, DebugOptions, Error, Versions};

pub struct Version<'s, Rt: RuntimeType> {
    pub chunk: type3::Chunk<'s, Rt>,
    pub memory_statistics: type2::memory_planning::Statistics,
}

/// The fresh Type3 IR after memory planning from Type2 IR.
pub struct FreshType3<'s, 'i, Rt: RuntimeType> {
    pub(super) versions: Versions<Version<'s, Rt>>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) constants_device: Heap<ConstantId, type3::Device>,
    pub(super) libs: Libs,
    pub(super) options: &'i DebugOptions,
    pub(super) hardware_info: &'i HardwareInfo,
    pub(super) config: &'i Config,
}

impl<'s, 'i, Rt: RuntimeType> FreshType3<'s, 'i, Rt> {
    /// Apply passes to obtain processed Type3 IR.
    pub fn apply_passes<'a>(self) -> Result<ProcessedType3<'s, 'i, Rt>, Error<'s, Rt>> {
        Ok(ProcessedType3 {
            versions: self.versions,
            uf_table: self.uf_table,
            constant_table: self.constant_table,
            constants_device: self.constants_device,
            libs: self.libs,
            options: self.options,
            hardware_info: self.hardware_info,
            config: self.config,
        })
    }
}
