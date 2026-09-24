use std::{collections::BTreeMap, ops::AddAssign, path::Path};

use super::Log;
use crate::{
    debug::TimedItem,
    functions::FuncMeta,
    instructions::{Instruction, InstructionNode, Track},
};
use plotters::{
    prelude::{self as plt, IntoSegmentedCoord},
    style::Color,
};
use zkpoly_common::devices::DeviceType;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Arith,
    MemoryManagement,
    SubgraphLoad(Track),
    SubgraphStore(Track),
    Transfer { from: DeviceType, to: DeviceType },
    F(String),
    Blind,
    Record,
    Fork,
    Wait,
    Other,
}

impl std::fmt::Debug for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Category::*;
        match self {
            Arith => write!(f, "Arith"),
            MemoryManagement => write!(f, "MemoryManagement"),
            SubgraphLoad(t) => write!(f, "SubgraphLoad({:?})", t),
            SubgraphStore(t) => write!(f, "SubgraphStore({:?})", t),
            Transfer { from, to } => {
                let device_str = |d: &DeviceType| -> String {
                    use DeviceType::*;
                    match d {
                        CPU => "CPU".to_string(),
                        GPU { device_id } => format!("GPU({device_id})"),
                        Disk => "Disk".to_string(),
                    }
                };
                write!(f, "{}->{}", device_str(from), device_str(to))
            }
            Blind => write!(f, "Blind"),
            Wait => write!(f, "Wait"),
            F(s) => write!(f, "{s}"),
            Record => write!(f, "Record"),
            Fork => write!(f, "Fork"),
            Other => write!(f, "Other"),
        }
    }
}

impl Category {
    pub fn of(item: &TimedItem, function_meta: &Option<FuncMeta>) -> Self {
        use TimedItem::*;
        match item {
            Inst(inst) => Self::of_inst(inst, function_meta),
            SubgraphLoad(t, ..) => Self::SubgraphLoad(t.clone()),
            SubgraphStore(t, ..) => Self::SubgraphStore(t.clone()),
        }
    }
    pub fn of_inst(inst: &Instruction, function_meta: &Option<FuncMeta>) -> Self {
        use InstructionNode::*;
        match inst.node() {
            FuncCall { .. } => {
                let fname = &function_meta.as_ref().unwrap().name;
                if fname.starts_with("fused_arith") {
                    Self::Arith
                } else {
                    Self::F(fname.clone())
                }
            }
            Allocate { .. } => Self::MemoryManagement,
            Deallocate { .. } => Self::MemoryManagement,
            Transfer {
                src_device,
                dst_device,
                ..
            } => Self::Transfer {
                from: src_device.clone(),
                to: dst_device.clone(),
            },
            Blind { .. } => Self::Blind,
            Wait { .. } => Self::Wait,
            Fork { .. } => Self::Fork,
            Record { .. } => Self::Record,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Statistics {
    categories: BTreeMap<Category, f64>,
}

impl Log {
    pub fn statistics(&self) -> Statistics {
        Statistics::from_log(self)
    }
}

impl Statistics {
    fn from_log(log: &Log) -> Statistics {
        if log.is_empty() {
            panic!("no log recorded");
        }

        let insts = log
            .executed_instructions
            .iter()
            .map(|ie| {
                (
                    Category::of(&ie.item, &ie.function_meta),
                    ie.duration_nanos() as f64 / 1000_000.0,
                )
            })
            .collect::<Vec<_>>();
        let categories = insts.iter().fold(BTreeMap::new(), |mut acc, (c, v)| {
            acc.entry(c.clone()).or_insert(0.0).add_assign(v);
            acc
        });

        Self { categories }
    }

    pub fn plot_percentage<DB: plt::DrawingBackend>(
        &self,
        cb: &mut plt::ChartBuilder<'_, '_, DB>,
        top_k: usize,
    ) -> Result<(), plt::DrawingAreaErrorKind<DB::ErrorType>> {
        let mut categories = self
            .categories
            .iter()
            .map(|(a, b)| (a.clone(), *b))
            .collect::<Vec<_>>();
        categories.sort_by(|(_, lhs), (_, rhs)| f64::total_cmp(lhs, rhs).reverse());
        categories.truncate(top_k);

        let x_range = 0.0..(categories
            .iter()
            .fold(f64::NAN, |max, (_, val)| val.max(max))
            * 1.1);

        let c = categories
            .iter()
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>();

        let mut cc = cb.build_cartesian_2d(x_range, c.into_segmented())?;

        cc.configure_mesh()
            .light_line_style(&plt::WHITE)
            .x_desc("Category")
            .y_desc("Accumulated Time")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        cc.draw_series(
            plt::Histogram::horizontal(&cc)
                .style(plt::BLUE.filled())
                .margin(20)
                .data(categories.iter().map(|(k, v)| (k, *v))),
        )?;

        Ok(())
    }

    pub fn plot_percentage_to_file<'a, T: AsRef<Path> + ?Sized>(
        &self,
        file: &'a T,
        top_k: usize,
    ) -> Result<
        (),
        plt::DrawingAreaErrorKind<<plt::SVGBackend<'a> as plt::DrawingBackend>::ErrorType>,
    > {
        let root = plt::SVGBackend::new(file, (512, 1024));
        let root = plt::DrawingArea::from(root);
        let mut cb = plt::ChartBuilder::on(&root);

        cb.x_label_area_size(35)
            .y_label_area_size(120)
            .margin(10)
            .caption("Runtime Statistics", ("sans-serif", 15.0));

        self.plot_percentage(&mut cb, top_k)?;

        root.present()
    }

    pub fn export(&self) -> Vec<(String, f64)> {
        self.categories
            .iter()
            .map(|(c, t)| (format!("{:?}", c), *t))
            .collect()
    }
}
