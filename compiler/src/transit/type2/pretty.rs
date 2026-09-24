use super::template::labelling::LabelT;
use super::template::pretty;
use crate::transit::{self, type2, SourceInfo};
use cytoscape_visualizer::{self as vis, pretty::*};
use std::fmt::{Debug, Display};
use std::io::Write;
use std::ops::DerefMut;
use zkpoly_common::digraph::internal::Digraph;
use zkpoly_common::heap::Heap;
use zkpoly_runtime::args::RuntimeType;

pub mod partial_typed {
    use super::super::{partial_typed, VertexId};
    use super::*;

    struct StylePartialTyped;

    impl<'s, T> pretty::Style<VertexId, partial_typed::Vertex<'s, Option<T>>> for StylePartialTyped
    where
        T: Debug,
    {
        fn style_vertex(
            &self,
            vid: VertexId,
            v: &partial_typed::Vertex<'s, Option<T>>,
        ) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!("{}: {}", &vid, LabelT(v.node()))).with_optional_info(
                v.typ()
                    .as_ref()
                    .map(|t| format!("{:?}\\n@{:?}", t, v.src())),
            )
        }
    }

    pub fn debug_graph<'s, T>(
        g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<T>>>,
        output: impl Write,
    ) -> std::io::Result<()>
    where
        T: Debug + Clone,
    {
        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &StylePartialTyped, builder.deref_mut());
        builder.emit("Debug Type Error", output)
    }
}

pub mod type_error {

    use super::super::{partial_typed, VertexId};
    use super::*;

    struct ErrorVertex(VertexId);

    impl<'s, T> pretty::Style<VertexId, partial_typed::Vertex<'s, Option<T>>> for ErrorVertex
    where
        T: Debug,
    {
        fn style_vertex(
            &self,
            vid: VertexId,
            v: &partial_typed::Vertex<'s, Option<T>>,
        ) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!("{}: {}", &vid, LabelT(v.node())))
                .with_color(if vid == self.0 {
                    vis::Color::red()
                } else {
                    vis::Color::white()
                })
                .with_optional_info(
                    v.typ()
                        .as_ref()
                        .map(|t| format!("{:?}\\n@{:?}", t, v.src())),
                )
        }
    }

    pub fn debug_graph<'s, T>(
        g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<T>>>,
        error_at: VertexId,
        output: impl Write,
    ) -> std::io::Result<()>
    where
        T: Debug + Clone,
    {
        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &ErrorVertex(error_at), builder.deref_mut());
        builder.emit("Debug Type Error", output)
    }
}

struct DebugCycle<'c>(&'c Heap<super::VertexId, bool>);

impl<'s, 'c, T, N> pretty::Style<super::VertexId, transit::Vertex<N, T, SourceInfo<'s>>>
    for DebugCycle<'c>
where
    T: Debug,
    for<'a> LabelT<'a, N>: Display,
{
    fn style_vertex(
        &self,
        vid: super::VertexId,
        v: &transit::Vertex<N, T, SourceInfo<'s>>,
    ) -> cytoscape_visualizer::Vertex {
        vis::Vertex::new(format!("{}: {}", &vid, LabelT(v.node())))
            .with_color(if self.0[vid] {
                vis::Color::red()
            } else {
                vis::Color::white()
            })
            .with_info(format!("{:?}\\n@{:?}", v.typ(), v.src()))
    }
}

pub mod unsubgraphed_cg {
    use super::super::{no_subgraph::Cg, VertexId};
    use super::*;

    pub fn debug_cycle<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        cycle: &Heap<VertexId, bool>,
        output: impl Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty_with_first_shown(
            &g.g,
            &pretty::WithStyle2Default(DebugCycle(cycle)),
            &mut builder,
            std::iter::once(
                cycle
                    .iter_with_id()
                    .filter(|(_, s)| **s)
                    .map(|(v, _)| v)
                    .next()
                    .unwrap(),
            ),
        );
        builder.emit("Debug Non-DAG", output)
    }

    pub fn debug_graph<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        output: impl Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty(&g.g, &pretty::DefaultVertexStyle, builder.deref_mut());
        builder.emit("Debug CG", output)
    }
}

pub mod subgraphed_cg {
    use std::collections::BTreeMap;

    use zkpoly_common::heap::UsizeId;

    use crate::transit::type2::template::pretty::Style;

    use super::super::{
        graph_scheduling::{Schedule, SubgraphsSchedule, WholeSchedule},
        object_analysis::{DefUse, Intermediates},
        sliceable_subgraph, VertexId, {Cg, CgSubgraph, Vertex},
    };
    use super::*;
    use std::fmt::Write;
    use std::io;

    pub fn debug_cycle<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        cycle: &Heap<VertexId, bool>,
        output: impl io::Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty_with_first_shown(
            &g.g,
            &pretty::WithStyle2Default(DebugCycle(cycle)),
            &mut builder,
            cycle.iter_with_id().filter(|(_, s)| **s).map(|(v, _)| v),
        );
        builder.emit("Debug Non-DAG", output)
    }

    pub fn debug_graph<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        output: impl io::Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty(
            &g.g,
            &pretty::WithStyle2Default(pretty::DefaultVertexStyle),
            builder.deref_mut(),
        );
        builder.emit("Debug CG", output)
    }

    struct WholeScheduleStyle<'sh>(&'sh WholeSchedule);
    struct SubgraphsScheduleStyle<'sh>(&'sh SubgraphsSchedule);
    struct ScheduleStyle<'sh, I: Eq + Ord>(&'sh Schedule<I>);

    struct OptionalSequenceNumber(Option<usize>);
    impl Display for OptionalSequenceNumber {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if let Some(seq) = self.0 {
                write!(f, "[{}]", seq)
            } else {
                Ok(())
            }
        }
    }

    impl<'s, 'sh, I, N, T> pretty::Style<I, transit::Vertex<N, T, SourceInfo<'s>>>
        for ScheduleStyle<'sh, I>
    where
        I: UsizeId + Display,
        T: Debug,
        for<'a> LabelT<'a, N>: Display,
    {
        fn style_vertex(
            &self,
            vid: I,
            v: &transit::Vertex<N, T, SourceInfo<'s>>,
        ) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!(
                "{} {}: {}",
                OptionalSequenceNumber(self.0.seq_inv(vid)),
                &vid,
                LabelT(v.node())
            ))
            .with_info(format!("{:?}\\n@{:?}", v.typ(), v.src()))
        }
    }

    impl<'s, 'sh, Rt: RuntimeType> pretty::Style<VertexId, Vertex<'s, Rt>> for WholeScheduleStyle<'sh> {
        fn style_vertex(&self, vid: VertexId, v: &Vertex<'s, Rt>) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!(
                "{} {}: {}",
                OptionalSequenceNumber(self.0.cg_schedule.seq_inv(vid)),
                &vid,
                LabelT(v.node())
            ))
            .with_info(format!("{:?}\\n@{:?}", v.typ(), v.src()))
        }
    }

    impl<'s, 'sh, Rt: RuntimeType>
        pretty::Style2<
            VertexId,
            sliceable_subgraph::VertexId,
            Vertex<'s, Rt>,
            sliceable_subgraph::Vertex<'s, Rt>,
        > for WholeScheduleStyle<'sh>
    {
        type SubStyle =
            pretty::WithDefaultStyleFromOuterEdge<ScheduleStyle<'sh, sliceable_subgraph::VertexId>>;

        fn sub(&self, vid: VertexId) -> Self::SubStyle {
            pretty::WithDefaultStyleFromOuterEdge(ScheduleStyle(
                self.0.subgraphs_schedule.for_subgraph(vid),
            ))
        }
    }

    pub fn debug_schedule<'g, 's, Rt: RuntimeType>(
        g: &CgSubgraph<'g, 's, Rt>,
        schedule: &WholeSchedule,
        output: impl io::Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &WholeScheduleStyle(schedule), builder.deref_mut());
        builder.emit("Debug CG SChedule", output)
    }

    impl<'s, 'sh, Rt: RuntimeType> pretty::Style<VertexId, Vertex<'s, Rt>>
        for SubgraphsScheduleStyle<'sh>
    {
        fn style_vertex(&self, vid: VertexId, v: &Vertex<'s, Rt>) -> cytoscape_visualizer::Vertex {
            vis::Vertex::new(format!(" {}: {}", &vid, LabelT(v.node()))).with_info(format!(
                "{:?}\\n@{:?}",
                v.typ(),
                v.src()
            ))
        }
    }

    impl<'s, 'sh, Rt: RuntimeType>
        pretty::Style2<
            VertexId,
            sliceable_subgraph::VertexId,
            Vertex<'s, Rt>,
            sliceable_subgraph::Vertex<'s, Rt>,
        > for SubgraphsScheduleStyle<'sh>
    {
        type SubStyle =
            pretty::WithDefaultStyleFromOuterEdge<ScheduleStyle<'sh, sliceable_subgraph::VertexId>>;

        fn sub(&self, vid: VertexId) -> Self::SubStyle {
            pretty::WithDefaultStyleFromOuterEdge(ScheduleStyle(self.0.for_subgraph(vid)))
        }
    }

    pub fn debug_subgraphs_schedule<'g, 's, Rt: RuntimeType>(
        g: &CgSubgraph<'g, 's, Rt>,
        schedule: &SubgraphsSchedule,
        output: impl io::Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &SubgraphsScheduleStyle(schedule), builder.deref_mut());
        builder.emit("Debug CG SChedule", output)
    }

    #[derive(Clone, Copy)]
    struct ObjectAnalysisStyle<'an>(
        &'an DefUse,
        &'an BTreeMap<VertexId, type2::DeviceCollection>,
    );
    struct SubgraphObjectAnalysisStyle<'an>(ObjectAnalysisStyle<'an>, VertexId);

    fn unwrap_err(f: impl FnOnce() -> std::fmt::Result) {
        f().unwrap()
    }

    impl<'s, 'an, Rt: RuntimeType>
        pretty::Style<sliceable_subgraph::VertexId, sliceable_subgraph::Vertex<'s, Rt>>
        for SubgraphObjectAnalysisStyle<'an>
    {
        fn style_vertex(
            &self,
            vid: sliceable_subgraph::VertexId,
            v: &sliceable_subgraph::Vertex<'s, Rt>,
        ) -> cytoscape_visualizer::Vertex {
            let mut info = String::new();
            let def_use = self.0 .0.subgraph_analysis(self.1);

            unwrap_err(|| {
                writeln!(&mut info, "{:?}\\n@{:?}", v.typ(), v.src())?;

                writeln!(&mut info, "Prologue Def =")?;
                writeln!(&mut info, "{:?}", def_use.prologue_def_use.value(vid))?;
                writeln!(&mut info, "Prologue Slice Out =")?;
                writeln!(
                    &mut info,
                    "{:?}",
                    def_use.prologue_itms.slice_outs(vid).collect::<Vec<_>>()
                )?;

                writeln!(&mut info, "Body Def =")?;
                writeln!(&mut info, "{:?}", def_use.body_def_use.value(vid))?;
                writeln!(&mut info, "Body Slice Out =")?;
                writeln!(
                    &mut info,
                    "{:?}",
                    def_use.body_itms.slice_outs(vid).collect::<Vec<_>>()
                )?;
                writeln!(&mut info, "Body Concat In =")?;
                writeln!(
                    &mut info,
                    "{:?}",
                    def_use.body_itms.concat_ins(vid).collect::<Vec<_>>()
                )?;

                Ok(())
            });

            vis::Vertex::new(format!(" {}: {}", &vid, LabelT(v.node()))).with_info(info)
        }

        fn style_edge(
            &self,
            from: sliceable_subgraph::VertexId,
            to: sliceable_subgraph::VertexId,
            label: Option<String>,
        ) -> cytoscape_visualizer::EdgeStyle {
            let mut info = String::new();
            let def_use = self.0 .0.subgraph_analysis(self.1);

            unwrap_err(|| {
                writeln!(&mut info, "Prologue Input = ")?;
                writeln!(
                    &mut info,
                    "{:?}",
                    def_use.prologue_def_use.input_from(to, from)
                )?;

                writeln!(&mut info, "Body Input = ")?;
                writeln!(&mut info, "{:?}", def_use.body_def_use.input_from(to, from))?;

                Ok(())
            });

            vis::EdgeStyle::default()
                .with_optional_label(label)
                .with_info(info)
        }
    }

    impl<'s, 'an, Rt: RuntimeType>
        pretty::StyleInner<
            sliceable_subgraph::VertexId,
            VertexId,
            sliceable_subgraph::Vertex<'s, Rt>,
        > for SubgraphObjectAnalysisStyle<'an>
    {
        fn style_from_outer_edge(&self, from: VertexId) -> cytoscape_visualizer::EdgeStyle {
            <ObjectAnalysisStyle as Style<_, Vertex<'s, Rt>>>::style_edge(
                &self.0, from, self.1, None,
            )
        }
    }

    impl<'s, 'an, Rt: RuntimeType> pretty::Style<VertexId, Vertex<'s, Rt>>
        for ObjectAnalysisStyle<'an>
    {
        fn style_vertex(&self, vid: VertexId, v: &Vertex<'s, Rt>) -> cytoscape_visualizer::Vertex {
            let mut info = String::new();

            unwrap_err(|| {
                writeln!(&mut info, "{:?}\\n@{:?}", v.typ(), v.src())?;

                writeln!(&mut info, "{:?}", self.0.value(vid))?;

                writeln!(&mut info, "Device = {:?}", &self.1[&vid])?;

                if let type2::VertexNode::Subgraph(..) = v.node() {
                    let subgraph_analysis = self.0.subgraph_analysis(vid);

                    writeln!(&mut info, "Chunk =")?;
                    writeln!(&mut info, "{:?}", &subgraph_analysis.chunker)?;

                    writeln!(&mut info, "Body Write Backs =")?;
                    writeln!(
                        &mut info,
                        "{:?}",
                        &subgraph_analysis
                            .body_itms
                            .write_backs()
                            .collect::<Vec<_>>()
                    )?;
                }
                Ok(())
            });

            vis::Vertex::new(format!(" {}: {}", &vid, LabelT(v.node()))).with_info(info)
        }

        fn style_edge(
            &self,
            from: VertexId,
            to: VertexId,
            label: Option<String>,
        ) -> cytoscape_visualizer::EdgeStyle {
            let mut info = String::new();

            let iv = self.0.input_from(to, from);

            unwrap_err(|| {
                writeln!(&mut info, "{:?}", iv)?;

                let relevant_cloned_slices = iv
                    .iter()
                    .filter_map(|oa| {
                        Some((oa.object(), self.0.object_is_cloned_slice(*oa.object())?))
                    })
                    .collect::<Vec<_>>();
                if relevant_cloned_slices.len() != 0 {
                    writeln!(&mut info, "Relevant Cloned Slices =")?;
                    for (o, (from, s)) in relevant_cloned_slices {
                        writeln!(&mut info, "{:?} from {:?}:{:?}", o, from, s)?;
                    }
                }

                Ok(())
            });

            vis::EdgeStyle::default()
                .with_optional_label(label)
                .with_info(info)
        }
    }

    impl<'s, 'an, Rt: RuntimeType>
        pretty::Style2<
            VertexId,
            sliceable_subgraph::VertexId,
            Vertex<'s, Rt>,
            sliceable_subgraph::Vertex<'s, Rt>,
        > for ObjectAnalysisStyle<'an>
    {
        type SubStyle = SubgraphObjectAnalysisStyle<'an>;

        fn sub(&self, vid: VertexId) -> Self::SubStyle {
            SubgraphObjectAnalysisStyle(*self, vid)
        }
    }

    pub fn debug_object_analysis<'g, 's, Rt: RuntimeType>(
        g: &CgSubgraph<'g, 's, Rt>,
        analysis: &DefUse,
        execution_devices: &BTreeMap<VertexId, type2::DeviceCollection>,
        output: impl io::Write,
    ) -> std::io::Result<()> {
        let mut builder = vis::DigraphBuilder::new();
        pretty(
            g,
            &ObjectAnalysisStyle(analysis, execution_devices),
            builder.deref_mut(),
        );
        builder.emit("Debug CG SChedule", output)
    }
}
