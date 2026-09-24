use super::prelude::*;
use zkpoly_common::digraph::internal::Adjacents;

pub trait DefUseDag<I> {
    /// The vertices where used objects of `i` are defined
    fn predecessors(&self, i: I) -> impl Iterator<Item = I>;
    /// The vertices where defined objects of `i` are used
    fn successors(&self, i: I) -> impl Iterator<Item = I>;
    /// The total size of defined objects of vertex `i`
    fn def_size(&self, i: I) -> u64;
    /// Whether vertex `i` actually defines anything
    fn valid(&self, i: I) -> bool;
}

pub struct ObjectDag<'d, 'g, 's, Rt: RuntimeType> {
    du: &'d DefUse,
    /// Not every object must be used
    used_at: BTreeMap<ObjectId, Vec<VertexId>>,
    objects: &'d ObjectDefs,
    cg: &'d type2::CgSubgraph<'g, 's, Rt>,
}

impl<'d, 'g, 's, Rt: RuntimeType> ObjectDag<'d, 'g, 's, Rt> {
    pub fn new(
        du: &'d DefUse,
        objects: &'d ObjectDefs,
        cg: &'d type2::CgSubgraph<'g, 's, Rt>,
    ) -> Self {
        let mut used_at: BTreeMap<ObjectId, Vec<VertexId>> = BTreeMap::new();

        for vid in cg.vertices().filter(|v| !cg.vertex(*v).is_virtual()) {
            du.input_of(vid).for_each(|(_, iv)| {
                iv.iter().for_each(|ia| {
                    if let Some((sliced_obj, _)) = du.object_is_cloned_slice(*ia.object()) {
                        used_at.entry(sliced_obj).or_default().push(vid);
                    } else {
                        used_at.entry(*ia.object()).or_default().push(vid);
                    }
                })
            })
        }

        Self {
            du,
            objects,
            cg,
            used_at,
        }
    }
}

impl<'d, 'g, 's, Rt: RuntimeType> DefUseDag<VertexId> for ObjectDag<'d, 'g, 's, Rt> {
    fn predecessors(&self, i: VertexId) -> impl Iterator<Item = VertexId> {
        let objs = self
            .du
            .input_of(i)
            .flat_map(|(_, iv)| iv.iter().map(|ia| *ia.object()))
            // We are ignoring cloned slices here, by viewing them as internal operations to vertex `i`
            .map(|oid| {
                self.du
                    .object_is_cloned_slice(oid)
                    .map_or(oid, |(oid, _)| oid)
            });
        let vids: BTreeSet<VertexId> = objs
            .map(|oid| {
                let vid = self.du.def_at(oid);
                assert!(
                    !self.cg.vertex(vid).is_virtual(),
                    "{:?} should not be defined at a virtual vertex {:?}",
                    oid,
                    vid
                );
                vid
            })
            .collect();
        vids.into_iter()
    }

    fn successors(&self, i: VertexId) -> impl Iterator<Item = VertexId> {
        use type2::template::VertexNode;
        let iter: Box<dyn Iterator<Item = _>> = match self.cg.vertex(i).node() {
            VertexNode::Print(..) | VertexNode::AssertEq(..) => Box::new(std::iter::empty()),
            _ => {
                let objs = self.du.value(i).iter().map(|ia| *ia.object());
                let vids: BTreeSet<VertexId> = objs
                    .filter_map(|oid| self.used_at.get(&oid).map(|vs| vs.iter().copied()))
                    .flatten()
                    .collect();
                Box::new(vids.into_iter())
            }
        };
        iter
    }

    fn def_size(&self, i: VertexId) -> u64 {
        self.du
            .value(i)
            .iter()
            .map(|ia| *ia.object())
            .filter(|oid| self.du.is_whole_object_immortal(*oid))
            .map(|oid| self.objects[oid].size::<Rt::Field, Rt::PointAffine>() as u64)
            .sum()
    }

    fn valid(&self, i: VertexId) -> bool {
        !self.cg.vertex(i).is_virtual()
    }
}

pub struct SubgraphObjectDag<'d, 's, Rt: RuntimeType> {
    cg: &'d sliceable_subgraph::Cg<'s, Rt>,
    successors: Adjacents<sliceable_subgraph::VertexId>,
    predecessors: Adjacents<sliceable_subgraph::VertexId>,
}

impl<'d, 's, Rt: RuntimeType> SubgraphObjectDag<'d, 's, Rt> {
    pub fn new(cg: &'d sliceable_subgraph::Cg<'s, Rt>) -> Self {
        Self {
            cg,
            successors: cg.g.successors(),
            predecessors: cg.g.predecessors_no_multiedge(),
        }
    }
}

impl<'d, 's, Rt: RuntimeType> DefUseDag<sliceable_subgraph::VertexId>
    for SubgraphObjectDag<'d, 's, Rt>
{
    fn predecessors(
        &self,
        i: sliceable_subgraph::VertexId,
    ) -> impl Iterator<Item = sliceable_subgraph::VertexId> {
        self.predecessors[i].iter().copied()
    }

    fn successors(
        &self,
        i: sliceable_subgraph::VertexId,
    ) -> impl Iterator<Item = sliceable_subgraph::VertexId> {
        self.successors[i].iter().copied()
    }

    fn def_size(&self, i: sliceable_subgraph::VertexId) -> u64 {
        self.cg.g.vertex(i).typ().inner().size()
    }

    fn valid(&self, _i: sliceable_subgraph::VertexId) -> bool {
        true
    }
}
