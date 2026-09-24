//! Implements algorithm 3 in Optimal and Heuristic Min-Reg Scheduling Algorithms for GPU Programs

use std::collections::HashSet;
use std::collections::VecDeque;

use super::prelude::*;
use super::{DefUseDag, StatisticsCollector};

pub struct Scheduler<'d, I, D> {
    dag: &'d D,
    /// MaxRP_i described in the paper
    max_rp: BTreeMap<I, u64>,
    /// Q in algorithm 3
    queue: MmHeap<I, u64, u64>,
    /// The schedule begin built, from last to first
    schedule_inv: Vec<I>,
    /// Scheduled vertices
    scheduled: HashSet<I>,
    /// The set of vertices that has lived until now
    lived: HashSet<I>,
    /// Number of unscheduled successors
    succ_count: BTreeMap<I, usize>,
    stc: StatisticsCollector,
}

impl<'d, I, D> Scheduler<'d, I, D>
where
    I: UsizeId,
    D: DefUseDag<I>,
{
    /// Create a scheduler from a DAG describing object def-use and a topology sequence of the CG
    pub fn new(dag: &'d D, topology_seq: impl Iterator<Item = I>) -> Self {
        let mut schd = Scheduler {
            dag,
            max_rp: BTreeMap::new(),
            queue: MmHeap::new(),
            scheduled: HashSet::new(),
            schedule_inv: Vec::new(),
            lived: HashSet::new(),
            succ_count: BTreeMap::new(),
            stc: StatisticsCollector::new(),
        };

        for vid in topology_seq.filter(|i| dag.valid(*i)) {
            struct Pred {
                max_rp: u64,
                def_size: u64,
            }

            let mut pred_seq: Vec<Pred> = dag
                .predecessors(vid)
                .map(|pred_id| Pred {
                    max_rp: *schd
                        .max_rp
                        .get(&pred_id)
                        .unwrap_or_else(|| panic!("no entry found for {:?}", pred_id)),
                    def_size: dag.def_size(pred_id),
                })
                .collect();

            pred_seq.sort_unstable_by_key(|x| x.max_rp as i64 - x.def_size as i64);

            let max_rp_unadjusted = pred_seq
                .iter()
                .fold(0, |acc, pred| pred.max_rp.max(pred.def_size + acc));
            let n_successors = dag.successors(vid).count();
            let max_rp = if n_successors == 0 {
                max_rp_unadjusted
            } else {
                max_rp_unadjusted / n_successors as u64
            };

            schd.max_rp.insert(vid, max_rp);

            if dag.successors(vid).next().is_none() {
                schd.live_if_not(vid);
                schd.queue.insert(vid, max_rp);
            }

            schd.succ_count.insert(vid, n_successors);
        }

        schd
    }

    /// Traverse successors of `i` in BFS order until a vertex is found to be in Q
    fn lookup_blocking(&self, i: I) -> I {
        let mut visited = HashSet::new();
        visited.insert(i);
        let mut stack = VecDeque::from_iter(std::iter::once(i));
        while let Some(i) = stack.pop_front() {
            if self.queue.contains(&i) {
                return i;
            }
            for succ in self.dag.successors(i) {
                if !visited.contains(&succ) {
                    visited.insert(succ);
                    stack.push_back(succ);
                }
            }
        }

        panic!("expect to find a ready but unexecuted successor")
    }

    /// Mark a vertex as living if not already
    fn live_if_not(&mut self, i: I) {
        if self.lived.insert(i) {
            self.stc.live(self.dag.def_size(i))
        }
    }

    fn kill(&mut self, i: I) {
        self.schedule_inv.push(i);
        self.scheduled.insert(i);
        self.stc.kill(self.dag.def_size(i));
    }

    fn step(&mut self) -> Option<()> {
        let (i, _) = self.queue.max()?;
        let i = *i;

        // cluster in algorithm 3
        let cluster = self
            .dag
            .predecessors(i)
            .flat_map(|pred_id| self.dag.successors(pred_id))
            .filter(|i| !self.scheduled.contains(i))
            .collect::<BTreeSet<_>>();

        // W in algorithm 3
        let mut worklist: VecDeque<I> = if let Some(unready_id) = cluster
            .iter()
            .filter(|&ci| !self.queue.contains(&ci))
            .next()
        {
            let blocking = self.lookup_blocking(*unready_id);
            self.queue.remove(&blocking);
            vec![blocking].into()
        } else {
            cluster.iter().for_each(|&i| {
                assert!(
                    self.queue.remove(&i).is_some(),
                    "all vertices in cluster should be ready"
                );
            });
            cluster.into_iter().collect()
        };

        while let Some(vid) = worklist.pop_back() {
            for pred in self.dag.predecessors(vid) {
                self.live_if_not(pred);
            }

            self.kill(vid);
            let d = if self.lived.contains(&vid) {
                self.dag.def_size(vid)
            } else {
                0
            };
            for pred in self.dag.predecessors(vid) {
                *self
                    .succ_count
                    .get_mut(&pred)
                    .expect("should find successor count") -= 1;
                if self.succ_count[&pred] == 0 {
                    let mut d = d as i64;
                    for m in self.dag.predecessors(pred) {
                        if !self.lived.contains(&m) {
                            d -= self.dag.def_size(m) as i64;
                        }
                    }

                    if d >= 0 {
                        worklist.push_back(pred);
                    } else {
                        self.queue.insert(pred, self.max_rp[&pred]);
                    }
                }
            }
        }

        Some(())
    }

    pub fn run(mut self) -> (Vec<I>, u64) {
        while let Some(_) = self.step() {}
        (
            self.schedule_inv.into_iter().rev().collect(),
            self.stc.max_occupied,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkpoly_common::define_usize_id;
    use zkpoly_common::digraph::internal::{Digraph, Predecessors, SubDigraph};
    use zkpoly_common::heap::Heap;

    define_usize_id!(VertexId);

    struct Vertex {
        predecessors: Vec<VertexId>,
        def_size: u64,
    }

    impl Vertex {
        fn new(predecessors: Vec<VertexId>, def_size: u64) -> Self {
            Self {
                predecessors,
                def_size,
            }
        }
    }

    impl Predecessors<VertexId> for Vertex {
        fn predecessors(&self) -> impl Iterator<Item = VertexId> {
            self.predecessors.iter().copied()
        }
    }

    impl<'g> DefUseDag<VertexId> for SubDigraph<'g, VertexId, Vertex> {
        fn predecessors(&self, i: VertexId) -> impl Iterator<Item = VertexId> {
            self.vertex(i).predecessors()
        }

        fn successors(&self, i: VertexId) -> impl Iterator<Item = VertexId> {
            self.successors_of(i)
        }

        fn def_size(&self, i: VertexId) -> u64 {
            self.vertex(i).def_size
        }

        fn valid(&self, _i: VertexId) -> bool {
            true
        }
    }

    fn test_digraph() -> Digraph<VertexId, Vertex> {
        let mut vertices = Heap::new();

        let in1 = vertices.push(Vertex::new(vec![], 0));
        let in2 = vertices.push(Vertex::new(vec![], 0));
        let in3 = vertices.push(Vertex::new(vec![], 0));

        let v4 = vertices.push(Vertex::new(vec![in1, in2], 1000));
        let v5 = vertices.push(Vertex::new(vec![in2, v4, in3], 2000));
        let v6 = vertices.push(Vertex::new(vec![in1, v4, v5], 2000));
        let v7 = vertices.push(Vertex::new(vec![in3, v6], 3000));
        let v8 = vertices.push(Vertex::new(vec![v4, v5, in1, in2], 3000));
        let _v9 = vertices.push(Vertex::new(vec![v7, v8], 5000));

        Digraph::from_vertices(vertices)
    }

    #[test]
    fn test_scheduler() {
        let cg = test_digraph();
        let dag = SubDigraph::all(&cg);
        let _ = Scheduler::new(&dag, dag.topology_sort().map(|(i, _)| i)).run();
    }
}
