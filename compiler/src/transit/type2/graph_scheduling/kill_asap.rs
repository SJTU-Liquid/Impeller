use super::prelude::*;
use super::Size;

type ActiveRanking = (bool, i64);

struct ActiveVertex {
    has_ready_successor: bool,
    score: i64,
    size: u64,
}

impl mm_heap::Priority<ActiveRanking> for ActiveVertex {
    fn priority(&self) -> ActiveRanking {
        (self.has_ready_successor, self.score)
    }
}

pub struct Scheduler<'a, I, V> {
    cg: &'a SubDigraph<'a, I, V>,
    /// The estimated benifit of trying to kill an active vertex ASAP
    active_vertices: MmHeap<I, ActiveRanking, ActiveVertex>,
    /// The estimated benefit of killing the vertex after it is activated
    unexecuted_score: BTreeMap<I, i64>,
    ready_vertices: BTreeSet<I>,
    n_unexecuted_successors: Heap<I, usize>,
    n_unexecuted_predecessors: Heap<I, usize>,
}

impl<'a, I, V> Scheduler<'a, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
    SubDigraph<'a, I, V>: super::Size<I>,
{
    pub fn new(cg: &'a SubDigraph<'a, I, V>) -> Self {
        let ready_vertices: BTreeSet<I> = cg
            .vertices()
            .filter_map(|vid| if cg.deg_in(vid) == 0 { Some(vid) } else { None })
            .collect();

        let unexecuted_score = cg
            .topology_sort_inv()
            .fold(BTreeMap::new(), |mut acc, (vid, _)| {
                let cost = cg
                    .successors_of(vid)
                    .map(|succ| cg.size_of(succ) as i64)
                    .sum::<i64>();
                let gain = cg.size_of(vid);
                let score = gain as i64 - cost as i64;

                acc.insert(vid, score);
                acc
            });

        Scheduler {
            active_vertices: MmHeap::new(),
            unexecuted_score,
            ready_vertices,
            n_unexecuted_successors: cg.degrees_out(),
            n_unexecuted_predecessors: cg.degrees_in().clone(),
            cg,
        }
    }

    fn active_ranking(&self, vid: I) -> ActiveVertex {
        // Sum of the size of the unexecuted successors of a vertex
        let cost: i64 = self
            .cg
            .successors_of(vid)
            .filter_map(|x| self.unexecuted_score.get(&x))
            .copied()
            .sum();
        let gain = self.cg.size_of(vid);
        let score = gain as i64 - cost;

        let has_ready_successor = self
            .cg
            .successors_of(vid)
            .any(|x| self.ready_vertices.contains(&x));

        ActiveVertex {
            has_ready_successor,
            score,
            size: self.cg.size_of(vid),
        }
    }
}

impl<'a, I, V> super::Scheduler<'a, I, V> for Scheduler<'a, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
    SubDigraph<'a, I, V>: super::Size<I>,
{
    fn on(cg: &'a SubDigraph<'a, I, V>) -> Self {
        Self::new(cg)
    }

    fn choose_next_vertex(&self) -> Option<I> {
        if self.ready_vertices.is_empty() {
            return None;
        }

        let next = if let Some(victim) = self
            .active_vertices
            .max()
            .map(|(vid, v)| {
                if v.has_ready_successor {
                    Some(*vid)
                } else {
                    None
                }
            })
            .flatten()
        {
            // Arbitrarily choose a ready successor of the victim
            self.cg
                .successors_of(victim)
                .filter(|x| self.ready_vertices.contains(x))
                .next()
                .unwrap()
                .clone()
        } else {
            // If no active vertex has unexecuted ready successor, arbitrarily choose a ready vertex
            self.ready_vertices.first().unwrap().clone()
        };

        Some(next)
    }

    fn begin(&mut self, vid: I) {
        self.ready_vertices.remove(&vid);

        let has_ready_successor = self
            .cg
            .successors_of(vid)
            .any(|x| self.ready_vertices.contains(&x));

        let score = self.unexecuted_score.remove(&vid).unwrap();
        self.active_vertices.insert(
            vid,
            ActiveVertex {
                has_ready_successor,
                score,
                size: self.cg.size_of(vid),
            },
        );
    }

    fn finish(&mut self, vid: I) {
        for pred in self
            .cg
            .vertex(vid)
            .predecessors()
            .collect::<BTreeSet<_>>()
            .into_iter()
        {
            // Update the active ranking of the predecessor
            self.active_vertices.insert(pred, self.active_ranking(pred));

            self.n_unexecuted_successors[pred] -= 1;
            if self.n_unexecuted_successors[pred] == 0 {
                self.active_vertices.remove(&pred);
            }
        }

        for succ in self.cg.successors_of(vid) {
            self.n_unexecuted_predecessors[succ] -= 1;
            if self.n_unexecuted_predecessors[succ] == 0 {
                self.ready_vertices.insert(succ);
            }
        }
    }

    fn total(&self) -> usize {
        self.cg.order()
    }

    fn current_occupied_size(&self) -> u64 {
        self.active_vertices.iter().map(|(_, v)| v.size).sum()
    }
}
