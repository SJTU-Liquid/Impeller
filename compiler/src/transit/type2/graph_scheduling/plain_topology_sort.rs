use super::prelude::*;
use super::{DefUseDag, StatisticsCollector};

pub struct Scheduler<'d, I, D> {
    dag: &'d D,
    seq: Vec<I>,
    succ_count: BTreeMap<I, usize>,
    stc: StatisticsCollector,
}

impl<'d, I, D> Scheduler<'d, I, D>
where
    I: UsizeId,
    D: DefUseDag<I>,
{
    pub fn new(dag: &'d D, topology_seq: impl Iterator<Item = I>) -> Self {
        let seq: Vec<I> = topology_seq.filter(|i| dag.valid(*i)).collect();
        let succ_count: BTreeMap<I, usize> = seq
            .iter()
            .map(|i| (*i, dag.successors(*i).count()))
            .collect();
        Self {
            dag,
            succ_count,
            seq,
            stc: StatisticsCollector::new(),
        }
    }

    pub fn run(mut self) -> (Vec<I>, u64) {
        for vid in self.seq.iter().cloned() {
            self.stc.live(self.dag.def_size(vid));

            for pred in self.dag.predecessors(vid) {
                *self
                    .succ_count
                    .get_mut(&pred)
                    .unwrap_or_else(|| panic!("no entry found for {:?}", pred)) -= 1;
                if self.succ_count[&pred] == 0 {
                    self.stc.kill(self.dag.def_size(pred));
                }
            }
        }

        (self.seq, self.stc.max_occupied)
    }
}
