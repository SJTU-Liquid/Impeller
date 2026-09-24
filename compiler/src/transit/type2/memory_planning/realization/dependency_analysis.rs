use super::super::prelude::*;
use std::ops::Deref;

#[derive(Clone, Debug)]
struct ResidentalObjectTable<P, T>(BTreeMap<P, ObjectTable<DeviceSpecific<Option<T>>>>, usize);

impl<P, T> ResidentalObjectTable<P, T>
where
    P: Ord + Eq + Clone,
{
    pub fn new(hd_info: &HardwareInfo) -> Self {
        Self(BTreeMap::new(), hd_info.n_gpus())
    }

    pub fn get(&self, p: P, o: &Object, d: Device) -> Option<&T> {
        self.0.get(&p)?.get(o)?.get(d).as_ref()
    }

    pub fn get_mut(&mut self, p: P, o: &Object, d: Device) -> Option<&mut T> {
        self.0.get_mut(&p)?.get_mut(o)?.get_mut(d).as_mut()
    }

    pub fn insert(&mut self, p: P, o: Object, d: Device, t: T) -> Option<T> {
        let ot = self.0.entry(p).or_insert_with(|| ObjectTable::new());
        if !ot.contains(&o) {
            ot.insert(o.clone(), DeviceSpecific::new(self.1, |_| None));
        }

        let r = ot.get_mut(&o).unwrap().get_mut(d).take();
        *ot.get_mut(&o).unwrap().get_mut(d) = Some(t);

        r
    }

    pub fn _remove(&mut self, p: P, o: &Object, d: Device) -> Option<T> {
        self.0.get_mut(&p)?.get_mut(o)?.get_mut(d).take()
    }
}

#[derive(Debug, Clone)]
pub struct Analyzer<P> {
    allocated_at: ResidentalObjectTable<Pointer<P>, Index>,
    write_at: ResidentalObjectTable<Pointer<P>, Index>,
    read_at: ResidentalObjectTable<Pointer<P>, Vec<Index>>,
}

impl<P> Analyzer<P>
where
    P: Ord + Eq + Copy + std::fmt::Debug,
{
    pub fn prepare_for<'s, 'r, R>(
        ops: impl Iterator<Item = (Index, &'r R)> + 'r,
        hd_info: &HardwareInfo,
    ) -> Self
    where
        R: ObjectAtRw<Object, Option<Pointer<P>>> + 'r,
    {
        let mut r = Analyzer {
            allocated_at: ResidentalObjectTable::new(hd_info),
            write_at: ResidentalObjectTable::new(hd_info),
            read_at: ResidentalObjectTable::new(hd_info),
        };

        for (index, op) in ops {
            op.object_at_writes().for_each(|(o, p, d)| {
                let _ = r
                    .write_at
                    .insert(p.unwrap(), (*o.deref()).clone(), d, index)
                    .is_some_and(|_| {
                        panic!(
                            "object {:?} at {:?} already written at {:?}",
                            o,
                            p.unwrap(),
                            index
                        )
                    });
            });

            op.object_at_reads().for_each(|(o, p, d)| {
                if r.read_at.get(p.unwrap(), o, d).is_none() {
                    r.read_at.insert(p.unwrap(), o.clone(), d, Vec::new());
                }
                r.read_at.get_mut(p.unwrap(), o, d).unwrap().push(index);
            });

            if let Some((o, p, d)) = op.object_at_allocates() {
                let _ = r
                    .allocated_at
                    .insert(p.unwrap(), o.clone(), d, index)
                    .is_some_and(|_| {
                        panic!(
                            "object {:?} at {:?} already allocated at {:?}",
                            o,
                            p.unwrap(),
                            index
                        )
                    });
            }
        }

        r
    }

    pub fn analyze<'s>(&self, ops: &phase1::OperationSeq<'s, P>) -> Dependencies {
        let mut dep = Heap::repeat_with(|| BTreeSet::new(), ops.len());

        for (index, op) in ops.iter() {
            op.object_at_reads().for_each(|(o, p, d)| {
                if let Some(written_at) = self.write_at.get(p.unwrap(), o, d) {
                    dep[index].insert(*written_at);
                } else if let Some(allocated_at) = self.allocated_at.get(p.unwrap(), o, d) {
                    dep[index].insert(*allocated_at);
                }
            });

            op.object_at_writes()
                .for_each(|(o, p, d)| match o.inplace_of() {
                    Some(taken) => {
                        if let Some(read_at) = self.read_at.get(p.unwrap(), taken, d) {
                            dep[index].extend(read_at.iter().copied().filter(|x| *x != index));
                        } else {
                            let written_at =
                                self.write_at.get(p.unwrap(), taken, d).unwrap_or_else(|| {
                                    panic!(
                                        "object {:?} at {:?} on {:?} is never written",
                                        taken,
                                        p.unwrap(),
                                        d
                                    )
                                });
                            dep[index].insert(*written_at);
                        }
                    }
                    None => {
                        if let Some(allocated_at) = self.allocated_at.get(p.unwrap(), *o, d) {
                            dep[index].insert(*allocated_at);
                        }
                    }
                });

            if let Some((o, p, d)) = op.object_at_deallocates() {
                if let Some(read_at) = self.read_at.get(p.unwrap(), o, d) {
                    dep[index].extend(read_at.iter().copied());
                } else if let Some(write_at) = self.write_at.get(p.unwrap(), o, d) {
                    dep[index].insert(*write_at);
                } else {
                    panic!("{:?} is neither written nor read", (o, p, d));
                }
            }
        }

        Dependencies {
            on: Seq::from_heap(dep),
        }
    }
}

/// An operation's write of some object depends on allocation of the object,
/// if the object does not take place of some other object and the object is not immortal.
/// If it did take someone's space, the write depends on all reads to the object taken, if any reads exist.
/// Otherwise, the write just depends on the write to the object taken.
///
/// An operation's read of some object depends on write of the object,
/// or the allocation directly if there is no write,
/// which should only happen in case of temporaries.
///
/// An deallocation of an object depends on last read of the object.
#[derive(Debug, Clone)]
pub struct Dependencies {
    on: Seq<BTreeSet<Index>>,
}

impl Dependencies {
    pub fn of<'a>(&'a self, idx: Index) -> impl Iterator<Item = Index> + Clone + 'a {
        self.on[idx].iter().cloned()
    }
}
