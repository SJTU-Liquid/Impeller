//! A module for min/max-heap

pub mod with_data {
    use std::{
        collections::{btree_set, BTreeMap, BTreeSet},
        fmt::Debug,
        iter::FusedIterator,
    };

    /// A random-access, random-remove capable min-heap or max-heap.
    /// [`K`] is the key type and [`S`] is the sequence number type.
    #[derive(Clone)]
    pub struct MmHeap<K, S, T> {
        heap: BTreeSet<(S, K)>,
        k2t: BTreeMap<K, T>,
    }

    impl<K, S, T> Default for MmHeap<K, S, T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<K, S, T> MmHeap<K, S, T> {
        pub fn new() -> Self {
            Self {
                heap: BTreeSet::new(),
                k2t: BTreeMap::new(),
            }
        }

        pub fn len(&self) -> usize {
            self.k2t.len()
        }

        pub fn is_empty(&self) -> bool {
            self.k2t.is_empty()
        }
    }

    pub struct InternalDebug<'a, K, S, T>(&'a MmHeap<K, S, T>);

    impl<'a, K, S, T> Debug for InternalDebug<'a, K, S, T>
    where
        K: Debug,
        S: Debug,
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MmHeap")
                .field("heap", &self.0.heap)
                .field("k2t", &self.0.k2t)
                .finish()
        }
    }

    impl<K, S, T> Debug for MmHeap<K, S, T>
    where
        K: Debug,
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut debug = f.debug_map();
            for (k, t) in self.k2t.iter() {
                debug.entry(k, t);
            }
            debug.finish()
        }
    }

    /// The priority of an item must be stable.
    /// That is, as long as the item is in heap,
    /// its priority must remain unchanged, even it is mutated through mutable reference.
    pub trait Priority<S> {
        fn priority(&self) -> S;
    }

    impl<S> Priority<S> for S
    where
        S: Clone,
    {
        fn priority(&self) -> S {
            self.clone()
        }
    }

    impl<K, S, T> MmHeap<K, S, T>
    where
        K: Ord + Clone,
        S: Ord + Clone,
        T: Priority<S>,
    {
        /// Inserts a new element with key `k` and sequence number `s`.
        /// If the element already exists, it will be removed first.
        pub fn insert(&mut self, k: K, t: T) -> Option<T> {
            let old = if let Some(old_t) = self.k2t.remove(&k) {
                assert!(self.heap.remove(&(old_t.priority(), k.clone())));
                Some(old_t)
            } else {
                None
            };

            self.heap.insert((t.priority(), k.clone()));
            self.k2t.insert(k.clone(), t);

            old
        }

        pub fn update_with(&mut self, k: K, f: impl FnOnce(T) -> T) -> Option<()> {
            let old_t = self.k2t.remove(&k)?;
            assert!(self.heap.remove(&(old_t.priority(), k.clone())));

            let t = f(old_t);
            self.heap.insert((t.priority(), k.clone()));
            self.k2t.insert(k.clone(), t);

            Some(())
        }

        /// Removes the element with key `k`, if found.
        pub fn remove(&mut self, k: &K) -> Option<T> {
            let t = self.k2t.remove(k)?;
            self.heap.remove(&(t.priority(), k.clone()));
            Some(t)
        }
    }

    impl<K, S, T> MmHeap<K, S, T>
    where
        K: Ord + Clone,
        S: Ord + Clone,
    {
        /// Returns the key of the element with maximum sequence number.
        pub fn max(&self) -> Option<(&K, &T)> {
            let (_, k) = self.heap.last()?;
            let t = &self.k2t[k];
            Some((k, t))
        }

        /// Returns the key of the element with minimum sequence number.
        pub fn min(&self) -> Option<(&K, &T)> {
            let (_, k) = self.heap.first()?;
            let t = &self.k2t[k];
            Some((k, t))
        }

        pub fn pop_max(&mut self) -> Option<(K, T)> {
            let (_, k) = self.heap.pop_first()?;
            let t = self.k2t.remove(&k).unwrap();
            Some((k, t))
        }

        pub fn pop_min(&mut self) -> Option<(K, T)> {
            let (_, k) = self.heap.pop_last()?;
            let t = self.k2t.remove(&k).unwrap();
            Some((k, t))
        }

        pub fn keys(&self) -> impl Iterator<Item = &K> {
            self.k2t.keys()
        }

        pub fn get(&self, k: &K) -> Option<&T> {
            self.k2t.get(k)
        }

        pub fn get_mut(&mut self, k: &K) -> Option<&mut T> {
            self.k2t.get_mut(k)
        }

        /// Keys and items sorted according to `S`
        pub fn iter<'a>(&'a self) -> Iter<'a, K, S, T> {
            Iter {
                keys: self.heap.iter(),
                items: &self.k2t,
            }
        }

        pub fn contains(&self, k: &K) -> bool {
            self.k2t.contains_key(k)
        }
    }

    pub struct Iter<'a, K: 'a, S: 'a, T: 'a> {
        keys: btree_set::Iter<'a, (S, K)>,
        items: &'a BTreeMap<K, T>,
    }

    impl<'a, K: 'a, S: 'a, T: 'a> Iterator for Iter<'a, K, S, T>
    where
        K: Ord,
    {
        type Item = (&'a K, &'a T);

        fn next(&mut self) -> Option<Self::Item> {
            let (_, k) = self.keys.next()?;
            Some((k, &self.items[k]))
        }
    }

    impl<'a, K: 'a, S: 'a, T: 'a> DoubleEndedIterator for Iter<'a, K, S, T>
    where
        K: Ord,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            let (_, k) = self.keys.next_back()?;
            Some((k, &self.items[k]))
        }
    }

    impl<'a, K: 'a, S: 'a, T: 'a> ExactSizeIterator for Iter<'a, K, S, T>
    where
        K: Ord,
    {
        fn len(&self) -> usize {
            self.keys.len()
        }
    }
    impl<'a, K: 'a, S: 'a, T: 'a> FusedIterator for Iter<'a, K, S, T> where K: Ord {}

    impl<K, S, T> std::ops::Index<&K> for MmHeap<K, S, T>
    where
        K: Ord,
    {
        type Output = T;

        fn index(&self, index: &K) -> &Self::Output {
            &self.k2t[index]
        }
    }
}

pub type MmHeap<K, S> = with_data::MmHeap<K, S, S>;
pub use with_data::{InternalDebug, Priority};
