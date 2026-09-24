//! Freshly created sliceable subgraph still operates on complete polynomials.
//! Here we must tailor them for operating on polynomial slices.
use super::prelude;
use prelude::*;

mod builder;
pub use builder::{Paragraph, ParagraphBuilder};

pub mod slice_analysis;

#[derive(Debug, Hash, Clone)]
pub struct Chunker {
    chunk_size: u64,
    deg: u64,
}

impl Chunker {
    pub fn new(size: u64, deg: u64) -> Self {
        Self {
            chunk_size: size,
            deg,
        }
    }

    pub fn chunk_len(&self) -> u64 {
        self.chunk_size
    }

    pub fn chunks(&self) -> u64 {
        self.deg / self.chunk_size
    }

    fn output_range(&self) -> Slice {
        Slice::new(0, self.chunk_size)
    }

    /// Return the smallest slice containing `slices`, and with begin and end aligned to chunk boundaries
    fn including_chunks(&self, slices: impl Iterator<Item = Slice>) -> Option<Slice> {
        let including = self.including_slice(slices)?;
        let (min, max) = (including.begin(), including.end());

        let start = min / self.chunk_size * self.chunk_size;
        let len = crate::utils::div_ceil_u64(max - start, self.chunk_size) * self.chunk_size;
        Some(Slice::new(start, len))
    }

    /// Return the smalles slice containing `slices`, if `slices` is not empty
    fn including_slice(&self, slices: impl Iterator<Item = Slice>) -> Option<Slice> {
        let mut slices: Vec<Slice> = slices.collect();
        if slices.len() == 0 {
            return None;
        }
        slices.sort();

        let mut merged_slices = Vec::new();
        while let Some(last) = slices.pop() {
            if let Some(ll) = slices.pop() {
                assert!(ll.begin() <= last.begin());
                if last.begin() - ll.begin() <= ll.len() {
                    slices.push(Slice::new(
                        ll.begin(),
                        (last.begin() - ll.begin() + last.len()).max(ll.len()),
                    ));
                } else {
                    merged_slices.push(last);
                    slices.push(ll);
                }
            } else {
                merged_slices.push(last);
            }
        }
        merged_slices.reverse();

        if merged_slices.len() >= 2 {
            let first = merged_slices.first().unwrap().clone();
            let last = merged_slices.last().unwrap().clone();
            let distance =
                (first.begin() as i64 - last.begin() as i64).rem_euclid(self.deg as i64) as u64;
            if distance <= last.len() {
                let first = merged_slices.remove(0);
                let last = merged_slices.pop().unwrap();
                merged_slices.push(Slice::new(
                    last.begin(),
                    (distance + first.len()).max(last.len()),
                ));
            }
        }

        let adjacents = merged_slices.iter().cloned().zip(
            merged_slices
                .iter()
                .skip(1)
                .chain(std::iter::once(merged_slices.first().unwrap()))
                .cloned(),
        );

        let (begin, gap) = adjacents
            .map(|(left, right)| {
                let gap = (right.begin() as i64 - left.end() as i64).rem_euclid(self.deg as i64);
                (right.begin(), gap)
            })
            .max_by_key(|(_, gap)| *gap)
            .unwrap();

        Some(Slice::new(begin, self.deg - gap as u64))
    }

    fn chunk_size(&self) -> u64 {
        self.chunk_size
    }

    fn rotate(&self, slice: &Slice, offset: i64) -> Slice {
        slice.rotated(offset, self.deg)
    }

    fn new_slice(&self, begin: u64, len: u64) -> Slice {
        Slice::new(begin % self.deg, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_including_slice_simple() {
        let chunker = Chunker::new(16, 128);
        let including_slice = chunker
            .including_slice([Slice::new(0, 16), Slice::new(124, 16)].into_iter())
            .unwrap();

        assert_eq!(including_slice, Slice::new(124, 20))
    }

    #[test]
    fn test_including_slice_containing() {
        let chunker = Chunker::new(16, 128);
        let including_slice = chunker
            .including_slice(
                [
                    Slice::new(0, 16),
                    Slice::new(124, 16),
                    Slice::new(4, 6),
                    Slice::new(120, 2),
                    Slice::new(100, 16),
                ]
                .into_iter(),
            )
            .unwrap();

        assert_eq!(including_slice, Slice::new(100, 44))
    }

    #[test]
    fn test_including_chunks() {
        let chunker = Chunker::new(16, 128);
        let including_slice = chunker
            .including_chunks(
                [
                    Slice::new(0, 16),
                    Slice::new(124, 16),
                    Slice::new(4, 16),
                    Slice::new(8, 16),
                    Slice::new(126, 16),
                ]
                .into_iter(),
            )
            .unwrap();

        assert_eq!(including_slice, Slice::new(112, 48))
    }

    #[test]
    fn test_including_slice_more() {
        let chunker = Chunker::new(16, 128);
        let including_slice = chunker
            .including_slice(
                [
                    Slice::new(0, 16),
                    Slice::new(124, 16),
                    Slice::new(4, 6),
                    Slice::new(120, 2),
                    Slice::new(100, 16),
                    Slice::new(124, 4),
                    Slice::new(32, 16),
                ]
                .into_iter(),
            )
            .unwrap();

        assert_eq!(including_slice, Slice::new(100, 76))
    }
}
