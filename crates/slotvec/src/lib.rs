pub use bitvec::BitVec;
pub use slot::{Indexing, OptionalIndexing, SlotVec};
mod bitvec {
    /// A compact growable bit vector, used to track which vertices are exposed.
    ///
    /// Bits are packed into `u64` blocks, so each vertex costs a single bit rather
    /// than the byte a `Vec<bool>` would use.
    #[derive(Default)]
    pub struct BitVec {
        blocks: Vec<u64>,
    }

    impl BitVec {
        const BITS: usize = u64::BITS as usize;

        pub fn new() -> Self {
            Self { blocks: Vec::new() }
        }

        /// Grows the vector to hold at least `len` bits, zero-filling new bits.
        pub fn grow_to(&mut self, len: usize) {
            let blocks = len.div_ceil(Self::BITS);
            if blocks > self.blocks.len() {
                self.blocks.resize(blocks, 0);
            }
        }

        #[expect(dead_code)]
        pub fn all_in_range(&self, range: std::ops::Range<usize>) -> bool {
            /// Iterator over the N-sized block indices and masks from bits `start` to `end` exclusively.
            struct BlockIter<const N: usize> {
                start: usize,
                end: usize,
            }

            impl<const N: usize> Iterator for BlockIter<N> {
                type Item = (usize, u64);

                fn next(&mut self) -> Option<Self::Item> {
                    if self.start >= self.end {
                        return None;
                    }

                    let block_idx = self.start.div_euclid(N);
                    let mask_start = self.start.rem_euclid(N);
                    let mask_size = if self.start + N < self.end {
                        N - mask_start
                    } else {
                        self.end - self.start
                    };

                    let mask = ((1u64 << mask_size) - 1) << mask_start;
                    self.start += mask_size;
                    Some((block_idx, mask))
                }
            }

            for (block_idx, mask) in {
                BlockIter::<{ u64::BITS as usize }> {
                    start: range.start,
                    end: range.end,
                }
            } {
                let block = self.blocks.get(block_idx).copied().unwrap_or(0);
                if block & mask != mask {
                    return false;
                }
            }

            true
        }

        /// Returns whether the bit at `index` is set; `false` if out of range.
        pub fn get(&self, index: usize) -> bool {
            let block = index / Self::BITS;
            self.blocks
                .get(block)
                .is_some_and(|bits| bits >> (index % Self::BITS) & 1 == 1)
        }

        pub fn remove(&mut self, index: usize, last: usize) {
            self.set(index, self.get(last));
            self.set(last, false);
        }

        /// Sets the bit at `index`, ignoring indices past the end.
        pub fn set(&mut self, index: usize, value: bool) {
            let block = index / Self::BITS;
            let Some(bits) = self.blocks.get_mut(block) else {
                return;
            };
            let mask = 1u64 << (index % Self::BITS);
            if value {
                *bits |= mask;
            } else {
                *bits &= !mask;
            }
        }
    }
}

pub mod slot {
    use std::mem::{self, ManuallyDrop};

    use crate::BitVec;

    pub trait OptionalIndexing<I>: Copy + Sized {
        const NONE: Self;
        fn into_option(self) -> Option<I>;
        fn some(value: I) -> Self;
    }

    pub trait Indexing: Sized + Copy {
        type Optional: OptionalIndexing<Self>;

        fn get(&self) -> usize;
        fn new(value: usize) -> Self;
        fn into_optional(self) -> Self::Optional {
            Self::Optional::some(self)
        }
    }

    impl<T: Copy> OptionalIndexing<T> for Option<T> {
        const NONE: Self = None;

        fn into_option(self) -> Option<T> {
            self
        }

        fn some(value: T) -> Self {
            Some(value)
        }
    }

    impl Indexing for usize {
        type Optional = Option<Self>;

        fn get(&self) -> usize {
            *self
        }

        fn new(value: usize) -> Self {
            value
        }
    }

    pub(crate) union Slot<T, U: Copy> {
        value: ManuallyDrop<T>,
        pub(crate) next: U,
    }

    pub struct SlotVec<T, I: Indexing = usize> {
        pub(crate) slots: Vec<Slot<T, I::Optional>>,
        pub(crate) occupancy: BitVec,
        pub(crate) first_free: I::Optional,
    }

    impl<T, I: Indexing> Drop for SlotVec<T, I> {
        fn drop(&mut self) {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                if self.occupancy.get(i) {
                    unsafe { ManuallyDrop::drop(&mut slot.value) }
                }
            }
        }
    }

    impl<T, I: Indexing> std::ops::Index<I> for SlotVec<T, I> {
        type Output = T;

        fn index(&self, index: I) -> &Self::Output {
            self.get(index).expect("index out of bounds")
        }
    }

    impl<T, I: Indexing> std::ops::IndexMut<I> for SlotVec<T, I> {
        fn index_mut(&mut self, index: I) -> &mut Self::Output {
            self.get_mut(index).expect("index out of bounds")
        }
    }

    impl<T, I: Indexing> SlotVec<T, I> {
        pub fn new() -> Self {
            Self {
                slots: Vec::new(),
                occupancy: BitVec::new(),
                first_free: I::Optional::NONE,
            }
        }

        pub fn push(&mut self, value: T) -> I {
            let index = if let Some(free) = self.first_free.into_option() {
                let free_index = free.get();
                self.first_free = unsafe { self.slots[free_index].next };
                self.slots[free_index] = Slot {
                    value: ManuallyDrop::new(value),
                };

                free
            } else {
                let index = I::new(self.slots.len());
                self.slots.push(Slot {
                    value: ManuallyDrop::new(value),
                });

                index
            };

            self.occupancy.grow_to(index.get() + 1);
            self.occupancy.set(index.get(), true);

            index
        }

        pub fn push_with<R>(&mut self, f: impl FnOnce(I) -> (T, R)) -> (I, R) {
            let (index, r) = if let Some(free) = self.first_free.into_option() {
                let free_index = free.get();
                self.first_free = unsafe { self.slots[free_index].next };

                let (value, r) = f(free);
                self.slots[free_index] = Slot {
                    value: ManuallyDrop::new(value),
                };

                (free, r)
            } else {
                // SAFETY: Vec cannot grow beyond isize::MAX elements.
                let index = I::new(self.slots.len());

                let (value, r) = f(index);
                self.slots.push(Slot {
                    value: ManuallyDrop::new(value),
                });

                (index, r)
            };

            self.occupancy.grow_to(index.get() + 1);
            self.occupancy.set(index.get(), true);

            (index, r)
        }

        pub fn remove(&mut self, index: I) -> Option<T> {
            let idx = index.get();
            if !self.occupancy.get(idx) {
                return None;
            }

            let elt = unsafe {
                let next_free = self.first_free;
                let elt = ManuallyDrop::take(&mut self.slots[idx].value);
                _ = mem::replace(&mut self.slots[idx], Slot { next: next_free });

                elt
            };

            self.occupancy.set(idx, false);
            if idx == self.slots.len() - 1 {
                self.slots.pop();
                // self.occupancy.shrink_to(self.slots.len());
            } else {
                self.first_free = index.into_optional();
            }

            Some(elt)
        }

        pub fn get(&self, index: I) -> Option<&T> {
            let idx = index.get();
            if !self.occupancy.get(idx) {
                return None;
            }

            unsafe { Some(&self.slots[idx].value) }
        }

        pub fn replace(&mut self, index: I, value: T) -> Option<T> {
            let idx = index.get();
            if !self.occupancy.get(idx) {
                return None;
            }

            let old_value = unsafe { ManuallyDrop::take(&mut self.slots[idx].value) };
            self.slots[idx] = Slot {
                value: ManuallyDrop::new(value),
            };

            Some(old_value)
        }

        pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
            let idx = index.get();
            if !self.occupancy.get(idx) {
                return None;
            }

            unsafe { Some(&mut self.slots[idx].value) }
        }

        pub fn for_each_mut(&mut self, mut f: impl FnMut(I, &mut T)) {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                let index = I::new(i);
                if self.occupancy.get(i) {
                    unsafe { f(index, &mut slot.value) }
                }
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
            self.slots.iter().enumerate().filter_map(move |(i, slot)| {
                let index = I::new(i);
                if self.occupancy.get(i) {
                    Some((index, unsafe { &*slot.value }))
                } else {
                    None
                }
            })
        }
    }

    impl<T> Default for SlotVec<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_all_in_range() {
        let mut bv = BitVec::new();
        bv.grow_to(100);
        bv.set(10, true);
        bv.set(11, true);
        bv.set(12, true);
        assert!(bv.all_in_range(10..13));
        assert!(!bv.all_in_range(9..13));
        assert!(!bv.all_in_range(10..14));
        assert!(!bv.all_in_range(13..15));
    }
}
