//! Dynamic top trees over a forest.
//!
//! A top tree maintains information about a dynamic forest of trees, where the
//! leaves of the top tree correspond to edges in the underlying forest, and
//! the internal nodes correspond to clusters of connected edges.
//!
//! The forest itself is stored in a [`Tree`], which also holds *labels*:
//! leaf-like objects attached to a single vertex, used to represent non-tree
//! edges of the graph that the forest spans (or arbitrary vertex marks).
//! [`TopTree::link`], [`TopTree::cut`], [`TopTree::attach`] and
//! [`TopTree::detach`] update the forest; [`TopTree::expose`] and
//! [`TopTree::expose_path`] reposition the external boundary vertices and
//! return the resulting root cluster.
//!
//! # Implementing an algorithm
//!
//! To maintain application data, implement [`Summary`]. A summary is stored for
//! every cluster and recomputed from the two children whenever the top tree is
//! rebalanced. Because a cluster is either a *path* cluster (two boundary
//! vertices) or a *point* cluster (fewer), [`Summary::combine`] receives a
//! [`MergeContext`] that tells it whether the join is a *compress*, a *rake*,
//! or a point merge. This is exactly the information needed by classic
//! applications:
//!
//! * **Connectivity / sizes**: a monoid over all edges ignoring the context.
//! * **Dynamic max/min on a path**: keep the aggregate on the cluster path and
//!   only incorporate the children that are path children
//!   ([`MergeContext::left_is_path_child`], [`MergeContext::right_is_path_child`]).
//! * **Bridge finding and 2-edge connectivity**: store, for each cluster, the
//!   minimum *cover level* of its path edges together with the corresponding
//!   edge, and use [`Summary::Tag`] to lazily apply a `cover(x, i)` to an
//!   exposed path ([`TopTree::expose_path_tagged`]). The tags are pushed only
//!   to path children, matching the lazy "extra" weights used in the
//!   literature.
//! * **Biconnectivity**: the same boundary aware information, additionally
//!   tracking the number and levels of back edges incident to each boundary
//!   vertex.
//!
//! # Example
//!
//! ```
//! use top_tree::{MergeContext, Summary, TopTree, VertexId};
//!
//! /// The number of edges on the cluster path.
//! #[derive(Clone, Copy, PartialEq, Eq, Debug)]
//! struct PathLen {
//!     len: u32,
//! }
//!
//! impl Summary for PathLen {
//!     type Tag = ();
//!
//!     fn tree_edge(_u: VertexId, _v: VertexId) -> Self {
//!         PathLen { len: 1 }
//!     }
//!
//!     fn label(_v: VertexId) -> Self {
//!         PathLen { len: 0 }
//!     }
//!
//!     fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
//!         if ctx.boundary.is_path() {
//!             PathLen {
//!                 len: u32::from(ctx.left_is_path_child()) * left.len
//!                     + u32::from(ctx.right_is_path_child()) * right.len,
//!             }
//!         } else {
//!             PathLen { len: 0 }
//!         }
//!     }
//! }
//!
//! let mut top_tree = TopTree::<PathLen>::new();
//! let a = top_tree.add_vertex();
//! let b = top_tree.add_vertex();
//! let c = top_tree.add_vertex();
//!
//! top_tree.link(a, b);
//! top_tree.link(b, c);
//!
//! assert_eq!(top_tree.expose_path(a, c), Some(PathLen { len: 2 }));
//! ```
#![expect(internal_features)]
#![feature(pattern_types, pattern_type_macro, structural_match)]

mod summary;
mod top_tree;
mod tree;

pub use summary::{Boundary, MergeContext, Summary};
pub use top_tree::{ClusterId, ClusterWeight, NodeKind, TopTree};
pub use tree::{Edge, EdgeId, Label, LabelId, Node, SwapResult, Tree, VertexId};

macro_rules! impl_nonmax_type {
    ($($vis:vis struct $name:ident($int:ident is $pat:pat)),* $(,)?) => {
        $(impl_nonmax_type!(@impl $vis struct $name($int is $pat));)*
    };
    (@impl $vis:vis struct $name:ident($int:ident is $pat:pat)) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        $vis struct $name(pattern_type!($int is $pat));

        const _: () = {
            assert!(core::mem::size_of::<$name>() == core::mem::size_of::<$int>());
            assert!(core::mem::size_of::<Option<$name>>() == core::mem::size_of::<$int>());
        };

        impl $name {
            pub const fn new(value: $int) -> Option<Self> {
                if let $pat = value {
                    // SAFETY: The value is guaranteed to be in the valid range for this type.
                    Some($name(unsafe { core::mem::transmute::<$int, pattern_type!($int is $pat)>(value) }))
                } else {
                    None
                }
            }

            /// # Safety
            /// The caller must ensure that `value` is in the valid range for this type.
            pub const unsafe fn new_unchecked(value: $int) -> Self {
                    // SAFETY: The value is guaranteed to be in the valid range
                    // for this type by the caller
                    $name(unsafe { core::mem::transmute::<$int, pattern_type!($int is $pat)>(value) })
            }

            pub const fn get(self) -> $int {
                unsafe { core::mem::transmute(self) }
            }

            pub const fn as_inner(self) -> $int {
                self.get()
            }
        }

        impl core::marker::StructuralPartialEq for $name {}
        impl Eq for $name {}
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.get() == other.get()
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                Ord::cmp(&self.get(), &other.get())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(Ord::cmp(self, other))
            }
        }

        impl core::hash::Hash for $name {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                core::hash::Hash::hash(&self.get(), state)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$int as core::fmt::Debug>::fmt(&self.get(), f)
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$int as core::fmt::Display>::fmt(&self.get(), f)
            }
        }
    };
}

impl_nonmax_type!(
    pub struct NonMaxUsize(usize is 0..=0xFFFFFFFFFFFFFFFE),
    pub struct NonMaxIsize(isize is 0..=0x7FFFFFFFFFFFFFFE),
    pub struct NonMaxU32(u32 is 0..=0xFFFFFFFE),
    pub struct NonMaxI32(i32 is 0..=0x7FFFFFFE)
);

#[repr(transparent)]
#[derive(Clone, Copy, Default, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(u32);

impl Generation {
    fn increment(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }

    fn current(&self) -> Self {
        *self
    }
}

/// A compact growable bit vector, used to track which vertices are exposed.
///
/// Bits are packed into `u64` blocks, so each vertex costs a single bit rather
/// than the byte a `Vec<bool>` would use.
#[derive(Default)]
struct BitVec {
    blocks: Vec<u64>,
}

impl BitVec {
    const BITS: usize = u64::BITS as usize;

    fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Grows the vector to hold at least `len` bits, zero-filling new bits.
    fn grow_to(&mut self, len: usize) {
        let blocks = len.div_ceil(Self::BITS);
        if blocks > self.blocks.len() {
            self.blocks.resize(blocks, 0);
        }
    }

    #[expect(dead_code)]
    fn all_in_range(&self, range: std::ops::Range<usize>) -> bool {
        /// Iterator over the N-sized block indices and masks from bits `start` to `end` exclusively.
        struct BlockIter<const N: usize> {
            start: usize,
            end: usize,
        }

        impl<const N: usize> Iterator for BlockIter<N> {
            type Item = (usize, u64);

            fn next(&mut self) -> Option<Self::Item> {
                if self.start > self.end {
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
    fn get(&self, index: usize) -> bool {
        let block = index / Self::BITS;
        self.blocks
            .get(block)
            .is_some_and(|bits| bits >> (index % Self::BITS) & 1 == 1)
    }

    fn remove(&mut self, index: usize, last: usize) {
        self.set(index, self.get(last));
        self.set(last, false);
    }

    /// Sets the bit at `index`, ignoring indices past the end.
    fn set(&mut self, index: usize, value: bool) {
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

pub mod slot {
    use std::mem::{self, ManuallyDrop};

    use crate::{BitVec, NonMaxUsize};

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

    impl Indexing for NonMaxUsize {
        type Optional = Option<Self>;

        fn get(&self) -> usize {
            NonMaxUsize::get(*self)
        }

        fn new(value: usize) -> Self {
            debug_assert!(
                value != usize::MAX,
                "value must be in the range 0..usize::MAX"
            );
            // SAFETY: The value is guaranteed to be in the valid range for this type.
            unsafe { NonMaxUsize::new_unchecked(value) }
        }
    }

    pub(crate) union Slot<T, U: Copy> {
        value: ManuallyDrop<T>,
        pub(crate) next: U,
    }

    pub struct SlotVec<T, I: Indexing = NonMaxUsize> {
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

        pub fn push_with(&mut self, f: impl FnOnce(I) -> T) -> I {
            let index = if let Some(free) = self.first_free.into_option() {
                let free_index = free.get();
                self.first_free = unsafe { self.slots[free_index].next };
                self.slots[free_index] = Slot {
                    value: ManuallyDrop::new(f(free)),
                };

                free
            } else {
                // SAFETY: Vec cannot grow beyond isize::MAX elements.
                let index = I::new(self.slots.len());
                self.slots.push(Slot {
                    value: ManuallyDrop::new(f(index)),
                });

                index
            };

            self.occupancy.grow_to(index.get() + 1);
            self.occupancy.set(index.get(), true);

            index
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

        pub fn for_each_mut(&mut self, mut f: impl FnMut(NonMaxUsize, &mut T)) {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                let index = unsafe { NonMaxUsize::new_unchecked(i) };
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
