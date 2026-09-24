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
//! use top_tree::{MergeContext, Summary, TopTree};
//!
//! /// The number of edges on the cluster path.
//! #[derive(Clone, Copy, PartialEq, Eq, Debug)]
//! struct PathLen {
//!     len: u32,
//! }
//!
//! impl Summary<()> for PathLen {
//!     type Tag = ();
//!
//!     fn tree_edge(_weight: &(), _u: usize, _v: usize) -> Self {
//!         PathLen { len: 1 }
//!     }
//!
//!     fn label(_weight: &(), _v: usize) -> Self {
//!         PathLen { len: 0 }
//!     }
//!
//!     fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
//!         if ctx.boundary == 2 {
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
//! let mut top_tree = TopTree::<u32, PathLen>::new();
//! let a = top_tree.add_vertex(0, ());
//! let b = top_tree.add_vertex(1, ());
//! let c = top_tree.add_vertex(2, ());
//!
//! top_tree.link(a, b, ());
//! top_tree.link(b, c, ());
//!
//! assert_eq!(top_tree.expose_path(a, c), Some(PathLen { len: 2 }));
//! ```
#![feature(pattern_types, pattern_type_macro, structural_match)]

mod summary;
mod top_tree;
mod tree;

pub use summary::{MergeContext, Summary};
pub use top_tree::{LabelId, TopTree};
pub use tree::{Edge, Label, Node, Tree};

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
                    Some(unsafe { $name(core::mem::transmute(value)) })
                } else {
                    None
                }
            }

            pub const unsafe fn new_unchecked(value: $int) -> Self {
                $name(unsafe {core::mem::transmute(value) })
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

const USIZE_MINUS_TWO_BITS_MASK: usize = usize::MAX >> 3;

impl_nonmax_type!(
    pub struct NonMaxUsize(usize is 0..=0xFFFFFFFFFFFFFFFE),
    pub struct NonMaxIsize(isize is 0..=0x7FFFFFFFFFFFFFFE),
);
