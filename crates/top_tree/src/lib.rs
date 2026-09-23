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

mod summary;
mod top_tree;
mod tree;

pub use summary::{MergeContext, Summary};
pub use top_tree::{LabelId, TopTree};
pub use tree::{Edge, Label, Node, Tree};
