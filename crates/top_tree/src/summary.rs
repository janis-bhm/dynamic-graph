//! Generic, boundary-aware cluster summaries.
//!
//! A [`TopTree`](crate::TopTree) stores one value of a user supplied
//! [`Summary`] type per cluster. Whenever two neighboring clusters are joined
//! (or split), the summary of the resulting cluster is computed from the
//! summaries of its children by [`Summary::combine`].
//!
//! The two children of a cluster are joined at a single *central* vertex and
//! their boundaries determine whether the join is a
//!
//! * **compress**: both children are path clusters and their paths
//!   concatenate, or
//! * **rake**: a point cluster is attached to a path cluster, or
//! * **point merge**: neither child is a path cluster (this only happens while
//!   exposing a vertex).
//!
//! Because the shape of a cluster is what makes those cases meaningful,
//! [`Summary::combine`] receives a [`MergeContext`] describing how many
//! boundary vertices each child and the resulting cluster have. From this a
//! summary can recover the case it is in:

/// Describes the boundary sizes involved in a join.
///
/// A cluster has at most two boundary vertices. A cluster with two boundary
/// vertices is called a *path cluster*; a cluster with fewer is a *point
/// cluster*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeContext {
    /// Number of boundary vertices of the left child.
    pub left_boundary: u8,
    /// Number of boundary vertices of the right child.
    pub right_boundary: u8,
    /// Number of boundary vertices of the merged cluster.
    pub boundary: u8,
}

impl MergeContext {
    /// The merged cluster is a path cluster and both children are path
    /// children: this is a *compress*.
    pub fn is_compress(&self) -> bool {
        self.boundary == 2 && self.left_boundary == 2 && self.right_boundary == 2
    }

    /// The merged cluster is a path cluster and exactly one child is a path
    /// child: this is a *rake*.
    pub fn is_rake(&self) -> bool {
        self.boundary == 2 && ((self.left_boundary == 2) ^ (self.right_boundary == 2))
    }

    /// Whether the left child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn left_is_path_child(&self) -> bool {
        self.boundary == 2 && self.left_boundary == 2
    }

    /// Whether the right child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn right_is_path_child(&self) -> bool {
        self.boundary == 2 && self.right_boundary == 2
    }
}

/// A user supplied summary of a cluster.
///
/// Implementors describe how to summarize a single edge or label leaf and how
/// to combine the summaries of two children. The top tree takes care of
/// calling these in the right order as it rebalances.
///
/// `W` is the weight type stored on the underlying [`Tree`](crate::Tree)
/// edges and labels, so that a summary may depend on application data such as
/// edge levels.
pub trait Summary<W = ()>: Sized {
    /// A lazy tag that can be applied to a whole cluster, e.g. to add a common
    /// value to every edge on an exposed path. Use `()` if no lazy
    /// propagation is needed.
    type Tag: Clone + Default;

    /// Summary of the leaf representing the tree edge `u`-`v`.
    ///
    /// `u` and `v` are given in the orientation stored in the leaf.
    fn tree_edge(weight: &W, u: usize, v: usize) -> Self;

    /// Summary of the leaf representing the label at `v`.
    fn label(weight: &W, v: usize) -> Self;

    /// Combine the summaries of the left and right child of a cluster.
    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self;

    /// Reverses the orientation of this cluster's summary.
    ///
    /// The top tree calls this whenever it flips a cluster. Summaries that
    /// depend on path direction should update their orientation-sensitive
    /// fields; summaries that do not depend on orientation can use this
    /// default no-op implementation.
    fn flip(&mut self) {}

    /// Apply a lazy `tag` to this summary.
    fn apply(&mut self, _tag: &Self::Tag) {}

    /// Compose a tag that is being pushed from a parent onto this (child)
    /// tag. `tag` is the child's current tag and `parent` is the tag being
    /// pushed down. The default implementation discards the parent tag, which
    /// is correct for the no-op `()` tag.
    fn compose(_tag: &mut Self::Tag, _parent: &Self::Tag) {}
}
