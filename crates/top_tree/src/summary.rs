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

/// The boundary vertices of a cluster, in its logical frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// The cluster has no boundary vertices.
    None,
    /// The cluster has one boundary vertex.
    One(usize),
    /// The cluster has two boundary vertices, ordered from left to right.
    Two { left: usize, right: usize },
}

impl Boundary {
    /// Returns the number of boundary vertices.
    pub fn count(self) -> u8 {
        match self {
            Boundary::None => 0,
            Boundary::One(_) => 1,
            Boundary::Two { .. } => 2,
        }
    }

    /// Returns whether the cluster is a path cluster.
    pub fn is_path(self) -> bool {
        matches!(self, Boundary::Two { .. })
    }

    /// Returns the boundary vertices in left-to-right slots.
    ///
    /// A one-vertex cluster repeats its vertex in both slots so that set-
    /// intersection helpers can treat point and path clusters uniformly.
    pub fn slots(self) -> [Option<usize>; 2] {
        match self {
            Boundary::None => [None, None],
            Boundary::One(v) => [Some(v), Some(v)],
            Boundary::Two { left, right } => [Some(left), Some(right)],
        }
    }
}

/// Describes the boundaries involved in a join.
///
/// A cluster has at most two boundary vertices. A cluster with two boundary
/// vertices is called a *path cluster*; a cluster with fewer is a *point
/// cluster*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeContext {
    /// Left child's boundary vertices, in the logical frame of `left.sum`.
    pub left_boundary: Boundary,
    /// Right child's boundary vertices, in the logical frame of `right.sum`.
    pub right_boundary: Boundary,
    /// Merged cluster's boundary vertices, in the logical frame of the new sum.
    pub boundary: Boundary,
    /// The vertex shared by the two children (their central vertex).
    pub central: usize,
}

impl MergeContext {
    /// The merged cluster is a path cluster and both children are path
    /// children: this is a *compress*.
    pub fn is_compress(&self) -> bool {
        self.boundary.is_path() && self.left_boundary.is_path() && self.right_boundary.is_path()
    }

    /// The merged cluster is a path cluster and exactly one child is a path
    /// child: this is a *rake*.
    pub fn is_rake(&self) -> bool {
        self.boundary.is_path()
            && ((self.left_boundary.is_path()) ^ (self.right_boundary.is_path()))
    }

    /// Whether the left child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn left_is_path_child(&self) -> bool {
        self.boundary.is_path() && self.left_boundary.is_path()
    }

    /// Whether the right child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn right_is_path_child(&self) -> bool {
        self.boundary.is_path() && self.right_boundary.is_path()
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

    /// Key type stored with labels in the underlying tree.
    type LabelKey;

    /// Summary of the leaf representing the tree edge `u`-`v`.
    ///
    /// `u` and `v` are given in the orientation stored in the leaf.
    fn tree_edge(weight: &W, u: usize, v: usize) -> Self;

    /// Summary of the leaf representing the label at `v`.
    fn label(key: &Self::LabelKey, v: usize) -> Self;

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
