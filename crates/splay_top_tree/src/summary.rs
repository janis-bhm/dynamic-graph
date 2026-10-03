use crate::{ClusterId, VertexId};

/// The boundary vertices of a cluster, in its logical frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// The cluster has no boundary vertices.
    None,
    /// The cluster has one boundary vertex.
    One(VertexId),
    /// The cluster has two boundary vertices, ordered from left to right.
    Two { left: VertexId, right: VertexId },
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
    pub fn slots(self) -> [Option<VertexId>; 2] {
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
    pub central: VertexId,
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

pub trait Summary: Sized {
    fn edge(e: ClusterId) -> Self;
    fn label(l: ClusterId) -> Self;
    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self;
    fn flip(&mut self) {}
}
