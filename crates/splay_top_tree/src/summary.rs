use crate::{VertexId, index};

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
    pub(crate) fn shared(&self, other: &Self) -> Option<VertexId> {
        match (self, other) {
            (Boundary::None, _) | (_, Boundary::None) => None,
            (Boundary::One(v1), Boundary::One(v2)) => {
                if v1 == v2 {
                    Some(*v1)
                } else {
                    None
                }
            }
            (Boundary::One(v), Boundary::Two { left, right })
            | (Boundary::Two { left, right }, Boundary::One(v)) => {
                if v == left || v == right {
                    Some(*v)
                } else {
                    None
                }
            }
            (
                Boundary::Two {
                    left: l1,
                    right: r1,
                },
                Boundary::Two {
                    left: l2,
                    right: r2,
                },
            ) => {
                if l1 == l2 || l1 == r2 {
                    Some(*l1)
                } else if r1 == l2 || r1 == r2 {
                    Some(*r1)
                } else {
                    None
                }
            }
        }
    }

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
    pub parent_boundary: Boundary,
    /// The vertex shared by the two children (their central vertex).
    pub central: VertexId,
}

impl MergeContext {
    /// The merged cluster is a path cluster and both children are path
    /// children: this is a *compress*.
    pub fn is_compress(&self) -> bool {
        self.parent_boundary.is_path()
            && self.left_boundary.is_path()
            && self.right_boundary.is_path()
    }

    /// The merged cluster is a path cluster and exactly one child is a path
    /// child: this is a *rake*.
    pub fn is_rake(&self) -> bool {
        self.parent_boundary.is_path()
            && ((self.left_boundary.is_path()) ^ (self.right_boundary.is_path()))
    }

    /// Whether the left child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn left_is_path_child(&self) -> bool {
        self.parent_boundary.is_path() && self.left_boundary.is_path()
    }

    /// Whether the right child's cluster path is a sub-path of the merged
    /// cluster's path.
    pub fn right_is_path_child(&self) -> bool {
        self.parent_boundary.is_path() && self.right_boundary.is_path()
    }
}

pub trait Summary: Sized {
    fn edge(e: index::EdgeId) -> Self;
    fn label(l: index::LabelId) -> Self;
    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self;
    fn update_boundary(&mut self, left: &Self, right: &Self, ctx: &MergeContext) {
        *self = Self::combine(left, right, ctx);
    }
    fn flip(&mut self) {}
}
