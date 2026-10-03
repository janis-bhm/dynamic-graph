use crate::tree::VertexId;

pub struct PackedBoundaryVertices {
    pub left: Option<VertexId>,
    pub right: Option<VertexId>,
}

impl PackedBoundaryVertices {
    fn unpack(&self) -> BoundaryVertices {
        match (self.left, self.right) {
            (None, None) => BoundaryVertices::None,
            (Some(v), None) | (None, Some(v)) => BoundaryVertices::One(v),
            (Some(left), Some(right)) => BoundaryVertices::Two { left, right },
        }
    }

    fn pack(boundary: BoundaryVertices) -> Self {
        match boundary {
            BoundaryVertices::None => Self {
                left: None,
                right: None,
            },
            BoundaryVertices::One(v) => Self {
                left: Some(v),
                right: None,
            },
            BoundaryVertices::Two { left, right } => Self {
                left: Some(left),
                right: Some(right),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryVertices {
    None,
    One(VertexId),
    Two { left: VertexId, right: VertexId },
}

impl BoundaryVertices {
    pub fn from_left_and_right(left: Option<VertexId>, right: Option<VertexId>) -> Self {
        match (left, right) {
            (None, None) => Self::None,
            (Some(v), None) | (None, Some(v)) => Self::One(v),
            (Some(v), Some(w)) => {
                if v == w {
                    Self::One(v)
                } else {
                    Self::Two { left: v, right: w }
                }
            }
        }
    }

    pub fn from_option(v: Option<VertexId>) -> Self {
        match v {
            None => Self::None,
            Some(v) => Self::One(v),
        }
    }

    pub fn from_children(left: Self, right: Self, is_path: bool) -> Self {
        match (left, right) {
            (Self::None, Self::None) => Self::None,
            (Self::None, _) | (_, Self::None) => {
                panic!("call expose on a vertex before linking or attaching an edge/label to it")
            }
            (Self::One(v), Self::One(w)) => {
                assert_eq!(
                    v, w,
                    "two clusters can only be merged if they share a boundary vertex"
                );
                assert!(
                    !is_path,
                    "two point clusters cannot be merged into a path cluster"
                );

                Self::One(v)
            }
            (Self::One(v), Self::Two { left, right }) => {
                assert_eq!(
                    v, left,
                    "two clusters can only be merged if they share a boundary vertex"
                );

                if is_path {
                    Self::Two { left, right }
                } else {
                    Self::One(right)
                }
            }
            (Self::Two { left, right }, Self::One(v)) => {
                assert_eq!(
                    v, right,
                    "two clusters can only be merged if they share a boundary vertex"
                );

                if is_path {
                    Self::Two { left, right }
                } else {
                    Self::One(left)
                }
            }
            (
                Self::Two { left, right },
                Self::Two {
                    left: l2,
                    right: r2,
                },
            ) => {
                debug_assert_eq!(
                    right, l2,
                    "two clusters can only be merged if they share a boundary vertex"
                );
                assert!(
                    is_path,
                    "two path clusters cannot be merged into a point cluster"
                );
                Self::Two { left, right: r2 }
            }
        }
    }

    pub fn remove(&mut self, v: VertexId) -> Option<VertexId> {
        match self {
            Self::None => {}
            Self::One(w) => {
                if *w == v {
                    *self = Self::None;
                    return Some(v);
                }
            }
            Self::Two { left, right } => {
                if *left == v {
                    *self = Self::One(*right);
                    return Some(v);
                } else if *right == v {
                    *self = Self::One(*left);
                    return Some(v);
                }
            }
        }

        None
    }

    pub fn add(&mut self, v: VertexId, left: bool) {
        if self.contains(v) {
            return;
        }

        match self {
            Self::None => *self = Self::One(v),
            Self::One(w) => {
                if *w != v {
                    if left {
                        *self = Self::Two { left: v, right: *w };
                    } else {
                        *self = Self::Two { left: *w, right: v };
                    }
                }
            }
            Self::Two { .. } => {
                panic!("cannot add a third boundary vertex to a cluster");
            }
        }
    }

    pub fn contains(&self, v: VertexId) -> bool {
        match self {
            Self::None => false,
            Self::One(w) => *w == v,
            Self::Two { left, right } => *left == v || *right == v,
        }
    }

    pub fn remap(&mut self, old: VertexId, new: VertexId) {
        match self {
            Self::None => {}
            Self::One(vertex) => {
                if *vertex == old {
                    *vertex = new;
                }
            }
            Self::Two { left, right } => {
                if *left == old {
                    *left = new;
                }
                if *right == old {
                    *right = new;
                }
            }
        }
    }

    pub fn left(&self) -> Option<VertexId> {
        match self {
            Self::None => None,
            Self::One(v) => Some(*v),
            Self::Two { left, .. } => Some(*left),
        }
    }

    pub fn right(&self) -> Option<VertexId> {
        match self {
            Self::None => None,
            Self::One(v) => Some(*v),
            Self::Two { right, .. } => Some(*right),
        }
    }

    pub fn is_path(&self) -> bool {
        matches!(self, Self::Two { .. })
    }

    pub fn is_point(&self) -> bool {
        !self.is_path()
    }

    pub fn count(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Two { .. } => 2,
        }
    }

    pub fn flip(&mut self) {
        if let Self::Two { left, right } = self {
            std::mem::swap(left, right);
        }
    }

    pub fn flipped(self) -> Self {
        match self {
            Self::None => Self::None,
            Self::One(v) => Self::One(v),
            Self::Two { left, right } => Self::Two {
                left: right,
                right: left,
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
}
