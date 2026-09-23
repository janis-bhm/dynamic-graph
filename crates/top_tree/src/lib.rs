mod tree;

pub use tree::{Edge, Label, Node, Tree};

/// A top tree maintains information about a dynamic forest of trees, where the leaves of the top tree correspond to edges in the underlying forest, and the internal nodes correspond to clusters of connected edges.
