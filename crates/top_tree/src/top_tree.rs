//! A dynamic top tree over a [`Tree`] forest.
//!
//! The top tree is stored as a binary tree of *clusters*. The leaves of this
//! cluster tree are exactly the tree edges and labels of the underlying
//! forest; every internal cluster is the union of its two children, which
//! share a single vertex. The structure is kept balanced with the splay top
//! tree operations of Holm, Rotenberg, and Ryhl.
//!
//! Each cluster additionally stores a value of the user supplied
//! [`Summary`] type, which is recomputed from its children whenever the
//! cluster changes. This is what lets a user maintain arbitrary aggregate
//! information (connectivity, bridge/cover levels, biconnectivity data, ...)
//! over a fully dynamic forest.

use std::hash::Hash;

use crate::{
    NonMaxUsize, Tree,
    summary::{Boundary, MergeContext, Summary},
    tree::SwapResult,
};

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

    /// Returns whether the bit at `index` is set; `false` if out of range.
    fn get(&self, index: usize) -> bool {
        let block = index / Self::BITS;
        self.blocks
            .get(block)
            .is_some_and(|bits| bits >> (index % Self::BITS) & 1 == 1)
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

/// What a cluster node represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClusterData {
    /// The cluster is a single tree edge, stored by its index in the forest.
    Edge(usize),
    /// The cluster is a single label, stored by its index in the forest.
    Label(usize),
    /// The cluster is the union of its two children.
    Internal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Children {
    left: NonMaxUsize,
    right: NonMaxUsize,
}

impl Children {
    fn flip(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right);
    }
}

/// A node of the top tree.
struct Cluster<S: Summary> {
    parent: Option<NonMaxUsize>,
    children: Option<Children>,
    /// Whether the logical orientation of the cluster is reversed.
    flipped: bool,
    /// What this cluster represents.
    data: ClusterData,
    /// The boundary vertices of the cluster.
    boundary_vertices: BoundaryVertices,
    /// The user supplied summary of this cluster.
    sum: S,
    /// A lazy tag waiting to be pushed to this cluster's descendants.
    tag: S::Tag,
}

impl<S: Summary> Cluster<S> {
    fn flipped_boundary_vertices(&self) -> BoundaryVertices {
        if self.flipped {
            self.boundary_vertices.flipped()
        } else {
            self.boundary_vertices
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryVertices {
    None,
    One(NonMaxUsize),
    Two {
        left: NonMaxUsize,
        right: NonMaxUsize,
    },
}

impl BoundaryVertices {
    fn from_left_and_right(left: Option<NonMaxUsize>, right: Option<NonMaxUsize>) -> Self {
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

    fn from_option(v: Option<NonMaxUsize>) -> Self {
        match v {
            None => Self::None,
            Some(v) => Self::One(v),
        }
    }

    fn from_children(left: Self, right: Self, is_path: bool) -> Self {
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

    fn remove(&mut self, v: NonMaxUsize) -> Option<NonMaxUsize> {
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

    fn add(&mut self, v: NonMaxUsize, left: bool) {
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

    fn contains(&self, v: NonMaxUsize) -> bool {
        match self {
            Self::None => false,
            Self::One(w) => *w == v,
            Self::Two { left, right } => *left == v || *right == v,
        }
    }

    fn left(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::One(v) => Some(v.get()),
            Self::Two { left, .. } => Some(left.get()),
        }
    }

    fn right(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::One(v) => Some(v.get()),
            Self::Two { right, .. } => Some(right.get()),
        }
    }

    fn is_path(&self) -> bool {
        matches!(self, Self::Two { .. })
    }

    fn is_point(&self) -> bool {
        !self.is_path()
    }

    fn count(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Two { .. } => 2,
        }
    }

    fn flip(&mut self) {
        if let Self::Two { left, right } = self {
            std::mem::swap(left, right);
        }
    }

    fn flipped(self) -> Self {
        match self {
            Self::None => Self::None,
            Self::One(v) => Self::One(v),
            Self::Two { left, right } => Self::Two {
                left: right,
                right: left,
            },
        }
    }

    fn to_boundary(self) -> Boundary {
        match self {
            Self::None => Boundary::None,
            Self::One(v) => Boundary::One(v.get()),
            Self::Two { left, right } => Boundary::Two {
                left: left.get(),
                right: right.get(),
            },
        }
    }
}

fn shared(a: Boundary, b: Boundary) -> usize {
    let b_slots = b.slots();
    a.slots()
        .into_iter()
        .flatten()
        .find(|vertex| b_slots.contains(&Some(*vertex)))
        .unwrap_or_else(|| {
            panic!("clusters must share a boundary vertex (left: {a:?}, right: {b:?})")
        })
}

/// A stable handle to a label attached with [`TopTree::attach`].
///
/// Unlike the underlying forest indices, a `LabelId` stays valid until the
/// corresponding [`TopTree::detach`] call, even when other labels are removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(pub usize);

/// An opaque handle to a node of a top tree, valid only while the tree is not
/// structurally modified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(usize);

/// What a cluster node represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeData {
    Edge(usize),
    Label(usize),
    Internal,
}

/// A dynamic top tree over a forest of trees.
///
/// `N` is the vertex key type, `S` the user supplied [`Summary`] and `W`/`V`
/// the edge/label and vertex weights of the underlying [`Tree`].
pub struct TopTree<N, L, S>
where
    S: Summary,
{
    tree: Tree<N, L, usize, ()>,
    nodes: Vec<Option<Cluster<S>>>,
    free: Vec<usize>,
    /// Whether each vertex is currently exposed, indexed by vertex index.
    exposed: BitVec,
}

impl<N, L, S> TopTree<N, L, S>
where
    S: Summary,
{
    /// Creates an empty top tree.
    pub fn new() -> Self {
        Self {
            tree: Tree::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            exposed: BitVec::new(),
        }
    }

    /// The underlying forest.
    pub fn forest(&self) -> &Tree<N, L, usize> {
        &self.tree
    }

    /// The number of tree edges currently in the forest.
    pub fn edge_count(&self) -> usize {
        self.tree.edge_count()
    }

    /// The number of labels currently in the forest.
    pub fn label_count(&self) -> usize {
        self.tree.label_count()
    }
}

impl<N, L, S> TopTree<N, L, S>
where
    S: Summary,
    N: Eq + Hash,
{
    /// Adds a vertex with the given key and weight and returns its index.
    ///
    /// Adding a key that is already present returns the existing index.
    pub fn add_vertex(&mut self, key: N) -> usize {
        let index = self.tree.add_node(key, ());
        self.exposed.grow_to(index + 1);
        index
    }

    /// Returns the index of the vertex with the given key.
    pub fn vertex_index(&self, key: &N) -> Option<usize> {
        self.tree.node_index_of(key)
    }

    /// Links the trees containing `u` and `v` with a new tree edge of the
    /// given weight and returns the index of the new edge.
    ///
    /// `u` and `v` must be in different trees.
    pub fn link(&mut self, u: usize, v: usize) -> usize {
        self.link_internal(u, v)
    }

    /// Cuts the tree edge connecting `u` and `v`, returning its weight.
    ///
    /// Returns `None` if there is no such edge.
    pub fn cut(&mut self, u: usize, v: usize) -> Option<()> {
        self.cut_internal(u, v)
    }

    /// Attaches a label with the given weight at `v` and returns a stable
    /// handle to it.
    ///
    /// Labels act as leaf edges that are not part of the spanning forest,
    /// e.g. non-tree edges in the underlying graph.
    pub fn attach(&mut self, v: usize, label: L)
    where
        L: Eq + Hash,
    {
        self.attach_internal(v, label)
    }

    /// Removes the label identified by `label`, returning its weight.
    ///
    /// The handle is invalidated; all other [`LabelId`]s remain valid.
    pub fn detach(&mut self, label: &L) -> Option<()>
    where
        L: Eq + Hash,
    {
        self.detach_internal(label);
        Some(())
    }
}

impl<N, L, S> TopTree<N, L, S>
where
    S: Summary,
    L: Eq + Hash,
{
    /// Mutates the summary of the live label identified by `key` and recomputes
    /// all of its ancestors. The label node itself is not re-linked, so the top
    /// tree's structure is unchanged.
    pub fn update_label_summary(&mut self, key: &L, update: impl FnOnce(&mut S)) {
        let node = *self.tree.label_weight(key).expect("label must exist");
        assert!(
            matches!(self.cl(node).data, ClusterData::Label(_)),
            "update_label_summary expects a label node"
        );
        update(&mut self.cl_mut(node).sum);
        let mut current = self.parent(node);
        while let Some(parent) = current {
            self.recompute(parent);
            current = self.parent(parent);
        }
    }
}

impl<N, L, S> TopTree<N, L, S>
where
    S: Summary,
{
    /// Exposes `u` and `v` and returns the resulting root cluster node.
    pub fn expose_path_node(&mut self, u: usize, v: usize) -> Option<NodeId> {
        self.expose_vertex(u);
        self.expose_vertex(v).map(NodeId)
    }

    /// The summary of `node`.
    pub fn node_summary(&self, node: NodeId) -> &S {
        &self.cl(node.0).sum
    }

    /// Whether `node` is a path cluster.
    pub fn node_is_path(&self, node: NodeId) -> bool {
        self.is_path(node.0)
    }

    /// The logical (flipped) boundary vertices of `node`.
    pub fn node_boundary(&self, node: NodeId) -> Boundary {
        self.cl(node.0).flipped_boundary_vertices().to_boundary()
    }

    /// The logical children of an internal `node`.
    pub fn node_children(&self, node: NodeId) -> Option<(NodeId, NodeId)> {
        self.cl(node.0).children?;
        let (left, right) = self.flipped_children(node.0);
        Some((NodeId(left), NodeId(right)))
    }

    /// The vertex shared by the two children of an internal `node`.
    pub fn node_central(&self, node: NodeId) -> Option<usize> {
        self.cl(node.0).children?;
        let (left, right) = self.flipped_children(node.0);
        let left_vertices = self.cl(left).flipped_boundary_vertices().to_boundary();
        let right_vertices = self.cl(right).flipped_boundary_vertices().to_boundary();
        Some(shared(left_vertices, right_vertices))
    }

    /// What `node` represents.
    pub fn node_leaf_data(&self, node: NodeId) -> NodeData {
        match self.cl(node.0).data {
            ClusterData::Edge(edge) => NodeData::Edge(edge),
            ClusterData::Label(index) => NodeData::Label(index),
            ClusterData::Internal => NodeData::Internal,
        }
    }

    /// The key of the label represented by `node`, if it is a label leaf.
    pub fn node_label_key(&self, node: NodeId) -> Option<&L> {
        match self.cl(node.0).data {
            ClusterData::Label(index) => self.tree.label_key(index),
            _ => None,
        }
    }

    /// Pushes `node`'s pending lazy tag to its path children. This does not
    /// change the top tree's structure and is used by traversal code that
    /// needs up-to-date child summaries.
    pub fn push_node_tag(&mut self, node: NodeId) {
        self.push_tag(node.0);
    }
}

impl<N, L, S> TopTree<N, L, S>
where
    S: Summary + Clone,
{
    /// Exposes `v`, making it an external boundary vertex, and returns the
    /// summary of the resulting root cluster.
    ///
    /// A tree can have at most two external boundary vertices. Callers must
    /// [`deexpose`](Self::deexpose) vertices to make room before exposing a
    /// vertex whose component already has two exposed vertices.
    ///
    /// Returns `None` if `v` has no incident edges or labels.
    pub fn expose(&mut self, v: usize) -> Option<S> {
        self.expose_vertex(v).map(|node| self.cl(node).sum.clone())
    }

    /// Exposes `u` and `v`, making them the two boundaries of the tree
    /// containing them, and returns the summary of the resulting path
    /// cluster.
    ///
    /// The path of the returned cluster is exactly the `u`-`v` path. Returns
    /// `None` unless `u` and `v` are both incident to an edge or label.
    pub fn expose_path(&mut self, u: usize, v: usize) -> Option<S> {
        self.expose_vertex(u);
        self.expose_vertex(v).map(|node| self.cl(node).sum.clone())
    }

    /// Applies a lazy tag to the path between `u` and `v` and returns the
    /// resulting summary.
    pub fn expose_path_tagged(&mut self, u: usize, v: usize, tag: S::Tag) -> Option<S> {
        let node = {
            self.expose_vertex(u);
            self.expose_vertex(v)
        };
        node.map(|node| {
            self.apply_tag(node, &tag);
            self.cl(node).sum.clone()
        })
    }

    /// Removes the external boundary status from `v`.
    pub fn deexpose(&mut self, v: usize) -> Option<S> {
        self.deexpose_vertex(v)
            .map(|node| self.cl(node).sum.clone())
    }

    /// Returns whether `u` and `v` are in the same tree.
    pub fn connected(&mut self, u: usize, v: usize) -> bool {
        if u == v {
            return true;
        }

        let root_u = self.find_root(u);
        let root_v = self.find_root(v);

        match (root_u, root_v) {
            (Some(u), Some(v)) => u == v,
            _ => false,
        }
    }

    /// Returns the summary of the root cluster of the tree containing `v`.
    ///
    /// The boundaries of that root are the currently exposed vertices.
    pub fn component_summary(&self, v: usize) -> Option<S> {
        let leaf = self.incident_leaves(v).next()?;
        let mut node = leaf;
        while let Some(parent) = self.parent(node) {
            node = parent;
        }
        Some(self.cl(node).sum.clone())
    }
}

impl<N, L, S> Default for TopTree<N, L, S>
where
    S: Summary,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Low level cluster tree operations.
impl<N, L, S> TopTree<N, L, S>
where
    S: Summary,
{
    #[inline]
    fn cl(&self, node: usize) -> &Cluster<S> {
        self.nodes[node]
            .as_ref()
            .expect("top tree node must be live")
    }

    #[inline]
    fn cl_mut(&mut self, node: usize) -> &mut Cluster<S> {
        self.nodes[node]
            .as_mut()
            .expect("top tree node must be live")
    }

    fn alloc(&mut self, cluster: Cluster<S>) -> usize {
        if let Some(index) = self.free.pop() {
            self.nodes[index] = Some(cluster);
            index
        } else {
            self.nodes.push(Some(cluster));
            self.nodes.len() - 1
        }
    }

    fn dealloc(&mut self, node: usize) {
        self.nodes[node] = None;
        self.free.push(node);
    }

    #[inline]
    fn parent(&self, node: usize) -> Option<usize> {
        self.cl(node).parent.map(NonMaxUsize::get)
    }

    #[inline]
    fn set_parent(&mut self, node: usize, parent: Option<usize>) {
        self.cl_mut(node).parent = parent.and_then(NonMaxUsize::new);
    }

    fn is_left_child(&self, node: usize) -> Option<bool> {
        let parent = self.parent(node)?;
        self.cl(parent)
            .children
            .map(|children| children.left.get() == node)
    }

    /// Returns the child of `node` on the given side.
    ///
    /// `left` refers to the logical (unflipped) orientation.
    fn child(&self, node: usize, is_left: bool) -> Option<usize> {
        let Children { left, right } = self.cl(node).children?;
        if is_left {
            Some(left.get())
        } else {
            Some(right.get())
        }
    }

    fn set_child(&mut self, node: usize, child: usize, is_left: bool) {
        let Some(Children {
            ref mut left,
            ref mut right,
        }) = self.cl_mut(node).children
        else {
            panic!("internal node must have children");
        };

        unsafe {
            if is_left {
                *left = NonMaxUsize::new_unchecked(child);
            } else {
                *right = NonMaxUsize::new_unchecked(child);
            }
        }

        self.cl_mut(child).parent = NonMaxUsize::new(node);
    }

    /// Returns the two children in logical (unflipped) order.
    fn flipped_children(&self, node: usize) -> (usize, usize) {
        let cluster = self.cl(node);
        let Some(Children { left, right }) = cluster.children else {
            panic!("internal node must have children");
        };

        let (left, right) = (left.get(), right.get());

        if cluster.flipped {
            (right, left)
        } else {
            (left, right)
        }
    }

    fn sibling(&self, node: usize) -> Option<usize> {
        let parent = self.cl(node).parent?.get();
        let cluster = self.cl(parent);

        let sibling = if cluster.children?.left.get() == node {
            cluster.children?.right.get()
        } else {
            cluster.children?.left.get()
        };

        Some(sibling)
    }

    fn set_flipped(&mut self, node: usize, flipped: bool) {
        let cluster = self.cl_mut(node);
        if cluster.flipped != flipped {
            cluster.flipped = flipped;
            S::flip(&mut cluster.sum);
        }
    }

    fn toggle_flipped(&mut self, node: usize) {
        self.set_flipped(node, !self.cl(node).flipped);
    }

    fn add_boundary(&mut self, node: usize, vertex: usize) {
        let v = unsafe { NonMaxUsize::new_unchecked(vertex) };

        let (data, children) = {
            let cluster = self.cl(node);
            (cluster.data, cluster.children)
        };

        match data {
            ClusterData::Edge(edge) => {
                // `boundary_vertices` is stored in the cluster's local
                // (unflipped) frame, so compare against the physical endpoints.
                let (left, right) = self.tree.edge_endpoints(edge).expect("edge must exist");
                if left == vertex {
                    self.cl_mut(node).boundary_vertices.add(v, true);
                } else if right == vertex {
                    self.cl_mut(node).boundary_vertices.add(v, false);
                } else {
                    panic!("vertex must be an endpoint of the edge");
                }
            }
            ClusterData::Label(_) => {
                self.cl_mut(node).boundary_vertices.add(v, false);
            }
            ClusterData::Internal => {
                let Children { left, right } = children.expect("internal node must have children");
                let (left, right) = (left.get(), right.get());
                let in_left = self.cl(left).boundary_vertices.contains(v);
                let in_right = self.cl(right).boundary_vertices.contains(v);
                if in_left && in_right {
                    // `vertex` is the central vertex. It sits on the side of the
                    // child that is a path cluster, because that child's outer
                    // boundary is the node's other boundary.
                    let is_left = !self.is_path(left);
                    self.cl_mut(node).boundary_vertices.add(v, is_left);
                } else if in_left {
                    self.cl_mut(node).boundary_vertices.add(v, true);
                } else if in_right {
                    self.cl_mut(node).boundary_vertices.add(v, false);
                } else {
                    panic!("vertex must be a boundary of one of the children");
                }
            }
        }
    }

    fn is_path(&self, node: usize) -> bool {
        self.cl(node).boundary_vertices.is_path()
    }

    fn is_point(&self, node: usize) -> bool {
        self.cl(node).boundary_vertices.is_point()
    }

    fn num_path_children(&self, node: usize) -> usize {
        let cluster = self.cl(node);
        if let Some(Children { left, right }) = cluster.children {
            usize::from(self.is_path(left.get())) + usize::from(self.is_path(right.get()))
        } else {
            0
        }
    }

    fn has_middle_boundary(&self, node: usize) -> bool {
        if matches!(self.cl(node).boundary_vertices, BoundaryVertices::None) {
            return false;
        }

        match self.cl(node).data {
            ClusterData::Internal => {
                self.cl(node).boundary_vertices.count() != self.num_path_children(node) as u8
            }
            _ => false,
        }
    }

    fn middle_boundary(&self, node: usize) -> Option<usize> {
        match self.cl(node).boundary_vertices {
            BoundaryVertices::None => None,
            BoundaryVertices::One(v) => {
                if self.num_path_children(node) == 0 {
                    Some(v.get())
                } else {
                    None
                }
            }
            BoundaryVertices::Two { left, right } => {
                let Children { left: l, right: r } = self
                    .cl(node)
                    .children
                    .expect("internal node must have children");
                let (l, r) = (self.is_path(l.get()), self.is_path(r.get()));

                match (l, r) {
                    (true, true) => None,
                    (true, false) => Some(right.get()),
                    (false, true) => Some(left.get()),
                    (false, false) => None,
                }
            }
        }
    }

    /// Returns the rightmost boundary vertex of `node`, or `None` if it has no boundary vertices.
    /// This is the rightmost vertex in the logical (unflipped) orientation of the cluster.
    fn rightmost_boundary(&self, node: usize) -> Option<usize> {
        self.cl(node).boundary_vertices.right()
    }

    /// Returns the leftmost boundary vertex of `node`, or `None` if it has no boundary vertices.
    /// This is the leftmost vertex in the logical (unflipped) orientation of the cluster.
    fn leftmost_boundary(&self, node: usize) -> Option<usize> {
        self.cl(node).boundary_vertices.left()
    }

    fn is_boundary_vertex(&self, vertex: usize) -> bool {
        self.exposed.get(vertex) || self.tree.degree(vertex) >= 2
    }

    fn has_left_boundary(&self, node: usize) -> bool {
        match self.cl(node).data {
            ClusterData::Edge(edge) => {
                let (left, right) = self.tree.edge_endpoints(edge).expect("edge must exist");
                let endpoint = if self.cl(node).flipped { right } else { left };
                self.is_boundary_vertex(endpoint)
            }
            ClusterData::Label(label) => {
                let vertex = self
                    .tree
                    .label_index(label)
                    .expect("label must exist")
                    .node_id();
                self.is_boundary_vertex(vertex)
            }
            ClusterData::Internal => {
                let left = self
                    .child(node, !self.cl(node).flipped)
                    .expect("internal node must have children");
                self.is_path(left)
            }
        }
    }

    fn has_right_boundary(&self, node: usize) -> bool {
        match self.cl(node).data {
            ClusterData::Edge(edge) => {
                let (left, right) = self.tree.edge_endpoints(edge).expect("edge must exist");
                let endpoint = if self.cl(node).flipped { left } else { right };
                self.is_boundary_vertex(endpoint)
            }
            ClusterData::Label(label) => {
                let vertex = self
                    .tree
                    .label_index(label)
                    .expect("label must exist")
                    .node_id();
                self.is_boundary_vertex(vertex)
            }
            ClusterData::Internal => {
                let right = self
                    .child(node, self.cl(node).flipped)
                    .expect("internal node must have children");
                self.is_path(right)
            }
        }
    }

    fn set_exposed(&mut self, vertex: usize, exposed: bool) {
        self.exposed.set(vertex, exposed);
    }

    fn incident_leaves(&self, vertex: usize) -> impl Iterator<Item = usize> + '_ {
        let edges = self.tree.incident_edge_weights(vertex);

        let labels = self.tree.incident_label_weights(vertex);

        edges.chain(labels).cloned()
    }

    fn has_at_most_one_incident_element(&self, vertex: usize) -> bool {
        self.tree.incident_edge_indices(vertex).count()
            + self.tree.incident_label_indices(vertex).count()
            <= 1
    }

    fn apply_tag(&mut self, node: usize, tag: &S::Tag) {
        S::apply(&mut self.cl_mut(node).sum, tag);
        S::compose(&mut self.cl_mut(node).tag, tag);
    }

    /// Pushes the pending tag of `node` down to its *path* children, which are
    /// exactly the children whose cluster path is contained in the path of
    /// `node`. Raked (point) children contain no path edges and are left
    /// untouched.
    fn push_tag(&mut self, node: usize) {
        let tag = std::mem::take(&mut self.cl_mut(node).tag);
        if !self.is_path(node) {
            return;
        }

        if let Some(Children { left, right }) = self.cl(node).children {
            for child in [left, right].map(NonMaxUsize::get) {
                if self.is_path(child) {
                    S::apply(&mut self.cl_mut(child).sum, &tag);
                    S::compose(&mut self.cl_mut(child).tag, &tag);
                }
            }
        }
    }

    fn push_flip(&mut self, node: usize) {
        self.push_tag(node);
        if self.cl(node).flipped {
            let cluster = self.cl_mut(node);
            cluster.flipped = false;
            cluster.children.as_mut().map(Children::flip);
            cluster.boundary_vertices.flip();

            if let Some(Children { left, right }) = cluster.children {
                self.toggle_flipped(left.get());
                self.toggle_flipped(right.get());
            }
        }
    }

    /// Recomputes the summary of an internal node from its children.
    fn recompute(&mut self, node: usize) {
        self.push_tag(node);
        let (left, right) = self.flipped_children(node);
        let left_vertices = self.cl(left).flipped_boundary_vertices().to_boundary();
        let right_vertices = self.cl(right).flipped_boundary_vertices().to_boundary();
        let parent_vertices = self.cl(node).flipped_boundary_vertices().to_boundary();
        let ctx = MergeContext {
            left_boundary: left_vertices,
            right_boundary: right_vertices,
            boundary: parent_vertices,
            central: shared(left_vertices, right_vertices),
        };
        let sum = S::combine(&self.cl(left).sum, &self.cl(right).sum, &ctx);
        self.cl_mut(node).sum = sum;
    }

    fn recompute_new_internal(
        &mut self,
        left: usize,
        right: usize,
        boundary_vertices: BoundaryVertices,
    ) -> Cluster<S> {
        let left_vertices = self.cl(left).flipped_boundary_vertices().to_boundary();
        let right_vertices = self.cl(right).flipped_boundary_vertices().to_boundary();
        let parent_vertices = boundary_vertices.to_boundary();
        let ctx = MergeContext {
            left_boundary: left_vertices,
            right_boundary: right_vertices,
            boundary: parent_vertices,
            central: shared(left_vertices, right_vertices),
        };
        let sum = S::combine(&self.cl(left).sum, &self.cl(right).sum, &ctx);
        Cluster {
            parent: None,
            children: unsafe {
                Some(Children {
                    left: NonMaxUsize::new_unchecked(left),
                    right: NonMaxUsize::new_unchecked(right),
                })
            },
            flipped: false,
            boundary_vertices,
            data: ClusterData::Internal,
            sum,
            tag: S::Tag::default(),
        }
    }

    /// Creates a new internal node from `left` and `right`, linking them.
    fn new_internal(
        &mut self,
        left: usize,
        right: usize,
        boundary_vertices: BoundaryVertices,
    ) -> usize {
        let cluster = self.recompute_new_internal(left, right, boundary_vertices);
        let node = self.alloc(cluster);
        self.set_parent(left, Some(node));
        self.set_parent(right, Some(node));
        node
    }

    fn new_leaf_edge(
        &mut self,
        edge: usize,
        u: usize,
        v: usize,
        boundary_vertices: BoundaryVertices,
    ) -> usize {
        let sum = S::tree_edge(u, v);
        let cluster = Cluster {
            parent: None,
            children: None,
            flipped: false,
            boundary_vertices,
            data: ClusterData::Edge(edge),
            sum,
            tag: S::Tag::default(),
        };

        self.alloc(cluster)
    }

    fn new_leaf_label(&mut self, label: usize, vertex: usize, boundary: BoundaryVertices) -> usize {
        let sum = S::label(vertex);
        let cluster = Cluster {
            parent: None,
            children: None,
            flipped: false,
            boundary_vertices: boundary,
            data: ClusterData::Label(label),
            sum,
            tag: S::Tag::default(),
        };

        self.alloc(cluster)
    }

    /// Rotates `node` up, above its parent and grandparent.
    /// Rotating a node up is legal under the following condition:
    /// sibling(node) \cup sibling(parent(node)) must be a valid cluster,
    /// i.e. its children must share a vertex, and it must not have more than 2 boundary vertices.
    ///
    /// Returns `None` if `node` has no grandparent.
    fn rotate_up(&mut self, node: usize) -> Option<()> {
        let sibling = self.sibling(node)?;
        let parent = self.parent(node)?;
        let uncle = self.sibling(parent)?;
        let grandparent = self.parent(parent)?;

        self.push_flip(grandparent);
        self.push_flip(parent);

        let uncle_is_left = self.is_left_child(uncle)?;
        let sibling_is_left = self.is_left_child(sibling)?;
        let same_sides = uncle_is_left == sibling_is_left;
        let sibling_is_path = self.is_path(sibling);
        let uncle_is_path = self.is_path(uncle);
        let grandparent_is_path = self.is_path(grandparent);

        // the new parent made up of the sibling and uncle will be a path
        // consider the star case:
        // the middle boundary of gp is shared by `node`, `sibling`, and `uncle`.
        // the new parent `sibling` \cup `uncle` will have the same middle boundary; if either `sibling` or `uncle` is a path, then the new parent will be a path.
        // both `sibling` and `uncle` cannot be paths.
        // consider the path case:
        // the shared vertex of gp is shared by `parent` and `uncle`, and must be the outward boundary of `sibling`, which is a path.
        // if gp has a middle boundary, then it must be this vertex, and it must be a boundary of the new parent.
        // if gp has no middle boundary but uncle is a path, then the boundary of the new parent will be the two non-shared boundaries of `sibling` and `uncle`.
        let new_parent_boundary;
        let flip_new_parent;
        let flip_grandparent;
        // The new parent's physical left child is `uncle` when `uncle_is_left`,
        // otherwise `sibling`. Merge the children's logical boundaries (the
        // orientation in which a child contributes to the merge) in that
        // physical order.
        let sibling_boundary = self.cl(sibling).flipped_boundary_vertices();
        let uncle_boundary = self.cl(uncle).flipped_boundary_vertices();
        let (left_boundary, right_boundary) = if uncle_is_left {
            (uncle_boundary, sibling_boundary)
        } else {
            (sibling_boundary, uncle_boundary)
        };
        if same_sides && sibling_is_path {
            // path rotation
            let grandparent_has_middle = self.has_middle_boundary(grandparent);
            new_parent_boundary = BoundaryVertices::from_children(
                left_boundary,
                right_boundary,
                grandparent_has_middle || uncle_is_path,
            );
            flip_new_parent = false;
            if grandparent_has_middle && !grandparent_is_path {
                match self.is_left_child(grandparent) {
                    Some(grandparent_is_left) => {
                        flip_grandparent = grandparent_is_left == uncle_is_left;
                    }
                    None => flip_grandparent = false,
                }
            } else {
                flip_grandparent = false;
            }
        } else {
            // star rotation
            if !same_sides {
                flip_new_parent = sibling_is_path;
                flip_grandparent = sibling_is_path;
                self.toggle_flipped(node);

                new_parent_boundary = BoundaryVertices::from_children(
                    left_boundary,
                    right_boundary,
                    sibling_is_path || uncle_is_path,
                );
            } else {
                // sibling is a point cluster on the same side as the uncle
                flip_new_parent = false;
                flip_grandparent = false;
                self.toggle_flipped(sibling);

                new_parent_boundary =
                    BoundaryVertices::from_children(left_boundary, right_boundary, uncle_is_path);
            }
        }

        self.set_child(parent, sibling, !uncle_is_left);
        self.set_child(parent, uncle, uncle_is_left);
        self.set_flipped(parent, flip_new_parent);
        self.cl_mut(parent).boundary_vertices = new_parent_boundary;

        self.set_child(grandparent, node, !uncle_is_left);
        self.set_child(grandparent, parent, uncle_is_left);
        self.set_flipped(grandparent, flip_grandparent);

        self.recompute(parent);
        self.recompute(grandparent);

        Some(())
    }

    /// Performs one splay step at `node`, returning the next node to splay if
    /// the splay is not finished.
    fn splay_step(&mut self, node: usize) -> Option<usize> {
        let mut node = node;
        loop {
            let parent = self.parent(node)?;
            let grandparent = self.parent(parent)?;

            if self.is_point(node) && self.is_point(grandparent) {
                self.rotate_up(node);
                return Some(grandparent);
            }

            let great_grandparent = self.parent(grandparent)?;

            if self.is_path(parent)
                && (self.is_path(grandparent) || self.is_point(great_grandparent))
            {
                self.push_flip(grandparent);
                self.push_flip(parent);

                let node_is_left = self.is_left_child(node)?;
                let parent_is_left = self.is_left_child(parent)?;
                let grandparent_is_left = self.is_left_child(grandparent)?;

                if node_is_left == parent_is_left {
                    self.rotate_up(node);
                    return Some(grandparent);
                }

                if parent_is_left == grandparent_is_left {
                    self.rotate_up(parent);
                    return Some(great_grandparent);
                }

                debug_assert_eq!(node_is_left, grandparent_is_left);
                let sibling = self.sibling(node)?;
                self.rotate_up(sibling);
                self.rotate_up(parent);
                return Some(great_grandparent);
            }

            node = parent;
        }
    }

    fn semi_splay(&mut self, node: usize) {
        let mut node = node;
        while let Some(next) = self.splay_step(node) {
            node = next;
        }
    }

    fn full_splay(&mut self, node: usize) {
        while let Some(next) = self.splay_step(node) {
            self.splay_step(next);
        }
    }

    fn find_root(&mut self, vertex: usize) -> Option<usize> {
        let mut node = self.incident_leaves(vertex).next()?;

        while let Some(parent) = self.parent(node) {
            node = parent;
        }

        Some(node)
    }

    /// Finds the least common ancestor of all leaves incident to `vertex`.
    fn find_consuming_node(&mut self, vertex: usize) -> Option<usize> {
        let node = self.incident_leaves(vertex).next()?;
        self.semi_splay(node);

        // if the vertex has exactly one incident leaf, then the consuming node is the incident leaf.
        if self.has_at_most_one_incident_element(vertex) {
            return Some(node);
        }

        let (left_endpoint, right_endpoint) = match self.cl(node).data {
            ClusterData::Edge(edge) => self.tree.edge_endpoints(edge).expect("edge must exist"),
            ClusterData::Label(label) => {
                let v = self
                    .tree
                    .label_index(label)
                    .expect("label must exist")
                    .node_id();
                (v, v)
            }
            ClusterData::Internal => unreachable!("consuming node must be a leaf"),
        };

        let flip = self.cl(node).flipped;
        let mut is_left = (left_endpoint == vertex) != flip;
        let mut is_right = (right_endpoint == vertex) != flip;
        let mut is_middle = false;
        let mut last_middle_node = None;
        let mut node = node;

        while let Some(parent) = self.parent(node) {
            let is_left_child = self
                .is_left_child(node)
                .expect("node with parent must have a side");
            is_middle = if is_left_child {
                is_right || (is_middle && !self.has_right_boundary(node))
            } else {
                is_left || (is_middle && !self.has_left_boundary(node))
            };
            is_left = (is_left_child != self.cl(parent).flipped) && !is_middle;
            is_right = (is_left_child == self.cl(parent).flipped) && !is_middle;
            node = parent;
            if is_middle {
                if !self.has_middle_boundary(node) {
                    return Some(node);
                }
                last_middle_node = Some(node);
            }
        }

        last_middle_node
    }

    /// Increments the boundary count from `node` up to the root, flipping
    /// children as needed so that the exposed vertex ends up on the correct
    /// side.
    fn expose_prepared(&mut self, node: usize, vertex: usize) -> usize {
        let mut left = false;
        let mut right = false;
        let mut node = node;
        loop {
            self.add_boundary(node, vertex);
            let Some(parent) = self.parent(node) else {
                if matches!(self.cl(node).data, ClusterData::Internal) {
                    self.recompute(node);
                }
                return node;
            };
            let is_left_child = self
                .is_left_child(node)
                .expect("node with parent must have a side");
            let is_right_child = !is_left_child;
            if (is_left_child && right) || (is_right_child && left) {
                self.toggle_flipped(node);
            }
            if matches!(self.cl(node).data, ClusterData::Internal) {
                self.recompute(node);
            }
            left = is_left_child != self.cl(parent).flipped;
            right = is_right_child != self.cl(parent).flipped;
            node = parent;
        }
    }

    /// There must be at most one exposed vertex in the tree containing `consuming_node`.
    fn prepare_expose(&mut self, consuming_node: usize) -> usize {
        let mut consuming_node = consuming_node;
        let mut node = consuming_node;

        // we want each cluster from the consuming node up to the root to hold
        // the following invariant:
        // the cluster should either be a point cluster, or a path cluster with
        // the exposed vertex as one of its boundary vertices.
        while let Some(parent) = self.parent(node) {
            if self.is_point(node) {
                node = parent;
            } else {
                // This iterations goal is to either raise the consuming node,
                // or a point cluster ancestor of the consuming node, up one
                // level.
                self.push_flip(parent);
                self.push_flip(node);

                let sibling = self.sibling(node).expect("node with parent has sibling");

                let sibling_is_left = self
                    .is_left_child(sibling)
                    .expect("sibling with parent has side");
                let same_side_child = self
                    .child(node, sibling_is_left)
                    .expect("internal node has children");

                // recall that the conditions for rotating up are that the
                // sibling and uncle must form a valid cluster.
                //
                // because of the orientational invariant, we know that
                // same_side_child and sibling share the sibling-ward most
                // boundary vertex of `node`.
                //
                // We want to produce a root-level cluster that is either a point cluster.
                // Alternatively, a path cluster with the to-be-exposed
                // vertex as one of its boundary vertices means the vertex
                // is already exposed.
                // The middle vertex of `parent` is not the to-be-exposed
                // vertex, or else it would have been the consuming node.
                // `node` is a path cluster, of which the to-be-exposed vertex.
                // Because there can be at most one exposed vertex in the
                // current tree, there must above `node` be a point cluster.
                if self.is_path(same_side_child) || self.is_point(sibling) {
                    // Rotating up the other-sided child of the consuming node
                    // will make `parent` the new consuming node and we don't
                    // care whether any of its children are point clusters.

                    // If both children of `node` are path clusters, then `node`
                    // is the consuming node.

                    // If `sibling` is a point cluster and `node` is not the
                    // consuming node, then `node` must have a point child which
                    // is an ancestor of the to-be-exposed vertex.
                    //
                    // If `same_side_child` is that ancestor, merging it with
                    // `sibling` will produce a point cluster, and we have
                    // achieved our goal.
                    //
                    // If `other_side_child` is that ancestor, then merging
                    // `same_side_child` with `sibling` will produce a cluster
                    // that does not contain the to-be-exposed vertex, and
                    // `other_side_child` is the ancestor of the to-be-exposed
                    // vertex and a point cluster, and we have achieved our
                    // goal.

                    let other_side_child = self
                        .child(node, !sibling_is_left)
                        .expect("internal node has children");
                    let _ = self.rotate_up(other_side_child);
                    if node == consuming_node {
                        consuming_node = parent;
                    }
                    node = parent;
                } else {
                    // we pull down the uncle of `node` into the sibling position and try again.
                    let uncle = self.sibling(parent).expect("parent has sibling");
                    let uncle_is_left = self
                        .is_left_child(uncle)
                        .expect("uncle with parent has side");
                    if sibling_is_left == uncle_is_left {
                        let _ = self.rotate_up(node);
                    } else {
                        let _ = self.rotate_up(sibling);
                    }
                }
            }
        }
        consuming_node
    }

    /// Exposes `vertex`, returning the root node of its top tree.
    fn expose_vertex(&mut self, vertex: usize) -> Option<usize> {
        // Exposing an already exposed vertex is a no-op: its boundary is
        // already present on the path to the root.
        if self.exposed.get(vertex) {
            return self.find_root(vertex);
        }

        match self.find_consuming_node(vertex) {
            Some(consuming) => {
                let consuming = self.prepare_expose(consuming);
                let node = self.expose_prepared(consuming, vertex);
                self.set_exposed(vertex, true);
                Some(node)
            }
            None => {
                self.set_exposed(vertex, true);
                None
            }
        }
    }

    /// Removes the exposed status of `vertex`, returning the root node.
    fn deexpose_vertex(&mut self, vertex: usize) -> Option<usize> {
        // Deexposing a vertex that is not exposed is a no-op, matching the
        // idempotent behaviour of `expose_vertex`.
        if !self.exposed.get(vertex) {
            return None;
        }

        let consuming = self.find_consuming_node(vertex);

        // Collect the path from the consuming node up to the root.
        let mut path = Vec::new();
        let mut node = consuming;
        while let Some(current) = node {
            path.push(current);
            node = self.parent(current);
        }

        // Flush pending tags from the root down. A tag can only be pushed onto
        // path children, so this must happen while the nodes on the path still
        // carry the boundary that makes them path clusters. Pushing bottom-up
        // would let a child lose its boundary before its parent's tag reaches
        // it, dropping the tag for the edges of that child.
        for &current in path.iter().rev() {
            self.push_tag(current);
        }

        let mut root = None;
        for &current in path.iter() {
            let before = self.cl(current).boundary_vertices;
            let _contains = self
                .cl_mut(current)
                .boundary_vertices
                .remove(unsafe { NonMaxUsize::new_unchecked(vertex) });
            assert!(
                _contains.is_some(),
                "boundary of node {current} must contain vertex {vertex}, but was {before:?} \
                 (consuming={consuming:?})"
            );

            if matches!(self.cl(current).data, ClusterData::Internal) {
                self.recompute(current);
            }
            root = Some(current);
        }
        self.set_exposed(vertex, false);
        root
    }

    fn delete_all_ancestors(&mut self, node: usize) {
        if let Some(parent) = self.parent(node) {
            let sibling = self.sibling(node).expect("node with parent has sibling");
            self.delete_all_ancestors(parent);
            self.set_parent(sibling, None);
        }
        self.dealloc(node);
    }

    fn link_internal(&mut self, u: usize, v: usize) -> usize {
        let mut root_u = self.expose_vertex(u);
        if let Some(node) = root_u
            && self.has_left_boundary(node)
        {
            self.toggle_flipped(node);
        }
        self.set_exposed(u, false);

        let mut root_v = self.expose_vertex(v);
        if let Some(node) = root_v
            && self.has_right_boundary(node)
        {
            self.toggle_flipped(node);
        }
        self.set_exposed(v, false);

        let leaf = self.new_leaf_edge(self.tree.edge_count(), u, v, unsafe {
            BoundaryVertices::from_left_and_right(
                root_u.map(|_| NonMaxUsize::new_unchecked(u)),
                root_v.map(|_| NonMaxUsize::new_unchecked(v)),
            )
        });
        let edge = self.tree.add_edge(u, v, leaf);

        let mut node = leaf;
        if let Some(root_u) = root_u.take() {
            node = self.new_internal(
                root_u,
                node,
                BoundaryVertices::from_option(
                    root_v.map(|_| unsafe { NonMaxUsize::new_unchecked(v) }),
                ),
            );
        }
        if let Some(root_v) = root_v.take() {
            node = self.new_internal(
                node,
                root_v,
                BoundaryVertices::from_option(
                    root_u.map(|_| unsafe { NonMaxUsize::new_unchecked(u) }),
                ),
            );
        }

        let _ = node;
        edge
    }

    fn cut_internal(&mut self, u: usize, v: usize) -> Option<()> {
        let edge = self.tree.edge_index_of(u, v)?;
        let leaf = *self.tree.edge_weight(edge).expect("edge must exist");

        self.full_splay(leaf);
        self.delete_all_ancestors(leaf);

        let (_weight, swap) = self.tree.remove_edge(edge).expect("edge must exist");

        if let SwapResult::Some { current, .. } = swap {
            let idx = self.tree.edge_index(current).unwrap().weight;
            self.cl_mut(idx).data = ClusterData::Edge(current);
        }

        self.set_exposed(u, true);
        self.set_exposed(v, true);
        self.deexpose_vertex(u);
        self.deexpose_vertex(v);

        Some(())
    }

    fn attach_internal(&mut self, vertex: usize, label: L)
    where
        L: Eq + Hash,
    {
        let root_v = self.expose_vertex(vertex);
        if let Some(node) = root_v
            && self.has_left_boundary(node)
        {
            self.toggle_flipped(node);
        }
        self.set_exposed(vertex, false);

        let leaf = self.new_leaf_label(self.tree.label_count(), vertex, unsafe {
            BoundaryVertices::from_option(root_v.map(|_| NonMaxUsize::new_unchecked(vertex)))
        });

        self.tree.add_label(vertex, label, leaf);

        if let Some(root_v) = root_v {
            self.new_internal(root_v, leaf, BoundaryVertices::None);
        }
    }

    fn detach_internal(&mut self, label: &L)
    where
        L: Eq + Hash,
    {
        let vertex = self.tree.label(label).expect("label must exist").node_id();
        let label_node = *self.tree.label_weight(label).expect("label must exist");
        // Expose the vertex first so that removing the label undoes the
        // boundary the exposure added. This keeps `num_boundary` consistent
        // even though it is maintained relative to each cluster's parent
        // context, which a plain path-based decrement cannot account for.
        let _ = self.expose_vertex(vertex);

        // Labels are never path components. Splay this leaf next to the root
        // so removing its parent leaves the rest of the component as one root
        // cluster.
        self.full_splay(label_node);
        if let Some(parent) = self.parent(label_node) {
            self.push_flip(parent);
            let sibling = self.sibling(label_node).expect("label has sibling");
            assert!(
                self.parent(parent).is_none(),
                "splayed label parent must be root"
            );
            self.set_parent(sibling, None);
            self.dealloc(parent);
        }
        self.dealloc(label_node);

        let (_weight, swap) = self.tree.remove_label(label).expect("label must exist");

        if let SwapResult::Some { current, .. } = swap {
            let idx = self.tree.label_index(current).unwrap().weight;
            self.cl_mut(idx).data = ClusterData::Label(current);
        }

        let _ = self.deexpose_vertex(vertex);
    }
}

#[cfg(test)]
mod tests;
