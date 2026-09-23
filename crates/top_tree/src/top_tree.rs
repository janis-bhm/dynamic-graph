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

use std::{hash::Hash, marker::PhantomData};

use crate::{
    Tree,
    summary::{MergeContext, Summary},
};

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

/// A node of the top tree.
struct Cluster<S: Summary<W>, W> {
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    /// Whether the logical orientation of the cluster is reversed.
    flipped: bool,
    /// The number of boundary vertices of the cluster (0, 1, or 2).
    num_boundary: u8,
    data: ClusterData,
    /// The user supplied summary of this cluster.
    sum: S,
    /// A lazy tag waiting to be pushed to this cluster's descendants.
    tag: S::Tag,
    _marker: PhantomData<fn() -> W>,
}

/// A stable handle to a label attached with [`TopTree::attach`].
///
/// Unlike the underlying forest indices, a `LabelId` stays valid until the
/// corresponding [`TopTree::detach`] call, even when other labels are removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(pub usize);

/// A dynamic top tree over a forest of trees.
///
/// `N` is the vertex key type, `S` the user supplied [`Summary`] and `W`/`V`
/// the edge/label and vertex weights of the underlying [`Tree`].
pub struct TopTree<N, S, W = (), V = ()>
where
    S: Summary<W>,
{
    tree: Tree<N, W, V>,
    nodes: Vec<Option<Cluster<S, W>>>,
    free: Vec<usize>,
    /// The top tree leaf of each forest edge, indexed by edge index.
    edge_leaf: Vec<Option<usize>>,
    /// The top tree leaf of each forest label, indexed by label index.
    label_leaf: Vec<Option<usize>>,
    /// Stable [`LabelId`] of each forest label, indexed by label index.
    tree_label_ids: Vec<Option<usize>>,
    /// Forest label index of each stable [`LabelId`].
    label_ids: Vec<Option<usize>>,
    /// Free slots in `label_ids`.
    free_label_ids: Vec<usize>,
    /// Whether each vertex is currently exposed, indexed by vertex index.
    exposed: Vec<bool>,
}

impl<N, S, W, V> TopTree<N, S, W, V>
where
    S: Summary<W>,
{
    /// Creates an empty top tree.
    pub fn new() -> Self {
        Self {
            tree: Tree::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            edge_leaf: Vec::new(),
            label_leaf: Vec::new(),
            tree_label_ids: Vec::new(),
            label_ids: Vec::new(),
            free_label_ids: Vec::new(),
            exposed: Vec::new(),
        }
    }

    /// The underlying forest.
    pub fn forest(&self) -> &Tree<N, W, V> {
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

impl<N: Eq + Hash, S, W, V> TopTree<N, S, W, V>
where
    S: Summary<W>,
{
    /// Adds a vertex with the given key and weight and returns its index.
    ///
    /// Adding a key that is already present returns the existing index.
    pub fn add_vertex(&mut self, key: N, weight: V) -> usize {
        let index = self.tree.add_node(key, weight);
        if self.exposed.len() <= index {
            self.exposed.resize(index + 1, false);
        }
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
    pub fn link(&mut self, u: usize, v: usize, weight: W) -> usize {
        self.link_internal(u, v, weight)
    }

    /// Cuts the tree edge connecting `u` and `v`, returning its weight.
    ///
    /// Returns `None` if there is no such edge.
    pub fn cut(&mut self, u: usize, v: usize) -> Option<W> {
        self.cut_internal(u, v)
    }

    /// Attaches a label with the given weight at `v` and returns a stable
    /// handle to it.
    ///
    /// Labels act as leaf edges that are not part of the spanning forest,
    /// e.g. non-tree edges in the underlying graph.
    pub fn attach(&mut self, v: usize, weight: W) -> LabelId {
        let tree_label = self.attach_internal(v, weight);

        let id = match self.free_label_ids.pop() {
            Some(id) => {
                self.label_ids[id] = Some(tree_label);
                id
            }
            None => {
                self.label_ids.push(Some(tree_label));
                self.label_ids.len() - 1
            }
        };
        self.tree_label_ids[tree_label] = Some(id);

        LabelId(id)
    }

    /// Removes the label identified by `label`, returning its weight.
    ///
    /// The handle is invalidated; all other [`LabelId`]s remain valid.
    pub fn detach(&mut self, label: LabelId) -> Option<W> {
        let tree_label = self.label_ids.get(label.0).copied().flatten()?;
        let weight = self.detach_internal(tree_label);
        self.label_ids[label.0] = None;
        self.free_label_ids.push(label.0);
        weight
    }
}

impl<N, S, W, V> TopTree<N, S, W, V>
where
    S: Summary<W> + Clone,
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

impl<N, S, W, V> Default for TopTree<N, S, W, V>
where
    S: Summary<W>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Low level cluster tree operations.
impl<N, S, W, V> TopTree<N, S, W, V>
where
    S: Summary<W>,
{
    #[inline]
    fn cl(&self, node: usize) -> &Cluster<S, W> {
        self.nodes[node]
            .as_ref()
            .expect("top tree node must be live")
    }

    #[inline]
    fn cl_mut(&mut self, node: usize) -> &mut Cluster<S, W> {
        self.nodes[node]
            .as_mut()
            .expect("top tree node must be live")
    }

    fn alloc(&mut self, cluster: Cluster<S, W>) -> usize {
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
        self.cl(node).parent
    }

    #[inline]
    fn set_parent(&mut self, node: usize, parent: Option<usize>) {
        self.cl_mut(node).parent = parent;
    }

    fn is_left_child(&self, node: usize) -> Option<bool> {
        let parent = self.cl(node).parent?;
        Some(self.cl(parent).left == Some(node))
    }

    /// Returns the child of `node` on the given side.
    ///
    /// `left` refers to the logical (unflipped) orientation.
    fn child(&self, node: usize, left: bool) -> Option<usize> {
        if left {
            self.cl(node).left
        } else {
            self.cl(node).right
        }
    }

    fn set_child(&mut self, node: usize, child: usize, left: bool) {
        if left {
            self.cl_mut(node).left = Some(child);
        } else {
            self.cl_mut(node).right = Some(child);
        }
        self.cl_mut(child).parent = Some(node);
    }

    /// Returns the two children in logical (unflipped) order.
    fn flipped_children(&self, node: usize) -> (usize, usize) {
        let cluster = self.cl(node);
        let (left, right) = (
            cluster.left.expect("internal node must have a left child"),
            cluster
                .right
                .expect("internal node must have a right child"),
        );
        if cluster.flipped {
            (right, left)
        } else {
            (left, right)
        }
    }

    fn sibling(&self, node: usize) -> Option<usize> {
        let parent = self.cl(node).parent?;
        let cluster = self.cl(parent);
        if cluster.left == Some(node) {
            cluster.right
        } else {
            cluster.left
        }
    }

    fn set_flipped(&mut self, node: usize, flipped: bool) {
        self.cl_mut(node).flipped = flipped;
    }

    fn toggle_flipped(&mut self, node: usize) {
        self.cl_mut(node).flipped = !self.cl(node).flipped;
    }

    fn set_num_boundary(&mut self, node: usize, num: u8) {
        debug_assert!(num <= 2, "num_boundary must be <= 2");
        self.cl_mut(node).num_boundary = num;
    }

    fn inc_num_boundary(&mut self, node: usize) {
        let num = self.cl(node).num_boundary + 1;
        self.set_num_boundary(node, num);
    }

    fn dec_num_boundary(&mut self, node: usize) {
        let num = self
            .cl(node)
            .num_boundary
            .checked_sub(1)
            .expect("num_boundary must be >= 0");
        self.set_num_boundary(node, num);
    }

    fn is_path(&self, node: usize) -> bool {
        self.cl(node).num_boundary == 2
    }

    fn is_point(&self, node: usize) -> bool {
        self.cl(node).num_boundary < 2
    }

    fn num_path_children(&self, node: usize) -> usize {
        let cluster = self.cl(node);
        usize::from(cluster.left.is_some_and(|c| self.is_path(c)))
            + usize::from(cluster.right.is_some_and(|c| self.is_path(c)))
    }

    fn has_middle_boundary(&self, node: usize) -> bool {
        if self.cl(node).num_boundary == 0 {
            return false;
        }
        match self.cl(node).data {
            ClusterData::Internal => {
                self.cl(node).num_boundary != self.num_path_children(node) as u8
            }
            _ => false,
        }
    }

    fn is_boundary_vertex(&self, vertex: usize) -> bool {
        self.exposed.get(vertex).copied().unwrap_or(false) || self.tree.degree(vertex) >= 2
    }

    fn has_left_boundary(&self, node: usize) -> bool {
        match self.cl(node).data {
            ClusterData::Edge(edge) => {
                let (left, right) = self.tree.edge_endpoints(edge).expect("edge must exist");
                let endpoint = if self.cl(node).flipped { right } else { left };
                self.is_boundary_vertex(endpoint)
            }
            ClusterData::Label(label) => {
                let vertex = self.tree.label(label).expect("label must exist").node_id();
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
                let vertex = self.tree.label(label).expect("label must exist").node_id();
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
        if let Some(slot) = self.exposed.get_mut(vertex) {
            *slot = exposed;
        }
    }

    fn incident_leaves(&self, vertex: usize) -> impl Iterator<Item = usize> + '_ {
        let edges = self
            .tree
            .incident_edge_indices(vertex)
            .filter_map(|edge| self.edge_leaf[edge]);
        let labels = self
            .tree
            .incident_label_indices(vertex)
            .filter_map(|label| self.label_leaf[label]);
        edges.chain(labels)
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
        let (left, right) = (self.cl(node).left, self.cl(node).right);
        for child in [left, right].into_iter().flatten() {
            if self.is_path(child) {
                S::apply(&mut self.cl_mut(child).sum, &tag);
                S::compose(&mut self.cl_mut(child).tag, &tag);
            }
        }
    }

    fn push_flip(&mut self, node: usize) {
        self.push_tag(node);
        if self.cl(node).flipped {
            let (left, right) = (self.cl(node).left, self.cl(node).right);
            let cluster = self.cl_mut(node);
            cluster.flipped = false;
            cluster.left = right;
            cluster.right = left;
            if let Some(child) = left {
                self.toggle_flipped(child);
            }
            if let Some(child) = right {
                self.toggle_flipped(child);
            }
        }
    }

    /// Recomputes the summary of an internal node from its children.
    fn recompute(&mut self, node: usize) {
        self.push_tag(node);
        let (left, right) = self.flipped_children(node);
        let ctx = MergeContext {
            left_boundary: self.cl(left).num_boundary,
            right_boundary: self.cl(right).num_boundary,
            boundary: self.cl(node).num_boundary,
        };
        let sum = S::combine(&self.cl(left).sum, &self.cl(right).sum, &ctx);
        self.cl_mut(node).sum = sum;
    }

    fn recompute_new_internal(
        &mut self,
        left: usize,
        right: usize,
        num_boundary: u8,
    ) -> Cluster<S, W> {
        let ctx = MergeContext {
            left_boundary: self.cl(left).num_boundary,
            right_boundary: self.cl(right).num_boundary,
            boundary: num_boundary,
        };
        let sum = S::combine(&self.cl(left).sum, &self.cl(right).sum, &ctx);
        Cluster {
            parent: None,
            left: Some(left),
            right: Some(right),
            flipped: false,
            num_boundary,
            data: ClusterData::Internal,
            sum,
            tag: S::Tag::default(),
            _marker: PhantomData,
        }
    }

    /// Creates a new internal node from `left` and `right`, linking them.
    fn new_internal(&mut self, left: usize, right: usize, num_boundary: u8) -> usize {
        let cluster = self.recompute_new_internal(left, right, num_boundary);
        let node = self.alloc(cluster);
        self.set_parent(left, Some(node));
        self.set_parent(right, Some(node));
        node
    }

    fn new_leaf_edge(&mut self, edge: usize, u: usize, v: usize, num_boundary: u8) -> usize {
        let weight = &self.tree.edge_index(edge).expect("edge must exist").weight;
        let sum = S::tree_edge(weight, u, v);
        let cluster = Cluster {
            parent: None,
            left: None,
            right: None,
            flipped: false,
            num_boundary,
            data: ClusterData::Edge(edge),
            sum,
            tag: S::Tag::default(),
            _marker: PhantomData,
        };
        let node = self.alloc(cluster);
        self.edge_leaf[edge] = Some(node);
        node
    }

    fn new_leaf_label(&mut self, label: usize, num_boundary: u8) -> usize {
        let sum = {
            let label_ref = self.tree.label(label).expect("label must exist");
            S::label(&label_ref.weight, label_ref.node_id())
        };
        let cluster = Cluster {
            parent: None,
            left: None,
            right: None,
            flipped: false,
            num_boundary,
            data: ClusterData::Label(label),
            sum,
            tag: S::Tag::default(),
            _marker: PhantomData,
        };
        let node = self.alloc(cluster);
        self.label_leaf[label] = Some(node);
        node
    }

    /// Rotates `node` up, above its parent and grandparent.
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

        let new_parent_is_path;
        let flip_new_parent;
        let flip_grandparent;
        if same_sides && sibling_is_path {
            // path rotation
            let grandparent_has_middle = self.has_middle_boundary(grandparent);
            new_parent_is_path = grandparent_has_middle || uncle_is_path;
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
        } else if !same_sides {
            // star rotation
            new_parent_is_path = sibling_is_path || uncle_is_path;
            flip_new_parent = sibling_is_path;
            flip_grandparent = sibling_is_path;
            self.toggle_flipped(node);
        } else {
            // sibling is a point cluster on the same side as the uncle
            new_parent_is_path = uncle_is_path;
            flip_new_parent = false;
            flip_grandparent = false;
            self.toggle_flipped(sibling);
        }

        self.set_child(parent, sibling, !uncle_is_left);
        self.set_child(parent, uncle, uncle_is_left);
        self.set_flipped(parent, flip_new_parent);
        self.set_num_boundary(parent, if new_parent_is_path { 2 } else { 1 });

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

    /// Finds the least common ancestor of all leaves incident to `vertex`.
    fn find_consuming_node(&mut self, vertex: usize) -> Option<usize> {
        let node = self.incident_leaves(vertex).next()?;
        self.semi_splay(node);

        if self.has_at_most_one_incident_element(vertex) {
            return Some(node);
        }

        let (left_endpoint, right_endpoint) = match self.cl(node).data {
            ClusterData::Edge(edge) => self.tree.edge_endpoints(edge).expect("edge must exist"),
            ClusterData::Label(label) => {
                let v = self.tree.label(label).expect("label must exist").node_id();
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
    fn expose_prepared(&mut self, node: usize) -> usize {
        let mut left = false;
        let mut right = false;
        let mut node = node;
        loop {
            self.inc_num_boundary(node);
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

    fn prepare_expose(&mut self, consuming_node: usize) -> usize {
        let mut consuming_node = consuming_node;
        let mut node = consuming_node;
        while let Some(parent) = self.parent(node) {
            if self.is_point(node) {
                node = parent;
            } else {
                self.push_flip(parent);
                self.push_flip(node);
                let sibling = self.sibling(node).expect("node with parent has sibling");
                let sibling_is_left = self
                    .is_left_child(sibling)
                    .expect("sibling with parent has side");
                let same_side_child = self
                    .child(node, sibling_is_left)
                    .expect("internal node has children");
                if self.is_path(same_side_child) || self.is_point(sibling) {
                    let other_side_child = self
                        .child(node, !sibling_is_left)
                        .expect("internal node has children");
                    let _ = self.rotate_up(other_side_child);
                    if node == consuming_node {
                        consuming_node = parent;
                    }
                    node = parent;
                } else {
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
        match self.find_consuming_node(vertex) {
            Some(consuming) => {
                let consuming = self.prepare_expose(consuming);
                let node = self.expose_prepared(consuming);
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
        let mut node = self.find_consuming_node(vertex);
        let mut root = None;
        while let Some(current) = node {
            // Flush any pending tag while the cluster is still a path cluster,
            // so that the affected edges keep their values once the cluster
            // loses a boundary.
            self.push_tag(current);
            self.dec_num_boundary(current);
            if matches!(self.cl(current).data, ClusterData::Internal) {
                self.recompute(current);
            }
            node = self.parent(current);
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

    fn link_internal(&mut self, u: usize, v: usize, weight: W) -> usize {
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

        let edge = self.tree.add_edge(u, v, weight);
        self.edge_leaf.push(None);
        let leaf = self.new_leaf_edge(
            edge,
            u,
            v,
            u8::from(root_u.is_some()) + u8::from(root_v.is_some()),
        );

        let mut node = leaf;
        if let Some(root_u) = root_u.take() {
            node = self.new_internal(root_u, node, u8::from(root_v.is_some()));
        }
        if let Some(root_v) = root_v.take() {
            node = self.new_internal(node, root_v, 0);
        }

        let _ = node;
        edge
    }

    fn cut_internal(&mut self, u: usize, v: usize) -> Option<W> {
        let edge = self.tree.edge_index_of(u, v)?;
        let leaf = self.edge_leaf[edge].expect("edge must have a top tree leaf");

        self.full_splay(leaf);
        self.delete_all_ancestors(leaf);

        let last = self.tree.edge_count() - 1;
        let weight = self.tree.remove_edge(edge);
        let moved = self.edge_leaf.pop().expect("edge leaf entry must exist");
        if edge < last {
            self.edge_leaf[edge] = moved;
            if let Some(node) = moved {
                self.cl_mut(node).data = ClusterData::Edge(edge);
            }
        }

        self.set_exposed(u, true);
        self.set_exposed(v, true);
        self.deexpose_vertex(u);
        self.deexpose_vertex(v);

        weight
    }

    fn attach_internal(&mut self, vertex: usize, weight: W) -> usize {
        let root_v = self.expose_vertex(vertex);
        if let Some(node) = root_v
            && self.has_left_boundary(node)
        {
            self.toggle_flipped(node);
        }
        self.set_exposed(vertex, false);

        let label = self.tree.add_label(vertex, weight);
        self.label_leaf.push(None);
        self.tree_label_ids.push(None);
        let leaf = self.new_leaf_label(label, u8::from(root_v.is_some()));

        if let Some(root_v) = root_v {
            self.new_internal(root_v, leaf, 0);
        }

        label
    }

    fn detach_internal(&mut self, label: usize) -> Option<W> {
        let label_node = self.label_leaf[label].expect("label must have a top tree leaf");

        if let Some(parent) = self.parent(label_node) {
            let sibling = self.sibling(label_node).expect("label has sibling");
            if let Some(grandparent) = self.parent(parent) {
                let parent_is_left = self
                    .is_left_child(parent)
                    .expect("parent with grandparent has side");
                let sibling_is_left = self
                    .is_left_child(sibling)
                    .expect("sibling with parent has side");
                let flip_sibling = (sibling_is_left != parent_is_left) ^ self.cl(parent).flipped;
                if flip_sibling {
                    self.toggle_flipped(sibling);
                }
                self.set_child(grandparent, sibling, parent_is_left);
            } else {
                self.set_parent(sibling, None);
            }
            self.dealloc(parent);
        }
        self.dealloc(label_node);

        let last = self.tree.label_count() - 1;
        let weight = self.tree.remove_label(label);
        let moved = self.label_leaf.pop().expect("label leaf entry must exist");
        let moved_id = self
            .tree_label_ids
            .pop()
            .expect("label id entry must exist");
        if label < last {
            self.label_leaf[label] = moved;
            if let Some(node) = moved {
                self.cl_mut(node).data = ClusterData::Label(label);
            }
            self.tree_label_ids[label] = moved_id;
            if let Some(id) = moved_id {
                self.label_ids[id] = Some(label);
            }
        }

        weight
    }
}

#[cfg(test)]
mod tests;
