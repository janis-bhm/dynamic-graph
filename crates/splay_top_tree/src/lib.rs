//! A Top Tree is a tree data structure representing a forest of trees.
//!
#![expect(internal_features)]
#![feature(pattern_types, pattern_type_macro, structural_match)]

use core::fmt;
use std::{
    alloc::{Allocator, Global},
    collections::BTreeSet,
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
};

use crate::{
    boundary::BoundaryVertices,
    node::{Handle, LabelNode, LeafNode, Node, marker},
    tree::{Endpoints, SwapResult},
};

pub use index::{EdgeId, LabelId, VertexId};
use slotvec::SlotVec;
pub use summary::{MergeContext, Summary};

mod boundary;
mod index;
mod iter;
mod node;
mod non_max;
#[cfg(test)]
mod oracle;
mod summary;
#[cfg(test)]
mod tests;
mod tree;
mod util;

fn incident_leaves<W, A: Allocator>(
    root: &TopTree<W, A>,
    v: tree::VertexId,
) -> impl Iterator<Item = Handle<W, marker::Leaf>> + '_ {
    root.tree
        .incident_label_weights(v)
        .map(|l| Handle::<W, marker::Leaf>::new(l.node.cast()))
        .chain(
            root.tree
                .incident_edge_weights(v)
                .map(|e| Handle::<W, marker::Leaf>::new(e.node.cast())),
        )
}

/// Finds the least common ancestor of all leaves incident to `v` in the tree rooted at `root`.
/// if `v` has no incident leaves, returns `None`.
pub(crate) fn find_consuming_node<W, A: Allocator>(
    root: &TopTree<W, A>,
    v: tree::VertexId,
) -> Option<Handle<W, marker::Either>>
where
    W: Summary,
{
    let node = incident_leaves(root, v).next()?;
    unsafe { ptr::read(&node) }.semi_splay();

    // if the vertex has exactly one incident edge, then the consuming node is the incident leaf.
    if root.tree.is_exactly_degree_n(v, 1) {
        return Some(node.forget_type());
    }

    let (mut is_left, mut is_right) = match node.force() {
        LeafOrInternal::Edge(edge) => {
            let endpoints = root.tree.edge_endpoints(edge.edge());
            let flip = node.is_flipped();
            (
                (endpoints.left() == v) != flip,
                (endpoints.right() == v) != flip,
            )
        }
        // A label is a point leaf: reversing it cannot move its vertex from
        // one end to the other. In particular, a flipped label still has the
        // vertex at both logical borders (unlike an edge leaf, whose endpoints
        // are exchanged by the flip).
        LeafOrInternal::Label(label) => {
            let vertex = root.tree.label_vertex(label.label());
            (vertex == v, vertex == v)
        }
        LeafOrInternal::Internal(_) => unreachable!("node must be a leaf"),
    };
    let mut is_middle = false;

    let mut last_middle_node = None;
    let mut node = node.forget_type();
    while let Some(parent) = node.parent() {
        let is_left_child = node.is_left_child().expect("node must have a parent");

        is_middle = if is_left_child {
            is_right || (is_middle && !node.has_right_boundary(root))
        } else {
            is_left || (is_middle && !node.has_left_boundary(root))
        };
        is_left = (is_left_child != parent.is_flipped()) && !is_middle;
        is_right = (is_left_child == parent.is_flipped()) && !is_middle;

        node = parent.forget_type();

        if is_middle {
            if !node.has_middle_boundary() {
                assert!(!root.is_exposed_internal(v));
                return Some(node);
            }
            last_middle_node = Some(unsafe { ptr::read(&node) });
        }
    }

    last_middle_node
}

/// Exposes the vertex `v` in the top tree rooted at `root`, returning a handle to the consuming node of `v` if it exists.
/// If `v` has no incident edges, then `None` is returned.
pub(crate) fn expose<W, A: Allocator>(
    v: VertexId,
    root: &mut TopTree<W, A>,
) -> Option<Handle<W, marker::Either>>
where
    W: Summary,
{
    fn expose_prepared<W, A: Allocator>(
        mut node: Handle<W, marker::Either>,
        tree: &mut TopTree<W, A>,
        v: VertexId,
        resolved: tree::VertexId,
    ) -> Handle<W, marker::Either>
    where
        W: Summary,
    {
        let mut left = false;
        let mut right = false;

        loop {
            node.add_boundary_vertex(tree, v, resolved);

            if let LeafOrInternal::Internal(mut internal) = node.force() {
                internal.recompute_weight();
            }

            let Some(parent) = node.parent() else {
                return node;
            };

            let is_left_child = node.is_left_child().expect("node must have a parent");
            let is_right_child = !is_left_child;

            if (is_left_child && right) || (is_right_child && left) {
                node.toggle_flipped();
            }

            left = is_left_child != parent.is_flipped();
            right = is_right_child != parent.is_flipped();
            node = parent.forget_type();
        }
    }

    /// precondition: root of the tree containing the to-be-exposed vertex is a point.
    fn prepare_expose<W>(mut consuming_node: Handle<W, marker::Either>) -> Handle<W, marker::Either>
    where
        W: Summary,
    {
        let mut node = unsafe { ptr::read(&consuming_node) };

        // we want each cluster from the consuming node up to the root to hold
        // the following invariant:
        // the cluster should either be a point cluster, or a path cluster with
        // the exposed vertex as one of its boundary vertices.
        while let Some(parent) = node.parent() {
            if node.is_point() {
                node = parent.forget_type();
            } else {
                // This iterations goal is to either raise the consuming node,
                // or a point cluster ancestor of the consuming node, up one
                // level.

                // we know that `node` is an internal node at this point:
                // let `node` be a leaf edge cluster with end points `v` and
                // `w`, where `v` is degree 1 and `w` is degree >= 2 and `v` is
                // the to-be-exposed vertex.
                // for `(v,w)` to have a parent, `w` must have degree >= 2.
                // it follows that `v` is not yet exposed, and the consuming
                // node of `v` is the leaf `(v,w)`.
                // because `v` is not exposed, `(v,w)` must be a point cluster.
                assert!(node.is_internal(), "{node:?} must be an internal node");
                let internal = unsafe { ptr::read(&node).into_internal() };
                parent.push_flip();
                internal.push_flip();

                let mut sibling = internal.sibling().expect("node must have a sibling");

                let sibling_is_left = sibling.is_left_child().expect("sibling must have a parent");
                let same_side_child = internal.child(sibling_is_left);

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
                if same_side_child.is_path() || sibling.is_point() {
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

                    let mut other_side_child = internal.child(!sibling_is_left);
                    other_side_child
                        .rotate_up()
                        .expect("rotate_up should succeed");

                    if node == consuming_node {
                        consuming_node = unsafe { ptr::read(&parent).forget_type() };
                    }
                    node = parent.forget_type();
                } else {
                    // we pull down the uncle of `node` into the sibling position and try again.
                    let uncle = parent.sibling().expect("parent must have a sibling");
                    let uncle_is_left = uncle.is_left_child().expect("uncle must have a parent");

                    if sibling_is_left == uncle_is_left {
                        node.rotate_up().expect("rotate_up should succeed");
                    } else {
                        sibling.rotate_up().expect("rotate_up should succeed");
                    }
                }
            }
        }

        consuming_node
    }

    let resolved = root.resolve_vertex(v);
    assert!(
        !root.is_exposed_internal(resolved),
        "vertex must not be exposed"
    );

    match find_consuming_node(root, resolved) {
        Some(mut consuming_node) => {
            consuming_node.recursively_split_down();

            let consuming_node = prepare_expose(consuming_node);

            let node = expose_prepared(consuming_node, root, v, resolved);
            root.set_exposed(resolved, true);

            Some(node)
        }
        None => {
            root.set_exposed(resolved, true);
            None
        }
    }
}

pub(crate) fn deexpose<W, A: Allocator>(
    v: VertexId,
    tree: &mut TopTree<W, A>,
) -> Option<Handle<W, marker::Either>>
where
    W: Summary,
{
    let resolved = tree.resolve_vertex(v);

    let consuming = find_consuming_node(tree, resolved);
    let mut root = None;

    let mut node = consuming;
    while let Some(mut some_node) = node {
        some_node.remove_from_boundary(v);
        node = some_node.parent().map(Handle::forget_type);

        if let LeafOrInternal::Internal(mut internal) = some_node.force() {
            internal.recompute_weight();
        }

        root = Some(some_node);
    }

    tree.set_exposed(resolved, false);

    root
}

pub(crate) fn link<W, A: Allocator>(
    u: VertexId,
    v: VertexId,
    weight: impl FnOnce(EdgeId) -> W,
    tree: &mut TopTree<W, A>,
) -> EdgeId
where
    W: Summary,
{
    let uu = tree.resolve_vertex(u);
    let vv = tree.resolve_vertex(v);

    let mut ru = expose(u, tree);
    if let Some(ref mut tu) = ru {
        if tu.has_left_boundary(tree) {
            tu.toggle_flipped();
        }

        assert_eq!(tu.flipped_boundary().right(), Some(u));
    }
    tree.set_exposed(uu, false);

    let mut rv = expose(v, tree);
    if let Some(ref mut tv) = rv {
        if tv.has_right_boundary(tree) {
            tv.toggle_flipped();
        }

        assert_eq!(tv.flipped_boundary().left(), Some(v));
    }
    tree.set_exposed(vv, false);

    let (leaf, _) =
        Box::into_non_null_with_allocator(Box::new_in(MaybeUninit::uninit(), &tree.allocator));

    let (edge, mut node) = tree.edge_ids.push_with(|id| {
        let edge = tree.tree.add_edge(
            uu,
            vv,
            EdgeInfo {
                node: leaf.cast(),
                id,
            },
        );
        let leaf = Handle::init_edge(
            leaf,
            weight(id),
            edge,
            BoundaryVertices::from_left_and_right(ru.as_ref().map(|_| u), rv.as_ref().map(|_| v)),
        );

        (edge, leaf.forget_type())
    });

    if let Some(ru) = ru {
        let left = ru;
        let right = node;
        let rv_boundary = BoundaryVertices::from_option(rv.as_ref().map(|_| v));
        node = Handle::alloc_internal(left, right, rv_boundary, &tree.allocator).forget_type();
    }

    if let Some(rv) = rv {
        let left = node;
        let right = rv;

        node = Handle::alloc_internal(left, right, BoundaryVertices::None, &tree.allocator)
            .forget_type();
    }

    _ = node;

    edge
}

#[expect(clippy::type_complexity)]
pub(crate) fn cut<W, A: Allocator>(
    u: VertexId,
    v: VertexId,
    tree: &mut TopTree<W, A>,
) -> (
    Option<Handle<W, marker::Either>>,
    Option<Handle<W, marker::Either>>,
)
where
    W: Summary,
{
    let uu = tree.resolve_vertex(u);
    let vv = tree.resolve_vertex(v);

    let Some((id, edge)) = tree
        .tree
        .find_edge_with_endpoints(uu, vv)
        .map(|id| (id, tree.tree.edge(id)))
    else {
        return (None, None);
    };

    let edge_handle = Handle::new_edge(edge.weight.node.cast::<Node<W>>());

    unsafe { ptr::read(&edge_handle) }.full_splay();
    edge_handle.delete_all_ancestors(&tree.allocator);

    let (weight, swap) = tree.tree.remove_edge(id);
    tree.edge_ids.remove(weight.id);

    if let SwapResult::Swapped { prev, next } = swap {
        _ = (prev, next);
        let id = tree.tree.edge_weight(next).id;
        tree.edge_ids[id] = next;
        unsafe { tree.tree.edge_weight_mut(next).node.as_mut().edge = next };
        // translate tree::EdgeId -> index::EdgeId
        // swap in self.edge_ids
    }

    tree.set_exposed(uu, true);
    tree.set_exposed(vv, true);

    let ru = deexpose(u, tree);
    let rv = deexpose(v, tree);

    (ru, rv)
}

pub(crate) fn attach<W, A: Allocator>(
    v: VertexId,
    weight: impl FnOnce(LabelId) -> W,
    tree: &mut TopTree<W, A>,
) -> (LabelId, NonNull<Node<W>>)
where
    W: Summary,
{
    let mut rv = expose(v, tree);
    if let Some(ref mut tv) = rv
        && Handle::has_left_boundary(tv, tree)
    {
        tv.toggle_flipped();
    }
    let resolved = tree.resolve_vertex(v);
    tree.set_exposed(resolved, false);

    let (label, mut node) = tree.label_ids.push_with(|id| {
        let (leaf, _) =
            Box::into_non_null_with_allocator(Box::new_in(MaybeUninit::uninit(), &tree.allocator));

        let label = tree.tree.add_label(
            resolved,
            LabelInfo {
                node: leaf.cast(),
                id,
            },
        );
        let leaf = Handle::init_label(
            leaf,
            weight(id),
            label,
            BoundaryVertices::from_option(rv.as_ref().map(|_| v)),
        );

        (label, leaf.forget_type())
    });

    if let Some(rv) = rv {
        let left = rv;
        let right = node;

        node = Handle::alloc_internal(left, right, BoundaryVertices::None, &tree.allocator)
            .forget_type();
    }

    (label, node.node)
}

pub(crate) fn detach<W, A: Allocator>(l: LabelId, tree: &mut TopTree<W, A>)
where
    W: Summary,
{
    // labels are never path components, so removing them can never disconnect the tree.
    // instead, we want to replace the label's parent with the label's sibling, then delete the label and parent.
    let resolved = tree.resolve_label(l);
    let node = tree.tree.label_weight(resolved).node;
    let v = *tree.tree.vertex_weight(tree.tree.label_vertex(resolved));
    let label = Handle::<W, _>::new_label(node.cast());

    let _ = expose(v, tree);
    unsafe { ptr::read(&label) }.full_splay();

    if let Some(parent) = label.parent() {
        parent.push_flip();
        let mut sibling = label.sibling().expect("has parent; must have sibling");
        assert!(parent.parent().is_none(), "parent must be root");
        sibling.set_parent(None);

        unsafe {
            Node::dealloc(label.node, &tree.allocator);
            Node::dealloc(parent.node, &tree.allocator);
        }
    }

    let (l, swap) = tree.tree.remove_label(resolved);
    tree.label_ids.remove(l.id);

    if let SwapResult::Swapped { prev, next } = swap {
        _ = (prev, next);
        let id = tree.tree.label_weight(next).id;
        tree.label_ids[id] = next;
        unsafe { tree.tree.label_weight_mut(next).node.as_mut().label = next };
    }

    let _ = deexpose(v, tree);
}

#[derive(Debug, PartialEq, Eq)]
enum LeafOrInternal<T, V, U> {
    Edge(T),
    Label(V),
    Internal(U),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Entry<'a, W> {
    handle: Handle<W, marker::Either>,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a, W> Entry<'a, W> {
    pub fn weight(&self) -> &W {
        unsafe { &self.handle.node.as_ref().weight }
    }

    pub fn into_root(self) -> Self {
        Self {
            handle: self.handle.root(),
            _marker: core::marker::PhantomData,
        }
    }
}

struct EdgeInfo<W> {
    /// The leaf node in the top tree structure corresponding to this edge
    node: NonNull<LeafNode<W>>,
    /// The edge id in the top tree structure
    id: index::EdgeId,
}

struct LabelInfo<W> {
    /// The leaf node in the top tree structure corresponding to this label
    node: NonNull<LabelNode<W>>,
    /// The label id in the top tree structure
    id: index::LabelId,
}

pub struct TopTree<W, A: Allocator = Global> {
    // the underlying forest stores the edges and per-edge/per-label information.
    // we keep the top-tree node pointers here, as well as the TopTree edge id for edges so that there is a O(1) mapping from tree::EdgeId -> index::EdgeId and vice versa.
    tree: tree::Tree<index::VertexId, EdgeInfo<W>, LabelInfo<W>>,
    // indexed by tree vertex index, true if the vertex is exposed in the top tree.
    exposed: slotvec::BitVec,
    // stable mapping of `tree::VertexId`s
    // the `tree::VertexId` is the index in the `tree::Tree` structure, which change when edges are removed.
    vertex_ids: SlotVec<tree::VertexId, VertexId>,
    edge_ids: SlotVec<tree::EdgeId, EdgeId>,
    label_ids: SlotVec<tree::LabelId, LabelId>,
    allocator: A,
}

unsafe impl<W, A: Allocator + Send> Send for TopTree<W, A> where W: Send {}
unsafe impl<W, A: Allocator + Sync> Sync for TopTree<W, A> where W: Sync {}

/// A [`TopTree`] with currently-exposed vertices, which does not allow the
/// removal of vertices or adding and removing edges or labels.
#[repr(transparent)]
pub struct ExposedTopTree<W, A: Allocator = Global> {
    inner: TopTree<W, A>,
}

/// The weight of a currently-exposed cluster, together with a dormant handle
/// to the [`TopTree`] it lives in.
///
/// While the weight is borrowed the tree is inaccessible; calling
/// [`DormantTopTreeWeightMut::awaken`] consumes the dormant value and hands
/// back the tree, so the weight borrow and a tree borrow can never overlap.
pub struct DormantTopTreeWeightMut<'a, W, A: Allocator> {
    inner: NonNull<TopTree<W, A>>,
    weight: &'a mut W,
}

impl<'a, W, A: Allocator> DormantTopTreeWeightMut<'a, W, A> {
    fn new(tree: &'a mut TopTree<W, A>, handle: &Handle<W, marker::Either>) -> Self {
        // SAFETY: `handle` points into `tree`, and `tree` is borrowed for `'a`,
        // so no other reference to the weight is reachable for `'a`.
        let weight = unsafe { &mut (*handle.node.as_ptr()).weight };

        DormantTopTreeWeightMut {
            inner: NonNull::from(tree),
            weight,
        }
    }

    pub fn weight(&mut self) -> &mut W {
        self.weight
    }

    /// Ends the weight borrow and returns the tree, which still has the same
    /// vertices exposed.
    pub fn awaken(self) -> &'a mut ExposedTopTree<W, A> {
        unsafe { (&mut *self.inner.as_ptr()).as_exposed_mut() }
    }
}

impl<W, A: Allocator> ExposedTopTree<W, A> {
    /// Returns the set of vertices in the connected component containing `v`.
    pub fn component_of(&self, v: VertexId) -> BTreeSet<VertexId> {
        self.inner.component_of(v)
    }

    pub fn is_boundary_vertex(&self, v: VertexId) -> bool {
        self.inner.is_boundary_vertex(v)
    }

    pub fn degree(&self, v: VertexId) -> usize {
        self.inner.degree(v)
    }

    pub fn edge_endpoints(&self, e: EdgeId) -> Option<[VertexId; 2]> {
        self.inner.edge_endpoints(e)
    }

    pub fn find_edge(&self, v: VertexId, w: VertexId) -> Option<EdgeId> {
        self.inner.find_edge(v, w)
    }

    pub fn label_vertex(&self, l: LabelId) -> VertexId {
        self.inner.label_vertex(l)
    }

    pub fn first_label(&self, v: VertexId) -> Option<LabelId> {
        self.inner.first_label(v)
    }

    pub fn is_exposed(&self, v: VertexId) -> bool {
        self.inner.is_exposed(v)
    }

    pub fn incident_edges(&self, v: VertexId) -> impl Iterator<Item = EdgeId> + '_ {
        self.inner.incident_edges(v)
    }

    pub fn is_connected(&self, v: VertexId, w: VertexId) -> bool {
        self.inner.is_connected(v, w)
    }

    pub fn add_vertex(&mut self) -> VertexId {
        self.inner.add_vertex()
    }
}

impl<W, A: Allocator> TopTree<W, A> {
    #[expect(dead_code)]
    fn as_exposed(&self) -> &ExposedTopTree<W, A> {
        unsafe { mem::transmute::<&Self, &ExposedTopTree<W, A>>(self) }
    }

    fn as_exposed_mut(&mut self) -> &mut ExposedTopTree<W, A> {
        unsafe { mem::transmute::<&mut Self, &mut ExposedTopTree<W, A>>(self) }
    }

    fn resolve_vertex(&self, v: VertexId) -> tree::VertexId {
        *self
            .vertex_ids
            .get(v)
            .expect("vertex must exist in the tree")
    }

    fn try_resolve_edge(&self, e: EdgeId) -> Option<tree::EdgeId> {
        self.edge_ids.get(e).copied()
    }

    fn resolve_label(&self, l: LabelId) -> tree::LabelId {
        *self.label_ids.get(l).expect("label must exist in the tree")
    }

    fn set_exposed(&mut self, v: tree::VertexId, exposed: bool) {
        self.exposed.set(v.index(), exposed);
    }

    fn is_exposed_internal(&self, v: tree::VertexId) -> bool {
        self.exposed.get(v.index())
    }

    fn is_boundary_vertex_internal(&self, v: tree::VertexId) -> bool {
        self.is_exposed_internal(v) || self.tree.is_at_least_degree_n(v, 2)
    }

    fn is_exposed(&self, v: VertexId) -> bool {
        let vv = self.resolve_vertex(v);
        self.is_exposed_internal(vv)
    }
}

impl<W, A: Allocator> TopTree<W, A> {
    pub fn new_in(allocator: A) -> Self {
        Self {
            tree: tree::Tree::new(),
            exposed: slotvec::BitVec::new(),
            vertex_ids: SlotVec::new(),
            edge_ids: SlotVec::new(),
            label_ids: SlotVec::new(),
            allocator,
        }
    }

    /// Returns the set of vertices in the connected component containing `v`.
    pub fn component_of(&self, v: impl IntoVertexId) -> BTreeSet<VertexId> {
        let v = v.into_vertex_id(self);
        let leaves = if let Some(root) = incident_leaves(self, self.resolve_vertex(v))
            .next()
            .map(Handle::root)
        {
            iter::TreeRange::new(root)
        } else {
            iter::TreeRange::empty()
        };

        let mut vertices = BTreeSet::new();
        for leaf in leaves {
            match leaf.force() {
                LeafOrInternal::Edge(edge) => {
                    let Endpoints([left, right]) = self.tree.edge_endpoints(edge.edge());
                    vertices.insert(*self.tree.vertex_weight(left));
                    vertices.insert(*self.tree.vertex_weight(right));
                }
                LeafOrInternal::Label(label) => {
                    let vertex = *self
                        .tree
                        .vertex_weight(self.tree.label_vertex(label.label()));
                    vertices.insert(vertex);
                }
                LeafOrInternal::Internal(_) => unreachable!("leaf cannot be internal"),
            }
        }

        vertices
    }

    pub fn is_boundary_vertex(&self, v: VertexId) -> bool {
        let vv = self.resolve_vertex(v);
        self.is_boundary_vertex_internal(vv)
    }

    /// Two distinct vertices with no incident leaves or labels are reported as
    /// disconnected. A leafless vertex is connected only to itself.
    pub fn is_connected(&self, v: impl IntoVertexId, w: impl IntoVertexId) -> bool {
        let v = v.into_vertex_id(self);
        let w = w.into_vertex_id(self);

        let vv = self.resolve_vertex(v);
        let ww = self.resolve_vertex(w);

        let rv = incident_leaves(self, vv).next().map(Handle::root);
        let rw = incident_leaves(self, ww).next().map(Handle::root);

        match (rv, rw) {
            (Some(a), Some(b)) => a == b,
            // Neither vertex has a consuming node, so no shared root can be
            // established. A vertex is still trivially connected to itself: it is
            // the sole member of its own component.
            (None, None) => v == w,
            _ => false,
        }
    }

    pub fn degree(&self, v: VertexId) -> usize {
        let vv = self.resolve_vertex(v);
        self.tree.degree(vv)
    }

    pub fn edge_endpoints(&self, e: EdgeId) -> Option<[VertexId; 2]> {
        let ee = self.try_resolve_edge(e)?;
        let Endpoints([left, right]) = self.tree.edge_endpoints(ee);

        Some([
            *self.tree.vertex_weight(left),
            *self.tree.vertex_weight(right),
        ])
    }

    pub fn find_edge(&self, v: impl IntoVertexId, w: impl IntoVertexId) -> Option<EdgeId> {
        let v = v.into_vertex_id(self);
        let w = w.into_vertex_id(self);

        let vv = self.resolve_vertex(v);
        let ww = self.resolve_vertex(w);

        self.tree
            .find_edge_with_endpoints(vv, ww)
            .map(|e| self.tree.edge_weight(e).id)
    }

    pub fn label_vertex(&self, l: LabelId) -> VertexId {
        let ll = self.resolve_label(l);
        *self.tree.vertex_weight(self.tree.label_vertex(ll))
    }

    pub fn first_label(&self, v: VertexId) -> Option<LabelId> {
        let vv = self.resolve_vertex(v);
        let label = self.tree.incident_label_weights(vv).next()?;

        Some(label.id)
    }

    pub fn incident_edges(&self, v: VertexId) -> impl Iterator<Item = EdgeId> + '_ {
        let vv = self.resolve_vertex(v);
        self.tree.incident_edge_weights(vv).map(|e| e.id)
    }

    /// Exposes `v` and `w`, calls `f` with the weight of their combined
    /// cluster and a dormant handle to the tree, and de-exposes both vertices.
    pub fn with_exposed_path<R, F>(&mut self, v: VertexId, w: VertexId, f: F) -> R
    where
        W: Summary,
        F: for<'a> FnOnce(DormantTopTreeWeightMut<'a, W, A>) -> R,
    {
        assert!(self.is_connected(v, w), "vertices must be connected");
        expose(v, self);
        let meet =
            expose(w, self).expect("vertices are connected, so w must have a consuming node");

        let result = f(DormantTopTreeWeightMut::new(self, &meet));

        deexpose(w, self);
        deexpose(v, self);

        result
    }

    /// Like [`TopTree::with_exposed_path`], but returns `None` without
    /// calling `f` when `w` has no incident leaves.
    pub fn try_with_exposed_path<R, F>(&mut self, v: VertexId, w: VertexId, f: F) -> Option<R>
    where
        W: Summary,
        F: for<'a> FnOnce(DormantTopTreeWeightMut<'a, W, A>) -> R,
    {
        assert!(self.is_connected(v, w), "vertices must be connected");
        expose(v, self);

        let result = expose(w, self).map(|meet| f(DormantTopTreeWeightMut::new(self, &meet)));

        deexpose(w, self);
        deexpose(v, self);

        result
    }

    /// Exposes `v`, calls `f` with the weight of `v`'s cluster and a dormant
    /// handle to the tree, and de-exposes `v`.
    ///
    /// # Panics
    /// Panics if `v` has no incident leaves.
    pub fn with_exposed_vertex<R, F>(&mut self, v: impl IntoVertexId, f: F) -> R
    where
        W: Summary,
        F: for<'a> FnOnce(DormantTopTreeWeightMut<'a, W, A>) -> R,
    {
        let v = v.into_vertex_id(self);
        let handle = expose(v, self).expect("vertex must have a consuming node");

        let result = f(DormantTopTreeWeightMut::new(self, &handle));

        deexpose(v, self);

        result
    }

    /// Like [`TopTree::with_exposed_vertex`], but returns `None` without
    /// calling `f` when `v` has no incident leaves.
    pub fn try_with_exposed_vertex<R, F>(&mut self, v: impl IntoVertexId, f: F) -> Option<R>
    where
        W: Summary,
        F: for<'a> FnOnce(DormantTopTreeWeightMut<'a, W, A>) -> R,
    {
        let v = v.into_vertex_id(self);
        let result = expose(v, self).map(|handle| f(DormantTopTreeWeightMut::new(self, &handle)));

        deexpose(v, self);

        result
    }

    /// Adds a new vertex to the top tree, returning its `VertexId`.
    pub fn add_vertex(&mut self) -> VertexId {
        let (id, vertex) = self.vertex_ids.push_with(|id| {
            let vertex = self.tree.add_vertex(id);

            (vertex, vertex)
        });

        self.exposed.grow_to(vertex.index() + 1);
        self.set_exposed(vertex, false);

        id
    }

    /// Removes the vertex `v` from the top tree, along with all incident edges and labels.
    ///
    /// # Panics
    /// Panics if `v` is not a vertex in the top tree.
    pub fn remove_vertex(&mut self, v: impl IntoVertexId)
    where
        W: Summary,
    {
        let v = v.into_vertex_id(self);
        let vv = self.resolve_vertex(v);

        while let Some(&EdgeInfo { id, .. }) = self.tree.incident_edge_weights(vv).next() {
            let [v, w] = self.edge_endpoints(id).unwrap();
            self.cut(v, w);
        }

        while let Some(&LabelInfo { id, .. }) = self.tree.incident_label_weights(vv).next() {
            self.detach(id);
        }

        let mut removal = self.tree.remove_vertex(vv);

        while let Some(x) = removal.next(&mut self.tree) {
            use tree::EdgeOrLabelId::*;
            match x {
                Edge(SwapResult::Swapped { next, prev }) => {
                    let id = self.tree.edge_weight(next).id;
                    let old = mem::replace(&mut self.edge_ids[id], next);
                    assert_eq!(old, prev, "tree::EdgeId should match the previous tree id");

                    unsafe { self.tree.edge_weight_mut(next).node.as_mut().edge = next };
                }
                Label(SwapResult::Swapped { prev, next }) => {
                    let id = self.tree.label_weight(next).id;
                    let old = mem::replace(&mut self.label_ids[id], next);
                    assert_eq!(old, prev, "tree::LabelId should match the previous tree id");

                    unsafe { self.tree.label_weight_mut(next).node.as_mut().label = next };
                }
                Vertex(SwapResult::Swapped { prev, next }) => {
                    let id = *self.tree.vertex_weight(next);
                    let old = mem::replace(&mut self.vertex_ids[id], next);

                    self.exposed
                        .set(next.index(), self.exposed.get(prev.index()));
                    self.exposed.set(prev.index(), false);

                    assert_eq!(
                        old, prev,
                        "tree::VertexId should match the previous tree id"
                    );
                }
                _ => {}
            }
        }

        let _ = removal.into_weight();
        self.vertex_ids.remove(v);
    }

    pub fn link(&mut self, v: impl IntoVertexId, w: impl IntoVertexId) -> EdgeId
    where
        W: Default + Summary,
    {
        let v = v.into_vertex_id(self);
        let w = w.into_vertex_id(self);

        link(v, w, |_| W::default(), self)
    }

    pub fn link_with(
        &mut self,
        v: impl IntoVertexId,
        w: impl IntoVertexId,
        weight: impl FnOnce(EdgeId) -> W,
    ) -> EdgeId
    where
        W: Summary,
    {
        let v = v.into_vertex_id(self);
        let w = w.into_vertex_id(self);

        link(v, w, weight, self)
    }

    pub fn cut(&mut self, v: impl IntoVertexId, w: impl IntoVertexId)
    where
        W: Summary,
    {
        let v = v.into_vertex_id(self);
        let w = w.into_vertex_id(self);

        cut(v, w, self);
    }

    pub fn attach(&mut self, v: VertexId) -> LabelId
    where
        W: Default + Summary,
    {
        let (label, _root) = attach(v, |_| W::default(), self);
        label
    }

    pub fn attach_with(&mut self, v: VertexId, weight: impl FnOnce(LabelId) -> W) -> LabelId
    where
        W: Summary,
    {
        let (label, _root) = attach(v, weight, self);
        label
    }

    pub fn update_label(&mut self, l: LabelId, weight: W)
    where
        W: Summary,
    {
        let ll = self.resolve_label(l);
        let mut handle = Handle::<W, _>::new_label(self.tree.label_weight(ll).node.cast());

        unsafe {
            *handle.weight_mut() = weight;
        }

        handle.recursively_recompute_weight_up();
    }

    pub fn attach_new(&mut self) -> (VertexId, LabelId)
    where
        W: Default + Summary,
    {
        let v = self.add_vertex();
        let l = self.attach(v);
        (v, l)
    }

    pub fn attach_new_with(&mut self, weight: impl FnOnce(LabelId) -> W) -> (VertexId, LabelId)
    where
        W: Summary,
    {
        let v = self.add_vertex();
        let l = self.attach_with(v, weight);
        (v, l)
    }

    pub fn detach(&mut self, l: LabelId)
    where
        W: Summary,
    {
        detach(l, self);
    }

    pub fn edge_indices(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edge_ids.iter().map(|(id, _)| id)
    }
}

pub trait IntoVertexId {
    fn into_vertex_id<W, A: Allocator>(self, tree: &TopTree<W, A>) -> VertexId;
}

impl IntoVertexId for VertexId {
    fn into_vertex_id<W, A: Allocator>(self, _: &TopTree<W, A>) -> VertexId {
        self
    }
}

impl IntoVertexId for LabelId {
    fn into_vertex_id<W, A: Allocator>(self, tree: &TopTree<W, A>) -> VertexId {
        tree.label_vertex(self)
    }
}

impl<W> TopTree<W, Global> {
    pub fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<W> Default for TopTree<W, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W, A: Allocator> Drop for TopTree<W, A> {
    fn drop(&mut self) {
        let allocator = &self.allocator;
        self.tree.edges.retain(|edge| {
            Handle::new_edge(edge.weight.node.cast::<Node<W>>()).delete_all_ancestors(allocator);

            true
        });
    }
}

impl<W, A: Allocator> fmt::Debug for TopTree<W, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let edges = self
            .edge_ids
            .iter()
            .map(|(id, _)| self.edge_endpoints(id).unwrap())
            .map(|[u, v]| (u, v))
            .collect::<Vec<_>>();

        let labels = self
            .label_ids
            .iter()
            .map(|(id, _)| (id, self.label_vertex(id)))
            .collect::<Vec<_>>();

        f.debug_struct("TopTree")
            .field("edges", &edges)
            .field("labels", &labels)
            .finish()
    }
}
