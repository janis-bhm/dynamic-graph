#![expect(internal_features)]
#![feature(
    ptr_as_uninit,
    cast_maybe_uninit,
    pattern_types,
    pattern_type_macro,
    structural_match
)]
#![allow(dead_code)]

use std::ptr::{self, NonNull};

use crate::{boundary::BoundaryVertices, tree::Endpoints, util::TaggedPtr};

pub use index::VertexId;

mod boundary;
mod index;
mod non_max;
mod summary;
#[cfg(test)]
mod tests;
mod tree;
mod util;

struct ClusterId(NonNull<Node<()>>);

type Tree<W> = tree::Tree<(), NonNull<LeafNode<W>>, NonNull<LabelNode<W>>>;

#[repr(u8)]
enum NodeKind {
    EdgeLeaf,
    Cluster,
    LabelLeaf,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct NodeFlags: u8 {
        const LEAF = 1 << 0;
        const FLIPPED = 1 << 1;
        const LABEL = 1 << 2;
    }
}

impl NodeFlags {
    fn is_edge(&self) -> bool {
        self.contains(NodeFlags::LEAF)
    }

    fn is_cluster(&self) -> bool {
        !self.contains(NodeFlags::LEAF) && !self.contains(NodeFlags::LABEL)
    }

    fn is_label(&self) -> bool {
        self.contains(NodeFlags::LABEL)
    }

    fn kind(&self) -> NodeKind {
        if self.contains(Self::LABEL) {
            NodeKind::LabelLeaf
        } else if self.contains(Self::LEAF) {
            NodeKind::EdgeLeaf
        } else {
            NodeKind::Cluster
        }
    }
}

unsafe impl util::Tag for NodeFlags {
    const BITS: u32 = 4;

    fn into_usize(self) -> usize {
        self.bits() as usize
    }

    unsafe fn from_usize(tag: usize) -> Self {
        Self::from_bits_truncate(tag as u8)
    }
}

#[repr(C, align(16))]
struct Node<W> {
    parent: TaggedPtr<InternalNode<W>, NodeFlags>,
    boundary: BoundaryVertices,
    weight: W,
}

impl<W> Node<W> {
    unsafe fn dealloc(this: NonNull<Node<W>>) {
        let leaf = unsafe { this.as_ref() }.is_edge();
        let ptr = this.as_ptr();
        if leaf {
            drop(unsafe { Box::from_raw(ptr as *mut LeafNode<W>) });
        } else {
            drop(unsafe { Box::from_raw(ptr as *mut InternalNode<W>) });
        }
    }
}

impl<W> Node<W> {
    fn flags(&self) -> NodeFlags {
        self.parent.tag()
    }

    fn is_edge(&self) -> bool {
        self.flags().is_edge()
    }

    fn is_flipped(&self) -> bool {
        self.flags().contains(NodeFlags::FLIPPED)
    }

    fn set_flipped(&mut self, flipped: bool) {
        self.parent.update_tag(|flags| {
            flags.set(NodeFlags::FLIPPED, flipped);
        });
    }

    fn toggle_flipped(&mut self) {
        self.parent.update_tag(|flags| {
            flags.toggle(NodeFlags::FLIPPED);
        });
    }

    fn num_boundary(&self) -> usize {
        self.boundary.count() as usize
    }

    fn is_path(&self) -> bool {
        self.boundary.is_path()
    }

    fn is_point(&self) -> bool {
        self.boundary.is_point()
    }

    fn parent(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent.as_non_null()
    }

    fn set_parent(&mut self, parent: Option<NonNull<InternalNode<W>>>) {
        self.parent
            .set_ptr(parent.map_or(ptr::null_mut(), NonNull::as_ptr));
    }

    fn parent_node(&self) -> Option<NonNull<Node<W>>> {
        self.parent().map(NonNull::cast)
    }

    fn grandparent(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent()
            .and_then(|p| unsafe { p.as_ref().node.parent() })
    }

    fn ggp(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent()
            .and_then(|p| unsafe { p.as_ref().node.parent() })
            .and_then(|gp| unsafe { gp.as_ref().node.parent() })
    }

    /// # Safety
    /// Creates a reference to the parent of this node.
    unsafe fn sibling(&self) -> Option<NonNull<Node<W>>> {
        self.parent().map(|p| unsafe { p.as_ref() }).map(|p| {
            let self_ptr = NonNull::from(self);
            if p.children.left == self_ptr {
                p.children.right
            } else {
                p.children.left
            }
        })
    }

    unsafe fn is_left_child(&self) -> Option<bool> {
        self.parent().map(|p| unsafe { p.as_ref() }).map(|p| {
            let self_ptr = NonNull::from(self);
            p.children.left == self_ptr
        })
    }

    fn force_ptr(
        this: NonNull<Self>,
    ) -> LeafOrInternal<NonNull<LeafNode<W>>, NonNull<LabelNode<W>>, NonNull<InternalNode<W>>> {
        match unsafe { (&*this.as_ptr()).flags().kind() } {
            NodeKind::EdgeLeaf => LeafOrInternal::Edge(this.cast()),
            NodeKind::Cluster => LeafOrInternal::Internal(this.cast()),
            NodeKind::LabelLeaf => LeafOrInternal::Label(this.cast()),
        }
    }
}

#[repr(C)]
struct LabelNode<W> {
    node: Node<W>,
    label: tree::LabelId,
}

#[repr(C)]
struct LeafNode<W> {
    node: Node<W>,
    edge: tree::EdgeId,
}

impl<W> LabelNode<W> {
    fn init(
        node: NonNull<LabelNode<W>>,
        weight: W,
        label: tree::LabelId,
        boundary: BoundaryVertices,
    ) {
        unsafe {
            let uninit = node.as_uninit_mut();
            uninit.write(LabelNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LABEL),
                    boundary,
                    weight,
                },
                label,
            });
        };
    }
}

impl<W> LeafNode<W> {
    fn init(node: NonNull<LeafNode<W>>, weight: W, edge: tree::EdgeId, boundary: BoundaryVertices) {
        unsafe {
            let uninit = node.as_uninit_mut();
            uninit.write(LeafNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LEAF),
                    boundary,
                    weight,
                },
                edge,
            });
        };
    }
}

struct Children<W> {
    left: NonNull<Node<W>>,
    right: NonNull<Node<W>>,
}

impl<W> IntoIterator for &Children<W> {
    type Item = NonNull<Node<W>>;
    type IntoIter = std::array::IntoIter<Self::Item, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [self.left, self.right].into_iter()
    }
}

impl<W> Children<W> {
    fn flip(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right);
    }

    fn get(&self, left: bool) -> NonNull<Node<W>> {
        if left { self.left } else { self.right }
    }
    fn get_mut(&mut self, left: bool) -> &mut NonNull<Node<W>> {
        if left {
            &mut self.left
        } else {
            &mut self.right
        }
    }
}

#[repr(C)]
struct InternalNode<W> {
    node: Node<W>,
    children: Children<W>,
}

impl<W> InternalNode<W> {
    fn alloc(
        weight: W,
        left: NonNull<Node<W>>,
        right: NonNull<Node<W>>,
        boundary: BoundaryVertices,
    ) -> NonNull<Self> {
        let node = Box::new(InternalNode {
            node: Node {
                parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::empty()),
                boundary,
                weight,
            },
            children: Children { left, right },
        });

        let node = Box::into_non_null(node);

        unsafe {
            (&mut *left.as_ptr()).set_parent(Some(node));
            (&mut *right.as_ptr()).set_parent(Some(node));
        }

        node
    }
}

impl<W> core::ops::Deref for LabelNode<W> {
    type Target = Node<W>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<W> core::ops::DerefMut for LabelNode<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

impl<W> core::ops::Deref for LeafNode<W> {
    type Target = Node<W>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<W> core::ops::DerefMut for LeafNode<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

impl<W> core::ops::Deref for InternalNode<W> {
    type Target = Node<W>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<W> core::ops::DerefMut for InternalNode<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

mod marker {

    pub trait IsLeaf {}
    impl IsLeaf for Edge {}
    impl IsLeaf for Label {}
    impl IsLeaf for Leaf {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Leaf;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Edge;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Label;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Internal;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Either;
}

#[derive(Debug, Clone, Copy)]
struct Handle<W, NodeType> {
    node: NonNull<Node<W>>,
    _marker: core::marker::PhantomData<NodeType>,
}

impl<W, NodeType> PartialEq for Handle<W, NodeType> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<W, NodeType> Eq for Handle<W, NodeType> {}

impl<W> Handle<W, marker::Either> {
    fn new_either(node: NonNull<Node<W>>) -> Handle<W, marker::Either> {
        Handle {
            node,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<W> Handle<W, marker::Edge> {
    fn new_edge(node: NonNull<Node<W>>) -> Handle<W, marker::Edge> {
        Handle {
            node,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<W, NodeType> Handle<W, NodeType> {
    fn new(node: NonNull<Node<W>>) -> Self {
        Self {
            node,
            _marker: core::marker::PhantomData,
        }
    }

    unsafe fn new_internal(node: NonNull<InternalNode<W>>) -> Self {
        Self {
            node: node.cast(),
            _marker: core::marker::PhantomData,
        }
    }

    fn force(
        &self,
    ) -> LeafOrInternal<
        Handle<W, marker::Edge>,
        Handle<W, marker::Label>,
        Handle<W, marker::Internal>,
    > {
        match unsafe { self.node.as_ref().flags().kind() } {
            NodeKind::EdgeLeaf => LeafOrInternal::Edge(Handle::new(self.node)),
            NodeKind::Cluster => LeafOrInternal::Internal(Handle::new(self.node)),
            NodeKind::LabelLeaf => LeafOrInternal::Label(Handle::new(self.node)),
        }
    }

    fn set_parent(&mut self, parent: Option<Handle<W, marker::Internal>>) {
        unsafe {
            self.node.as_mut().set_parent(parent.map(|p| p.node.cast()));
        }
    }

    fn parent(&self) -> Option<Handle<W, marker::Internal>> {
        unsafe { self.node.as_ref().parent().map(|p| Handle::new_internal(p)) }
    }

    fn is_left_child(&self) -> Option<bool> {
        unsafe { self.node.as_ref().is_left_child() }
    }

    fn sibling(&self) -> Option<Handle<W, NodeType>> {
        let parent = self.parent()?;
        let [left, right] = parent.children();
        if left.node == self.node {
            Some(Self::new(right.node))
        } else {
            Some(Self::new(left.node))
        }
    }

    fn num_boundary(&self) -> usize {
        unsafe { self.node.as_ref().num_boundary() }
    }

    fn has_middle_boundary(&self) -> bool {
        use LeafOrInternal::*;
        if self.num_boundary() == 0 {
            return false;
        }
        match self.force() {
            Label(_) | Edge(_) => false,
            Internal(internal) => internal.num_boundary() != internal.num_path_children(),
        }
    }

    fn boundary(&self) -> BoundaryVertices {
        unsafe { self.node.as_ref().boundary }
    }

    fn set_boundary(&mut self, boundary: BoundaryVertices) {
        unsafe { self.node.as_mut().boundary = boundary }
    }

    fn remove_from_boundary(&mut self, v: tree::VertexId) {
        unsafe {
            self.node.as_mut().boundary.remove(v);
        }
    }

    fn add_boundary_vertex(&mut self, tree: &TopTree<W>, v: tree::VertexId) {
        match self.force() {
            LeafOrInternal::Edge(edge) => {
                let Endpoints([left, right]) = tree.tree.edge_endpoints(edge.edge());

                if left == v {
                    unsafe { self.node.as_mut().boundary.add(v, true) }
                } else if right == v {
                    unsafe { self.node.as_mut().boundary.add(v, false) }
                } else {
                    panic!("vertex not found in edge endpoints")
                }
            }
            LeafOrInternal::Label(_) => unsafe { self.node.as_mut().boundary.add(v, true) },
            LeafOrInternal::Internal(internal) => {
                let [left, right] = internal.children();
                let in_left = left.boundary().contains(v);
                let in_right = right.boundary().contains(v);

                if in_left && in_right {
                    let is_left = !left.is_path();
                    unsafe { self.node.as_mut().boundary.add(v, is_left) }
                } else if in_left {
                    unsafe { self.node.as_mut().boundary.add(v, true) }
                } else if in_right {
                    unsafe { self.node.as_mut().boundary.add(v, false) }
                } else {
                    panic!("vertex not found in either child")
                }
            }
        }
    }

    fn is_path(&self) -> bool {
        unsafe { self.node.as_ref().is_path() }
    }

    fn is_point(&self) -> bool {
        unsafe { self.node.as_ref().is_point() }
    }

    fn is_flipped(&self) -> bool {
        unsafe { self.node.as_ref().is_flipped() }
    }

    fn set_flipped(&mut self, flipped: bool) {
        unsafe { self.node.as_mut().set_flipped(flipped) }
    }

    fn toggle_flipped(&mut self) {
        unsafe { self.node.as_mut().toggle_flipped() }
    }

    /// returns `None` if the node has no grandparent (i.e. is a child of/or the root)
    fn rotate_up(&mut self) -> Option<()>
    where
        W: Reduce,
    {
        let mut sibling = self.sibling()?;
        let mut parent = self.parent()?;
        let uncle = parent.sibling()?;
        let mut gp = parent.parent()?;

        gp.push_flip();
        parent.push_flip();

        let uncle_is_left = uncle.is_left_child().expect("uncle must have a parent");
        let sibling_is_left = sibling.is_left_child().expect("sibling must have a parent");
        let same_sides = uncle_is_left == sibling_is_left;
        let sibling_is_path = sibling.is_path();
        let uncle_is_path = uncle.is_path();
        let gp_is_path = gp.is_path();

        // the new parent made up of the sibling and uncle will be a path
        // consider the star case:
        // the middle boundary of gp is shared by `node`, `sibling`, and `uncle`.
        // the new parent `sibling` \cup `uncle` will have the same middle boundary; if either `sibling` or `uncle` is a path, then the new parent will be a path.
        // both `sibling` and `uncle` cannot be paths.
        // consider the path case:
        // the shared vertex of gp is shared by `parent` and `uncle`, and must be the outward boundary of `sibling`, which is a path.
        // if gp has a middle boundary, then it must be this vertex, and it must be a boundary of the new parent.
        // if gp has no middle boundary but uncle is a path, then the boundary of the new parent will be the two non-shared boundaries of `sibling` and `uncle`.
        let new_parent_boundary: BoundaryVertices;
        let flip_new_parent: bool;
        let flip_gp: bool;

        // The new parent's physical left child is `uncle` when `uncle_is_left`,
        // otherwise `sibling`. Merge the children's logical boundaries (the
        // orientation in which a child contributes to the merge) in that
        // physical order.
        let sibling_boundary = sibling.boundary();
        let uncle_boundary = uncle.boundary();
        let (left_boundary, right_boundary) = if uncle_is_left {
            (uncle_boundary, sibling_boundary)
        } else {
            (sibling_boundary, uncle_boundary)
        };
        if same_sides && sibling_is_path {
            // path
            let gp_has_middle = gp.has_middle_boundary();
            new_parent_boundary = BoundaryVertices::from_children(
                left_boundary,
                right_boundary,
                gp_has_middle || uncle_is_path,
            );

            flip_new_parent = false;
            if gp_has_middle
                && !gp_is_path
                && let Some(gp_is_left) = gp.is_left_child()
            {
                flip_gp = gp_is_left == uncle_is_left;
            } else {
                flip_gp = false;
            }
        } else {
            // star
            if !same_sides {
                flip_new_parent = sibling_is_path;
                flip_gp = sibling_is_path;
                self.toggle_flipped();

                new_parent_boundary = BoundaryVertices::from_children(
                    left_boundary,
                    right_boundary,
                    sibling_is_path || uncle_is_path,
                );
            } else {
                flip_new_parent = false;
                flip_gp = false;
                sibling.toggle_flipped();

                new_parent_boundary =
                    BoundaryVertices::from_children(left_boundary, right_boundary, uncle_is_path);
            }
        }

        parent.set_child(sibling.node, !uncle_is_left);
        parent.set_child(uncle.node, uncle_is_left);
        parent.set_flipped(flip_new_parent);
        parent.set_boundary(new_parent_boundary);

        gp.set_child(self.node, !uncle_is_left);
        gp.set_child(parent.node, uncle_is_left);
        gp.set_flipped(flip_gp);

        // recompute W for parent and gp
        parent.recompute_weight();
        gp.recompute_weight();

        Some(())
    }

    fn forget_type(self) -> Handle<W, marker::Either> {
        Handle {
            node: self.node,
            _marker: core::marker::PhantomData,
        }
    }

    fn splay_step(self) -> Option<Handle<W, marker::Either>>
    where
        W: Reduce,
    {
        let mut node = self.forget_type();
        loop {
            let mut p = node.parent()?;
            let gp = p.parent()?;

            if node.is_point() && gp.is_point() {
                node.rotate_up().expect("rotate_up should succeed");
                return Some(Handle::new(gp.node));
            }

            let ggp = gp.parent()?;

            if p.is_path() && (gp.is_path() || ggp.is_point()) {
                gp.push_flip();
                p.push_flip();

                let node_is_left = node.is_left_child().expect("node must have a parent");
                let p_is_left = p.is_left_child().expect("p must have a parent");
                let gp_is_left = gp.is_left_child().expect("gp must have a parent");

                if node_is_left == p_is_left {
                    node.rotate_up().expect("rotate_up should succeed");
                    return Some(Handle::new(gp.node));
                }

                if p_is_left == gp_is_left {
                    p.rotate_up().expect("rotate_up should succeed");
                    return Some(Handle::new(ggp.node));
                }

                debug_assert_eq!(node_is_left, gp_is_left);
                let mut sibling = node.sibling().expect("node must have a sibling");
                sibling.rotate_up().expect("rotate_up should succeed");
                p.rotate_up().expect("rotate_up should succeed");
                return Some(Handle::new(ggp.node));
            }

            node = p.forget_type();
        }
    }

    fn semi_splay(self)
    where
        W: Reduce,
    {
        let mut node = self.forget_type();
        while let Some(next_node) = node.splay_step() {
            node = next_node;
        }
    }

    fn full_splay(self)
    where
        W: Reduce,
    {
        while let Some(next_node) = unsafe { ptr::read(&self) }.splay_step() {
            next_node.splay_step();
        }
    }

    fn has_left_boundary(&self, root: &TopTree<W>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.tree.edge_endpoints(leaf.edge())[self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Label(label) => root.is_boundary_vertex(root.tree.label_vertex(label.label())),
            Internal(internal) => internal.child(!self.is_flipped()).is_path(),
        }
    }

    fn has_right_boundary(&self, root: &TopTree<W>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.tree.edge_endpoints(leaf.edge())[!self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Label(label) => root.is_boundary_vertex(root.tree.label_vertex(label.label())),
            Internal(internal) => internal.child(self.is_flipped()).is_path(),
        }
    }

    fn is_internal(&self) -> bool {
        !unsafe { self.node.as_ref().is_edge() }
    }

    unsafe fn into_internal(self) -> Handle<W, marker::Internal> {
        debug_assert!(self.is_internal());
        unsafe { Handle::new_internal(self.node.cast()) }
    }

    unsafe fn into_leaf(self) -> Handle<W, marker::Edge> {
        debug_assert!(!self.is_internal());
        Handle::new_edge(self.node.cast())
    }

    fn delete_all_ancestors(self) {
        if let Some(parent) = self.parent() {
            let mut sibling = self.sibling().expect("node must have a sibling");
            parent.delete_all_ancestors();
            unsafe {
                sibling.node.as_mut().set_parent(None);
            }
        }

        unsafe {
            Node::dealloc(self.node);
        }
    }

    fn root(self) -> Handle<W, marker::Either> {
        let mut node = self.forget_type();
        while let Some(parent) = node.parent() {
            node = parent.forget_type();
        }
        node
    }
}

impl<W> Handle<W, marker::Label> {
    fn label(&self) -> tree::LabelId {
        unsafe { self.node.cast::<LabelNode<W>>().as_ref().label }
    }
}

impl<W> Handle<W, marker::Leaf> {
    fn endpoints(&self, root: &TopTree<W>) -> [tree::VertexId; 2] {
        match self.force() {
            LeafOrInternal::Edge(edge) => unsafe {
                let edge = edge.node.cast::<LeafNode<W>>().as_ref().edge;

                root.tree.edge_endpoints(edge).0
            },
            LeafOrInternal::Label(label) => {
                let label = label.label();
                let vertex = root.tree.label_vertex(label);
                [vertex, vertex]
            }
            LeafOrInternal::Internal(_) => {
                unreachable!("Handle<W, marker::Leaf> cannot be Internal")
            }
        }
    }
}

impl<W> Handle<W, marker::Edge> {
    fn edge(&self) -> tree::EdgeId {
        unsafe { self.node.cast::<LeafNode<W>>().as_ref().edge }
    }
}

impl<W> Handle<W, marker::Internal> {
    fn push_flip(&self) {
        if self.is_flipped() {
            unsafe {
                let node = self.node.cast::<InternalNode<W>>().as_mut();

                node.set_flipped(false);
                node.children.flip();

                for mut child in &node.children {
                    child.as_mut().toggle_flipped();
                }
            }
        }
    }

    fn num_path_children(&self) -> usize {
        self.children()
            .iter()
            .filter(|child| child.is_path())
            .count()
    }

    fn children(&self) -> [Handle<W, marker::Internal>; 2] {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            [
                Handle::new(internal.children.left),
                Handle::new(internal.children.right),
            ]
        }
    }

    fn flipped_children(&self) -> [Handle<W, marker::Internal>; 2] {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            if self.is_flipped() {
                [
                    Handle::new(internal.children.right),
                    Handle::new(internal.children.left),
                ]
            } else {
                [
                    Handle::new(internal.children.left),
                    Handle::new(internal.children.right),
                ]
            }
        }
    }

    fn child(&self, left: bool) -> Handle<W, marker::Internal> {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            Handle::new(internal.children.get(left))
        }
    }

    fn set_child(&mut self, mut child: NonNull<Node<W>>, left: bool) {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_mut();
            *internal.children.get_mut(left) = child;
            child
                .as_mut()
                .set_parent(Some(self.node.cast::<InternalNode<W>>()));
        }
    }

    fn recompute_weight(&mut self)
    where
        W: Reduce,
    {
        let [left, right] = self.flipped_children();
        unsafe {
            let lw = &left.node.as_ref().weight;
            let rw = &right.node.as_ref().weight;
            let new_weight = W::reduce(lw, rw);
            self.node.as_mut().weight = new_weight;
        }
    }
}

fn incident_leaves<W>(
    root: &Tree<W>,
    v: tree::VertexId,
) -> impl Iterator<Item = Handle<W, marker::Leaf>> + '_ {
    root.incident_label_weights(v)
        .map(|l| Handle::<W, marker::Leaf>::new(l.cast()))
        .chain(
            root.incident_edge_weights(v)
                .map(|e| Handle::<W, marker::Leaf>::new(e.cast())),
        )
}

/// Finds the least common ancestor of all leaves incident to `v` in the tree rooted at `root`.
pub(crate) fn find_consuming_node<W>(
    root: &TopTree<W>,
    v: tree::VertexId,
) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    let node = incident_leaves(&root.tree, v).next()?;
    unsafe { ptr::read(&node) }.semi_splay();

    // if the vertex has exactly one incident edge, then the consuming node is the incident leaf.
    if root.tree.is_exactly_degree_n(v, 1) {
        return Some(node.forget_type());
    }

    let endpoints = match node.force() {
        LeafOrInternal::Edge(edge) => root.tree.edge_endpoints(edge.edge()),
        LeafOrInternal::Label(label) => {
            let vertex = root.tree.label_vertex(label.label());
            tree::Endpoints([vertex, vertex])
        }
        LeafOrInternal::Internal(_) => unreachable!("node must be a leaf"),
    };

    let flip = node.is_flipped();
    let mut is_left = (endpoints.left() == v) != flip;
    let mut is_right = (endpoints.right() == v) != flip;
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
                return Some(node);
            }
            last_middle_node = Some(unsafe { ptr::read(&node) });
        }
    }

    last_middle_node
}

pub(crate) fn expose<W>(
    v: tree::VertexId,
    root: &mut TopTree<W>,
) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    fn expose_prepared<W>(
        mut node: Handle<W, marker::Either>,
        tree: &mut TopTree<W>,
        v: tree::VertexId,
    ) -> Handle<W, marker::Either>
    where
        W: Reduce,
    {
        let mut left = false;
        let mut right = false;

        loop {
            node.add_boundary_vertex(tree, v);

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
        W: Reduce,
    {
        let mut node = unsafe { ptr::read(&consuming_node) };
        while let Some(parent) = node.parent() {
            if node.is_point() {
                node = parent.forget_type();
            } else {
                assert!(node.is_internal(), "node must be internal");
                let internal = unsafe { ptr::read(&node).into_internal() };
                parent.push_flip();
                internal.push_flip();

                let mut sibling = internal.sibling().expect("node must have a sibling");

                let sibling_is_left = sibling.is_left_child().expect("sibling must have a parent");
                let same_side_child = internal.child(sibling_is_left);

                if same_side_child.is_path() || sibling.is_point() {
                    let mut other_side_child = internal.child(!sibling_is_left);
                    other_side_child
                        .rotate_up()
                        .expect("rotate_up should succeed");

                    if node == consuming_node {
                        consuming_node = unsafe { ptr::read(&parent).forget_type() };
                    }
                    node = parent.forget_type();
                } else {
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

    match find_consuming_node(root, v) {
        Some(consuming_node) => {
            let consuming_node = prepare_expose(consuming_node);

            let node = expose_prepared(consuming_node, root, v);
            root.exposed.set(v.index(), true);

            Some(node)
        }
        None => {
            root.exposed.set(v.index(), true);
            None
        }
    }
}

pub(crate) fn deexpose<W>(
    v: tree::VertexId,
    tree: &mut TopTree<W>,
) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    let mut node = find_consuming_node(tree, v);
    let mut root = None;

    while let Some(mut some_node) = node {
        some_node.remove_from_boundary(v);
        node = some_node.parent().map(Handle::forget_type);
        root = Some(some_node);
    }

    tree.exposed.set(v.index(), false);

    root
}

pub(crate) fn link<W>(
    u: tree::VertexId,
    v: tree::VertexId,
    weight: W,
    tree: &mut TopTree<W>,
) -> NonNull<Node<W>>
where
    W: Reduce,
{
    let mut ru = expose(u, tree);
    if let Some(ref mut tu) = ru
        && Handle::has_left_boundary(tu, tree)
    {
        tu.toggle_flipped();
    }
    tree.exposed.set(u.index(), false);

    let mut rv = expose(v, tree);
    if let Some(ref mut tv) = rv
        && Handle::has_right_boundary(tv, tree)
    {
        tv.toggle_flipped();
    }
    tree.exposed.set(u.index(), false);

    let leaf = Box::into_non_null(Box::new_uninit()).cast_init();
    let edge = tree.tree.add_edge(u, v, leaf);
    LeafNode::init(
        leaf,
        weight,
        edge,
        BoundaryVertices::from_left_and_right(ru.as_ref().map(|_| u), rv.as_ref().map(|_| v)),
    );

    let mut node = leaf.cast::<Node<W>>();
    if let Some(ru) = ru {
        let left = ru;
        let right = node.cast::<Node<W>>();
        let weight = unsafe {
            let wl = &left.node.as_ref().weight;
            let wr = &right.as_ref().weight;
            W::reduce(wl, wr)
        };
        node = InternalNode::alloc(
            weight,
            left.node,
            right,
            BoundaryVertices::from_option(rv.as_ref().map(|_| v)),
        )
        .cast();
    }

    if let Some(rv) = rv {
        let left = node.cast::<Node<W>>();
        let right = rv;
        let weight = unsafe {
            let wl = &left.as_ref().weight;
            let wr = &right.node.as_ref().weight;
            W::reduce(wl, wr)
        };
        InternalNode::alloc(
            weight,
            left,
            right.node,
            BoundaryVertices::from_option((node.cast() != leaf).then_some(u)),
        );
    }

    leaf.cast()
}

#[expect(clippy::type_complexity)]
pub(crate) fn cut<W>(
    u: tree::VertexId,
    v: tree::VertexId,
    tree: &mut TopTree<W>,
) -> (
    Option<Handle<W, marker::Either>>,
    Option<Handle<W, marker::Either>>,
)
where
    W: Reduce,
{
    let Some((id, edge)) = tree
        .tree
        .find_edge_with_endpoints(u, v)
        .map(|id| (id, tree.tree.edge(id)))
    else {
        return (None, None);
    };

    let edge_handle = Handle::new_edge(edge.weight.cast::<Node<W>>());

    unsafe { ptr::read(&edge_handle) }.full_splay();
    edge_handle.delete_all_ancestors();

    _ = tree.tree.remove_edge(id);

    tree.exposed.set(u.index(), true);
    tree.exposed.set(v.index(), true);

    let ru = deexpose(u, tree);
    let rv = deexpose(v, tree);

    (ru, rv)
}

// pub(crate) fn attach<W>(
//     v: tree::VertexId,
//     weight: W,
//     tree: &mut TopTree<W>,
// ) -> (tree::LabelId, NonNull<Node<W>>)
// where
//     W: Reduce,
// {
//     let mut rv = expose(v, tree);
//     if let Some(ref mut tv) = rv
//         && Handle::has_left_boundary(tv, tree)
//     {
//         tv.toggle_flipped();
//     }
//     tree.exposed.set(v.index(), false);

//     let node: NonNull<LabelNode<W>> = Box::into_non_null(Box::new_uninit()).cast_init();
//     let label = tree.tree.add_label(v, node.cast());
//     LabelNode::init(node, weight, label, usize::from(rv.is_some()));

//     let root: NonNull<Node<W>> = match rv {
//         Some(rv) => {
//             let left = rv;
//             let right = node.cast::<Node<W>>();
//             let weight = unsafe {
//                 let wl = &left.node.as_ref().weight;
//                 let wr = &right.as_ref().weight;
//                 W::reduce(wl, wr)
//             };
//             InternalNode::alloc(weight, left.node, right, 0).cast()
//         }
//         None => node.cast(),
//     };

//     (label, root)
// }

// pub(crate) fn detach<W>(v: tree::LabelId, tree: &mut Tree<W>) {
//     // labels are never path components, so removing them can never disconnect the tree.
//     // instead, we want to replace the label's parent with the label's sibling, then delete the label and parent.
//     let (label, swap) = tree.remove_label(v);
//     let label_handle = Handle::<W, marker::Label>::new(label.cast());
//     if let Some(parent) = label_handle.parent() {
//         let mut sibling = label_handle.sibling().expect("label must have a sibling");

//         if let Some(mut gp) = parent.parent() {
//             let parent_is_left = parent.is_left_child().expect("parent has parent");
//             // we need to flip the sibling if it is on the opposite side of the parent.
//             // if the parent was flipped, then flip the sibling (again).
//             let flip_sibling = (sibling.is_left_child().expect("sibling has parent")
//                 != parent_is_left)
//                 ^ parent.is_flipped();

//             if flip_sibling {
//                 sibling.toggle_flipped();
//             }

//             gp.set_child(sibling.node, parent_is_left);
//         } else {
//             // parent is root, so we just make the sibling the new root.
//             sibling.set_parent(None);
//         }

//         unsafe {
//             Node::<W>::dealloc(parent.node);
//             Node::<W>::dealloc(label_handle.node);
//         }
//     }
// }

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

pub struct TopTree<W> {
    tree: Tree<W>,
    // indexed by tree vertex index, true if the vertex is exposed in the top tree.
    exposed: util::BitVec,
}

impl<W> TopTree<W> {
    pub fn new() -> Self {
        Self {
            tree: Tree::new(),
            exposed: util::BitVec::new(),
        }
    }

    fn is_boundary_vertex(&self, v: tree::VertexId) -> bool {
        self.exposed.get(v.index()) || self.tree.is_at_least_degree_n(v, 2)
    }
}

impl<W> Default for TopTree<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W> Drop for TopTree<W> {
    fn drop(&mut self) {
        self.tree.edges.retain(|edge| {
            Handle::new_edge(edge.weight.cast::<Node<W>>()).delete_all_ancestors();

            unsafe { Node::<W>::dealloc(edge.weight.cast()) };
            true
        });
    }
}

pub trait Reduce {
    fn reduce(&self, other: &Self) -> Self;
}
