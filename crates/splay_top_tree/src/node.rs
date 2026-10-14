use core::fmt;
use std::{
    alloc::{Allocator, Layout, handle_alloc_error},
    mem,
    ptr::{self, NonNull},
};

use crate::{
    LeafOrInternal, Summary, TopTree, VertexId, boundary::BoundaryVertices, summary, tree,
    tree::Endpoints, util::TaggedPtr,
};

#[repr(u8)]
pub enum NodeKind {
    EdgeLeaf,
    Cluster,
    LabelLeaf,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NodeFlags: u8 {
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

    pub fn kind(&self) -> NodeKind {
        if self.contains(Self::LABEL) {
            NodeKind::LabelLeaf
        } else if self.contains(Self::LEAF) {
            NodeKind::EdgeLeaf
        } else {
            NodeKind::Cluster
        }
    }
}

unsafe impl crate::util::Tag for NodeFlags {
    const BITS: u32 = 4;

    fn into_usize(self) -> usize {
        self.bits() as usize
    }

    unsafe fn from_usize(tag: usize) -> Self {
        Self::from_bits_truncate(tag as u8)
    }
}

#[repr(C, align(16))]
pub struct Node<W> {
    parent: TaggedPtr<InternalNode<W>, NodeFlags>,
    pub boundary: BoundaryVertices,
    pub weight: W,
}

impl<W> fmt::Debug for Node<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("parent", &self.parent)
            .field("boundary", &self.boundary)
            .finish_non_exhaustive()
    }
}

impl<W> Node<W> {
    pub unsafe fn dealloc<A: Allocator>(this: NonNull<Node<W>>, allocator: &A) {
        match unsafe { this.as_ref() }.flags().kind() {
            NodeKind::EdgeLeaf => unsafe {
                _ = Box::from_raw_in(this.cast::<LeafNode<W>>().as_ptr(), allocator);
            },
            NodeKind::LabelLeaf => unsafe {
                _ = Box::from_raw_in(this.cast::<LabelNode<W>>().as_ptr(), allocator);
            },
            NodeKind::Cluster => unsafe {
                _ = Box::from_raw_in(this.cast::<InternalNode<W>>().as_ptr(), allocator);
            },
        }
    }
}

impl<W> Node<W> {
    pub fn flags(&self) -> NodeFlags {
        self.parent.tag()
    }

    pub fn is_edge(&self) -> bool {
        self.flags().is_edge()
    }

    #[expect(dead_code)]
    fn is_label(&self) -> bool {
        self.flags().is_label()
    }

    fn is_internal(&self) -> bool {
        self.flags().is_cluster()
    }

    pub fn is_flipped(&self) -> bool {
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

    pub fn num_boundary(&self) -> usize {
        self.boundary.count() as usize
    }

    pub fn is_path(&self) -> bool {
        self.boundary.is_path()
    }

    fn is_point(&self) -> bool {
        self.boundary.is_point()
    }

    pub fn parent(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent.as_non_null()
    }

    fn set_parent(&mut self, parent: Option<NonNull<InternalNode<W>>>) {
        self.parent
            .set_ptr(parent.map_or(ptr::null_mut(), NonNull::as_ptr));
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) fn parent_node(&self) -> Option<NonNull<Node<W>>> {
        self.parent().map(NonNull::cast)
    }

    #[cfg_attr(not(test), expect(dead_code))]
    #[expect(clippy::type_complexity)]
    pub(crate) fn force_ptr(
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
pub struct LabelNode<W> {
    node: Node<W>,
    pub label: tree::LabelId,
}

impl<W> fmt::Debug for LabelNode<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabelNode")
            .field("node", &self.node)
            .field("label", &self.label)
            .finish()
    }
}

#[repr(C)]
pub struct LeafNode<W> {
    node: Node<W>,
    pub edge: tree::EdgeId,
}

impl<W> fmt::Debug for LeafNode<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafNode")
            .field("node", &self.node)
            .field("edge", &self.edge)
            .finish()
    }
}

pub struct Children<W> {
    pub left: NonNull<Node<W>>,
    pub right: NonNull<Node<W>>,
}

impl<W> fmt::Debug for Children<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Children")
            .field("left", &self.left)
            .field("right", &self.right)
            .finish()
    }
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

    pub fn get(&self, left: bool) -> NonNull<Node<W>> {
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
pub struct InternalNode<W> {
    node: Node<W>,
    pub children: Children<W>,
}

impl<W> fmt::Debug for InternalNode<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalNode")
            .field("node", &self.node)
            .field("children", &self.children)
            .finish()
    }
}

impl<W> InternalNode<W> {
    fn alloc<A: Allocator>(
        weight: W,
        left: NonNull<Node<W>>,
        right: NonNull<Node<W>>,
        boundary: BoundaryVertices,
        allocator: &A,
    ) -> NonNull<Self> {
        let node = Box::new_in(
            InternalNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::empty()),
                    boundary,
                    weight,
                },
                children: Children { left, right },
            },
            allocator,
        );

        let (node, _) = Box::into_non_null_with_allocator(node);

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

pub mod marker {

    #[expect(dead_code)]
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

#[derive(Clone, Copy)]
pub struct Handle<W, NodeType> {
    pub node: NonNull<Node<W>>,
    _marker: core::marker::PhantomData<NodeType>,
}

impl<W, NodeType: std::fmt::Debug> std::fmt::Debug for Handle<W, NodeType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            match self.force() {
                LeafOrInternal::Edge(n) => f
                    .debug_struct("Edge")
                    .field("node", &n.node.cast::<LeafNode<W>>().as_ref())
                    .finish(),
                LeafOrInternal::Label(n) => f
                    .debug_struct("Label")
                    .field("node", &n.node.cast::<LabelNode<W>>().as_ref())
                    .finish(),
                LeafOrInternal::Internal(n) => f
                    .debug_struct("Internal")
                    .field("node", &n.node.cast::<InternalNode<W>>().as_ref())
                    .finish(),
            }
        }
    }
}

impl<W, NodeType> PartialEq for Handle<W, NodeType> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<W, NodeType> Eq for Handle<W, NodeType> {}

impl<W> Handle<W, marker::Label> {
    pub fn new_label(node: NonNull<Node<W>>) -> Handle<W, marker::Label> {
        Handle {
            node,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<W> Handle<W, marker::Edge> {
    pub fn new_edge(node: NonNull<Node<W>>) -> Handle<W, marker::Edge> {
        Handle {
            node,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<W> Handle<W, marker::Either> {
    pub fn init_edge(
        mut node: NonNull<mem::MaybeUninit<LeafNode<W>>>,
        weight: W,
        edge: tree::EdgeId,
        boundary: BoundaryVertices,
    ) -> Handle<W, marker::Leaf> {
        unsafe {
            node.as_mut().write(LeafNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LEAF),
                    boundary,
                    weight,
                },
                edge,
            });

            Handle {
                node: node.cast(),
                _marker: core::marker::PhantomData::<marker::Leaf>,
            }
        }
    }
    pub fn init_label(
        mut node: NonNull<mem::MaybeUninit<LabelNode<W>>>,
        weight: W,
        label: tree::LabelId,
        boundary: BoundaryVertices,
    ) -> Handle<W, marker::Leaf> {
        unsafe {
            node.as_mut().write(LabelNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LABEL),
                    boundary,
                    weight,
                },
                label,
            });

            Handle {
                node: node.cast(),
                _marker: core::marker::PhantomData::<marker::Leaf>,
            }
        }
    }
}

impl<W> Handle<W, marker::Internal>
where
    W: Summary,
{
    pub fn alloc_internal<A: Allocator>(
        left: Handle<W, marker::Either>,
        right: Handle<W, marker::Either>,
        boundary: BoundaryVertices,
        allocator: &A,
    ) -> Handle<W, marker::Internal> {
        let left_boundary = left.flipped_boundary().into_boundary();
        let right_boundary = right.flipped_boundary().into_boundary();
        let parent_boundary = boundary.into_boundary();
        let central = left_boundary.shared(&right_boundary).unwrap();
        let ctx = summary::MergeContext {
            left_boundary,
            right_boundary,
            parent_boundary,
            central,
        };

        let weight = W::combine(left.weight(), right.weight(), &ctx);

        let node = InternalNode::alloc(weight, left.node, right.node, boundary, allocator);

        unsafe { Self::new_internal(node) }
    }
}

impl<W, NodeType> Handle<W, NodeType> {
    pub fn new(node: NonNull<Node<W>>) -> Self {
        Self {
            node,
            _marker: core::marker::PhantomData,
        }
    }

    pub unsafe fn new_internal(node: NonNull<InternalNode<W>>) -> Self {
        Self {
            node: node.cast(),
            _marker: core::marker::PhantomData,
        }
    }

    pub fn force(
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

    pub fn set_parent(&mut self, parent: Option<Handle<W, marker::Internal>>) {
        unsafe {
            self.node.as_mut().set_parent(parent.map(|p| p.node.cast()));
        }
    }

    pub fn parent(&self) -> Option<Handle<W, marker::Internal>> {
        unsafe { self.node.as_ref().parent().map(|p| Handle::new_internal(p)) }
    }

    pub fn is_left_child(&self) -> Option<bool> {
        let parent = self.parent()?;
        Some(parent.children()[0].node == self.node)
    }

    pub fn sibling(&self) -> Option<Handle<W, NodeType>> {
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

    pub fn has_middle_boundary(&self) -> bool {
        use crate::LeafOrInternal::*;
        if self.num_boundary() == 0 {
            return false;
        }

        match self.force() {
            Label(_) | Edge(_) => false,
            Internal(internal) => internal.num_boundary() != internal.num_path_children(),
        }
    }

    pub fn weight(&self) -> &W {
        unsafe { &self.node.as_ref().weight }
    }

    fn boundary(&self) -> BoundaryVertices {
        unsafe { self.node.as_ref().boundary }
    }

    pub fn flipped_boundary(&self) -> BoundaryVertices {
        let boundary = self.boundary();
        if self.is_flipped() {
            boundary.flipped()
        } else {
            boundary
        }
    }

    fn set_boundary(&mut self, boundary: BoundaryVertices) {
        unsafe { self.node.as_mut().boundary = boundary }
    }

    pub fn remove_from_boundary(&mut self, v: VertexId) {
        unsafe {
            self.node.as_mut().boundary.remove(v);
        }
    }

    pub fn add_boundary_vertex<A: Allocator>(&mut self, tree: &TopTree<W, A>, v: VertexId) {
        let resolved = tree.resolve_vertex(v);

        match self.force() {
            LeafOrInternal::Edge(edge) => {
                let Endpoints([left, right]) = tree.tree.edge_endpoints(edge.edge());

                if left == resolved {
                    unsafe { self.node.as_mut().boundary.add(v, true) }
                } else if right == resolved {
                    unsafe { self.node.as_mut().boundary.add(v, false) }
                } else {
                    panic!("vertex not found in edge endpoints")
                }
            }
            LeafOrInternal::Label(_) => unsafe { self.node.as_mut().boundary.add(v, false) },
            LeafOrInternal::Internal(internal) => {
                let [left, right] = internal.children();
                let in_left = left.boundary().contains(v);
                let in_right = right.boundary().contains(v);

                if in_left && in_right {
                    assert!(!left.is_path() || !right.is_path(),);
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

    pub fn is_path(&self) -> bool {
        unsafe { self.node.as_ref().is_path() }
    }

    pub fn is_point(&self) -> bool {
        unsafe { self.node.as_ref().is_point() }
    }

    pub fn is_flipped(&self) -> bool {
        unsafe { self.node.as_ref().is_flipped() }
    }

    fn set_flipped(&mut self, flipped: bool) {
        unsafe { self.node.as_mut().set_flipped(flipped) }
    }

    pub fn toggle_flipped(&mut self) {
        unsafe { self.node.as_mut().toggle_flipped() }
    }

    /// returns `None` if the node has no grandparent (i.e. is a child of/or the root)
    pub fn rotate_up(&mut self) -> Option<()>
    where
        W: Summary,
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
        let sibling_boundary = sibling.flipped_boundary();
        let uncle_boundary = uncle.flipped_boundary();
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

    pub fn forget_type(self) -> Handle<W, marker::Either> {
        Handle {
            node: self.node,
            _marker: core::marker::PhantomData,
        }
    }

    fn splay_step(self) -> Option<Handle<W, marker::Either>>
    where
        W: Summary,
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

    pub fn semi_splay(self)
    where
        W: Summary,
    {
        let mut node = unsafe { ptr::read(&self) }.forget_type();
        while let Some(next) = node.splay_step() {
            node = next;
        }
    }

    pub fn full_splay(self)
    where
        W: Summary,
    {
        while let Some(next_node) = unsafe { ptr::read(&self) }.splay_step() {
            next_node.splay_step();
        }
    }

    pub fn has_left_boundary<A: Allocator>(&self, root: &TopTree<W, A>) -> bool {
        use crate::LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.tree.edge_endpoints(leaf.edge())[self.is_flipped()];

                root.is_boundary_vertex_internal(v)
            }
            Label(label) => root.is_boundary_vertex_internal(root.tree.label_vertex(label.label())),
            Internal(internal) => internal.child(!self.is_flipped()).is_path(),
        }
    }

    pub fn has_right_boundary<A: Allocator>(&self, root: &TopTree<W, A>) -> bool {
        use crate::LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.tree.edge_endpoints(leaf.edge())[!self.is_flipped()];

                root.is_boundary_vertex_internal(v)
            }
            Label(label) => root.is_boundary_vertex_internal(root.tree.label_vertex(label.label())),
            Internal(internal) => internal.child(self.is_flipped()).is_path(),
        }
    }

    pub fn is_internal(&self) -> bool {
        unsafe { self.node.as_ref().is_internal() }
    }

    pub unsafe fn into_internal(self) -> Handle<W, marker::Internal> {
        debug_assert!(self.is_internal());
        unsafe { Handle::new_internal(self.node.cast()) }
    }

    pub fn delete_all_ancestors<A: Allocator>(self, allocator: &A) {
        if let Some(parent) = self.parent() {
            let mut sibling = self.sibling().expect("node must have a sibling");
            parent.delete_all_ancestors(allocator);
            unsafe {
                sibling.node.as_mut().set_parent(None);
            }
        }

        unsafe {
            Node::dealloc(self.node, allocator);
        }
    }

    pub fn root(self) -> Handle<W, marker::Either> {
        let mut node = self.forget_type();
        while let Some(parent) = node.parent() {
            node = parent.forget_type();
        }
        node
    }
}

impl<W> Handle<W, marker::Label> {
    pub fn label(&self) -> tree::LabelId {
        unsafe { self.node.cast::<LabelNode<W>>().as_ref().label }
    }
}

impl<W> Handle<W, marker::Edge> {
    pub fn edge(&self) -> tree::EdgeId {
        unsafe { self.node.cast::<LeafNode<W>>().as_ref().edge }
    }
}

impl<W> Handle<W, marker::Internal> {
    pub fn push_flip(&self) {
        if self.is_flipped() {
            unsafe {
                let node = self.node.cast::<InternalNode<W>>().as_mut();

                node.set_flipped(false);
                node.boundary.flip();
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

    pub fn children(&self) -> [Handle<W, marker::Either>; 2] {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            [
                Handle::new(internal.children.left),
                Handle::new(internal.children.right),
            ]
        }
    }

    pub fn flipped_children(&self) -> [Handle<W, marker::Either>; 2] {
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

    pub fn child(&self, left: bool) -> Handle<W, marker::Internal> {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            Handle::new(internal.children.get(left))
        }
    }

    pub fn set_child(&mut self, mut child: NonNull<Node<W>>, left: bool) {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_mut();
            *internal.children.get_mut(left) = child;
            child
                .as_mut()
                .set_parent(Some(self.node.cast::<InternalNode<W>>()));
        }
    }

    pub fn recompute_boundary(&mut self)
    where
        W: Summary,
    {
        let [left, right] = self.flipped_children();
        let left_boundary = left.flipped_boundary().into_boundary();
        let right_boundary = right.flipped_boundary().into_boundary();
        let parent_boundary = self.flipped_boundary().into_boundary();
        let ctx = summary::MergeContext {
            left_boundary,
            right_boundary,
            parent_boundary,
            central: left_boundary.shared(&right_boundary).unwrap(),
        };
        unsafe {
            self.node
                .as_mut()
                .weight
                .update_boundary(left.weight(), right.weight(), &ctx);
        }
    }

    fn recompute_weight(&mut self)
    where
        W: Summary,
    {
        let [left, right] = self.flipped_children();
        let left_boundary = left.flipped_boundary().into_boundary();
        let right_boundary = right.flipped_boundary().into_boundary();
        let parent_boundary = self.flipped_boundary().into_boundary();
        let ctx = summary::MergeContext {
            left_boundary,
            right_boundary,
            parent_boundary,
            central: left_boundary.shared(&right_boundary).unwrap(),
        };
        unsafe {
            let lw = &left.weight();
            let rw = &right.weight();
            let new_weight = W::combine(lw, rw, &ctx);
            self.node.as_mut().weight = new_weight;
        }
    }
}
