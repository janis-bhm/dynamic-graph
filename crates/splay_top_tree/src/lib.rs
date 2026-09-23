#![feature(ptr_as_uninit, cast_maybe_uninit)]

use std::{
    mem,
    ptr::{self, NonNull},
};

use crate::{
    index::Index,
    tree::{EdgeEndpoints, EdgeKey, OwningEdgeKey, Tree},
    util::TaggedPtr,
};

#[cfg(test)]
mod tests;
mod util;

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
        const BOUNDARY_LOW = 1 << 2;
        const BOUNDARY_HIGH = 1 << 3;
        const LABEL = 1 << 2 | 1 << 3;
    }
}

impl NodeFlags {
    fn num_boundary(&self) -> usize {
        let mut count = 0;
        count += usize::from(self.contains(NodeFlags::BOUNDARY_LOW));
        count += 2 * usize::from(self.contains(NodeFlags::BOUNDARY_HIGH));

        if count == 3 {
            usize::from(self.contains(NodeFlags::LEAF))
        } else {
            count
        }
    }

    fn set_num_boundary(&mut self, num: usize) {
        if self.is_label() {
            debug_assert!(num <= 1);
            self.set(NodeFlags::LEAF, num == 1);
        } else {
            debug_assert!(num <= 2);
            self.set(NodeFlags::BOUNDARY_LOW, num & 1 != 0);
            self.set(NodeFlags::BOUNDARY_HIGH, num & 2 != 0);
        }
    }

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

#[repr(C)]
struct LabelNode<W> {
    node: Node<W>,
    vertex: Index,
    label: Index,
}

#[repr(C)]
struct LeafNode<W> {
    node: Node<W>,
    edge: OwningEdgeKey,
}

impl<W> LabelNode<W> {
    fn init(
        node: NonNull<LabelNode<W>>,
        weight: W,
        vertex: Index,
        label: Index,
        num_boundary: usize,
    ) {
        unsafe {
            let uninit = node.as_uninit_mut();
            uninit.write(LabelNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LABEL),
                    weight,
                },
                vertex,
                label,
            });

            uninit.assume_init_mut().node.set_num_boundary(num_boundary)
        };
    }
}

impl<W> LeafNode<W> {
    fn init(node: NonNull<LeafNode<W>>, weight: W, edge: OwningEdgeKey, num_boundary: usize) {
        unsafe {
            let uninit = node.as_uninit_mut();
            uninit.write(LeafNode {
                node: Node {
                    parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LEAF),
                    weight,
                },
                edge,
            });

            uninit.assume_init_mut().node.set_num_boundary(num_boundary)
        };
    }
}

#[repr(C)]
struct InternalNode<W> {
    node: Node<W>,
    children: [NonNull<Node<W>>; 2],
}

impl<W> InternalNode<W> {
    fn alloc(
        weight: W,
        left: NonNull<Node<W>>,
        right: NonNull<Node<W>>,
        num_boundary: usize,
    ) -> NonNull<Self> {
        let mut node = Box::new(InternalNode {
            node: Node {
                parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::empty()),
                weight,
            },
            children: [left, right],
        });
        node.node.set_num_boundary(num_boundary);

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
        self.parent.tag().num_boundary()
    }

    fn set_num_boundary(&mut self, num: usize) {
        assert!(num <= 2, "num_boundary must be <= 2");
        self.parent.update_tag(|flags| {
            flags.set_num_boundary(num);
        });
    }

    fn inc_num_boundary(&mut self) {
        let new = self.num_boundary() + 1;
        assert!(new <= 2, "num_boundary must be <= 2");
        self.set_num_boundary(new);
    }

    fn dec_num_boundary(&mut self) {
        let new = self
            .num_boundary()
            .checked_sub(1)
            .expect("num_boundary must be >= 0");
        self.set_num_boundary(new);
    }

    fn is_path(&self) -> bool {
        self.num_boundary() == 2
    }

    fn is_point(&self) -> bool {
        self.num_boundary() < 2
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
            if p.children[0] == self_ptr {
                p.children[1]
            } else {
                p.children[0]
            }
        })
    }

    unsafe fn is_left_child(&self) -> Option<bool> {
        self.parent().map(|p| unsafe { p.as_ref() }).map(|p| {
            let self_ptr = NonNull::from(self);
            p.children[0] == self_ptr
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

    fn set_num_boundary(&mut self, num: usize) {
        unsafe { self.node.as_mut().set_num_boundary(num) }
    }

    fn inc_num_boundary(&mut self) {
        unsafe { self.node.as_mut().inc_num_boundary() }
    }

    fn dec_num_boundary(&mut self) {
        unsafe { self.node.as_mut().dec_num_boundary() }
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

        let new_parent_is_path: bool;
        let flip_new_parent: bool;
        let flip_gp: bool;
        if same_sides && sibling_is_path {
            // path
            let gp_has_middle = gp.has_middle_boundary();
            new_parent_is_path = gp_has_middle || uncle_is_path;
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
                new_parent_is_path = sibling_is_path || uncle_is_path;
                flip_new_parent = sibling_is_path;
                flip_gp = sibling_is_path;
                self.toggle_flipped();
            } else {
                new_parent_is_path = uncle_is_path;
                flip_new_parent = false;
                flip_gp = false;
                sibling.toggle_flipped();
            }
        }

        parent.set_child(sibling.node, !uncle_is_left);
        parent.set_child(uncle.node, uncle_is_left);
        parent.set_flipped(flip_new_parent);
        parent.set_num_boundary(if new_parent_is_path { 2 } else { 1 });

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

    fn has_left_boundary(&self, root: &Tree<W>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.endpoints(leaf.edge())[self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Label(label) => root.is_boundary_vertex(label.vertex()),
            Internal(internal) => internal.child(!self.is_flipped()).is_path(),
        }
    }

    fn has_right_boundary(&self, root: &Tree<W>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Edge(leaf) => {
                let v = root.endpoints(leaf.edge())[!self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Label(label) => root.is_boundary_vertex(label.vertex()),
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
    fn label(&self) -> Index {
        unsafe { self.node.cast::<LabelNode<W>>().as_ref().label }
    }
    fn vertex(&self) -> Index {
        unsafe { self.node.cast::<LabelNode<W>>().as_ref().vertex }
    }
}

impl<W> Handle<W, marker::Leaf> {
    fn endpoints(&self) -> [Index; 2] {
        match self.force() {
            LeafOrInternal::Edge(edge) => unsafe {
                edge.node.cast::<LeafNode<W>>().as_ref().edge.endpoints()
            },
            LeafOrInternal::Label(label) => [label.vertex(), label.vertex()],
            LeafOrInternal::Internal(_) => {
                unreachable!("Handle<W, marker::Leaf> cannot be Internal")
            }
        }
    }
}

impl<W> Handle<W, marker::Edge> {
    fn edge(&self) -> &OwningEdgeKey {
        unsafe { &self.node.cast::<LeafNode<W>>().as_ref().edge }
    }
}

impl<W> Handle<W, marker::Internal> {
    fn push_flip(&self) {
        if self.is_flipped() {
            unsafe {
                let node = self.node.cast::<InternalNode<W>>().as_mut();

                node.set_flipped(false);
                node.children.swap(0, 1);

                for mut child in node.children {
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
                Handle::new(internal.children[0]),
                Handle::new(internal.children[1]),
            ]
        }
    }

    fn flipped_children(&self) -> [Handle<W, marker::Internal>; 2] {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            if self.is_flipped() {
                [
                    Handle::new(internal.children[1]),
                    Handle::new(internal.children[0]),
                ]
            } else {
                [
                    Handle::new(internal.children[0]),
                    Handle::new(internal.children[1]),
                ]
            }
        }
    }

    fn child(&self, left: bool) -> Handle<W, marker::Internal> {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_ref();
            Handle::new(internal.children[(!left) as usize])
        }
    }

    fn set_child(&mut self, mut child: NonNull<Node<W>>, left: bool) {
        unsafe {
            let internal = self.node.cast::<InternalNode<W>>().as_mut();
            internal.children[(!left) as usize] = child;
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
    v: Index,
) -> impl Iterator<Item = Handle<W, marker::Leaf>> + '_ {
    root.vertices.get(&v).into_iter().flat_map(|v| {
        let labels = v
            .labels
            .iter()
            .filter_map(|key| root.labels.get(key))
            .map(|label| Handle::<W, marker::Leaf>::new(label.cast()));
        v.edges
            .iter()
            .filter_map(|key| root.edges.get(key))
            .map(|edge| Handle::<W, marker::Leaf>::new(edge.edge_node.cast()))
            .chain(labels)
    })
}

/// Finds the least common ancestor of all leaves incident to `v` in the tree rooted at `root`.
pub(crate) fn find_consuming_node<W>(root: &Tree<W>, v: Index) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    let node = incident_leaves(root, v).next()?;
    unsafe { ptr::read(&node) }.semi_splay();

    // if the vertex has exactly one incident edge, then the consuming node is the incident leaf.
    if root.has_at_most_one_incident_element(v) {
        return Some(node.forget_type());
    }

    let endpoints = match node.force() {
        LeafOrInternal::Edge(edge) => root.endpoints(edge.edge()),
        LeafOrInternal::Label(label) => EdgeEndpoints {
            left: label.vertex(),
            right: label.vertex(),
        },
        LeafOrInternal::Internal(_) => unreachable!("node must be a leaf"),
    };

    let flip = node.is_flipped();
    let mut is_left = (endpoints.left == v) != flip;
    let mut is_right = (endpoints.right == v) != flip;
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

pub(crate) fn expose<W>(v: Index, root: &mut Tree<W>) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    fn expose_prepared<W>(mut node: Handle<W, marker::Either>) -> Handle<W, marker::Either>
    where
        W: Reduce,
    {
        let mut left = false;
        let mut right = false;

        loop {
            node.inc_num_boundary();

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

            let node = expose_prepared(consuming_node);
            root.expose_vertex(v);

            Some(node)
        }
        None => {
            root.expose_vertex(v);
            None
        }
    }
}

pub(crate) fn deexpose<W>(v: Index, tree: &mut Tree<W>) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    let mut node = find_consuming_node(tree, v);
    let mut root = None;

    while let Some(mut some_node) = node {
        some_node.dec_num_boundary();
        node = some_node.parent().map(Handle::forget_type);
        root = Some(some_node);
    }

    if let Some(vert) = tree.vertices.get_mut(&v) {
        vert.exposed = false;
    }

    root
}

pub(crate) fn link<W>(u: Index, v: Index, weight: W, tree: &mut Tree<W>) -> NonNull<Node<W>>
where
    W: Reduce,
{
    let mut ru = expose(u, tree);
    if let Some(ref mut tu) = ru
        && Handle::has_left_boundary(tu, tree)
    {
        tu.toggle_flipped();
    }
    tree.set_vertex_exposed(u, false);

    let mut rv = expose(v, tree);
    if let Some(ref mut tv) = rv
        && Handle::has_right_boundary(tv, tree)
    {
        tv.toggle_flipped();
    }
    tree.set_vertex_exposed(v, false);

    let node = Box::into_non_null(Box::new_uninit()).cast_init();
    let edge = tree.add_edge(u, v, node);
    LeafNode::init(
        node,
        weight,
        edge,
        ru.is_some() as usize + rv.is_some() as usize,
    );

    let mut node = node.cast::<Node<W>>();
    if let Some(ru) = ru {
        let left = ru;
        let right = node.cast::<Node<W>>();
        let weight = unsafe {
            let wl = &left.node.as_ref().weight;
            let wr = &right.as_ref().weight;
            W::reduce(wl, wr)
        };
        node = InternalNode::alloc(weight, left.node, right, rv.is_some() as usize).cast();
    }

    if let Some(rv) = rv {
        let left = node.cast::<Node<W>>();
        let right = rv;
        let weight = unsafe {
            let wl = &left.as_ref().weight;
            let wr = &right.node.as_ref().weight;
            W::reduce(wl, wr)
        };
        node = InternalNode::alloc(weight, left, right.node, 0).cast();
    }

    node
}

#[expect(clippy::type_complexity)]
pub(crate) fn cut<W>(
    u: Index,
    v: Index,
    tree: &mut Tree<W>,
) -> (
    Option<Handle<W, marker::Either>>,
    Option<Handle<W, marker::Either>>,
)
where
    W: Reduce,
{
    let edge = EdgeKey::<()>::new(u, v);
    let Some(edge) = tree.try_edge(&edge) else {
        return (None, None);
    };

    let edge = Handle::new_edge(edge.edge_node.cast::<Node<W>>());

    let key = unsafe { ptr::read(edge.edge()) };
    unsafe { ptr::read(&edge) }.full_splay();
    edge.delete_all_ancestors();

    _ = tree.remove_edge(key);

    tree.expose_vertex(u);
    tree.expose_vertex(v);

    let ru = deexpose(u, tree);
    let rv = deexpose(v, tree);

    (ru, rv)
}

pub(crate) fn attach<W>(v: Index, weight: W, tree: &mut Tree<W>) -> (Index, NonNull<Node<W>>)
where
    W: Reduce,
{
    let mut rv = expose(v, tree);
    if let Some(ref mut tv) = rv
        && Handle::has_left_boundary(tv, tree)
    {
        tv.toggle_flipped();
    }
    tree.set_vertex_exposed(v, false);

    let node: NonNull<LabelNode<W>> = Box::into_non_null(Box::new_uninit()).cast_init();
    let label = tree.add_vertex_label(v, node.cast()).expect("asdf");
    LabelNode::init(node, weight, v, label, usize::from(rv.is_some()));

    let root: NonNull<Node<W>> = match rv {
        Some(rv) => {
            let left = rv;
            let right = node.cast::<Node<W>>();
            let weight = unsafe {
                let wl = &left.node.as_ref().weight;
                let wr = &right.as_ref().weight;
                W::reduce(wl, wr)
            };
            InternalNode::alloc(weight, left.node, right, 0).cast()
        }
        None => node.cast(),
    };

    (label, root)
}

pub(crate) fn detach<W>(v: Index, tree: &mut Tree<W>) {
    // labels are never path components, so removing them can never disconnect the tree.
    // instead, we want to replace the label's parent with the label's sibling, then delete the label and parent.
    let label = tree.labels.remove(&v).expect("label must exist");
    let label_handle = Handle::<W, marker::Label>::new(label.cast());
    if let Some(parent) = label_handle.parent() {
        let mut sibling = label_handle.sibling().expect("label must have a sibling");

        if let Some(mut gp) = parent.parent() {
            let parent_is_left = parent.is_left_child().expect("parent has parent");
            // we need to flip the sibling if it is on the opposite side of the parent.
            // if the parent was flipped, then flip the sibling (again).
            let flip_sibling = (sibling.is_left_child().expect("sibling has parent")
                != parent_is_left)
                ^ parent.is_flipped();

            if flip_sibling {
                sibling.toggle_flipped();
            }

            gp.set_child(sibling.node, parent_is_left);
        } else {
            // parent is root, so we just make the sibling the new root.
            sibling.set_parent(None);
        }

        unsafe {
            Node::<W>::dealloc(parent.node);
            Node::<W>::dealloc(label_handle.node);
        }
    }
}

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
}

impl<W> TopTree<W> {
    pub fn new() -> Self {
        Self { tree: Tree::new() }
    }

    pub fn add_vertex(&mut self) -> Index {
        self.tree.add_vertex()
    }

    pub fn attach(&mut self, index: Index, label: W)
    where
        W: Reduce,
    {
        attach(index, label, &mut self.tree);
    }

    pub fn remove_vertex(&mut self, index: Index)
    where
        W: Reduce,
    {
        use std::collections::btree_map::Entry::Occupied;
        let Occupied(mut entry) = self.tree.vertices.entry(index) else {
            return;
        };

        let edges = mem::take(&mut entry.get_mut().edges);
        let labels = mem::take(&mut entry.get_mut().labels);

        for edge in edges {
            let [u, v] = edge.endpoints();
            cut(u, v, &mut self.tree);
        }

        for label in labels {
            detach(label, &mut self.tree);
        }

        self.tree.vertices.remove(&index);
    }

    pub fn link(&mut self, u: Index, v: Index, weight: W)
    where
        W: Reduce,
    {
        link(u, v, weight, &mut self.tree);
    }

    pub fn cut(&mut self, u: Index, v: Index) -> (Option<Entry<'_, W>>, Option<Entry<'_, W>>)
    where
        W: Reduce,
    {
        let (ru, rv) = cut(u, v, &mut self.tree);

        let ru = ru.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        });
        let rv = rv.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        });

        (ru, rv)
    }

    pub fn expose(&mut self, v: Index) -> Option<Entry<'_, W>>
    where
        W: Reduce,
    {
        let node = expose(v, &mut self.tree);
        node.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        })
    }

    pub fn expose2(&mut self, u: Index, v: Index) -> (Option<Entry<'_, W>>, Option<Entry<'_, W>>)
    where
        W: Reduce,
    {
        let ru = expose(u, &mut self.tree);
        _ = ru.as_ref().map(|h| assert!(h.is_point()));
        let rv = expose(v, &mut self.tree);

        let ru = ru.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        });
        let rv = rv.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        });

        (ru, rv)
    }

    pub fn deexpose(&mut self, v: Index) -> Option<Entry<'_, W>>
    where
        W: Reduce,
    {
        let node = deexpose(v, &mut self.tree);
        node.map(|h| Entry {
            handle: h,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<W> Default for TopTree<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W> Drop for TopTree<W> {
    fn drop(&mut self) {
        self.tree.edges.retain(|_, edge| {
            Handle::new_edge(edge.edge_node.cast::<Node<W>>()).delete_all_ancestors();

            unsafe { Node::<W>::dealloc(edge.edge_node.cast()) };
            true
        });
    }
}

mod tree {
    use std::{
        collections::BTreeMap,
        marker::PhantomData,
        ptr::{self, NonNull},
    };

    use crate::{LabelNode, LeafNode, index::Index, util::WithDropExt};

    use super::index::IndexAllocator;

    pub struct Tree<W> {
        index_allocator: IndexAllocator,
        pub vertices: BTreeMap<Index, Vertex>,
        pub edges: BTreeMap<EdgeKey<Private>, Edge<W>>,
        pub labels: BTreeMap<Index, NonNull<LabelNode<W>>>,
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Private(pub(self) ());

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct EdgeKey<T = ()>(pub [Index; 2], PhantomData<T>);

    pub type OwningEdgeKey = EdgeKey<Private>;

    impl<T> EdgeKey<T> {
        pub fn new(u: Index, v: Index) -> EdgeKey<T> {
            let mut inner = [u, v];
            inner.sort();
            Self(inner, unsafe { std::mem::zeroed() })
        }

        unsafe fn transmute<U>(self) -> EdgeKey<U> {
            EdgeKey(self.0, PhantomData)
        }

        unsafe fn transmute_ref<U>(&self) -> &EdgeKey<U> {
            unsafe { &*(self as *const EdgeKey<T> as *const EdgeKey<U>) }
        }

        pub fn endpoints(&self) -> [Index; 2] {
            self.0
        }
    }

    impl EdgeKey<Private> {
        fn new_private(u: Index, v: Index) -> EdgeKey<Private> {
            let mut inner = [u, v];
            inner.sort();
            Self(inner, PhantomData)
        }

        pub fn downcast(&self) -> EdgeKey<()> {
            unsafe { *self.transmute_ref() }
        }
    }

    #[derive(Default)]
    pub struct Vertex {
        pub(crate) edges: Vec<EdgeKey<Private>>,
        pub(crate) labels: Vec<Index>,
        pub(crate) exposed: bool,
    }

    pub struct Edge<W> {
        pub edge_node: NonNull<LeafNode<W>>,
        pub endpoints: [Index; 2],
    }

    #[derive(Debug)]
    pub struct EdgeEndpoints {
        pub left: Index,
        pub right: Index,
    }

    pub enum EdgeOrLabel<W> {
        Edge(Edge<W>),
        Label(Index),
    }

    impl core::ops::Index<bool> for EdgeEndpoints {
        type Output = Index;

        fn index(&self, index: bool) -> &Self::Output {
            if index { &self.right } else { &self.left }
        }
    }

    impl<W> Tree<W> {
        pub fn new() -> Self {
            Self {
                index_allocator: IndexAllocator::new(),
                vertices: BTreeMap::new(),
                edges: BTreeMap::new(),
                labels: BTreeMap::new(),
            }
        }

        pub fn endpoints(&self, edge: &EdgeKey<Private>) -> EdgeEndpoints {
            self.edges
                .get(edge)
                .map(|e| EdgeEndpoints {
                    left: e.endpoints[0],
                    right: e.endpoints[1],
                })
                .unwrap()
        }

        pub fn edge(&self, edge: &EdgeKey<Private>) -> &Edge<W> {
            self.edges.get(edge).expect("edge must exist in tree")
        }

        pub fn try_edge<T>(&self, edge: &EdgeKey<T>) -> Option<&Edge<W>> {
            self.edges.get(unsafe { edge.transmute_ref::<Private>() })
        }

        pub fn is_boundary_vertex(&self, index: Index) -> bool {
            self.vertices.get(&index).is_some_and(|v| v.exposed)
                || self.total_degree(index).is_some_and(|d| d >= 2)
        }

        pub fn degree(&self, index: Index) -> Option<usize> {
            self.vertices.get(&index).map(|v| v.edges.len())
        }

        pub fn total_degree(&self, index: Index) -> Option<usize> {
            self.vertices
                .get(&index)
                .map(|v| v.edges.len() + v.labels.len())
        }

        pub fn expose_vertex(&mut self, index: Index) {
            if let Some(vertex) = self.vertices.get_mut(&index) {
                vertex.exposed = true;
            }
        }

        pub fn set_vertex_exposed(&mut self, index: Index, exposed: bool) {
            if let Some(vertex) = self.vertices.get_mut(&index) {
                vertex.exposed = exposed;
            }
        }

        pub fn incident_edges(&self, index: Index) -> impl Iterator<Item = &Edge<W>> {
            self.vertices
                .get(&index)
                .into_iter()
                .flat_map(|v| v.edges.iter())
                .filter_map(move |key| self.edges.get(key))
        }

        pub fn has_at_most_one_incident_element(&self, index: Index) -> bool {
            self.vertices
                .get(&index)
                .map(|v| v.edges.len() + v.labels.len() <= 1)
                .unwrap_or(true)
        }

        pub fn has_at_most_one_incident_edge(&self, index: Index) -> bool {
            self.vertices
                .get(&index)
                .map(|v| v.edges.len() <= 1)
                .unwrap_or(true)
        }

        pub fn add_vertex(&mut self) -> Index {
            let index = self.index_allocator.allocate();
            self.vertices.insert(
                index,
                Vertex {
                    edges: Vec::new(),
                    labels: Vec::new(),
                    exposed: false,
                },
            );
            index
        }

        pub fn add_vertex_label(
            &mut self,
            v: Index,
            label_node: NonNull<LabelNode<W>>,
        ) -> Option<Index> {
            if let Some(v) = self.vertices.get_mut(&v) {
                let index = self.index_allocator.allocate();
                self.labels.insert(index, label_node);
                v.labels.push(index);
                Some(index)
            } else {
                None
            }
        }

        pub fn remove_vertex(&mut self, index: Index) -> impl Iterator<Item = EdgeOrLabel<W>> {
            let edges = &mut self.edges;
            let labels = &mut self.labels;
            let (vertex_edges, vertex_labels) = self
                .vertices
                .remove(&index)
                .map(|v| (v.edges, v.labels))
                .unwrap_or_default();

            let edges = vertex_edges
                .into_iter()
                .filter_map(move |key| edges.remove(&key).map(|e| (key, e)));

            let labels = vertex_labels.into_iter().inspect(move |label_index| {
                labels.remove(label_index);
            });

            let vertices = &mut self.vertices;
            let edges = edges.map(move |(key, edge)| {
                let other_index = if edge.endpoints[0] == index {
                    edge.endpoints[1]
                } else {
                    edge.endpoints[0]
                };
                if let Some(other_vertex) = vertices.get_mut(&other_index) {
                    other_vertex.edges.retain(|e| e != &key);
                }
                edge
            });

            let index_allocator = &mut self.index_allocator;

            let edges = edges.with_drop(move |edges| {
                // drain iter to ensure that all edges are removed from the tree before deallocating the index
                for _ in edges {}
                index_allocator.deallocate(index);
            });

            edges
                .map(|edge| EdgeOrLabel::Edge(edge))
                .chain(labels.map(EdgeOrLabel::Label))
        }

        pub fn add_edge(
            &mut self,
            left: Index,
            right: Index,
            edge_node: NonNull<LeafNode<W>>,
        ) -> EdgeKey<Private> {
            let key = EdgeKey::new_private(left, right);
            let edge = Edge {
                edge_node,
                endpoints: [left, right],
            };

            self.edges.insert(unsafe { ptr::read(&key) }, edge);

            self.vertices
                .entry(left)
                .or_default()
                .edges
                .push(unsafe { ptr::read(&key) });

            self.vertices
                .entry(right)
                .or_default()
                .edges
                .push(unsafe { ptr::read(&key) });

            key
        }

        pub fn remove_edge(&mut self, key: OwningEdgeKey) -> Edge<W> {
            unsafe { self.remove_edge_unchecked(key).unwrap() }
        }

        unsafe fn remove_edge_unchecked<T>(&mut self, key: EdgeKey<T>) -> Option<Edge<W>> {
            let key = unsafe { key.transmute::<Private>() };
            if let Some(edge) = self.edges.remove(&key) {
                for endpoint in edge.endpoints.iter() {
                    if let Some(vertex) = self.vertices.get_mut(endpoint) {
                        vertex.edges.retain(|e| e != &key);
                    }
                }

                Some(edge)
            } else {
                None
            }
        }
    }
}

pub mod index {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Index(pub u32);

    pub struct IndexAllocator {
        next_index: u32,
        free_indices: Vec<u32>,
    }

    impl IndexAllocator {
        pub fn new() -> Self {
            Self {
                next_index: 0,
                free_indices: Vec::new(),
            }
        }

        pub fn allocate(&mut self) -> Index {
            if let Some(index) = self.free_indices.pop() {
                Index(index)
            } else {
                let index = self.next_index;
                self.next_index += 1;
                Index(index)
            }
        }

        pub fn deallocate(&mut self, index: Index) {
            self.free_indices.push(index.0);
        }
    }

    impl Default for IndexAllocator {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub trait Reduce {
    fn reduce(&self, other: &Self) -> Self;
}
