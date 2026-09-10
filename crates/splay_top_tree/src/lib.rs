#![feature(ptr_as_uninit, cast_maybe_uninit)]

use std::ptr::{self, NonNull};

use crate::{
    index::Index,
    tree::{EdgeKey, OwningEdgeKey, Tree},
    util::TaggedPtr,
};

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct NodeFlags: u8 {
        const LEAF = 1 << 0;
        const FLIPPED = 1 << 1;
        const BOUNDARY_LOW = 1 << 2;
        const BOUNDARY_HIGH = 1 << 3;
    }
}

impl NodeFlags {
    fn num_boundary(&self) -> usize {
        let mut count = 0;
        count += usize::from(self.contains(NodeFlags::BOUNDARY_LOW));
        count += 2 * usize::from(self.contains(NodeFlags::BOUNDARY_HIGH));
        count
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

fn boundary_bits(num: usize) -> NodeFlags {
    debug_assert!(num <= 2);
    let mut flags = NodeFlags::empty();
    flags.set(NodeFlags::BOUNDARY_LOW, num & 1 != 0);
    flags.set(NodeFlags::BOUNDARY_HIGH, num & 2 != 0);
    flags
}

#[repr(C, align(16))]
struct Node<W> {
    parent: TaggedPtr<InternalNode<W>, NodeFlags>,
    weight: W,
}

impl<W> Node<W> {
    unsafe fn dealloc(this: NonNull<Node<W>>) {
        let leaf = unsafe { this.as_ref() }.is_leaf();
        let ptr = this.as_ptr();
        if leaf {
            drop(unsafe { Box::from_raw(ptr as *mut LeafNode<W>) });
        } else {
            drop(unsafe { Box::from_raw(ptr as *mut InternalNode<W>) });
        }
    }
}

#[repr(C)]
struct LeafNode<W> {
    node: Node<W>,
    edge: OwningEdgeKey,
}

impl<W> LeafNode<W> {
    fn alloc(weight: W, edge: OwningEdgeKey, num_boundary: usize) -> NonNull<Self> {
        let mut node = Box::new(LeafNode {
            node: Node {
                parent: TaggedPtr::new(ptr::null_mut(), NodeFlags::LEAF),
                weight,
            },
            edge,
        });
        node.node.set_num_boundary(num_boundary);
        Box::into_non_null(node)
    }

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
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Leaf;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Internal;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Either;
}

impl<W> Node<W> {
    fn flags(&self) -> NodeFlags {
        self.parent.tag()
    }
    fn is_leaf(&self) -> bool {
        self.flags().contains(NodeFlags::LEAF)
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
            flags.remove(NodeFlags::BOUNDARY_LOW | NodeFlags::BOUNDARY_HIGH);
            flags.insert(boundary_bits(num));
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
    ) -> LeafOrInternal<NonNull<LeafNode<W>>, NonNull<InternalNode<W>>> {
        if unsafe { (&*this.as_ptr()).is_leaf() } {
            LeafOrInternal::Leaf(this.cast())
        } else {
            LeafOrInternal::Internal(this.cast())
        }
    }
}

impl<W> InternalNode<W> {
    /// # Safety
    /// Creates mutable references to the children of this node.
    unsafe fn push_flip(&mut self) {
        if self.node.is_flipped() {
            self.node.set_flipped(false);
            self.children.swap(0, 1);
            for mut child in self.children {
                unsafe {
                    child.as_mut().toggle_flipped();
                }
            }
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

impl<W> Handle<W, marker::Leaf> {
    fn new_leaf(node: NonNull<Node<W>>) -> Handle<W, marker::Leaf> {
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

    fn force(&self) -> LeafOrInternal<Handle<W, marker::Leaf>, Handle<W, marker::Internal>> {
        if unsafe { self.node.as_ref().is_leaf() } {
            LeafOrInternal::Leaf(Handle::new(self.node))
        } else {
            LeafOrInternal::Internal(Handle::new(self.node))
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
            Leaf(_) => false,
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

        parent.set_child(sibling.node, uncle_is_left);
        parent.set_child(uncle.node, !uncle_is_left);
        parent.set_flipped(flip_new_parent);
        parent.set_num_boundary(if new_parent_is_path { 2 } else { 1 });

        gp.set_child(self.node, uncle_is_left);
        gp.set_child(parent.node, !uncle_is_left);
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

            if p.is_point() && (gp.is_point() || ggp.is_point()) {
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

    fn has_left_boundary(&self, root: &Tree<NonNull<LeafNode<W>>>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Leaf(leaf) => {
                let v = root.endpoints(leaf.edge())[self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Internal(internal) => internal.child(!self.is_flipped()).is_path(),
        }
    }

    fn has_right_boundary(&self, root: &Tree<NonNull<LeafNode<W>>>) -> bool {
        use LeafOrInternal::*;
        match self.force() {
            Leaf(leaf) => {
                let v = root.endpoints(leaf.edge())[!self.is_flipped()];

                root.is_boundary_vertex(v)
            }
            Internal(internal) => internal.child(self.is_flipped()).is_path(),
        }
    }

    fn is_internal(&self) -> bool {
        !unsafe { self.node.as_ref().is_leaf() }
    }

    unsafe fn into_internal(self) -> Handle<W, marker::Internal> {
        debug_assert!(self.is_internal());
        unsafe { Handle::new_internal(self.node.cast()) }
    }

    unsafe fn into_leaf(self) -> Handle<W, marker::Leaf> {
        debug_assert!(!self.is_internal());
        Handle::new_leaf(self.node.cast())
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
}

impl<W> Handle<W, marker::Leaf> {
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

pub fn find_consuming_node<W>(
    root: &Tree<NonNull<LeafNode<W>>>,
    v: Index,
) -> Option<Handle<W, marker::Either>>
where
    W: Reduce,
{
    let edge = root.incident_edges(v).next()?;
    let node = Handle::new_leaf(edge.weight.cast());
    unsafe { ptr::read(&node) }.semi_splay();

    if root.has_at_most_one_incident_edge(v) {
        return Some(node.forget_type());
    }

    let endpoints = root.endpoints(node.edge());

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

pub fn expose<W>(
    v: Index,
    root: &mut Tree<NonNull<LeafNode<W>>>,
) -> Option<Handle<W, marker::Either>>
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

pub fn deexpose<W>(
    v: Index,
    tree: &mut Tree<NonNull<LeafNode<W>>>,
) -> Option<Handle<W, marker::Either>>
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

pub fn link<W>(
    u: Index,
    v: Index,
    weight: W,
    tree: &mut Tree<NonNull<LeafNode<W>>>,
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
        node = InternalNode::alloc(weight, left, right.node, 1).cast();
    }

    node
}

pub fn cut<W>(
    u: Index,
    v: Index,
    tree: &mut Tree<NonNull<LeafNode<W>>>,
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

    let edge = Handle::new_leaf(edge.weight.cast::<Node<W>>());

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

enum LeafOrInternal<T, U> {
    Leaf(T),
    Internal(U),
}

mod tree {
    use std::{collections::BTreeMap, marker::PhantomData, ptr};

    use crate::{index::Index, util::WithDropExt};

    use super::index::IndexAllocator;

    pub struct Tree<W> {
        index_allocator: IndexAllocator,
        pub vertices: BTreeMap<Index, Vertex>,
        pub edges: BTreeMap<EdgeKey<Private>, Edge<W>>,
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

    pub struct Vertex {
        edges: Vec<EdgeKey<Private>>,
        pub(crate) exposed: bool,
    }

    pub struct Edge<W> {
        pub weight: W,
        pub endpoints: [Index; 2],
    }

    #[derive(Debug)]
    pub struct EdgeEndpoints {
        pub left: Index,
        pub right: Index,
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
                || self.degree(index).is_some_and(|d| d >= 2)
        }

        pub fn degree(&self, index: Index) -> Option<usize> {
            self.vertices.get(&index).map(|v| v.edges.len())
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
                    exposed: false,
                },
            );
            index
        }

        pub fn remove_vertex(&mut self, index: Index) -> impl Iterator<Item = Edge<W>> {
            let edges = &mut self.edges;
            let edges = self
                .vertices
                .remove(&index)
                .into_iter()
                .flat_map(|vertex| vertex.edges.into_iter())
                .filter_map(move |key| edges.remove(&key).map(|e| (key, e)));

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

            edges.with_drop(move |edges| {
                // drain iter to ensure that all edges are removed from the tree before deallocating the index
                for _ in edges {}
                index_allocator.deallocate(index);
            })
        }

        pub fn add_edge(&mut self, left: Index, right: Index, weight: W) -> EdgeKey<Private> {
            let key = EdgeKey::new_private(left, right);
            let edge = Edge {
                weight,
                endpoints: [left, right],
            };

            self.edges.insert(unsafe { ptr::read(&key) }, edge);

            self.vertices
                .entry(left)
                .or_insert_with(|| Vertex {
                    edges: Vec::new(),
                    exposed: false,
                })
                .edges
                .push(unsafe { ptr::read(&key) });

            self.vertices
                .entry(right)
                .or_insert_with(|| Vertex {
                    edges: Vec::new(),
                    exposed: false,
                })
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

mod util {
    use std::{mem::ManuallyDrop, num::NonZero, ptr::NonNull};

    pub struct WithDrop<F: FnOnce(&mut T), T> {
        t: T,
        f: ManuallyDrop<F>,
    }

    impl<F: FnOnce(&mut T), T> Iterator for WithDrop<F, T>
    where
        T: Iterator,
    {
        type Item = T::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.t.next()
        }
    }

    impl<F: FnOnce(&mut T), T> std::ops::Deref for WithDrop<F, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.t
        }
    }

    impl<F: FnOnce(&mut T), T> std::ops::DerefMut for WithDrop<F, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.t
        }
    }

    impl<F: FnOnce(&mut T), T> WithDrop<F, T> {
        fn new(t: T, f: F) -> Self {
            Self {
                t,
                f: ManuallyDrop::new(f),
            }
        }
    }

    pub trait WithDropExt<T> {
        fn with_drop<F: FnOnce(&mut T)>(self, f: F) -> WithDrop<F, T>;
    }

    impl<T> WithDropExt<T> for T {
        fn with_drop<F: FnOnce(&mut T)>(self, f: F) -> WithDrop<F, T> {
            WithDrop::new(self, f)
        }
    }

    impl<F: FnOnce(&mut T), T> Drop for WithDrop<F, T> {
        fn drop(&mut self) {
            let f = unsafe { ManuallyDrop::take(&mut self.f) };
            f(&mut self.t);
        }
    }

    pub unsafe trait Tag: Copy {
        const BITS: u32;
        fn into_usize(self) -> usize;
        unsafe fn from_usize(tag: usize) -> Self;
    }

    const fn align_of<T: ?Sized + Aligned>() -> usize {
        T::ALIGN
    }

    pub unsafe trait Aligned {
        /// Alignment of `Self`.
        const ALIGN: usize;
    }

    unsafe impl<T> Aligned for T {
        const ALIGN: usize = core::mem::align_of::<Self>();
    }

    unsafe impl<T> Aligned for [T] {
        const ALIGN: usize = core::mem::align_of::<T>();
    }

    const fn bits_for<T: ?Sized + Aligned>() -> u32 {
        let align = align_of::<T>();
        align.trailing_zeros()
    }

    const fn bits_for_tags(mut tags: &[usize]) -> u32 {
        let mut bits = 0;
        while let &[tag, ref rest @ ..] = tags {
            tags = rest;
            let b = usize::BITS - tag.leading_zeros();
            if b > bits {
                bits = b;
            }
        }

        bits
    }

    #[derive(Debug, Clone, Copy)]
    pub struct TaggedPtr<P: Aligned + ?Sized, T> {
        packed: *mut P,
        _marker: core::marker::PhantomData<T>,
    }

    impl<P, T> TaggedPtr<P, T>
    where
        T: Tag,
        P: Aligned + ?Sized,
    {
        pub fn new(ptr: *mut P, tag: T) -> Self {
            Self {
                packed: Self::pack(ptr, tag),
                _marker: core::marker::PhantomData,
            }
        }

        const ASSERTION: () =
            { assert!(T::BITS <= bits_for::<P>(), "Not enough bits to store tag") };
        const TAG_BIT_SHIFT: u32 = usize::BITS - T::BITS;

        pub fn pack(ptr: *mut P, tag: T) -> *mut P {
            let () = Self::ASSERTION;

            let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

            ptr.map_addr(|addr| (addr >> T::BITS) | packed_tag)
        }

        pub fn tag(&self) -> T {
            let packed_addr = self.packed.addr();
            let tag_bits = packed_addr >> Self::TAG_BIT_SHIFT;
            unsafe { T::from_usize(tag_bits) }
        }

        pub fn set_tag(&mut self, tag: T) {
            let ptr = self.as_ptr();
            self.packed = Self::pack(ptr, tag);
        }

        pub fn update_tag<F: FnOnce(&mut T)>(&mut self, f: F) {
            let mut tag = self.tag();
            f(&mut tag);
            self.set_tag(tag);
        }

        pub fn as_ptr(&self) -> *mut P {
            self.packed.map_addr(|addr| addr << T::BITS)
        }

        pub fn set_ptr(&mut self, ptr: *mut P) {
            let tag = self.tag();
            self.packed = Self::pack(ptr, tag);
        }

        pub fn as_non_null(&self) -> Option<NonNull<P>> {
            NonNull::new(self.as_ptr())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_tagged_ptr() {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            enum MyTag {
                A,
                B,
                C,
            }

            unsafe impl Tag for MyTag {
                const BITS: u32 = 2;

                fn into_usize(self) -> usize {
                    match self {
                        MyTag::A => 0,
                        MyTag::B => 1,
                        MyTag::C => 2,
                    }
                }

                unsafe fn from_usize(tag: usize) -> Self {
                    match tag {
                        0 => MyTag::A,
                        1 => MyTag::B,
                        2 => MyTag::C,
                        _ => panic!("Invalid tag value"),
                    }
                }
            }

            let mut x = 42;
            let mut tagged_ptr = TaggedPtr::<i32, MyTag>::new(&mut x as *mut i32, MyTag::A);

            assert_eq!(tagged_ptr.tag(), MyTag::A);
            assert_eq!(unsafe { *tagged_ptr.as_ptr() }, 42);

            tagged_ptr.set_tag(MyTag::B);
            assert_eq!(tagged_ptr.tag(), MyTag::B);

            tagged_ptr.update_tag(|tag| {
                if let MyTag::B = tag {
                    *tag = MyTag::C;
                }
            });
            assert_eq!(tagged_ptr.tag(), MyTag::C);
        }
    }
}

trait Reduce {
    fn reduce(&self, other: &Self) -> Self;
}

#[cfg(test)]
mod tests;
