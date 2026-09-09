use std::{
    ops::Not as _,
    ptr::{self, NonNull},
};

use crate::{index::Index, tree::EdgeKey};

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct NodeFlags: u8 {
        const LEAF = 1 << 0;
        const FLIPPED = 1 << 1;
        const BOUNDARY_LOW = 1 << 2;
        const BOUNDARY_HIGH = 1 << 3;
    }
}

#[repr(C, align(16))]
struct Node<W> {
    parent: *mut Node<W>,
    weight: W,
}

impl<W> Node<W> {
    unsafe fn dealloc(this: NonNull<Self>) {
        use LeafOrInternal::*;
        unsafe {
            match this.as_ref().force() {
                Leaf(_) => _ = Box::<LeafNode<W>>::from_non_null(this.cast()),
                Internal(_) => _ = Box::<InternalNode<W>>::from_non_null(this.cast()),
            }
        }
    }
}

#[repr(C)]
struct LeafNode<W> {
    node: Node<W>,
    edge: EdgeKey,
}

impl<W> LeafNode<W> {
    fn alloc(weight: W, edge: EdgeKey, num_boundary: usize) -> NonNull<Self> {
        let mut node = Box::new(LeafNode {
            node: Node {
                parent: std::ptr::null_mut(),
                weight,
            },
            edge,
        });
        node.set_flags(NodeFlags::LEAF);
        node.set_num_boundary(num_boundary);

        Box::into_non_null(node)
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
                parent: std::ptr::null_mut(),
                weight,
            },
            children: [left, right],
        });
        node.set_flags(NodeFlags::empty());
        node.set_num_boundary(num_boundary);

        Box::into_non_null(node)
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

impl<W> Node<W> {
    fn flags(&self) -> NodeFlags {
        NodeFlags::from_bits_truncate((self.parent.addr() >> 60) as u8)
    }
    fn set_flags(&mut self, flags: NodeFlags) {
        let bits = (flags.bits() as usize) << 60;
        self.parent = self.parent.map_addr(|addr| bits | ((addr >> 4) << 4));
    }

    fn is_leaf(&self) -> bool {
        self.flags().contains(NodeFlags::LEAF)
    }

    fn is_flipped(&self) -> bool {
        self.flags().contains(NodeFlags::FLIPPED)
    }

    fn num_boundary(&self) -> usize {
        let flags = self.flags();
        (flags.contains(NodeFlags::BOUNDARY_LOW) as usize)
            + (2 * flags.contains(NodeFlags::BOUNDARY_HIGH) as usize)
    }

    fn set_num_boundary(&mut self, num: usize) {
        let mut flags = self.flags();
        flags.set(NodeFlags::BOUNDARY_LOW, num & 1 != 0);
        flags.set(NodeFlags::BOUNDARY_HIGH, num & 2 != 0);
        self.set_flags(flags);
    }

    fn update_num_boundary(&mut self, f: impl FnOnce(usize) -> usize) {
        let num = self.num_boundary();
        let new_num = f(num);
        assert!(new_num <= 2, "num_boundary must be <= 2");
        self.set_num_boundary(new_num);
    }

    fn is_path(&self) -> bool {
        self.num_boundary() == 2
    }

    fn is_point(&self) -> bool {
        self.num_boundary() < 2
    }

    fn update_flipped(&mut self, f: impl FnOnce(bool) -> bool) {
        self.set_flags(self.flags().union(if f(self.is_flipped()) {
            NodeFlags::FLIPPED
        } else {
            NodeFlags::empty()
        }));
    }

    fn push_flip(&mut self) {
        if !self.is_leaf() {
            let internal_node = unsafe { &mut *(self as *mut Node<W> as *mut InternalNode<W>) };
            internal_node.push_flip();
        }
    }

    unsafe fn as_leaf_unchecked(&self) -> &LeafNode<W> {
        assert!(self.is_leaf());
        unsafe { &*(self as *const Node<W> as *const LeafNode<W>) }
    }

    unsafe fn as_internal_unchecked(&self) -> &InternalNode<W> {
        assert!(!self.is_leaf());
        unsafe { &*(self as *const Node<W> as *const InternalNode<W>) }
    }

    unsafe fn as_leaf_mut_unchecked(&mut self) -> &mut LeafNode<W> {
        assert!(self.is_leaf());
        unsafe { &mut *(self as *mut Node<W> as *mut LeafNode<W>) }
    }

    unsafe fn as_internal_mut_unchecked(&mut self) -> &mut InternalNode<W> {
        assert!(!self.is_leaf());
        unsafe { &mut *(self as *mut Node<W> as *mut InternalNode<W>) }
    }

    fn force(&self) -> LeafOrInternal<&LeafNode<W>, &InternalNode<W>> {
        if self.is_leaf() {
            LeafOrInternal::Leaf(unsafe { self.as_leaf_unchecked() })
        } else {
            LeafOrInternal::Internal(unsafe { self.as_internal_unchecked() })
        }
    }

    fn force_mut(&mut self) -> LeafOrInternal<&mut LeafNode<W>, &mut InternalNode<W>> {
        if self.is_leaf() {
            LeafOrInternal::Leaf(unsafe { self.as_leaf_mut_unchecked() })
        } else {
            LeafOrInternal::Internal(unsafe { self.as_internal_mut_unchecked() })
        }
    }

    fn parent(&self) -> Option<NonNull<InternalNode<W>>> {
        NonNull::new(self.parent.map_addr(|addr| addr >> 4).cast())
    }

    unsafe fn parent_ref(&self) -> Option<&InternalNode<W>> {
        self.parent().map(|p| unsafe { p.as_ref() })
    }

    unsafe fn parent_mut<'a>(&mut self) -> Option<&'a mut InternalNode<W>> {
        self.parent().map(|mut p| unsafe { p.as_mut() })
    }

    fn grandparent(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent()
            .and_then(|p| unsafe { p.as_ref().node.parent() })
    }

    unsafe fn grandparent_ref(&self) -> Option<&InternalNode<W>> {
        self.grandparent().map(|gp| unsafe { gp.as_ref() })
    }

    unsafe fn grandparent_mut(&mut self) -> Option<&mut InternalNode<W>> {
        self.grandparent().map(|mut gp| unsafe { gp.as_mut() })
    }

    fn ggp(&self) -> Option<NonNull<InternalNode<W>>> {
        self.parent()
            .and_then(|p| unsafe { p.as_ref().node.parent() })
            .and_then(|gp| unsafe { gp.as_ref().node.parent() })
    }

    unsafe fn ggp_ref(&self) -> Option<&InternalNode<W>> {
        self.ggp().map(|gp| unsafe { gp.as_ref() })
    }

    unsafe fn ggp_mut(&mut self) -> Option<&mut InternalNode<W>> {
        self.ggp().map(|mut gp| unsafe { gp.as_mut() })
    }

    fn set_parent(&mut self, parent: Option<NonNull<InternalNode<W>>>) {
        let flags = (self.flags().bits() as usize) << 60;
        self.parent = parent
            .map_or(ptr::null_mut(), NonNull::as_ptr)
            .cast::<Node<W>>()
            .map_addr(|addr| flags | (addr >> 4));
    }

    fn sibling(&self) -> Option<NonNull<Node<W>>> {
        self.parent().map(|parent| {
            let internal_node = unsafe { parent.as_ref() };
            let index = if internal_node.children[0] == NonNull::from(self) {
                1
            } else {
                0
            };
            internal_node.children[index]
        })
    }

    fn is_left_child(&self) -> Option<bool> {
        self.parent().map(|parent| {
            let internal_node = unsafe { parent.as_ref() };
            internal_node.children[0] == NonNull::from(self)
        })
    }
}

impl<W> InternalNode<W> {
    fn push_flip(&mut self) {
        if self.node.is_flipped() {
            self.node.update_flipped(|_| false);
            self.children.swap(0, 1);
            for child in self.children.iter_mut() {
                unsafe { child.as_mut().update_flipped(|f| !f) };
            }
        }
    }
}

struct TopTreeCtx<'a, W> {
    tree: &'a mut tree::Tree<NonNull<LeafNode<W>>>,
}

impl<'a, W> TopTreeCtx<'a, W> {
    fn node_has_left_boundary(&self, node: &Node<W>) -> bool {
        match node.force() {
            LeafOrInternal::Leaf(leaf) => {
                let edge = leaf.edge;
                let v =
                    self.tree.endpoints(&edge).expect("edge must exist in tree")[node.is_flipped()];

                self.tree
                    .vertices
                    .get(&v)
                    .expect("vertex must exist in tree")
                    .exposed
                    || self.tree.degree(v).expect("vertex must exist in tree") <= 1
            }
            LeafOrInternal::Internal(internal) => {
                let child = unsafe { internal.children[node.is_flipped() as usize].as_ref() };

                child.is_path()
            }
        }
    }

    fn node_has_right_boundary(&self, node: &Node<W>) -> bool {
        match node.force() {
            LeafOrInternal::Leaf(leaf) => {
                let edge = leaf.edge;
                let v = self.tree.endpoints(&edge).expect("edge must exist in tree")
                    [!node.is_flipped()];

                self.tree
                    .vertices
                    .get(&v)
                    .expect("vertex must exist in tree")
                    .exposed
                    || self.tree.degree(v).expect("vertex must exist in tree") <= 1
            }
            LeafOrInternal::Internal(internal) => {
                let child = unsafe { internal.children[!node.is_flipped() as usize].as_ref() };

                child.is_path()
            }
        }
    }

    fn node_has_middle_boundary(&self, node: &Node<W>) -> bool {
        if node.is_leaf() || node.num_boundary() == 0 {
            return false;
        }

        let internal = unsafe { node.as_internal_unchecked() };

        0 != node.num_boundary()
            - internal
                .children
                .iter()
                .map(|child| unsafe { child.as_ref() })
                .map(|child| child.is_path() as usize)
                .sum::<usize>()
    }

    /// # Safety
    /// creats mutable references to the parent, grandparent, sibling and uncle
    /// of the given node, which must not alias with any other references to
    /// those nodes.
    unsafe fn rotate_up(&self, node: &mut Node<W>) -> Option<()> {
        let parent = unsafe { node.parent()?.as_mut() };
        let gp = unsafe { parent.node.parent()?.as_mut() };
        let sibling = unsafe { node.sibling()?.as_mut() };
        let uncle = unsafe { parent.node.sibling()?.as_mut() };

        gp.push_flip();
        parent.push_flip();

        let uncle_is_left = uncle.is_left_child().expect("uncle must have a parent");
        let sibling_is_left = sibling.is_left_child().expect("sibling must have a parent");
        let same_sides = uncle_is_left == sibling_is_left;
        let sibling_is_path = sibling.is_path();
        let uncle_is_path = uncle.is_path();
        let gp_is_path = gp.node.is_path();

        let new_parent_is_path: bool;
        let flip_new_parent: bool;
        let flip_gp: bool;
        if same_sides && sibling_is_path {
            // path
            let gp_has_middle = self.node_has_middle_boundary(&gp.node);
            new_parent_is_path = gp_has_middle || uncle_is_path;
            flip_new_parent = false;
            if gp_has_middle
                && !gp_is_path
                && let Some(gp_is_left) = gp.node.is_left_child()
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
                node.update_flipped(|f| f ^ true);
            } else {
                new_parent_is_path = uncle_is_path;
                flip_new_parent = false;
                flip_gp = false;
                sibling.update_flipped(|f| f ^ true);
            }
        }

        parent.children[uncle_is_left as usize] = NonNull::from(sibling);
        parent.children[uncle_is_left.not() as usize] = NonNull::from(&mut *uncle);
        parent.node.update_flipped(|_| flip_new_parent);
        parent
            .node
            .set_num_boundary(if new_parent_is_path { 2 } else { 1 });

        gp.children[uncle_is_left as usize] = NonNull::from(&mut *node);
        gp.children[uncle_is_left.not() as usize] = NonNull::from(&parent.node);
        gp.node.update_flipped(|_| flip_gp);

        // recompute W for parent and gp

        node.set_parent(Some(NonNull::from(gp)));
        uncle.set_parent(Some(NonNull::from(parent)));

        todo!()
    }

    fn splay_step<'n>(&self, mut node: &'n mut Node<W>) -> Option<&'n mut Node<W>> {
        loop {
            let mut p = node.parent()?;
            let mut gp = node.grandparent()?;

            if node.is_point() && unsafe { gp.as_mut().is_point() } {
                unsafe {
                    self.rotate_up(node).expect("rotate_up should succeed");
                    return Some(&mut gp.as_mut().node);
                };
            }

            let mut ggp = node.ggp()?;

            if unsafe {
                p.as_ref().node.is_point()
                    && (gp.as_ref().node.is_point() || ggp.as_ref().node.is_point())
            } {
                unsafe {
                    gp.as_mut().push_flip();
                    p.as_mut().push_flip();

                    let node_is_left = node.is_left_child().expect("node must have a parent");
                    let p_is_left = p
                        .as_ref()
                        .node
                        .is_left_child()
                        .expect("p must have a parent");
                    let gp_is_left = gp
                        .as_ref()
                        .node
                        .is_left_child()
                        .expect("gp must have a parent");

                    if node_is_left == p_is_left {
                        self.rotate_up(node).expect("rotate_up should succeed");
                        return Some(&mut gp.as_mut().node);
                    }

                    if p_is_left == gp_is_left {
                        self.rotate_up(&mut p.as_mut().node)
                            .expect("rotate_up should succeed");
                        return Some(&mut ggp.as_mut().node);
                    }

                    assert_eq!(node_is_left, gp_is_left);
                    self.rotate_up(node.sibling().expect("node must have a sibling").as_mut())
                        .expect("rotate_up should succeed");
                    self.rotate_up(&mut p.as_mut().node)
                        .expect("rotate_up should succeed");

                    return Some(&mut ggp.as_mut().node);
                }
            }

            node = unsafe { &mut p.as_mut().node };
        }
    }

    fn semi_splay(&self, node: &mut Node<W>) {
        let mut node = Some(node);
        while let Some(next_node) = node {
            node = self.splay_step(next_node);
        }
    }

    fn splay(&self, node: &mut Node<W>) {
        while let Some(next_node) = self.splay_step(node) {
            self.splay_step(next_node);
        }
    }

    fn find_consuming_node(&self, vert: Index) -> Option<NonNull<Node<W>>> {
        let edge = self.tree.incident_edges(vert).next()?;
        let mut node = edge.weight;

        self.semi_splay(unsafe { &mut node.as_mut().node });
        if self.tree.has_at_most_one_incident_edge(vert) {
            return Some(node.cast());
        }

        let node_ref = unsafe { node.as_ref() };
        let mut node = node.cast::<Node<W>>();

        // Determine if the vertex is the left or right boundary of the edge-node
        let endpoints = self.tree.endpoints(&node_ref.edge)?;
        let flip = node_ref.is_flipped();
        let mut is_left = (endpoints.left == vert) != flip;
        let mut is_right = (endpoints.right == vert) != flip;
        let mut is_middle = false;

        let mut last_middle_node = None;
        while let Some(parent) = unsafe { node.as_ref().parent() } {
            let node_ref = unsafe { node.as_ref() };
            let parent_ref = unsafe { parent.as_ref() };
            let is_left_child = node_ref.is_left_child().unwrap();

            is_middle = if is_left_child {
                is_right || (is_middle && !self.node_has_right_boundary(node_ref))
            } else {
                is_left || (is_middle && !self.node_has_left_boundary(node_ref))
            };
            is_left = (is_left_child != parent_ref.is_flipped()) && !is_middle;
            is_right = (is_left_child == parent_ref.is_flipped()) && !is_middle;

            node = parent.cast();

            if is_middle {
                if !self.node_has_middle_boundary(&parent_ref.node) {
                    return Some(node);
                }
                last_middle_node = Some(node);
            }
        }

        last_middle_node
    }
}

impl<'a, W> TopTreeCtx<'a, W> {
    pub fn expose(&mut self, vert: Index) -> Option<&'a mut InternalNode<W>> {
        let Some(mut node) = self.find_consuming_node(vert) else {
            if let Some(vert) = self.tree.vertices.get_mut(&vert) {
                vert.exposed = true;
            }
            return None;
        };

        let mut node_ref = unsafe { node.as_mut().as_internal_mut_unchecked() };
        while node_ref.is_path() {
            let mut parent = node_ref.parent().unwrap();
            node_ref.push_flip();
            let is_right = !node_ref.is_left_child().unwrap();
            let child = unsafe { node_ref.children[is_right as usize].as_mut() };
            unsafe { self.rotate_up(child).expect("rotate_up should succeed") };
            node_ref = unsafe { parent.as_mut() };
        }

        self.splay(node_ref);

        let mut node = Some(node_ref);
        let mut root = None;
        while let Some(node_ref) = node {
            node_ref.update_num_boundary(|n| n + 1);
            let parent = unsafe { node_ref.parent_mut() };
            root = Some(node_ref);
            node = parent;
        }

        if let Some(vert) = self.tree.vertices.get_mut(&vert) {
            vert.exposed = true;
        }

        root
    }

    pub fn deexpose(&mut self, vert: Index) -> &'a mut InternalNode<W> {
        let mut root = None;
        let node = self.find_consuming_node(vert);

        while let Some(mut node) = node {
            let node_ref = unsafe { node.as_mut() };
            node_ref.update_num_boundary(|n| n - 1);
            root = Some(node_ref);
        }

        if let Some(vert) = self.tree.vertices.get_mut(&vert) {
            vert.exposed = false;
        }

        unsafe { root.unwrap().as_internal_mut_unchecked() }
    }

    pub fn link(&mut self, u: Index, v: Index, weight: W) -> &'a mut Node<W>
    where
        W: Reduce,
    {
        let mut root_u = self.expose(u);
        if let Some(ref mut root) = root_u
            && self.node_has_left_boundary(root)
        {
            root.update_flipped(|f| !f);
        }
        self.tree
            .vertices
            .get_mut(&u)
            .expect("vertex must exist in tree")
            .exposed = true;

        let mut root_v = self.expose(v);
        if let Some(ref mut root) = root_v
            && self.node_has_right_boundary(root)
        {
            root.update_flipped(|f| !f);
        }
        self.tree
            .vertices
            .get_mut(&v)
            .expect("vertex must exist in tree")
            .exposed = true;

        let edge_key = EdgeKey::from((u, v));
        let node = LeafNode::alloc(
            weight,
            edge_key,
            root_u.is_some() as usize + root_v.is_some() as usize,
        );
        self.tree.add_edge(u, v, node);

        let mut node = node.cast::<Node<W>>();
        if let Some(root) = root_u {
            node = InternalNode::alloc(
                W::reduce(unsafe { &node.as_ref().weight }, &root.weight),
                node,
                NonNull::from(root).cast(),
                root_v.is_some() as usize,
            )
            .cast::<Node<W>>();
        }

        if let Some(root) = root_v {
            node = InternalNode::alloc(
                W::reduce(&root.weight, unsafe { &node.as_ref().weight }),
                NonNull::from(root).cast(),
                node,
                1,
            )
            .cast::<Node<W>>();
        }

        unsafe { node.as_mut() }
    }

    fn delete_all_ancestors(&mut self, node: NonNull<Node<W>>) {
        let node_ref = unsafe { node.as_ref() };
        while let Some(parent) = node_ref.parent() {
            let mut sibling = node_ref.sibling().expect("node must have a sibling");
            self.delete_all_ancestors(parent.cast::<Node<W>>());
            unsafe { sibling.as_mut().set_parent(None) };
        }

        unsafe {
            Node::dealloc(node);
        }
    }

    pub fn cut(
        &mut self,
        u: Index,
        v: Index,
    ) -> (
        Option<&'a mut InternalNode<W>>,
        Option<&'a mut InternalNode<W>>,
    ) {
        let edge = EdgeKey::from((u, v));
        let edge = self
            .tree
            .edges
            .get(&edge)
            .expect("edge must exist in tree")
            .weight;

        self.splay(unsafe { edge.cast::<Node<W>>().as_mut() });
        self.delete_all_ancestors(edge.cast::<Node<W>>());
        self.tree.remove_edge(u, v);

        self.tree
            .vertices
            .get_mut(&u)
            .expect("vertex must exist in tree")
            .exposed = true;
        self.tree
            .vertices
            .get_mut(&v)
            .expect("vertex must exist in tree")
            .exposed = true;

        let ru = self.expose(u);
        let rv = self.expose(v);

        (ru, rv)
    }
}

enum LeafOrInternal<T, U> {
    Leaf(T),
    Internal(U),
}

mod tree {
    use std::collections::BTreeMap;

    use crate::{index::Index, util::WithDropExt};

    use super::index::IndexAllocator;

    pub struct Tree<W> {
        index_allocator: IndexAllocator,
        pub vertices: BTreeMap<Index, Vertex>,
        pub edges: BTreeMap<EdgeKey, Edge<W>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct EdgeKey(pub [Index; 2]);

    impl From<(Index, Index)> for EdgeKey {
        fn from((a, b): (Index, Index)) -> Self {
            let mut inner = [a, b];
            inner.sort();
            Self(inner)
        }
    }

    pub struct Vertex {
        edges: Vec<EdgeKey>,
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

        pub fn endpoints(&self, edge: &EdgeKey) -> Option<EdgeEndpoints> {
            self.edges.get(edge).map(|e| EdgeEndpoints {
                left: e.endpoints[0],
                right: e.endpoints[1],
            })
        }

        pub fn degree(&self, index: Index) -> Option<usize> {
            self.vertices.get(&index).map(|v| v.edges.len())
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

        pub fn add_edge(&mut self, left: Index, right: Index, weight: W) {
            let key = EdgeKey::from((left, right));
            let edge = Edge {
                weight,
                endpoints: [left, right],
            };

            self.edges.insert(key, edge);

            self.vertices
                .entry(left)
                .or_insert_with(|| Vertex {
                    edges: Vec::new(),
                    exposed: false,
                })
                .edges
                .push(key);

            self.vertices
                .entry(right)
                .or_insert_with(|| Vertex {
                    edges: Vec::new(),
                    exposed: false,
                })
                .edges
                .push(key);
        }

        pub fn remove_edge(&mut self, left: Index, right: Index) {
            let key = EdgeKey::from((left, right));
            if let Some(edge) = self.edges.remove(&key) {
                for endpoint in edge.endpoints.iter() {
                    if let Some(vertex) = self.vertices.get_mut(endpoint) {
                        vertex.edges.retain(|e| e != &key);
                    }
                }
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

            ptr.map_addr(|addr| addr | packed_tag)
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

        pub fn as_ptr(&self) -> *mut P {
            self.packed.map_addr(|addr| addr << T::BITS)
        }

        pub fn as_non_null(&self) -> Option<NonNull<P>> {
            NonNull::new(self.as_ptr())
        }
    }
}

trait Reduce {
    fn reduce(&self, other: &Self) -> Self;
}

#[cfg(test)]
mod tests;
