//! An augmented B-tree (a `BTreeMap` that keeps a cached aggregate/measure for
//! every subtree).
//!
//! The layout mirrors the standard library's `BTreeMap` representation:
//! every node begins with a `LeafNode` header (so a pointer to an
//! `InternalNode` can be interpreted as a pointer to its leaf portion), and
//! leaf nodes carry parent pointers so that a node can be turned into a
//! traversal entry point.
//!
//! In addition to the usual key/value storage each node caches an
//! [`Aggregate`] over the values of its entire subtree. Structural operations
//! (split, merge, steal, insert, remove) recompute the affected aggregates so
//! that `aggregate` queries are O(1) for a whole subtree.

// The node layer intentionally exposes a fairly complete set of low-level
// primitives that mirror the standard library's BTree implementation; some are
// only used by a subset of the tree operations. Pointer-heavy code also tends
// to produce complex signatures and index-based loops that read more clearly
// than their iterator equivalents here.
#![allow(clippy::type_complexity, clippy::needless_range_loop)]

use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    borrow::Borrow,
    cmp::Ordering,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
    slice::SliceIndex,
};

use detail::{move_to_slice, slice_insert, slice_remove, slice_shl, slice_shr};

mod detail;
mod entry;
#[cfg(test)]
mod tests;

pub use entry::{Entry, OccupiedEntry, OccupiedValue, VacantEntry};

const B: usize = 4;
const CAPACITY: usize = 2 * B - 1;
#[allow(dead_code)]
const MAX_KEYS: usize = CAPACITY;
const MAX_EDGES: usize = 2 * B;
const MIN_LEN_AFTER_SPLIT: usize = B - 1;
const MIN_LEN: usize = MIN_LEN_AFTER_SPLIT;

const KV_IDX_CENTER: usize = B - 1;
const EDGE_IDX_LEFT_OF_CENTER: usize = B - 1;
const EDGE_IDX_RIGHT_OF_CENTER: usize = B;

/// A monoid over the values stored in the tree.
///
/// An aggregate is cached for every node: `node.aggregate` is the reduction of
/// all values in the subtree rooted at `node`, in key order. `identity` is the
/// neutral element and must satisfy `identity().reduce(x) == *x` and
/// `x.reduce(&identity()) == *x`; `reduce` must be associative.
pub trait Aggregate {
    /// Combines two aggregates into one.
    fn reduce(&self, other: &Self) -> Self;

    /// The neutral element of the monoid.
    fn identity() -> Self;

    /// Reduces a slice of values into a single aggregate.
    /// Returns `Self::identity()` if the slice is empty.
    fn reduce_slice(values: &[Self]) -> Self
    where
        Self: Sized,
    {
        values.iter().fold(Self::identity(), |acc, v| acc.reduce(v))
    }
}

struct LeafNode<K, V> {
    /// The parent node, if any.
    parent: Option<NonNull<InternalNode<K, V>>>,
    /// This node's index into the parent's `edges` array. Only valid when
    /// `parent` is `Some`.
    parent_idx: MaybeUninit<u16>,
    /// The number of keys, values and (for internal nodes) children.
    len: u16,
    /// The cached aggregate over this node's entire subtree.
    aggregate: V,
    keys: [MaybeUninit<K>; CAPACITY],
    values: [MaybeUninit<V>; CAPACITY],
}

/// An internal node. `repr(C)` guarantees that `leaf` is located at offset 0,
/// so that `NonNull<InternalNode>` and `NonNull<LeafNode>` are interchangeable
/// for the purposes of reading the shared header.
#[repr(C)]
struct InternalNode<K, V> {
    leaf: LeafNode<K, V>,
    edges: [MaybeUninit<NonNull<InternalNode<K, V>>>; MAX_EDGES],
}

#[allow(dead_code)]
type BoxedNode<K, V> = NonNull<LeafNode<K, V>>;
type Root<K, V> = NodeRef<marker::Owned, K, V, marker::Either>;

mod marker {
    pub struct Leaf;
    pub struct Internal;
    pub struct Either;

    pub struct KV;
    pub struct Edge;

    pub struct Immut<'a>(#[allow(dead_code)] pub &'a ());
    pub struct Mut<'a>(#[allow(dead_code)] pub &'a mut ());
    pub struct Owned;
    pub struct DormantMut;

    pub trait BorrowType {
        /// Whether node references of this borrow type may traverse to other
        /// nodes in the tree.
        const TRAVERSAL_PERMIT: bool = true;
    }

    impl BorrowType for Owned {
        const TRAVERSAL_PERMIT: bool = false;
    }
    impl<'a> BorrowType for Immut<'a> {}
    impl<'a> BorrowType for Mut<'a> {}
    impl BorrowType for DormantMut {}
}

struct NodeRef<BorrowType, K, V, NodeType> {
    node: NonNull<LeafNode<K, V>>,
    height: usize,
    _marker: PhantomData<(BorrowType, NodeType)>,
}

struct Handle<Node, HandleType> {
    node: Node,
    index: usize,
    _marker: PhantomData<HandleType>,
}

enum Force<Leaf, Internal> {
    Leaf(Leaf),
    Internal(Internal),
}

enum LeftOrRight<T> {
    Left(T),
    Right(T),
}

/// Result of an insertion/split: the (modified) left node, the median pair and
/// a newly allocated, still unattached right node.
struct SplitResult<'a, K, V, NodeType> {
    left: NodeRef<marker::Mut<'a>, K, V, NodeType>,
    kv: (K, V),
    right: NodeRef<marker::Owned, K, V, NodeType>,
}

enum SearchResult<BorrowType, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<BorrowType, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<BorrowType, K, V, GoDownType>, marker::Edge>),
}

enum IndexResult {
    KV(usize),
    Edge(usize),
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

unsafe fn alloc_node<T>() -> NonNull<T> {
    let layout = Layout::new::<T>();
    // SAFETY: `layout` has non-zero size for all node types.
    let ptr = unsafe { alloc(layout) };
    match NonNull::new(ptr.cast::<T>()) {
        Some(p) => p,
        None => handle_alloc_error(layout),
    }
}

unsafe fn dealloc_node<T>(ptr: NonNull<T>) {
    // SAFETY: `ptr` was allocated with `alloc_node::<T>` and has not been freed.
    unsafe { dealloc(ptr.as_ptr().cast(), Layout::new::<T>()) }
}

impl<K, V: Aggregate> LeafNode<K, V> {
    /// Initializes the shared header in place, leaving keys and values
    /// uninitialized.
    ///
    /// # Safety
    /// `this` must point to allocated (possibly uninitialized) storage.
    unsafe fn init(this: *mut Self) {
        unsafe {
            (&raw mut (*this).parent).write(None);
            (&raw mut (*this).len).write(0);
            (&raw mut (*this).aggregate).write(V::identity());
        }
    }

    fn new() -> NonNull<Self> {
        let node = unsafe { alloc_node::<Self>() };
        unsafe { LeafNode::init(node.as_ptr()) };
        node
    }
}

impl<K, V: Aggregate> InternalNode<K, V> {
    /// # Safety
    /// The new internal node has no initialized edge; the caller must set at
    /// least `edges[0]`.
    unsafe fn empty() -> NonNull<Self> {
        let node = unsafe { alloc_node::<Self>() };
        unsafe { LeafNode::init(&raw mut (*node.as_ptr()).leaf) };
        node
    }
}

/// Recomputes the aggregate of a node from its values and its children.
fn reduce_leaf<K, V: Aggregate>(leaf: &LeafNode<K, V>) -> V {
    let len = usize::from(leaf.len);
    // SAFETY: the first `len` values are initialized.
    V::reduce_slice(unsafe { leaf.values[..len].assume_init_ref() })
}

// ---------------------------------------------------------------------------
// NodeRef basics
// ---------------------------------------------------------------------------

impl<BorrowType, K, V, NodeType> NodeRef<BorrowType, K, V, NodeType> {
    fn as_leaf_ptr(this: &Self) -> *mut LeafNode<K, V> {
        this.node.as_ptr()
    }

    fn len(&self) -> usize {
        // SAFETY: the `len` field is always initialized.
        unsafe { usize::from((*Self::as_leaf_ptr(self)).len) }
    }

    fn reborrow(&self) -> NodeRef<marker::Immut<'_>, K, V, NodeType> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    fn force(
        self,
    ) -> Force<NodeRef<BorrowType, K, V, marker::Leaf>, NodeRef<BorrowType, K, V, marker::Internal>>
    {
        if self.height == 0 {
            Force::Leaf(NodeRef {
                node: self.node,
                height: self.height,
                _marker: PhantomData,
            })
        } else {
            Force::Internal(NodeRef {
                node: self.node,
                height: self.height,
                _marker: PhantomData,
            })
        }
    }

    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<BorrowType: marker::BorrowType, K, V, NodeType> NodeRef<BorrowType, K, V, NodeType> {
    /// Returns the parent edge handle, or gives back `self` if this is the
    /// root.
    fn ascend(
        self,
    ) -> Result<Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge>, Self> {
        const { assert!(BorrowType::TRAVERSAL_PERMIT) };

        let leaf_ptr = Self::as_leaf_ptr(&self);
        // SAFETY: parent pointers are maintained by the tree.
        unsafe { (*leaf_ptr).parent }
            .map(|parent| Handle {
                node: NodeRef::from_internal(parent, self.height + 1),
                // SAFETY: `parent_idx` is initialized when `parent` is set.
                index: unsafe { usize::from((*leaf_ptr).parent_idx.assume_init()) },
                _marker: PhantomData,
            })
            .ok_or(self)
    }

    fn first_edge(self) -> Handle<Self, marker::Edge> {
        // SAFETY: edge index 0 is always valid.
        unsafe { Handle::new_edge(self, 0) }
    }

    fn last_edge(self) -> Handle<Self, marker::Edge> {
        let len = self.len();
        // SAFETY: edge index `len` is always valid.
        unsafe { Handle::new_edge(self, len) }
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    fn from_internal(node: NonNull<InternalNode<K, V>>, height: usize) -> Self {
        debug_assert!(height > 0);
        NodeRef {
            node: node.cast(),
            height,
            _marker: PhantomData,
        }
    }

    fn as_internal_ptr(this: &Self) -> *mut InternalNode<K, V> {
        this.node.as_ptr().cast()
    }
}

// ---------------------------------------------------------------------------
// Immutable access
// ---------------------------------------------------------------------------

impl<'a, K: 'a, V: 'a, NodeType> NodeRef<marker::Immut<'a>, K, V, NodeType> {
    fn into_leaf(self) -> &'a LeafNode<K, V> {
        let ptr = Self::as_leaf_ptr(&self);
        // SAFETY: no mutable references into an `Immut` tree may exist.
        unsafe { &*ptr }
    }

    fn keys(&self) -> &'a [K] {
        let leaf = Self::as_leaf_ptr(self);
        // SAFETY: the first `len` keys are initialized.
        unsafe {
            (*leaf)
                .keys
                .get_unchecked(..usize::from((*leaf).len))
                .assume_init_ref()
        }
    }

    fn aggregate(&self) -> &'a V {
        // SAFETY: the aggregate is always initialized.
        unsafe { &(*Self::as_leaf_ptr(self)).aggregate }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Copy for NodeRef<marker::Immut<'a>, K, V, NodeType> {}
impl<'a, K: 'a, V: 'a, NodeType> Clone for NodeRef<marker::Immut<'a>, K, V, NodeType> {
    fn clone(&self) -> Self {
        *self
    }
}

// ---------------------------------------------------------------------------
// Owned nodes / root operations
// ---------------------------------------------------------------------------

impl<K, V: Aggregate> NodeRef<marker::Owned, K, V, marker::Leaf> {
    fn new_leaf() -> Self {
        Self::from_new_leaf(LeafNode::new())
    }

    fn from_new_leaf(leaf: NonNull<LeafNode<K, V>>) -> Self {
        let mut this = NodeRef {
            node: leaf,
            height: 0,
            _marker: PhantomData,
        };
        this.borrow_mut().recompute_aggregate();
        this
    }
}

impl<K, V: Aggregate> NodeRef<marker::Owned, K, V, marker::Internal> {
    fn new_internal(child: Root<K, V>) -> Self {
        let mut new_node = unsafe { InternalNode::empty() };
        // SAFETY: the edge array is uninitialized, writing element 0 is valid.
        unsafe { new_node.as_mut().edges[0].write(child.node.cast()) };
        Self::from_new_internal(new_node, child.height + 1)
    }

    fn from_new_internal(internal: NonNull<InternalNode<K, V>>, height: usize) -> Self {
        let mut this = NodeRef {
            node: internal.cast(),
            height,
            _marker: PhantomData,
        };
        // Every edge `0..=len` must already be initialized by the caller.
        {
            let mut node = this.borrow_mut();
            node.correct_all_childrens_parent_links();
            node.recompute_aggregate();
        }
        this
    }
}

impl<K, V: Aggregate, NodeType> NodeRef<marker::Owned, K, V, NodeType> {
    fn borrow_mut(&mut self) -> NodeRef<marker::Mut<'_>, K, V, NodeType> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }
}

impl<K, V: Aggregate> NodeRef<marker::Owned, K, V, marker::Either> {
    /// Creates a new, empty tree.
    fn new() -> Self {
        NodeRef::new_leaf().forget_type()
    }

    /// Adds a new internal root above the current one, increasing the height.
    fn push_internal_level(&mut self) -> NodeRef<marker::Mut<'_>, K, V, marker::Internal> {
        let old_root = NodeRef::<marker::Owned, K, V, marker::Either> {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        };
        let new_root = NodeRef::<marker::Owned, K, V, marker::Internal>::new_internal(old_root);
        self.node = new_root.node;
        self.height = new_root.height;
        // `new_root` is `Owned` and has no destructor; the pointer is now owned
        // by `self`.
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    /// Removes an empty internal root, using its first child as the new root.
    fn pop_internal_level(&mut self) {
        assert!(self.height > 0);
        let top = self.node;
        // SAFETY: `self.height > 0`, so the node is an `InternalNode`.
        let first = unsafe {
            let internal = &*top.as_ptr().cast::<InternalNode<K, V>>();
            internal.edges[0].assume_init_read()
        };
        self.node = first.cast();
        self.height -= 1;
        self.clear_parent_link();
        // SAFETY: the top node was allocated as an `InternalNode` and is empty,
        // so only its cached aggregate needs to be dropped before deallocating.
        unsafe {
            ptr::drop_in_place(&raw mut (*top.as_ptr()).aggregate);
            dealloc_node(top.cast::<InternalNode<K, V>>());
        }
    }

    fn clear_parent_link(&mut self) {
        let leaf = Self::as_leaf_ptr(self);
        // SAFETY: we have unique access to the root.
        unsafe { (*leaf).parent = None };
    }
}

// ---------------------------------------------------------------------------
// Mutable access
// ---------------------------------------------------------------------------

impl<'a, K, V, NodeType> NodeRef<marker::Mut<'a>, K, V, NodeType> {
    /// # Safety
    /// The caller must ensure no other reference to this node is used while the
    /// returned reference is alive.
    unsafe fn reborrow_mut(&mut self) -> NodeRef<marker::Mut<'_>, K, V, NodeType> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    fn as_leaf_mut(&mut self) -> &mut LeafNode<K, V> {
        let ptr = Self::as_leaf_ptr(self);
        // SAFETY: we have exclusive access to the node.
        unsafe { &mut *ptr }
    }

    fn dormant(&self) -> NodeRef<marker::DormantMut, K, V, NodeType> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    fn len_mut(&mut self) -> &mut u16 {
        &mut self.as_leaf_mut().len
    }

    /// # Safety
    /// `index` must be in bounds of `0..CAPACITY`.
    unsafe fn key_area_mut<I, Output: ?Sized>(&mut self, index: I) -> &mut Output
    where
        I: SliceIndex<[MaybeUninit<K>], Output = Output>,
    {
        // SAFETY: guaranteed by the caller.
        unsafe {
            self.as_leaf_mut()
                .keys
                .as_mut_slice()
                .get_unchecked_mut(index)
        }
    }

    /// # Safety
    /// `index` must be in bounds of `0..CAPACITY`.
    unsafe fn val_area_mut<I, Output: ?Sized>(&mut self, index: I) -> &mut Output
    where
        I: SliceIndex<[MaybeUninit<V>], Output = Output>,
    {
        // SAFETY: guaranteed by the caller.
        unsafe {
            self.as_leaf_mut()
                .values
                .as_mut_slice()
                .get_unchecked_mut(index)
        }
    }

    /// Recomputes the cached aggregate of this node from its values and its
    /// children's aggregates.
    fn recompute_aggregate(&mut self)
    where
        V: Aggregate,
    {
        if self.height == 0 {
            let agg = reduce_leaf(self.as_leaf_mut());
            self.as_leaf_mut().aggregate = agg;
        } else {
            let leaf_ptr = Self::as_leaf_ptr(self);
            // SAFETY: `height > 0`, so this is an `InternalNode`.
            let internal = unsafe { &mut *leaf_ptr.cast::<InternalNode<K, V>>() };
            let len = usize::from(internal.leaf.len);

            let mut agg = V::reduce_slice(unsafe { internal.leaf.values[..len].assume_init_ref() });

            for i in 0..=len {
                // SAFETY: the first `len` edges and values are initialized.
                let child = unsafe { internal.edges[i].assume_init() };
                agg = agg.reduce(unsafe { &(*child.cast::<LeafNode<K, V>>().as_ptr()).aggregate });
            }

            internal.leaf.aggregate = agg;
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    fn as_internal_mut(&mut self) -> &mut InternalNode<K, V> {
        let ptr = Self::as_internal_ptr(self);
        // SAFETY: we have exclusive access to the node.
        unsafe { &mut *ptr }
    }

    /// # Safety
    /// `index` must be in bounds of `0..MAX_EDGES`.
    unsafe fn edge_area_mut<I, Output: ?Sized>(&mut self, index: I) -> &mut Output
    where
        I: SliceIndex<[MaybeUninit<NonNull<InternalNode<K, V>>>], Output = Output>,
    {
        // SAFETY: guaranteed by the caller.
        unsafe {
            self.as_internal_mut()
                .edges
                .as_mut_slice()
                .get_unchecked_mut(index)
        }
    }

    /// # Safety
    /// Every item yielded by `range` must be a valid edge index.
    unsafe fn correct_childrens_parent_links<R: Iterator<Item = usize>>(&mut self, range: R) {
        for i in range {
            debug_assert!(i <= self.len());
            // SAFETY: `i` is a valid edge index.
            unsafe { Handle::new_edge(self.reborrow_mut(), i) }.correct_parent_link();
        }
    }

    fn correct_all_childrens_parent_links(&mut self) {
        let len = self.len();
        // SAFETY: `0..=len` are all valid edge indices.
        unsafe { self.correct_childrens_parent_links(0..=len) };
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Either> {
    fn set_parent_link(&mut self, parent: NonNull<InternalNode<K, V>>, parent_idx: usize) {
        let leaf = Self::as_leaf_ptr(self);
        // SAFETY: we have exclusive access to the node.
        unsafe {
            (*leaf).parent = Some(parent);
            (*leaf).parent_idx.write(parent_idx as u16);
        }
    }

    unsafe fn cast_to_leaf_unchecked(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Leaf> {
        debug_assert!(self.height == 0);
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    unsafe fn cast_to_internal_unchecked(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
        debug_assert!(self.height > 0);
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }

    /// Recomputes this node's aggregate and then all of its ancestors'.
    fn recompute_and_ascend(mut self)
    where
        V: Aggregate,
    {
        loop {
            self.recompute_aggregate();
            match self.ascend() {
                Ok(parent_edge) => self = parent_edge.into_node().forget_type(),
                Err(_) => break,
            }
        }
    }
}

impl<K, V, NodeType> NodeRef<marker::DormantMut, K, V, NodeType> {
    /// # Safety
    /// The reborrow used to create this node must have ended.
    unsafe fn awaken<'a>(self) -> NodeRef<marker::Mut<'a>, K, V, NodeType> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

impl<Node: Copy, HandleType> Copy for Handle<Node, HandleType> {}
impl<Node: Copy, HandleType> Clone for Handle<Node, HandleType> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Node, HandleType> Handle<Node, HandleType> {
    fn into_node(self) -> Node {
        self.node
    }

    fn idx(&self) -> usize {
        self.index
    }
}

impl<BorrowType, K, V, NodeType> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV> {
    /// # Safety
    /// `index < node.len()`.
    unsafe fn new_kv(node: NodeRef<BorrowType, K, V, NodeType>, index: usize) -> Self {
        debug_assert!(index < node.len());
        Handle {
            node,
            index,
            _marker: PhantomData,
        }
    }

    fn left_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        // SAFETY: `index < len`, so `index` is a valid edge.
        unsafe { Handle::new_edge(self.node, self.index) }
    }

    fn right_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        // SAFETY: `index + 1 <= len` is a valid edge.
        unsafe { Handle::new_edge(self.node, self.index + 1) }
    }
}

impl<BorrowType, K, V, NodeType> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
    /// # Safety
    /// `index <= node.len()`.
    unsafe fn new_edge(node: NodeRef<BorrowType, K, V, NodeType>, index: usize) -> Self {
        debug_assert!(index <= node.len());
        Handle {
            node,
            index,
            _marker: PhantomData,
        }
    }

    fn left_kv(self) -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {
        if self.index > 0 {
            // SAFETY: `index - 1 < len`.
            Ok(unsafe { Handle::new_kv(self.node, self.index - 1) })
        } else {
            Err(self)
        }
    }

    fn right_kv(self) -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {
        if self.index < self.node.len() {
            // SAFETY: `index < len`.
            Ok(unsafe { Handle::new_kv(self.node, self.index) })
        } else {
            Err(self)
        }
    }
}

impl<BorrowType, K, V, NodeType, HandleType> PartialEq
    for Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
    fn eq(&self, other: &Self) -> bool {
        self.node.eq(&other.node) && self.index == other.index
    }
}

impl<BorrowType, K, V, NodeType, HandleType>
    Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
    fn reborrow(&self) -> Handle<NodeRef<marker::Immut<'_>, K, V, NodeType>, HandleType> {
        Handle {
            node: self.node.reborrow(),
            index: self.index,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V, NodeType, HandleType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, HandleType> {
    /// # Safety
    /// The caller must ensure no other reference to this node is used while the
    /// returned handle is alive.
    unsafe fn reborrow_mut(
        &mut self,
    ) -> Handle<NodeRef<marker::Mut<'_>, K, V, NodeType>, HandleType> {
        // SAFETY: guaranteed by the caller.
        Handle {
            node: unsafe { self.node.reborrow_mut() },
            index: self.index,
            _marker: PhantomData,
        }
    }

    fn dormant(&self) -> Handle<NodeRef<marker::DormantMut, K, V, NodeType>, HandleType> {
        Handle {
            node: self.node.dormant(),
            index: self.index,
            _marker: PhantomData,
        }
    }
}

impl<K, V, NodeType, HandleType> Handle<NodeRef<marker::DormantMut, K, V, NodeType>, HandleType> {
    /// # Safety
    /// The reborrow used to create this handle must have ended.
    unsafe fn awaken<'a>(self) -> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, HandleType> {
        // SAFETY: guaranteed by the caller.
        Handle {
            node: unsafe { self.node.awaken() },
            index: self.index,
            _marker: PhantomData,
        }
    }
}

// kv_mut is only sound on a node we can mutate.
impl<'a, K: 'a, V: 'a, Type> Handle<NodeRef<marker::Mut<'a>, K, V, Type>, marker::KV> {
    fn kv_mut(&mut self) -> (&mut K, &mut V) {
        debug_assert!(self.index < self.node.len());
        // SAFETY: `index < len`, and we have exclusive access.
        unsafe {
            let leaf = self.node.as_leaf_mut();
            let key = leaf.keys.get_unchecked_mut(self.index).assume_init_mut();
            let val = leaf.values.get_unchecked_mut(self.index).assume_init_mut();
            (key, val)
        }
    }

    fn replace_kv(&mut self, k: K, v: V) -> (K, V)
    where
        V: Aggregate,
    {
        let old = {
            let (key, val) = self.kv_mut();
            (mem::replace(key, k), mem::replace(val, v))
        };
        self.node.recompute_aggregate();
        old
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Immut<'a>, K, V, NodeType>, marker::KV> {
    fn into_kv(self) -> (&'a K, &'a V) {
        debug_assert!(self.index < self.node.len());
        let leaf = self.node.into_leaf();
        // SAFETY: `index < len`.
        let k = unsafe { leaf.keys.get_unchecked(self.index).assume_init_ref() };
        let v = unsafe { leaf.values.get_unchecked(self.index).assume_init_ref() };
        (k, v)
    }
}

impl<BorrowType, K, V, Type> Handle<NodeRef<BorrowType, K, V, marker::Either>, Type> {
    fn force(
        self,
    ) -> Force<
        Handle<NodeRef<BorrowType, K, V, marker::Leaf>, Type>,
        Handle<NodeRef<BorrowType, K, V, marker::Internal>, Type>,
    > {
        match self.node.force() {
            Force::Leaf(node) => Force::Leaf(Handle {
                node,
                index: self.index,
                _marker: PhantomData,
            }),
            Force::Internal(node) => Force::Internal(Handle {
                node,
                index: self.index,
                _marker: PhantomData,
            }),
        }
    }
}

impl<'a, K, V, Type> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, Type> {
    unsafe fn cast_to_leaf_unchecked(
        self,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, Type> {
        // SAFETY: guaranteed by the caller.
        let node = unsafe { self.node.cast_to_leaf_unchecked() };
        Handle {
            node,
            index: self.index,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Type erasure helpers
// ---------------------------------------------------------------------------

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Leaf> {
    fn forget_type(self) -> NodeRef<BorrowType, K, V, marker::Either> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    fn forget_type(self) -> NodeRef<BorrowType, K, V, marker::Either> {
        NodeRef {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::KV> {
    fn forget_node_type(self) -> Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::KV> {
        // SAFETY: `index < len`.
        unsafe { Handle::new_kv(self.node.forget_type(), self.index) }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
    fn forget_node_type(self) -> Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::Edge> {
        // SAFETY: `index <= len`.
        unsafe { Handle::new_edge(self.node.forget_type(), self.index) }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge> {
    fn forget_node_type(self) -> Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::Edge> {
        // SAFETY: `index <= len`.
        unsafe { Handle::new_edge(self.node.forget_type(), self.index) }
    }
}

impl<'a, K, V> SplitResult<'a, K, V, marker::Leaf> {
    fn forget_node_type(self) -> SplitResult<'a, K, V, marker::Either> {
        SplitResult {
            left: self.left.forget_type(),
            kv: self.kv,
            right: self.right.forget_type(),
        }
    }
}

impl<'a, K, V> SplitResult<'a, K, V, marker::Internal> {
    fn forget_node_type(self) -> SplitResult<'a, K, V, marker::Either> {
        SplitResult {
            left: self.left.forget_type(),
            kv: self.kv,
            right: self.right.forget_type(),
        }
    }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge>
{
    /// Descends to the child pointed at by this edge.
    fn descend(self) -> NodeRef<BorrowType, K, V, marker::Either> {
        const { assert!(BorrowType::TRAVERSAL_PERMIT) };

        let parent_ptr = NodeRef::as_internal_ptr(&self.node);
        // SAFETY: `index` is a valid edge.
        let node = unsafe {
            (*parent_ptr)
                .edges
                .get_unchecked(self.index)
                .assume_init()
                .cast::<LeafNode<K, V>>()
        };
        NodeRef {
            node,
            height: self.node.height - 1,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    fn correct_parent_link(self) {
        // SAFETY: the pointer is to the parent internal node.
        let ptr = unsafe { NonNull::new_unchecked(NodeRef::as_internal_ptr(&self.node)) };
        let idx = self.index;
        let mut child = self.descend();
        child.set_parent_link(ptr, idx);
    }
}

impl<BorrowType: marker::BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Either> {
    fn first_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        let mut node = self;
        loop {
            match node.force() {
                Force::Leaf(leaf) => return leaf.first_edge(),
                Force::Internal(internal) => node = internal.first_edge().descend(),
            }
        }
    }

    fn last_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        let mut node = self;
        loop {
            match node.force() {
                Force::Leaf(leaf) => return leaf.last_edge(),
                Force::Internal(internal) => node = internal.last_edge().descend(),
            }
        }
    }
}

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>
{
    /// The next key/value pair to the right of this edge, if any.
    fn next_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::Either>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.right_kv() {
                Ok(kv) => return Ok(kv),
                Err(last_edge) => last_edge.into_node().ascend()?.forget_node_type(),
            };
        }
    }

    /// The next key/value pair to the left of this edge, if any.
    fn next_back_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::Either>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.left_kv() {
                Ok(kv) => return Ok(kv),
                Err(last_edge) => last_edge.into_node().ascend()?.forget_node_type(),
            };
        }
    }
}

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::Either>, marker::KV>
{
    fn next_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        match self.force() {
            Force::Leaf(leaf_kv) => leaf_kv.right_edge(),
            Force::Internal(internal_kv) => internal_kv.right_edge().descend().first_leaf_edge(),
        }
    }

    fn next_back_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        match self.force() {
            Force::Leaf(leaf_kv) => leaf_kv.left_edge(),
            Force::Internal(internal_kv) => internal_kv.left_edge().descend().last_leaf_edge(),
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

impl<BorrowType, K, V, NodeType> NodeRef<BorrowType, K, V, NodeType> {
    fn search_node<Q: ?Sized + Ord>(
        self,
        key: &Q,
    ) -> SearchResult<BorrowType, K, V, NodeType, NodeType>
    where
        K: Borrow<Q>,
    {
        match unsafe { self.find_key_index(key, 0) } {
            // SAFETY: the returned index is `< len`.
            IndexResult::KV(idx) => SearchResult::Found(unsafe { Handle::new_kv(self, idx) }),
            // SAFETY: the returned index is `<= len`.
            IndexResult::Edge(idx) => SearchResult::GoDown(unsafe { Handle::new_edge(self, idx) }),
        }
    }

    /// # Safety
    /// `start_index <= node.len()`.
    unsafe fn find_key_index<Q: ?Sized + Ord>(&self, key: &Q, start_index: usize) -> IndexResult
    where
        K: Borrow<Q>,
    {
        let node = self.reborrow();
        let keys = node.keys();
        debug_assert!(start_index <= keys.len());
        // SAFETY: `start_index <= len`.
        for (offset, k) in unsafe { keys.get_unchecked(start_index..) }
            .iter()
            .enumerate()
        {
            match key.cmp(k.borrow()) {
                Ordering::Greater => {}
                Ordering::Equal => return IndexResult::KV(start_index + offset),
                Ordering::Less => return IndexResult::Edge(start_index + offset),
            }
        }
        IndexResult::Edge(keys.len())
    }
}

impl<BorrowType: marker::BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Either> {
    /// Searches the subtree rooted at this node for `key`.
    fn search_tree<Q: ?Sized + Ord>(
        mut self,
        key: &Q,
    ) -> SearchResult<BorrowType, K, V, marker::Either, marker::Leaf>
    where
        K: Borrow<Q>,
    {
        loop {
            self = match self.search_node(key) {
                SearchResult::Found(handle) => return SearchResult::Found(handle),
                SearchResult::GoDown(handle) => match handle.force() {
                    Force::Leaf(leaf) => return SearchResult::GoDown(leaf),
                    Force::Internal(internal) => internal.descend(),
                },
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Insertion and splitting
// ---------------------------------------------------------------------------

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Appends a pair and a right edge to the node.
    fn push(&mut self, key: K, val: V, edge: Root<K, V>)
    where
        V: Aggregate,
    {
        assert_eq!(edge.height, self.height - 1);
        let idx = self.len();
        assert!(idx < CAPACITY);
        *self.len_mut() += 1;
        // SAFETY: there is room for one more pair and edge.
        unsafe {
            self.key_area_mut(idx).write(key);
            self.val_area_mut(idx).write(val);
            self.edge_area_mut(idx + 1).write(edge.node.cast());
            Handle::new_edge(self.reborrow_mut(), idx + 1).correct_parent_link();
        }
        self.recompute_aggregate();
    }
}

/// Computes where a node that is about to be split should put its median, and
/// on which side of the split the new element belongs.
fn splitpoint(edge_idx: usize) -> (usize, LeftOrRight<usize>) {
    debug_assert!(edge_idx <= CAPACITY);
    match edge_idx {
        0..EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER - 1, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_RIGHT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Right(0)),
        _ => (
            KV_IDX_CENTER + 1,
            LeftOrRight::Right(edge_idx - (KV_IDX_CENTER + 1 + 1)),
        ),
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts into a node that has room. Returns a handle to the new element.
    ///
    /// # Safety
    /// The node must have fewer than `CAPACITY` elements.
    unsafe fn insert_fit(
        mut self,
        key: K,
        val: V,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV>
    where
        V: Aggregate,
    {
        debug_assert!(self.node.len() < CAPACITY);
        let new_len = self.node.len() + 1;

        // SAFETY: there is room for one more element.
        unsafe {
            slice_insert(self.node.key_area_mut(..new_len), self.index, key);
            slice_insert(self.node.val_area_mut(..new_len), self.index, val);
            *self.node.len_mut() = new_len as u16;
        }
        self.node.recompute_aggregate();

        // SAFETY: `index < new_len`.
        unsafe { Handle::new_kv(self.node, self.index) }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts into the node, splitting it if necessary.
    fn insert(
        self,
        key: K,
        val: V,
    ) -> (
        Option<SplitResult<'a, K, V, marker::Leaf>>,
        Handle<NodeRef<marker::DormantMut, K, V, marker::Leaf>, marker::KV>,
    )
    where
        V: Aggregate,
    {
        if self.node.len() < CAPACITY {
            // SAFETY: there is room.
            let handle = unsafe { self.insert_fit(key, val) };
            (None, handle.dormant())
        } else {
            let (middle_kv_idx, insertion) = splitpoint(self.index);
            // SAFETY: `middle_kv_idx < len`.
            let middle = unsafe { Handle::new_kv(self.node, middle_kv_idx) };
            let mut result = middle.split();
            let insertion_edge = match insertion {
                LeftOrRight::Left(insert_idx) => {
                    // SAFETY: `insert_idx <= result.left.len()`.
                    unsafe { Handle::new_edge(result.left.reborrow_mut(), insert_idx) }
                }
                LeftOrRight::Right(insert_idx) => {
                    // SAFETY: `insert_idx <= result.right.len()`.
                    unsafe { Handle::new_edge(result.right.borrow_mut(), insert_idx) }
                }
            };
            // SAFETY: we just split, so there is room.
            let handle = unsafe { insertion_edge.insert_fit(key, val).dormant() };
            (Some(result), handle)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Inserts a pair and a right edge into a node that has room.
    fn insert_fit(&mut self, key: K, val: V, edge: Root<K, V>)
    where
        V: Aggregate,
    {
        debug_assert!(self.node.len() < CAPACITY);
        debug_assert_eq!(edge.height, self.node.height - 1);
        let new_len = self.node.len() + 1;

        // SAFETY: there is room for one more element and edge.
        unsafe {
            slice_insert(self.node.key_area_mut(..new_len), self.index, key);
            slice_insert(self.node.val_area_mut(..new_len), self.index, val);
            slice_insert(
                self.node.edge_area_mut(..new_len + 1),
                self.index + 1,
                edge.node.cast(),
            );
            *self.node.len_mut() = new_len as u16;
            self.node
                .correct_childrens_parent_links(self.index + 1..new_len + 1);
        }
        self.node.recompute_aggregate();
    }

    /// Inserts into the node, splitting it if necessary.
    fn insert(
        mut self,
        key: K,
        val: V,
        edge: Root<K, V>,
    ) -> Option<SplitResult<'a, K, V, marker::Internal>>
    where
        V: Aggregate,
    {
        assert_eq!(edge.height, self.node.height - 1);
        if self.node.len() < CAPACITY {
            self.insert_fit(key, val, edge);
            None
        } else {
            let (middle_kv_idx, insertion) = splitpoint(self.index);
            // SAFETY: `middle_kv_idx < len`.
            let middle = unsafe { Handle::new_kv(self.node, middle_kv_idx) };
            let mut result = middle.split();
            let mut insertion_edge = match insertion {
                LeftOrRight::Left(insert_idx) => {
                    // SAFETY: `insert_idx <= result.left.len()`.
                    unsafe { Handle::new_edge(result.left.reborrow_mut(), insert_idx) }
                }
                LeftOrRight::Right(insert_idx) => {
                    // SAFETY: `insert_idx <= result.right.len()`.
                    unsafe { Handle::new_edge(result.right.borrow_mut(), insert_idx) }
                }
            };
            insertion_edge.insert_fit(key, val, edge);
            Some(result)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts a pair, propagating splits up the tree. The `split_root`
    /// callback is invoked exactly when the root itself needs to be split.
    fn insert_recursing(
        self,
        key: K,
        value: V,
        split_root: impl FnOnce(SplitResult<'a, K, V, marker::Either>),
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV>
    where
        V: Aggregate,
    {
        let (mut split, handle) = match self.insert(key, value) {
            // SAFETY: splitting is over; the handle is stable.
            (None, handle) => return unsafe { handle.awaken() },
            (Some(split), handle) => (split.forget_node_type(), handle),
        };

        loop {
            split = match split.left.ascend() {
                Ok(parent) => match parent.insert(split.kv.0, split.kv.1, split.right) {
                    // SAFETY: splitting is over; the handle is stable.
                    None => return unsafe { handle.awaken() },
                    Some(split) => split.forget_node_type(),
                },
                Err(root) => {
                    split_root(SplitResult {
                        left: root,
                        ..split
                    });
                    // SAFETY: splitting is over; the handle is stable.
                    return unsafe { handle.awaken() };
                }
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {
    /// Moves everything to the right of `self` into `new_node`, returning the
    /// pair at `self`.
    fn split_leaf_data(&mut self, new_node: &mut LeafNode<K, V>) -> (K, V) {
        debug_assert!(self.index < self.node.len());
        let old_len = self.node.len();
        let new_len = old_len - self.index - 1;
        new_node.len = new_len as u16;
        // SAFETY: the source ranges are initialized, the destination is fresh.
        unsafe {
            let k = self.node.key_area_mut(self.index).assume_init_read();
            let v = self.node.val_area_mut(self.index).assume_init_read();

            move_to_slice(
                self.node.key_area_mut(self.index + 1..old_len),
                &mut new_node.keys[..new_len],
            );
            move_to_slice(
                self.node.val_area_mut(self.index + 1..old_len),
                &mut new_node.values[..new_len],
            );

            *self.node.len_mut() = self.index as u16;
            (k, v)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    /// Splits the node at this pair; the pair is returned separately.
    fn split(mut self) -> SplitResult<'a, K, V, marker::Leaf>
    where
        V: Aggregate,
    {
        let mut new_node = LeafNode::<K, V>::new();
        // SAFETY: freshly allocated and uniquely owned.
        let new_leaf = unsafe { new_node.as_mut() };

        let kv = self.split_leaf_data(new_leaf);

        self.node.recompute_aggregate();

        let right = NodeRef::<marker::Owned, K, V, marker::Leaf>::from_new_leaf(new_node);

        SplitResult {
            left: self.node,
            kv,
            right,
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Splits the node at this pair; the pair is returned separately.
    fn split(mut self) -> SplitResult<'a, K, V, marker::Internal>
    where
        V: Aggregate,
    {
        let old_len = self.node.len();
        // SAFETY: freshly allocated and uniquely owned.
        unsafe {
            let mut new_node = InternalNode::<K, V>::empty();
            let new_internal = new_node.as_mut();
            let kv = self.split_leaf_data(&mut new_internal.leaf);
            let new_len = usize::from(new_internal.leaf.len);
            move_to_slice(
                self.node.edge_area_mut(self.index + 1..old_len + 1),
                &mut new_internal.edges[..new_len + 1],
            );

            self.node.recompute_aggregate();

            let height = self.node.height;
            let right = NodeRef::<marker::Owned, K, V, marker::Internal>::from_new_internal(
                new_node, height,
            );

            SplitResult {
                left: self.node,
                kv,
                right,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Removal
// ---------------------------------------------------------------------------

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    /// Removes and returns this pair, together with the edge that collapsed
    /// into its place.
    fn remove(
        mut self,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    )
    where
        V: Aggregate,
    {
        let old_len = self.node.len();
        // SAFETY: `index < old_len`.
        let (k, v) = unsafe {
            let k = slice_remove(self.node.key_area_mut(..old_len), self.index);
            let v = slice_remove(self.node.val_area_mut(..old_len), self.index);
            *self.node.len_mut() = (old_len - 1) as u16;
            (k, v)
        };
        self.node.recompute_aggregate();
        ((k, v), self.left_edge())
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, marker::KV> {
    /// Removes this pair from the tree, rebalancing along the way. Calls
    /// `handle_emptied_internal_root` if the internal root became empty and
    /// must be popped by the caller.
    fn remove_kv_tracking<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    )
    where
        V: Aggregate,
    {
        match self.force() {
            Force::Leaf(node) => node.remove_leaf_kv(handle_emptied_internal_root),
            Force::Internal(node) => node.remove_internal_kv(handle_emptied_internal_root),
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    fn remove_leaf_kv<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    )
    where
        V: Aggregate,
    {
        let (old_kv, mut pos) = self.remove();
        let len = pos.reborrow().into_node().len();
        if len < MIN_LEN {
            let idx = pos.idx();
            let new_pos = match pos.into_node().forget_type().choose_parent_kv() {
                Ok(LeftOrRight::Left(left_parent_kv)) => {
                    debug_assert_eq!(left_parent_kv.right_child_len(), MIN_LEN - 1);
                    if left_parent_kv.can_merge() {
                        left_parent_kv.merge_tracking_child_edge(LeftOrRight::Right(idx))
                    } else {
                        left_parent_kv.steal_left(idx)
                    }
                }
                Ok(LeftOrRight::Right(right_parent_kv)) => {
                    debug_assert_eq!(right_parent_kv.left_child_len(), MIN_LEN - 1);
                    if right_parent_kv.can_merge() {
                        right_parent_kv.merge_tracking_child_edge(LeftOrRight::Left(idx))
                    } else {
                        right_parent_kv.steal_right(idx)
                    }
                }
                Err(pos) => {
                    // SAFETY: `idx <= pos.len()`.
                    unsafe { Handle::new_edge(pos, idx) }
                }
            };
            // SAFETY: the tracked position is a leaf edge.
            pos = unsafe { new_pos.cast_to_leaf_unchecked() };

            // SAFETY: `pos` is not affected by fixing its ancestors, at worst
            // its parent link changes.
            if let Ok(parent) = unsafe { pos.reborrow_mut() }.into_node().ascend()
                && !parent
                    .into_node()
                    .forget_type()
                    .fix_node_and_affected_ancestors()
            {
                handle_emptied_internal_root();
            }
        }
        // Recompute along this path too: the caller may end up returning a
        // position in a different subtree (see `remove_internal_kv`).
        // SAFETY: `pos` is a valid leaf edge.
        unsafe { pos.reborrow_mut() }
            .into_node()
            .forget_type()
            .recompute_and_ascend();
        (old_kv, pos)
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    fn remove_internal_kv<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    )
    where
        V: Aggregate,
    {
        // Replace with the in-order predecessor from the left subtree.
        let left_leaf_kv = self.left_edge().descend().last_leaf_edge().left_kv();
        // SAFETY: there is always a predecessor.
        let left_leaf_kv = unsafe { left_leaf_kv.ok().unwrap_unchecked() };
        let (left_kv, left_hole) = left_leaf_kv.remove_leaf_kv(handle_emptied_internal_root);

        // The internal node may have moved; find the pair again.
        // SAFETY: the hole is followed by a pair.
        let mut internal = unsafe { left_hole.next_kv().ok().unwrap_unchecked() };
        let old_kv = internal.replace_kv(left_kv.0, left_kv.1);
        let pos = internal.next_leaf_edge();
        (old_kv, pos)
    }
}

// ---------------------------------------------------------------------------
// Balancing
// ---------------------------------------------------------------------------

struct BalancingContext<'a, K, V> {
    parent: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV>,
    left_child: NodeRef<marker::Mut<'a>, K, V, marker::Either>,
    right_child: NodeRef<marker::Mut<'a>, K, V, marker::Either>,
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Either> {
    /// Chooses a balancing context involving this node as a child.
    fn choose_parent_kv(self) -> Result<LeftOrRight<BalancingContext<'a, K, V>>, Self>
    where
        V: Aggregate,
    {
        // SAFETY: `NodeRef` is not `Copy`, so read out the pointer fields.
        match unsafe { ptr::read(&self) }.ascend() {
            Ok(parent_edge) => match parent_edge.left_kv() {
                Ok(left_parent_kv) => Ok(LeftOrRight::Left(BalancingContext {
                    // SAFETY: `Handle` is only `Copy` for immutable nodes.
                    parent: unsafe { ptr::read(&left_parent_kv) },
                    left_child: left_parent_kv.left_edge().descend(),
                    right_child: self,
                })),
                Err(parent_edge) => match parent_edge.right_kv() {
                    Ok(right_parent_kv) => Ok(LeftOrRight::Right(BalancingContext {
                        // SAFETY: as above.
                        parent: unsafe { ptr::read(&right_parent_kv) },
                        left_child: self,
                        right_child: right_parent_kv.right_edge().descend(),
                    })),
                    Err(_) => unreachable!("empty internal node"),
                },
            },
            Err(root) => Err(root),
        }
    }
}

impl<'a, K, V> BalancingContext<'a, K, V> {
    fn left_child_len(&self) -> usize {
        self.left_child.len()
    }

    fn right_child_len(&self) -> usize {
        self.right_child.len()
    }

    /// Whether the central pair and both children fit into a single node.
    fn can_merge(&self) -> bool {
        self.left_child.len() + 1 + self.right_child.len() <= CAPACITY
    }
}

impl<'a, K: 'a, V: 'a> BalancingContext<'a, K, V> {
    fn do_merge<R>(
        self,
        result: impl FnOnce(
            NodeRef<marker::Mut<'a>, K, V, marker::Internal>,
            NodeRef<marker::Mut<'a>, K, V, marker::Either>,
        ) -> R,
    ) -> R
    where
        V: Aggregate,
    {
        let Handle {
            node: mut parent_node,
            index: parent_idx,
            _marker,
        } = self.parent;
        let old_parent_len = parent_node.len();
        let mut left_node = self.left_child;
        let old_left_len = left_node.len();
        let mut right_node = self.right_child;
        let right_len = right_node.len();
        // The right node is destroyed by the merge, but its cached aggregate is
        // a distinct value that still needs to be dropped.
        let right_leaf = right_node.node;
        let new_left_len = old_left_len + 1 + right_len;

        assert!(new_left_len <= CAPACITY);

        // SAFETY: all accesses below are within initialized storage or freshly
        // freed after being moved out of.
        unsafe {
            *left_node.len_mut() = new_left_len as u16;

            let parent_key = slice_remove(parent_node.key_area_mut(..old_parent_len), parent_idx);
            left_node.key_area_mut(old_left_len).write(parent_key);
            move_to_slice(
                right_node.key_area_mut(..right_len),
                left_node.key_area_mut(old_left_len + 1..new_left_len),
            );

            let parent_val = slice_remove(parent_node.val_area_mut(..old_parent_len), parent_idx);
            left_node.val_area_mut(old_left_len).write(parent_val);
            move_to_slice(
                right_node.val_area_mut(..right_len),
                left_node.val_area_mut(old_left_len + 1..new_left_len),
            );

            slice_remove(
                parent_node.edge_area_mut(..old_parent_len + 1),
                parent_idx + 1,
            );
            parent_node.correct_childrens_parent_links(parent_idx + 1..old_parent_len);
            *parent_node.len_mut() -= 1;

            if parent_node.height > 1 {
                // SAFETY: children are one level below the parent, hence internal.
                let mut left_internal = left_node.reborrow_mut().cast_to_internal_unchecked();
                let mut right_internal = right_node.cast_to_internal_unchecked();
                move_to_slice(
                    right_internal.edge_area_mut(..right_len + 1),
                    left_internal.edge_area_mut(old_left_len + 1..new_left_len + 1),
                );
                left_internal.correct_childrens_parent_links(old_left_len + 1..new_left_len + 1);
                ptr::drop_in_place(&raw mut (*right_leaf.as_ptr()).aggregate);
                dealloc_node(right_leaf.cast::<InternalNode<K, V>>());
            } else {
                ptr::drop_in_place(&raw mut (*right_leaf.as_ptr()).aggregate);
                dealloc_node(right_leaf);
            }

            left_node.recompute_aggregate();
            parent_node.recompute_aggregate();
        }
        result(parent_node, left_node)
    }

    /// Merges the central pair and both children into the left child, returning
    /// the shrunk parent.
    fn merge_tracking_parent(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Internal>
    where
        V: Aggregate,
    {
        self.do_merge(|parent, _child| parent)
    }

    /// Like `merge_tracking_parent`, but returns the merged child.
    fn merge_tracking_child(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Either>
    where
        V: Aggregate,
    {
        self.do_merge(|_parent, child| child)
    }

    /// Like `merge_tracking_child`, but tracks an edge through the merge.
    fn merge_tracking_child_edge(
        self,
        track_edge_idx: LeftOrRight<usize>,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, marker::Edge>
    where
        V: Aggregate,
    {
        let old_left_len = self.left_child.len();
        let right_len = self.right_child.len();
        assert!(match track_edge_idx {
            LeftOrRight::Left(idx) => idx <= old_left_len,
            LeftOrRight::Right(idx) => idx <= right_len,
        });
        let child = self.merge_tracking_child();
        let new_idx = match track_edge_idx {
            LeftOrRight::Left(idx) => idx,
            LeftOrRight::Right(idx) => old_left_len + 1 + idx,
        };
        // SAFETY: `new_idx <= child.len()`.
        unsafe { Handle::new_edge(child, new_idx) }
    }

    /// Steals one element from the left child into the right child, tracking an
    /// edge in the right child.
    fn steal_left(
        mut self,
        track_right_edge_idx: usize,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, marker::Edge>
    where
        V: Aggregate,
    {
        self.bulk_steal_left(1);
        // SAFETY: index shifted by one.
        unsafe { Handle::new_edge(self.right_child, 1 + track_right_edge_idx) }
    }

    /// Steals one element from the right child into the left child, tracking an
    /// edge in the left child.
    fn steal_right(
        mut self,
        track_left_edge_idx: usize,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, marker::Edge>
    where
        V: Aggregate,
    {
        self.bulk_steal_right(1);
        // SAFETY: index unchanged.
        unsafe { Handle::new_edge(self.left_child, track_left_edge_idx) }
    }

    fn bulk_steal_left(&mut self, count: usize)
    where
        V: Aggregate,
    {
        assert!(count > 0);
        // SAFETY: the node lengths are checked below.
        unsafe {
            let left_node = &mut self.left_child;
            let old_left_len = left_node.len();
            let right_node = &mut self.right_child;
            let old_right_len = right_node.len();

            assert!(old_right_len + count <= CAPACITY);
            assert!(old_left_len >= count);

            let new_left_len = old_left_len - count;
            let new_right_len = old_right_len + count;
            *left_node.len_mut() = new_left_len as u16;
            *right_node.len_mut() = new_right_len as u16;

            {
                slice_shr(right_node.key_area_mut(..new_right_len), count);
                slice_shr(right_node.val_area_mut(..new_right_len), count);

                move_to_slice(
                    left_node.key_area_mut(new_left_len + 1..old_left_len),
                    right_node.key_area_mut(..count - 1),
                );
                move_to_slice(
                    left_node.val_area_mut(new_left_len + 1..old_left_len),
                    right_node.val_area_mut(..count - 1),
                );

                let k = left_node.key_area_mut(new_left_len).assume_init_read();
                let v = left_node.val_area_mut(new_left_len).assume_init_read();
                let (k, v) = self.parent.replace_kv(k, v);

                right_node.key_area_mut(count - 1).write(k);
                right_node.val_area_mut(count - 1).write(v);
            }

            match (
                left_node.reborrow_mut().force(),
                right_node.reborrow_mut().force(),
            ) {
                (Force::Internal(mut left), Force::Internal(mut right)) => {
                    slice_shr(right.edge_area_mut(..new_right_len + 1), count);
                    move_to_slice(
                        left.edge_area_mut(new_left_len + 1..old_left_len + 1),
                        right.edge_area_mut(..count),
                    );
                    right.correct_childrens_parent_links(0..new_right_len + 1);
                }
                (Force::Leaf(_), Force::Leaf(_)) => {}
                _ => unreachable!(),
            }

            left_node.recompute_aggregate();
            right_node.recompute_aggregate();
            // The parent's cached aggregate depends on both children, so it must
            // be recomputed after them (not merely when the central pair changed).
            self.parent.node.recompute_aggregate();
        }
    }

    fn bulk_steal_right(&mut self, count: usize)
    where
        V: Aggregate,
    {
        assert!(count > 0);
        // SAFETY: the node lengths are checked below.
        unsafe {
            let left_node = &mut self.left_child;
            let old_left_len = left_node.len();
            let right_node = &mut self.right_child;
            let old_right_len = right_node.len();

            assert!(old_left_len + count <= CAPACITY);
            assert!(old_right_len >= count);

            let new_left_len = old_left_len + count;
            let new_right_len = old_right_len - count;
            *left_node.len_mut() = new_left_len as u16;
            *right_node.len_mut() = new_right_len as u16;

            {
                let k = right_node.key_area_mut(count - 1).assume_init_read();
                let v = right_node.val_area_mut(count - 1).assume_init_read();
                let (k, v) = self.parent.replace_kv(k, v);

                left_node.key_area_mut(old_left_len).write(k);
                left_node.val_area_mut(old_left_len).write(v);

                move_to_slice(
                    right_node.key_area_mut(..count - 1),
                    left_node.key_area_mut(old_left_len + 1..new_left_len),
                );
                move_to_slice(
                    right_node.val_area_mut(..count - 1),
                    left_node.val_area_mut(old_left_len + 1..new_left_len),
                );

                slice_shl(right_node.key_area_mut(..old_right_len), count);
                slice_shl(right_node.val_area_mut(..old_right_len), count);
            }

            match (
                left_node.reborrow_mut().force(),
                right_node.reborrow_mut().force(),
            ) {
                (Force::Internal(mut left), Force::Internal(mut right)) => {
                    move_to_slice(
                        right.edge_area_mut(..count),
                        left.edge_area_mut(old_left_len + 1..new_left_len + 1),
                    );
                    slice_shl(right.edge_area_mut(..old_right_len + 1), count);
                    left.correct_childrens_parent_links(old_left_len + 1..new_left_len + 1);
                    right.correct_childrens_parent_links(0..new_right_len + 1);
                }
                (Force::Leaf(_), Force::Leaf(_)) => {}
                _ => unreachable!(),
            }

            left_node.recompute_aggregate();
            right_node.recompute_aggregate();
            // See `bulk_steal_left`: the parent must be recomputed last.
            self.parent.node.recompute_aggregate();
        }
    }
}

// ---------------------------------------------------------------------------
// Fixing underfull nodes
// ---------------------------------------------------------------------------

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Either> {
    /// Stocks up this node from a sibling, or merges it away. Returns the
    /// shrunk parent if the parent lost an element.
    fn fix_node_through_parent(
        self,
    ) -> Result<Option<NodeRef<marker::Mut<'a>, K, V, marker::Internal>>, Self>
    where
        V: Aggregate,
    {
        let len = self.len();
        if len >= MIN_LEN {
            Ok(None)
        } else {
            match self.choose_parent_kv() {
                Ok(LeftOrRight::Left(mut left_parent_kv)) => {
                    if left_parent_kv.can_merge() {
                        Ok(Some(left_parent_kv.merge_tracking_parent()))
                    } else {
                        left_parent_kv.bulk_steal_left(MIN_LEN - len);
                        Ok(None)
                    }
                }
                Ok(LeftOrRight::Right(mut right_parent_kv)) => {
                    if right_parent_kv.can_merge() {
                        Ok(Some(right_parent_kv.merge_tracking_parent()))
                    } else {
                        right_parent_kv.bulk_steal_right(MIN_LEN - len);
                        Ok(None)
                    }
                }
                Err(root) => {
                    if len > 0 {
                        Ok(None)
                    } else {
                        Err(root)
                    }
                }
            }
        }
    }

    /// Repeatedly stocks up this node and its ancestors. Returns `false` if the
    /// root became empty.
    fn fix_node_and_affected_ancestors(mut self) -> bool
    where
        V: Aggregate,
    {
        loop {
            match self.fix_node_through_parent() {
                Ok(Some(parent)) => self = parent.forget_type(),
                Ok(None) => return true,
                Err(_) => return false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive destruction
// ---------------------------------------------------------------------------

/// Drops all keys, values and cached aggregates in the subtree and deallocates
/// every node.
///
/// # Safety
/// `node` must be a valid subtree root of the given `height`.
unsafe fn drop_subtree<K, V>(node: NonNull<LeafNode<K, V>>, height: usize) {
    // SAFETY: the caller guarantees validity.
    unsafe {
        let leaf = &mut *node.as_ptr();
        let len = usize::from(leaf.len);
        if height == 0 {
            for i in 0..len {
                leaf.keys[i].assume_init_drop();
                leaf.values[i].assume_init_drop();
            }
            ptr::drop_in_place(&raw mut (*node.as_ptr()).aggregate);
            dealloc_node(node);
        } else {
            let internal = &mut *node.as_ptr().cast::<InternalNode<K, V>>();
            for i in 0..=len {
                let child = internal.edges[i].assume_init_read();
                drop_subtree(child.cast::<LeafNode<K, V>>(), height - 1);
            }
            for i in 0..len {
                internal.leaf.keys[i].assume_init_drop();
                internal.leaf.values[i].assume_init_drop();
            }
            ptr::drop_in_place(&raw mut (*node.as_ptr()).aggregate);
            dealloc_node(node.cast::<InternalNode<K, V>>());
        }
    }
}

// ---------------------------------------------------------------------------
// Dormant mutable reference
// ---------------------------------------------------------------------------

/// A reborrow of a unique reference whose original borrow can be recovered
/// later, even through control flow the borrow checker cannot follow.
struct DormantMutRef<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> DormantMutRef<'a, T> {
    fn new(t: &'a mut T) -> (&'a mut T, Self) {
        let ptr = NonNull::from(t);
        // SAFETY: the borrow is held for `'a` through `_marker`; we only expose
        // this one reference.
        let new_ref = unsafe { &mut *ptr.as_ptr() };
        (
            new_ref,
            Self {
                ptr,
                _marker: PhantomData,
            },
        )
    }

    /// # Safety
    /// The reborrow returned by `new` must no longer be used.
    unsafe fn awaken(self) -> &'a mut T {
        // SAFETY: guaranteed by the caller.
        unsafe { &mut *self.ptr.as_ptr() }
    }

    /// # Safety
    /// The reborrow returned by `new` must no longer be used.
    unsafe fn reborrow(&mut self) -> &'a mut T {
        // SAFETY: guaranteed by the caller.
        unsafe { &mut *self.ptr.as_ptr() }
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// An iterator over the entries of a [`BTree`], in ascending key order.
pub struct Iter<'a, K, V> {
    front: Option<Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    back: Option<Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    remaining: usize,
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let edge = self.front?;
        match edge.next_kv() {
            Ok(kv) => {
                self.front = Some(kv.next_leaf_edge());
                self.remaining -= 1;
                Some(kv.into_kv())
            }
            Err(_) => {
                self.remaining = 0;
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let edge = self.back?;
        match edge.next_back_kv() {
            Ok(kv) => {
                self.back = Some(kv.next_back_leaf_edge());
                self.remaining -= 1;
                Some(kv.into_kv())
            }
            Err(_) => {
                self.remaining = 0;
                None
            }
        }
    }
}

impl<'a, K: 'a, V: 'a> ExactSizeIterator for Iter<'a, K, V> {}

// ---------------------------------------------------------------------------
// The public tree
// ---------------------------------------------------------------------------

/// An ordered map that caches an [`Aggregate`] for every subtree.
///
/// It behaves like a `BTreeMap` for `K: Ord` keys while maintaining, for every
/// subtree, the reduction of the values it contains. `BTree::aggregate`
/// therefore runs in constant time.
///
/// # Examples
/// ```
/// use augmented_tree::{Aggregate, BTree};
///
/// #[derive(Clone)]
/// struct Sum(i64);
/// impl Aggregate for Sum {
///     fn reduce(&self, other: &Self) -> Self { Sum(self.0 + other.0) }
///     fn identity() -> Self { Sum(0) }
/// }
///
/// let mut tree = BTree::new();
/// tree.insert(1, Sum(5));
/// tree.insert(2, Sum(7));
/// assert_eq!(tree.aggregate().0, 12);
/// ```
pub struct BTree<K, V> {
    root: Root<K, V>,
    length: usize,
}

unsafe impl<K: Send, V: Send> Send for BTree<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for BTree<K, V> {}

impl<K, V: Aggregate> BTree<K, V> {
    /// Creates a new, empty tree.
    pub fn new() -> Self {
        BTree {
            root: Root::new(),
            length: 0,
        }
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The reduction of every value in the tree, in key order. The identity is
    /// returned for an empty tree.
    pub fn aggregate(&self) -> &V {
        self.root.reborrow().aggregate()
    }

    /// Inserts a key-value pair, returning the previous value if the key was
    /// already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        let (map, mut dormant_map) = DormantMutRef::new(self);
        match map.root.borrow_mut().search_tree(&key) {
            SearchResult::Found(mut handle) => {
                let old = {
                    let (_, slot) = handle.kv_mut();
                    mem::replace(slot, value)
                };
                handle.into_node().recompute_and_ascend();
                Some(old)
            }
            SearchResult::GoDown(edge) => {
                let handle = edge.insert_recursing(key, value, |ins| {
                    // SAFETY: the split recursed through the root, and no other
                    // reference to the map is used here.
                    let map = unsafe { dormant_map.reborrow() };
                    map.root
                        .push_internal_level()
                        .push(ins.kv.0, ins.kv.1, ins.right);
                });
                handle.into_node().forget_type().recompute_and_ascend();

                // SAFETY: same as above; the insertion handle is no longer used.
                let map = unsafe { dormant_map.awaken() };
                map.length += 1;
                None
            }
        }
    }

    /// Looks up a key.
    pub fn get<Q: ?Sized + Ord>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        match self.root.reborrow().search_tree(key) {
            SearchResult::Found(handle) => Some(handle.into_kv().1),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Looks up a key, returning both the key and the value.
    pub fn get_key_value<Q: ?Sized + Ord>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
    {
        match self.root.reborrow().search_tree(key) {
            SearchResult::Found(handle) => Some(handle.into_kv()),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Whether the key is present.
    pub fn contains_key<Q: ?Sized + Ord>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.get(key).is_some()
    }

    /// Returns a guard that allows mutation of the value for `key`.
    ///
    /// When the guard is dropped, the cached aggregates along the path to the
    /// root are recomputed, so the tree stays consistent.
    pub fn get_mut<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<OccupiedValue<'_, K, V>>
    where
        K: Borrow<Q>,
    {
        let found = match self.root.borrow_mut().search_tree(key) {
            SearchResult::Found(handle) => handle,
            SearchResult::GoDown(_) => return None,
        };
        Some(OccupiedValue {
            node: found.node.node,
            height: found.node.height,
            index: found.index,
            _marker: PhantomData,
        })
    }

    /// Applies `f` to the value stored for `key`, updating the cached
    /// aggregates.
    pub fn update<Q: ?Sized + Ord, F>(&mut self, key: &Q, f: F) -> Option<()>
    where
        K: Borrow<Q>,
        F: FnOnce(&mut V),
    {
        let mut guard = self.get_mut(key)?;
        f(guard.get_mut());
        Some(())
    }

    /// Removes a key, returning its value.
    pub fn remove<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key, returning the stored pair.
    pub fn remove_entry<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let found = match map.root.borrow_mut().search_tree(key) {
            SearchResult::Found(handle) => handle,
            SearchResult::GoDown(_) => return None,
        };

        let mut emptied_internal_root = false;
        let (old_kv, pos) = found.remove_kv_tracking(|| emptied_internal_root = true);
        pos.into_node().forget_type().recompute_and_ascend();

        // SAFETY: the handle from the search is no longer used.
        let map = unsafe { dormant_map.awaken() };
        map.length -= 1;
        if emptied_internal_root {
            map.root.pop_internal_level();
        }
        Some(old_kv)
    }

    /// The smallest entry, if any.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        let edge = self.root.reborrow().first_leaf_edge();
        edge.right_kv().ok().map(|kv| kv.into_kv())
    }

    /// The largest entry, if any.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let edge = self.root.reborrow().last_leaf_edge();
        edge.left_kv().ok().map(|kv| kv.into_kv())
    }

    /// Removes and returns the smallest entry.
    pub fn pop_first(&mut self) -> Option<(K, V)>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let found = match map.root.borrow_mut().first_leaf_edge().right_kv() {
            Ok(kv) => kv.forget_node_type(),
            Err(_) => return None,
        };

        let mut emptied_internal_root = false;
        let (old_kv, pos) = found.remove_kv_tracking(|| emptied_internal_root = true);
        pos.into_node().forget_type().recompute_and_ascend();

        // SAFETY: the handle is no longer used.
        let map = unsafe { dormant_map.awaken() };
        map.length -= 1;
        if emptied_internal_root {
            map.root.pop_internal_level();
        }
        Some(old_kv)
    }

    /// Removes and returns the largest entry.
    pub fn pop_last(&mut self) -> Option<(K, V)>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let found = match map.root.borrow_mut().last_leaf_edge().left_kv() {
            Ok(kv) => kv.forget_node_type(),
            Err(_) => return None,
        };

        let mut emptied_internal_root = false;
        let (old_kv, pos) = found.remove_kv_tracking(|| emptied_internal_root = true);
        pos.into_node().forget_type().recompute_and_ascend();

        // SAFETY: the handle is no longer used.
        let map = unsafe { dormant_map.awaken() };
        map.length -= 1;
        if emptied_internal_root {
            map.root.pop_internal_level();
        }
        Some(old_kv)
    }

    /// Removes every entry.
    pub fn clear(&mut self) {
        // SAFETY: the root is a valid subtree.
        unsafe { drop_subtree(self.root.node, self.root.height) };
        self.root = Root::new();
        self.length = 0;
    }

    /// An iterator over the entries in ascending key order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        let root = self.root.reborrow();
        Iter {
            front: Some(root.first_leaf_edge()),
            back: Some(root.last_leaf_edge()),
            remaining: self.length,
        }
    }

    /// An iterator over the keys in ascending order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// An iterator over the values in ascending key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }
}

impl<K, V: Aggregate> Default for BTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for BTree<K, V> {
    fn drop(&mut self) {
        // SAFETY: the root is a valid subtree and is not used afterwards.
        unsafe { drop_subtree(self.root.node, self.root.height) };
    }
}
