use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

const B: usize = 4;
const CAPACITY: usize = 2 * B - 1;
const MAX_KEYS: usize = CAPACITY;
const MAX_EDGES: usize = 2 * B;
const MIN_LEN_AFTER_SPLIT: usize = B - 1;

trait Aggregate {
    fn reduce(&self, other: &Self) -> Self;
    fn reduce_slice(&self, others: &[Self]) -> Self
    where
        Self: Sized + Clone,
    {
        let Some((a, rest)) = others.split_first() else {
            panic!("Cannot reduce an empty slice");
        };

        rest.iter().fold(self.reduce(a), |acc, x| acc.reduce(x))
    }
}

struct LeafNode<K, V> {
    parent: Option<NonNull<InternalNode<K, V>>>,
    parent_idx: MaybeUninit<u16>,
    len: u16,
    aggregate: V,
    keys: [MaybeUninit<K>; CAPACITY],
    values: [MaybeUninit<V>; CAPACITY],
}

struct InternalNode<K, V> {
    leaf: LeafNode<K, V>,
    edges: [MaybeUninit<NonNull<InternalNode<K, V>>>; MAX_EDGES],
}

type BoxedNode<K, V> = NonNull<LeafNode<K, V>>;

mod marker {
    pub struct Leaf;
    pub struct Internal;
    pub struct Either;

    pub struct KV;
    pub struct Edge;

    pub struct Immut<'a>(pub &'a ());
    pub struct Mut<'a>(pub &'a mut ());
    pub struct Owned;
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

impl<BorrowType, K, V, NodeType> NodeRef<BorrowType, K, V, NodeType> {
    #[expect(clippy::type_complexity)]
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
}

enum Force<Leaf, Internal> {
    Leaf(Leaf),
    Internal(Internal),
}
