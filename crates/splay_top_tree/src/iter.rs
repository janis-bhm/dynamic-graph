use std::ptr::NonNull;

use crate::{Handle, InternalNode, Node, NodeKind, marker};

/// A range over all leaves below a node of a top tree.
///
/// The range covers every leaf of the subtree it was constructed from, in
/// left-to-right (logical) order, where leaves are either edge leaves or label
/// leaves. It is delimited by the leaf on the logical left border of the
/// subtree and the leaf on its logical right border, so a subtree consisting of
/// a single leaf yields a range holding exactly that leaf.
///
/// The yielded handles are only meaningful while the leaves they point to are
/// part of the top tree: the range borrows nothing, so it must not outlive the
/// operations that may re-shape the subtree it covers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeRange<W> {
    /// the next leaf to be yielded when iterating forwards
    front: Option<Handle<W, marker::Leaf>>,
    /// the next leaf to be yielded when iterating backwards
    back: Option<Handle<W, marker::Leaf>>,
}

impl<W> TreeRange<W> {
    /// Creates the range covering all leaves below `parent`.
    ///
    /// Both ends of the range are initialized to the border leaves of the
    /// subtree: the least significant leaf is found by descending the logical
    /// left border of `parent`, the most significant leaf by descending its
    /// logical right border, taking the flip of every visited cluster node into
    /// account.
    ///
    /// `parent` may be a cluster node or a leaf. In the latter case both ends
    /// are that leaf, which happens in practice when `expose` returns the leaf
    /// itself for a vertex of degree one.
    pub fn new(parent: Handle<W, marker::Either>) -> Self {
        Self {
            front: Some(border_leaf(parent.node, true)),
            back: Some(border_leaf(parent.node, false)),
        }
    }

    /// Creates an empty range, which yields no leaves.
    pub fn empty() -> Self {
        Self {
            front: None,
            back: None,
        }
    }
}

/// Descends from `node` along its logical left border if `left` is set, or along
/// its logical right border otherwise, until the leaf ending that border is
/// reached. Every subtree of a top tree has a leaf on each of its borders.
fn border_leaf<W>(mut node: NonNull<Node<W>>, left: bool) -> Handle<W, marker::Leaf> {
    loop {
        match unsafe { node.as_ref().flags().kind() } {
            NodeKind::EdgeLeaf | NodeKind::LabelLeaf => break Handle::new(node),
            NodeKind::Cluster => {
                let internal = unsafe { node.cast::<InternalNode<W>>().as_ref() };
                node = internal.children.get(left != internal.is_flipped());
            }
        }
    }
}

/// Returns the leaf following `leaf` in leaf order, if any.
///
/// Cluster nodes hold no leaf data, so the search ascends until it leaves a
/// cluster through its logical left child; the successor then lies in the
/// logical right child of that cluster, at its leftmost leaf. `None` is
/// returned when the topmost cluster is left through its logical right child,
/// since no leaf follows the rightmost leaf of the tree.
fn successor<W>(leaf: Handle<W, marker::Leaf>) -> Option<Handle<W, marker::Leaf>> {
    let mut node = leaf.forget_type();

    loop {
        let parent = node.parent()?;
        let is_left = node.is_left_child().expect("node must have a parent") != parent.is_flipped();

        if is_left {
            let right = parent.child(parent.is_flipped());

            return Some(border_leaf(right.node, true));
        }

        node = parent.forget_type();
    }
}

/// Returns the leaf preceding `leaf` in leaf order, if any.
///
/// This mirrors `successor`: the search ascends until it leaves a cluster
/// through its logical right child, and then descends to the rightmost leaf of
/// that cluster's logical left child.
fn predecessor<W>(leaf: Handle<W, marker::Leaf>) -> Option<Handle<W, marker::Leaf>> {
    let mut node = leaf.forget_type();

    loop {
        let parent = node.parent()?;
        let is_left = node.is_left_child().expect("node must have a parent") != parent.is_flipped();

        if !is_left {
            let left = parent.child(!parent.is_flipped());

            return Some(border_leaf(left.node, false));
        }

        node = parent.forget_type();
    }
}

impl<W> Iterator for TreeRange<W> {
    type Item = Handle<W, marker::Leaf>;

    /// Yields the leaf at the front of the range and moves the front to the next
    /// leaf. The last leaf of the range is yielded once, exhausting the range.
    ///
    /// The yielded handle is rebuilt from the node of the previous front, as
    /// `successor` consumes the handle it is given.
    fn next(&mut self) -> Option<Self::Item> {
        let front = self.front.take()?;
        let leaf = Handle::new(front.node);
        let is_last = self
            .back
            .as_ref()
            .is_some_and(|back| back.node == front.node);

        if is_last {
            self.back = None;
        } else {
            self.front = successor(front);
        }

        Some(leaf)
    }
}

impl<W> DoubleEndedIterator for TreeRange<W> {
    /// Yields the leaf at the back of the range and moves the back to the
    /// previous leaf. The first leaf of the range is yielded once, exhausting
    /// the range.
    ///
    /// The yielded handle is rebuilt from the node of the previous back, as
    /// `predecessor` consumes the handle it is given.
    fn next_back(&mut self) -> Option<Self::Item> {
        let back = self.back.take()?;
        let leaf = Handle::new(back.node);
        let is_first = self
            .front
            .as_ref()
            .is_some_and(|front| front.node == back.node);

        if is_first {
            self.front = None;
        } else {
            self.back = predecessor(back);
        }

        Some(leaf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LeafOrInternal, Summary, TopTree, VertexId, expose};
    use std::collections::BTreeSet;
    use std::mem::ManuallyDrop;

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct TestSummary(u64);

    impl Summary for TestSummary {
        fn edge(_e: crate::index::EdgeId) -> Self {
            todo!()
        }

        fn label(_l: crate::index::LabelId) -> Self {
            todo!()
        }

        fn combine(left: &Self, right: &Self, _ctx: &crate::summary::MergeContext) -> Self {
            Self(left.0 + right.0)
        }
    }

    /// Builds a star with `arms` tips around a center, returning the center and
    /// the tips, without any labels.
    ///
    /// The tree is wrapped in a `ManuallyDrop`, because the current `Drop`
    /// implementation of `TopTree` frees every leaf twice and thus aborts the
    /// test process; that is pre-existing breakage unrelated to `TreeRange`.
    fn star(arms: usize) -> (ManuallyDrop<TopTree<TestSummary>>, VertexId, Vec<VertexId>) {
        let mut tree = TopTree::<TestSummary>::new();
        let center = tree.add_vertex();
        let mut tips = Vec::new();

        for _ in 0..arms {
            let tip = tree.add_vertex();
            tree.link(center, tip);
            tips.push(tip);
        }

        (ManuallyDrop::new(tree), center, tips)
    }

    /// Builds a star whose label is attached to the center before the arms are
    /// linked. Linking the arms onto an already labelled center leaves flipped
    /// cluster nodes behind, so the logical order of the children of those
    /// clusters differs from the physical order.
    fn star_with_early_label(
        arms: usize,
    ) -> (ManuallyDrop<TopTree<TestSummary>>, VertexId, Vec<VertexId>) {
        let mut tree = TopTree::<TestSummary>::new();
        let center = tree.add_vertex();
        tree.attach(center);
        let mut tips = Vec::new();

        for _ in 0..arms {
            let tip = tree.add_vertex();
            tree.link(center, tip);
            tips.push(tip);
        }

        (ManuallyDrop::new(tree), center, tips)
    }

    /// The node a leaf handle points to, used to compare and order leaves.
    fn node_of(handle: Handle<TestSummary, marker::Leaf>) -> NonNull<Node<TestSummary>> {
        handle.node
    }

    /// The leaves below `node` in logical order, collected by recursing over the
    /// flipped children of every cluster node. This walks the children only, so
    /// it is independent of the parent chain that `successor` and `predecessor`
    /// ascend.
    fn logical_leaves(
        node: Handle<TestSummary, marker::Either>,
        leaves: &mut Vec<NonNull<Node<TestSummary>>>,
    ) {
        match node.force() {
            LeafOrInternal::Edge(_) | LeafOrInternal::Label(_) => leaves.push(node.node),
            LeafOrInternal::Internal(internal) => {
                for child in internal.flipped_children() {
                    logical_leaves(child.forget_type(), leaves);
                }
            }
        }
    }

    /// Whether `node` is, or has below it, a flipped cluster node.
    fn has_flipped_cluster(node: Handle<TestSummary, marker::Either>) -> bool {
        match node.force() {
            LeafOrInternal::Edge(_) | LeafOrInternal::Label(_) => false,
            LeafOrInternal::Internal(internal) => {
                if internal.is_flipped() {
                    return true;
                }

                for child in internal.flipped_children() {
                    if has_flipped_cluster(child.forget_type()) {
                        return true;
                    }
                }

                false
            }
        }
    }

    #[test]
    fn range_covers_all_leaves() {
        let (mut tree, center, _tips) = star(3);
        tree.attach(center);

        let node = expose(center, &mut tree).expect("center must have a consuming node");
        let leaves: Vec<_> = TreeRange::new(node).collect();

        assert_eq!(leaves.len(), 4, "three edges and one label must be covered");
        assert!(
            leaves
                .iter()
                .any(|leaf| matches!(leaf.force(), LeafOrInternal::Edge(_))),
            "edge leaves must be visited"
        );
        assert!(
            leaves
                .iter()
                .any(|leaf| matches!(leaf.force(), LeafOrInternal::Label(_))),
            "label leaves must be visited"
        );
    }

    #[test]
    fn range_follows_the_logical_leaf_order() {
        let (mut tree, center, _tips) = star(6);
        tree.attach(center);

        let node = expose(center, &mut tree).expect("center must have a consuming node");

        let mut expected = Vec::new();
        logical_leaves(node, &mut expected);
        assert_eq!(expected.len(), 7, "six edges and one label must be covered");

        let forwards: Vec<_> = TreeRange::new(node).map(node_of).collect();
        let mut backwards: Vec<_> = TreeRange::new(node).rev().map(node_of).collect();
        backwards.reverse();

        assert_eq!(
            forwards, expected,
            "forwards iteration must visit every leaf in logical order"
        );
        assert_eq!(
            backwards, expected,
            "backwards iteration must visit every leaf in logical order"
        );
        assert_eq!(
            forwards.iter().copied().collect::<BTreeSet<_>>().len(),
            7,
            "every leaf must be visited exactly once"
        );
    }

    #[test]
    fn range_follows_the_logical_leaf_order_of_flipped_clusters() {
        let (mut tree, center, _tips) = star_with_early_label(4);

        let node = expose(center, &mut tree).expect("center must have a consuming node");
        assert!(
            has_flipped_cluster(node),
            "the tree must contain a flipped cluster node"
        );

        let mut expected = Vec::new();
        logical_leaves(node, &mut expected);
        assert_eq!(
            expected.len(),
            5,
            "four edges and one label must be covered"
        );

        let forwards: Vec<_> = TreeRange::new(node).map(node_of).collect();
        let mut backwards: Vec<_> = TreeRange::new(node).rev().map(node_of).collect();
        backwards.reverse();

        assert_eq!(
            forwards, expected,
            "forwards iteration must visit every leaf in logical order"
        );
        assert_eq!(
            backwards, expected,
            "backwards iteration must visit every leaf in logical order"
        );
    }

    #[test]
    fn backwards_iteration_mirrors_forwards_iteration() {
        let (mut tree, center, _tips) = star(4);
        tree.attach(center);

        let node = expose(center, &mut tree).expect("center must have a consuming node");

        let forwards: Vec<_> = TreeRange::new(node).map(node_of).collect();
        let mut backwards: Vec<_> = TreeRange::new(node).rev().map(node_of).collect();
        backwards.reverse();

        assert_eq!(
            forwards, backwards,
            "backwards iteration must yield the reversed forwards order"
        );
    }

    #[test]
    fn range_from_leaf_handle_covers_a_single_leaf() {
        let (mut tree, _center, tips) = star(1);

        let node = expose(tips[0], &mut tree).expect("tip must have a consuming node");
        assert!(
            matches!(node.force(), LeafOrInternal::Edge(_)),
            "exposing a vertex of degree one returns its leaf"
        );

        let leaves: Vec<_> = TreeRange::new(node).collect();

        assert_eq!(leaves.len(), 1, "a leaf must cover itself only");
        assert_eq!(
            leaves[0].node, node.node,
            "the single leaf must be the leaf the range was built from"
        );
    }

    #[test]
    fn root_handle_range_of_a_single_leaf_has_equal_borders() {
        let (mut tree, _center, tips) = star(1);

        let leaf = expose(tips[0], &mut tree).expect("tip must have a consuming node");
        let root = leaf.root();

        let mut forwards = TreeRange::new(root);
        let mut backwards = TreeRange::new(root);

        assert_eq!(
            forwards.next(),
            backwards.next_back(),
            "the only leaf must be both the first and the last leaf"
        );
        assert_eq!(
            forwards.next(),
            None,
            "the range must be exhausted after its single leaf"
        );
        assert_eq!(
            backwards.next_back(),
            None,
            "the range must be exhausted after its single leaf"
        );
    }

    #[test]
    fn range_from_root_handle_matches_the_leaves_of_the_component() {
        let (mut tree, _center, tips) = star(5);

        let leaf = expose(tips[0], &mut tree).expect("tip must have a consuming node");
        let root = leaf.root();

        let mut expected = Vec::new();
        logical_leaves(root, &mut expected);

        let forwards: Vec<_> = TreeRange::new(root).map(node_of).collect();

        assert_eq!(forwards, expected, "a root handle must cover every leaf");
    }
}
