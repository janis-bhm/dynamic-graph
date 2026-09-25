use std::{
    hash::Hash,
    mem,
    ops::{Index, IndexMut},
};

use indexmap::IndexMap;

use crate::NonMaxUsize;

pub struct Node<V> {
    /// The first edge in the list of edges incident to this node.
    /// This is an index into the `edges` vector of the tree.
    next_edge: usize,
    /// The first label in the list of labels incident to this node.
    /// This is an index into the `labels` vector of the tree.
    next_label: Option<NonMaxUsize>,
    /// weight associated with a vertex.
    pub weight: V,
}

pub struct Edge<W> {
    /// The two endpoints of the edge. These are indices into the `nodes` map of the tree.
    endpoints: Endpoints,
    /// next edge in the list of edges incident to the [first, second] endpoint.
    next: EdgeLinks,
    /// weight associated with an edge.
    pub weight: W,
}

impl<W> Edge<W> {
    #[expect(unused)]
    fn next_for(&self, endpoint: usize) -> usize {
        self.endpoints
            .direction_of(endpoint)
            .map(|dir| self.next[dir])
            .unwrap_or_else(|| {
                panic!(
                    "Node index {} not found in edge endpoints {:?}",
                    endpoint, self.endpoints
                )
            })
    }

    pub fn endpoints(&self) -> (usize, usize) {
        (
            self.endpoints[Direction::Left],
            self.endpoints[Direction::Right],
        )
    }
}

pub struct Label<W> {
    /// A label attached to a single vertex, used to represent non-tree edges or
    /// arbitrary vertex marks.
    ///
    /// Labels are keyed by a globally unique id `L`, so a label can be looked up in
    /// constant time without disturbing the ids of any other label. The vertex's
    /// labels form a singly linked list through `next`.
    pub weight: W,
    node: usize,
    /// next label in the list of labels incident to attache node.
    next: Option<NonMaxUsize>,
}

impl<W> Label<W> {
    pub fn node_id(&self) -> usize {
        self.node
    }
}

/// A tree data structure representing a forest of spanning trees over some
/// graph. Each edge and node can have an associated weight, and nodes can have
/// associated labels which act like leaf edges and may represent non-tree edges
/// in the underlying graph.
///
/// Labels are stored in an [`IndexMap`] keyed by a globally unique id `L`, so a
/// label can be looked up directly by its id and removing one does not disturb
/// the ids of any other label. The label's payload has the same weight type `W`
/// as a tree edge. The labels at a vertex are additionally chained through a
/// singly linked list (`Node::first_label` / `Label::next`) for enumeration.
pub struct Tree<N, L, W = (), V = ()> {
    edges: Vec<Edge<W>>,
    labels: IndexMap<L, Label<W>>,
    nodes: IndexMap<N, Node<V>>,
}

impl<N, W, L, V> Tree<N, L, W, V> {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            labels: IndexMap::new(),
            nodes: IndexMap::new(),
        }
    }

    pub fn add_node(&mut self, node: N, weight: V) -> usize
    where
        N: Eq + Hash,
    {
        if let Some(index) = self.nodes.get_index_of(&node) {
            return index;
        }

        self.nodes.insert(
            node,
            Node {
                weight,
                next_edge: NO_EDGE,
                next_label: None,
            },
        );

        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, u: usize, v: usize, w: W) -> usize {
        let [(_, un), (_, vn)] = self.nodes.get_disjoint_indices_mut([u, v]).unwrap();

        let edge = self.edges.push_idx(Edge {
            weight: w,
            endpoints: Endpoints([u, v]),
            next: EdgeLinks([un.next_edge, vn.next_edge]),
        });

        un.next_edge = edge;
        vn.next_edge = edge;

        edge
    }

    pub fn node(&self, node: &N) -> Option<&Node<V>>
    where
        N: Eq + Hash,
    {
        self.nodes.get(node)
    }

    pub fn node_index(&self, node: usize) -> Option<(&N, &Node<V>)> {
        self.nodes.get_index(node)
    }

    pub fn node_index_of(&self, node: &N) -> Option<usize>
    where
        N: Eq + Hash,
    {
        self.nodes.get_index_of(node)
    }

    /// Returns the weight of the label with the given id.
    pub fn label_weight(&self, label: &L) -> Option<&W>
    where
        L: Eq + Hash,
    {
        self.labels.get(label).map(|l| &l.weight)
    }

    /// Returns a mutable reference to the weight of the label with the given id.
    pub fn label_weight_mut(&mut self, label: &L) -> Option<&mut W>
    where
        L: Eq + Hash,
    {
        self.labels.get_mut(label).map(|l| &mut l.weight)
    }

    pub fn edge_weight(&self, edge: usize) -> Option<&W> {
        self.edges.get(edge).map(|e| &e.weight)
    }

    pub fn edge_weight_mut(&mut self, edge: usize) -> Option<&mut W> {
        self.edges.get_mut(edge).map(|e| &mut e.weight)
    }

    pub fn edge_index(&self, edge: usize) -> Option<&Edge<W>> {
        self.edges.get(edge)
    }

    /// Returns the index of the edge connecting `u` and `v`, if any.
    pub fn edge_index_of(&self, u: usize, v: usize) -> Option<usize> {
        for edge in self.incident_edge_indices(u) {
            let (a, b) = self.edges[edge].endpoints();
            if (a == u && b == v) || (a == v && b == u) {
                return Some(edge);
            }
        }

        None
    }

    /// Returns the endpoints of the edge at `edge`.
    pub fn edge_endpoints(&self, edge: usize) -> Option<(usize, usize)> {
        self.edges.get(edge).map(Edge::endpoints)
    }

    /// Returns the label with the given id.
    pub fn label(&self, label: &L) -> Option<&Label<W>>
    where
        L: Eq + Hash,
    {
        self.labels.get(label)
    }

    pub fn incident_edges_weights(&self, node: usize) -> impl Iterator<Item = &W> + '_ {
        let first = self
            .nodes
            .get_index(node)
            .map(|(_, n)| n.next_edge)
            .unwrap_or(NO_EDGE);

        EdgeWalker {
            edges: &self.edges,
            current_edge: first,
            node,
        }
        .map(|edge| &edge.weight)
    }

    /// Returns an iterator over the weights of the labels incident to `node`.
    pub fn incident_label_weights(&self, node: usize) -> impl Iterator<Item = &W> + '_
    where
        L: Eq + Hash,
    {
        self.incident_label_keys(node)
            .filter_map(|key| self.labels.get(key))
            .map(|label| &label.weight)
    }

    /// Returns an iterator over the indices of the edges incident to `node`.
    pub fn incident_edge_indices(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let first = self
            .nodes
            .get_index(node)
            .map(|(_, n)| n.next_edge)
            .unwrap_or(NO_EDGE);

        EdgeIndexWalker {
            edges: &self.edges,
            current_edge: first,
            node,
        }
    }

    /// Returns an iterator over the indices of the labels incident to `node`.
    pub fn incident_label_indices(&self, node: usize) -> impl Iterator<Item = usize> + '_
    where
        L: Eq + Hash,
    {
        let first = self.nodes.get_index(node).and_then(|(_, n)| n.next_label);

        LabelIndexWalker {
            labels: &self.labels,
            current_label: first,
        }
    }

    /// Returns an iterator over the globally unique ids of the labels incident
    /// to `node`.
    pub fn incident_label_keys(&self, node: usize) -> impl Iterator<Item = &L> + '_
    where
        L: Eq + Hash,
    {
        let first = self.nodes.get_index(node).and_then(|(_, n)| n.next_label);

        LabelKeyWalker {
            labels: &self.labels,
            current: first,
        }
    }

    /// The number of edges and labels incident to `node`.
    pub fn degree(&self, node: usize) -> usize
    where
        L: Eq + Hash,
    {
        self.incident_edge_indices(node).count() + self.incident_label_indices(node).count()
    }

    pub fn is_degree_leq_two(&self, node: usize) -> bool
    where
        L: Eq + Hash,
    {
        self.incident_edge_indices(node).take(3).count()
            + self.incident_label_indices(node).take(3).count()
            <= 2
    }

    /// The number of vertices in the forest.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The number of tree edges in the forest.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The number of labels in the forest.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    pub fn remove_node(&mut self, node: &N) -> Option<(V, SwapResult)>
    where
        N: Eq + Hash,
        L: Eq + Hash,
    {
        let idx = self.nodes.get_index_of(node)?;

        loop {
            let next = self.nodes[idx].next_edge;
            if next == NO_EDGE {
                break;
            }

            let Some(_) = self.remove_edge(next) else {
                panic!("Edge index {} not found in tree", next);
            };
        }

        let node = self.nodes.swap_remove_index(idx).unwrap().1;

        if self.nodes.get_index(idx).is_some() {
            let old_index = self.nodes.len();

            // The node formerly at `old_index` now lives at `idx`, so remap
            // the endpoints of its incident edges and the node of its labels.

            for current in EdgeWalkerMut::from_edge_and_endpoint(
                &mut self.edges,
                self.nodes[idx].next_edge,
                old_index,
            ) {
                current.endpoints.replace(old_index, idx);
            }

            for (_, current) in LabelWalkerMut::from_node(self, idx) {
                current.node = idx;
            }

            Some((
                node.weight,
                SwapResult::Some {
                    prev: old_index,
                    current: idx,
                },
            ))
        } else {
            Some((node.weight, SwapResult::None))
        }
    }

    /// Attaches a label with the globally unique id `key` to `node`.
    ///
    /// # Panics
    ///
    /// Panics if a label with `key` is already attached. Label ids are
    /// required to be globally unique.
    pub fn add_label(&mut self, node: usize, key: L, weight: W)
    where
        L: Eq + Hash,
    {
        assert!(
            !self.labels.contains_key(&key),
            "label id must be globally unique"
        );

        let (_, n) = self.nodes.get_index_mut(node).unwrap();
        let first = mem::replace(&mut n.next_label, NonMaxUsize::new(self.labels.len()));

        self.labels.insert(
            key,
            Label {
                weight,
                node,
                next: first,
            },
        );
    }

    /// Removes the label with the given id, returning its weight.
    ///
    /// The ids of all other labels remain valid. Unlinking the label from the
    /// singly linked list of labels at its vertex takes time linear in the
    /// number of labels attached to that vertex.
    pub fn remove_label(&mut self, key: &L) -> Option<W>
    where
        L: Eq + Hash,
    {
        let label_idx = self.labels.get_index_of(key)?;
        let label = self.labels.swap_remove_index(label_idx).unwrap().1;
        let next = label.next;

        self.fix_label_links(
            label.node,
            unsafe { NonMaxUsize::new_unchecked(label_idx) },
            next,
        );

        Some(label.weight)
    }

    pub fn remove_edge(&mut self, edge: usize) -> Option<(W, SwapResult)> {
        let (node, next) = {
            let e = self.edges.get(edge)?;
            (e.endpoints, e.next)
        };

        self.fix_edge_links(node, edge, next);
        Some(self.swap_remove_edge(edge))
    }

    /// Removes the edge at `idx` from the `edges` vector by swapping it with
    /// the last edge and popping it off, returning the weight of the removed
    /// edge and fixing any links that referenced the swapped-in edge.
    fn swap_remove_edge(&mut self, idx: usize) -> (W, SwapResult) {
        let edge = self.edges.swap_remove(idx);

        match self.edges.get(idx) {
            None => (edge.weight, SwapResult::None),
            Some(e) => {
                self.fix_edge_links(e.endpoints, self.edges.len(), EdgeLinks([idx, idx]));
                (
                    edge.weight,
                    SwapResult::Some {
                        prev: self.edges.len(),
                        current: idx,
                    },
                )
            }
        }
    }

    /// Fix all links referencing `edge` in the edge lists of `nodes` to point to `next` instead.
    ///
    /// `nodes[dir]` is the endpoint whose outgoing link for the removed edge
    /// is `next[dir]`, i.e. `edge.endpoints[dir] == nodes[dir]`.
    fn fix_edge_links(&mut self, nodes: Endpoints, edge: usize, next: EdgeLinks) {
        for dir in Directions {
            let endpoint = nodes[dir];
            let replacement = next[dir];

            let first = self
                .nodes
                .get_index(endpoint)
                .expect("Node index not found in tree")
                .1
                .next_edge;

            if first == edge {
                self.nodes
                    .get_index_mut(endpoint)
                    .expect("Node index not found in tree")
                    .1
                    .next_edge = replacement;
            } else {
                for current in
                    EdgeWalkerMut::from_edge_and_endpoint(&mut self.edges, first, endpoint)
                {
                    let current_dir = current
                        .endpoints
                        .direction_of(endpoint)
                        .expect("Node index not found in edge endpoints");
                    if current.next[current_dir] == edge {
                        current.next[current_dir] = replacement;
                        break;
                    }
                }
            }
        }
    }

    fn fix_label_links(
        &mut self,
        node: usize,
        label_idx: NonMaxUsize,
        next_idx: Option<NonMaxUsize>,
    ) {
        let first = self
            .nodes
            .get_index(node)
            .expect("Node index not found in tree")
            .1
            .next_label
            .unwrap();

        if first == label_idx {
            self.nodes
                .get_index_mut(node)
                .expect("Node index not found in tree")
                .1
                .next_label = next_idx;
        } else {
            for (_, current) in LabelWalkerMut::new(&mut self.labels, first) {
                if current.next == Some(label_idx) {
                    current.next = next_idx;
                    break;
                }
            }
        }
    }
}

impl<N, L, W, V> Default for Tree<N, L, W, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub enum SwapResult {
    None,
    Some {
        /// Previous index of the swapped element.
        prev: usize,
        /// New index of the swapped element.
        current: usize,
    },
}

const NO_EDGE: usize = usize::MAX;

pub struct LabelKeyWalker<'a, L, W> {
    labels: &'a IndexMap<L, Label<W>>,
    current: Option<NonMaxUsize>,
}

impl<'a, L, W> LabelKeyWalker<'a, L, W> {
    fn from_node<N, V>(tree: &'a Tree<N, L, W, V>, node_index: usize) -> Self {
        let node = tree
            .node_index(node_index)
            .expect("Node index not found in tree")
            .1;
        Self {
            labels: &tree.labels,
            current: node.next_label,
        }
    }
}

impl<'a, L, W> Iterator for LabelKeyWalker<'a, L, W> {
    type Item = &'a L;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current.map(NonMaxUsize::get) else {
            return None;
        };

        let (key, label) = self
            .labels
            .get_index(label_index)
            .expect("Label index not found in tree");
        self.current = label.next;

        Some(key)
    }
}

pub struct LabelWalker<'a, W> {
    labels: &'a Vec<Label<W>>,
    current_label: Option<NonMaxUsize>,
}

impl<'a, W> LabelWalker<'a, W> {
    fn new(labels: &'a Vec<Label<W>>, first_label: usize) -> Self {
        Self {
            labels,
            current_label: NonMaxUsize::new(first_label),
        }
    }
}

impl<'a, W> Iterator for LabelWalker<'a, W> {
    type Item = &'a Label<W>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current_label.map(NonMaxUsize::get) else {
            return None;
        };

        let label = self
            .labels
            .get(label_index)
            .expect("Label index not found in tree");
        self.current_label = label.next;

        Some(label)
    }
}

pub struct LabelWalkerMut<'a, W, L> {
    labels: &'a mut IndexMap<L, Label<W>>,
    current_label: Option<NonMaxUsize>,
}

impl<'a, W, L> LabelWalkerMut<'a, W, L> {
    fn new(labels: &'a mut IndexMap<L, Label<W>>, first_label: NonMaxUsize) -> Self {
        Self {
            labels,
            current_label: Some(first_label),
        }
    }

    fn from_node<N, V>(tree: &'a mut Tree<N, L, W, V>, node_index: usize) -> Self {
        let node = tree
            .node_index(node_index)
            .expect("Node index not found in tree")
            .1;
        Self {
            current_label: node.next_label,
            labels: &mut tree.labels,
        }
    }
}

impl<'a, W, L> Iterator for LabelWalkerMut<'a, W, L> {
    type Item = (&'a L, &'a mut Label<W>);

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current_label.map(NonMaxUsize::get) else {
            return None;
        };

        let (key, label) = self
            .labels
            .get_index_mut(label_index)
            .expect("Label index not found in tree");
        self.current_label = label.next;

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of labels mutable for 'a, and we guarantee that we will not
        // return the same label twice in this iterator.
        let label = unsafe { &mut *(label as *mut Label<W>) };
        let key = unsafe { &*(key as *const L) };
        Some((key, label))
    }
}

#[allow(dead_code)]
struct EdgeWalker<'a, W> {
    edges: &'a Vec<Edge<W>>,
    current_edge: usize,
    node: usize,
}

impl<'a, W> EdgeWalker<'a, W> {
    #[allow(dead_code)]
    fn from_node<N, L, V>(tree: &'a Tree<N, L, W, V>, node_index: usize) -> Self {
        let node = tree
            .node_index(node_index)
            .expect("Node index not found in tree")
            .1;
        Self {
            edges: &tree.edges,
            current_edge: node.next_edge,
            node: node_index,
        }
    }
}

impl<'a, W> Iterator for EdgeWalker<'a, W> {
    type Item = &'a Edge<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_edge == NO_EDGE {
            return None;
        }

        let edge_index = self.current_edge;
        let edge = self
            .edges
            .get(edge_index)
            .expect("Edge index not found in tree");
        let direction = edge
            .endpoints
            .direction_of(self.node)
            .expect("Node index not found in edge endpoints");
        self.current_edge = edge.next[direction];

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of edges mutable for 'a, and we guarantee that we will not
        // return the same edge twice in this iterator.
        let edge = unsafe { &*(edge as *const Edge<W>) };
        Some(edge)
    }
}

struct EdgeIndexWalker<'a, W> {
    edges: &'a Vec<Edge<W>>,
    current_edge: usize,
    node: usize,
}

impl<'a, W> Iterator for EdgeIndexWalker<'a, W> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_edge == NO_EDGE {
            return None;
        }

        let edge_index = self.current_edge;
        let edge = self
            .edges
            .get(edge_index)
            .expect("Edge index not found in tree");
        let direction = edge
            .endpoints
            .direction_of(self.node)
            .expect("Node index not found in edge endpoints");
        self.current_edge = edge.next[direction];

        Some(edge_index)
    }
}

struct LabelIndexWalker<'a, L, W> {
    labels: &'a IndexMap<L, Label<W>>,
    current_label: Option<NonMaxUsize>,
}

impl<'a, L, W> Iterator for LabelIndexWalker<'a, L, W> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current_label.map(NonMaxUsize::get) else {
            return None;
        };
        let (_, label) = self
            .labels
            .get_index(label_index)
            .expect("Label index not found in tree");
        self.current_label = label.next;

        Some(label_index)
    }
}

struct EdgeWalkerMut<'a, W> {
    edges: &'a mut Vec<Edge<W>>,
    current_edge: usize,
    node_index: usize,
}

impl<'a, W> EdgeWalkerMut<'a, W> {
    fn from_edge_and_endpoint(
        edges: &'a mut Vec<Edge<W>>,
        edge_index: usize,
        endpoint: usize,
    ) -> Self {
        Self {
            edges,
            current_edge: edge_index,
            node_index: endpoint,
        }
    }
}

impl<'a, W> Iterator for EdgeWalkerMut<'a, W> {
    type Item = &'a mut Edge<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_edge == NO_EDGE {
            return None;
        }

        let edge_index = self.current_edge;
        let edge = self
            .edges
            .get_mut(edge_index)
            .expect("Edge index not found in tree");
        let direction = edge
            .endpoints
            .direction_of(self.node_index)
            .expect("Node index not found in edge endpoints");
        self.current_edge = edge.next[direction];

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of edges mutable for 'a, and we guarantee that we will not
        // return the same edge twice in this iterator.
        let edge = unsafe { &mut *(edge as *mut Edge<W>) };
        Some(edge)
    }
}

struct Directions;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Endpoints([usize; 2]);

impl Endpoints {
    fn direction_of(&self, node: usize) -> Option<Direction> {
        if self.0[0] == node {
            Some(Direction::Left)
        } else if self.0[1] == node {
            Some(Direction::Right)
        } else {
            None
        }
    }

    fn replace(&mut self, old: usize, new: usize) {
        if self.0[0] == old {
            self.0[0] = new;
        } else if self.0[1] == old {
            self.0[1] = new;
        } else {
            panic!("Node index {} not found in edge endpoints {:?}", old, self);
        }
    }

    #[cfg(test)]
    fn contains(&self, node: usize) -> bool {
        self.0[0] == node || self.0[1] == node
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct EdgeLinks([usize; 2]);

impl Index<Direction> for Endpoints {
    type Output = usize;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::Left => &self.0[0],
            Direction::Right => &self.0[1],
        }
    }
}

impl IndexMut<Direction> for Endpoints {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::Left => &mut self.0[0],
            Direction::Right => &mut self.0[1],
        }
    }
}

impl Index<Direction> for EdgeLinks {
    type Output = usize;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::Left => &self.0[0],
            Direction::Right => &self.0[1],
        }
    }
}

impl IndexMut<Direction> for EdgeLinks {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::Left => &mut self.0[0],
            Direction::Right => &mut self.0[1],
        }
    }
}

impl IntoIterator for Directions {
    type Item = Direction;

    type IntoIter = std::array::IntoIter<Direction, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [Direction::Left, Direction::Right].into_iter()
    }
}

trait VecExt {
    type Item;
    fn push_idx(&mut self, item: Self::Item) -> usize;

    #[expect(unused)]
    /// Removes the element at `index` from the vector by swapping it with the last element and popping it off, returning the value and an optional mutable reference to the new element at the swapped-out index, if it exists.
    ///
    /// For example, `swap_remove_idx(1)` on `[1, 2, 3]` returns `(2, Some(&mut 3))`.
    fn swap_remove_idx(&mut self, index: usize) -> (Self::Item, Option<&mut Self::Item>);
}
impl<T> VecExt for Vec<T> {
    type Item = T;

    fn push_idx(&mut self, item: Self::Item) -> usize {
        let idx = self.len();
        self.push(item);
        idx
    }

    fn swap_remove_idx(&mut self, index: usize) -> (Self::Item, Option<&mut Self::Item>) {
        let t = self.swap_remove(index);

        (t, self.get_mut(index))
    }
}

#[cfg(test)]
mod tests;
