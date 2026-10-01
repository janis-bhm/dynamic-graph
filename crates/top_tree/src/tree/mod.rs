use core::fmt;
use std::{
    mem,
    ops::{Index, IndexMut},
};

use crate::{Generation, NonMaxU32, NonMaxUsize};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct VertexId {
    index: NonMaxU32,
    generation: Generation,
}

impl fmt::Display for VertexId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.index.get(), self.generation.0)
    }
}

impl VertexId {
    fn new(index: usize, generation: Generation) -> Self {
        Self {
            index: NonMaxU32::new(u32::try_from(index).expect("out of bounds index")).unwrap(),
            generation,
        }
    }

    pub fn index(&self) -> usize {
        self.index.get() as usize
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn swap(&mut self, swap: SwapResult<Self>) {
        if let Some(new) = swap.try_swap(*self) {
            *self = new;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelId {
    index: NonMaxU32,
    generation: Generation,
}

impl fmt::Display for LabelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "l{}.{}", self.index.get(), self.generation.0)
    }
}

impl LabelId {
    fn new(index: usize, generation: Generation) -> Self {
        Self {
            index: NonMaxU32::new(u32::try_from(index).expect("out of bounds index")).unwrap(),
            generation,
        }
    }

    pub fn index(&self) -> usize {
        self.index.get() as usize
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn swap(&mut self, swap: SwapResult<Self>) {
        if let Some(new) = swap.try_swap(*self) {
            *self = new;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeId {
    index: NonMaxU32,
    generation: Generation,
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}.{}", self.index.get(), self.generation.0)
    }
}

impl EdgeId {
    fn new(index: usize, generation: Generation) -> Self {
        Self {
            index: NonMaxU32::new(u32::try_from(index).unwrap()).unwrap(),
            generation,
        }
    }

    pub fn index(&self) -> usize {
        self.index.get() as usize
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn swap(&mut self, swap: SwapResult<Self>) {
        if let Some(new) = swap.try_swap(*self) {
            *self = new;
        }
    }
}

pub struct Node<V> {
    /// The first edge in the list of edges incident to this node.
    /// This is an index into the `edges` vector of the tree.
    next_edge: Option<NonMaxUsize>,
    /// The first label in the list of labels incident to this node.
    /// This is an index into the `labels` vector of the tree.
    next_label: Option<NonMaxUsize>,
    generation: Generation,
    /// weight associated with a vertex.
    pub weight: V,
}

pub struct Edge<W> {
    /// The two endpoints of the edge. These are indices into the `nodes` map of the tree.
    endpoints: Endpoints,
    /// next edge in the list of edges incident to the [first, second] endpoint.
    next: EdgeLinks,
    generation: Generation,
    /// weight associated with an edge.
    pub weight: W,
}

pub struct Label<W> {
    node: usize,
    /// next label in the list of labels incident to attache node.
    next: Option<NonMaxUsize>,
    generation: Generation,
    /// A label attached to a single vertex, used to represent non-tree edges or
    /// arbitrary vertex marks.
    pub weight: W,
}

impl<W> Edge<W> {
    #[expect(unused)]
    fn next_for(&self, endpoint: usize) -> Option<NonMaxUsize> {
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

impl<W> Label<W> {
    pub fn node_index(&self) -> usize {
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
pub struct Tree<V, E, L> {
    pub(crate) nodes: Vec<Node<V>>,
    pub(crate) labels: Vec<Label<L>>,
    pub(crate) edges: Vec<Edge<E>>,
    generation: Generation,
}

impl<V, E, L> Tree<V, E, L> {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            labels: Vec::new(),
            nodes: Vec::new(),
            generation: Generation::default(),
        }
    }

    pub fn add_node(&mut self, weight: V) -> VertexId {
        let generation = self.generation.current();
        let index = self.nodes.push_idx(Node {
            weight,
            next_edge: None,
            next_label: None,
            generation,
        });

        VertexId::new(index, generation)
    }

    pub fn add_edge(&mut self, u: VertexId, v: VertexId, w: E) -> EdgeId {
        let [un, vn] = self.nodes.get_disjoint_mut([u.index(), v.index()]).unwrap();

        if un.generation != u.generation() || vn.generation != v.generation() {
            panic!("Node generation mismatch");
        }

        let generation = self.generation.current();
        let index = self.edges.push_idx(Edge {
            weight: w,
            endpoints: Endpoints([u.index(), v.index()]),
            next: EdgeLinks([un.next_edge, vn.next_edge]),
            generation,
        });

        un.next_edge = NonMaxUsize::new(index);
        vn.next_edge = NonMaxUsize::new(index);

        EdgeId::new(index, generation)
    }

    pub fn node(&self, node: VertexId) -> Option<&Node<V>> {
        self.nodes
            .get(node.index())
            .filter(|n| n.generation == node.generation())
    }

    /// Returns the weight of the label with the given id.
    pub fn label_weight(&self, label: LabelId) -> Option<&L> {
        self.labels
            .get(label.index())
            .filter(|l| l.generation == label.generation())
            .map(|l| &l.weight)
    }

    /// Returns a mutable reference to the weight of the label with the given id.
    pub fn label_weight_mut(&mut self, label: LabelId) -> Option<&mut L> {
        self.labels
            .get_mut(label.index())
            .filter(|l| l.generation == label.generation())
            .map(|l| &mut l.weight)
    }

    pub fn edge_weight(&self, edge: EdgeId) -> Option<&E> {
        self.edges
            .get(edge.index())
            .filter(|e| e.generation == edge.generation())
            .map(|e| &e.weight)
    }

    pub fn edge_weight_mut(&mut self, edge: EdgeId) -> Option<&mut E> {
        self.edges
            .get_mut(edge.index())
            .filter(|e| e.generation == edge.generation())
            .map(|e| &mut e.weight)
    }

    /// Returns the index of the edge connecting `u` and `v`, if any.
    pub fn edge_index_of(&self, u: VertexId, v: VertexId) -> Option<EdgeId> {
        for edge in self.incident_edge_indices(u) {
            let (a, b) = self.edges[edge].endpoints();
            let (u, v) = (u.index(), v.index());
            if (a == u && b == v) || (a == v && b == u) {
                return Some(EdgeId::new(edge, self.edges[edge].generation));
            }
        }

        None
    }

    pub(crate) fn vertex_id_from_index(&self, idx: usize) -> VertexId {
        VertexId::new(idx, self.nodes[idx].generation)
    }

    /// Returns the endpoints of the edge at `edge`.
    pub fn edge_endpoints(&self, edge: EdgeId) -> Option<(VertexId, VertexId)> {
        self.edges
            .get(edge.index())
            .filter(|e| e.generation == edge.generation())
            .map(Edge::endpoints)
            .map(|(u, v)| (self.vertex_id_from_index(u), self.vertex_id_from_index(v)))
    }

    /// Returns the label with the given id.
    pub fn label(&self, label: LabelId) -> Option<&Label<L>> {
        self.labels
            .get(label.index())
            .filter(|l| l.generation == label.generation())
    }

    pub fn label_vertex(&self, label: LabelId) -> Option<VertexId> {
        self.labels
            .get(label.index())
            .filter(|l| l.generation == label.generation())
            .map(|l| VertexId::new(l.node, self.nodes[l.node].generation))
    }

    pub fn incident_edge_weights(&self, node: VertexId) -> impl Iterator<Item = &E> + '_ {
        let first = self
            .nodes
            .get(node.index())
            .filter(|n| n.generation == node.generation())
            .and_then(|n| n.next_edge);

        EdgeWalker {
            edges: &self.edges,
            current_edge: first,
            node: node.index(),
        }
        .map(|edge| &edge.weight)
    }

    /// Returns an iterator over the weights of the labels incident to `node`.
    pub fn incident_label_weights(&self, node: VertexId) -> impl Iterator<Item = &L> + '_ {
        self.incident_label_indices(node)
            .filter_map(|key| self.labels.get(key))
            .map(|label| &label.weight)
    }

    /// Returns an iterator over the indices of the edges incident to `node`.
    pub fn incident_edge_indices(&self, node: VertexId) -> impl Iterator<Item = usize> + '_ {
        let first = self
            .nodes
            .get(node.index())
            .filter(|n| n.generation == node.generation())
            .and_then(|n| n.next_edge);

        EdgeIndexWalker {
            edges: &self.edges,
            current_edge: first,
            node: node.index(),
        }
    }

    /// Returns an iterator over the ids of the labels incident to `node`.
    pub fn incident_edge_ids(&self, node: VertexId) -> impl Iterator<Item = EdgeId> + '_ {
        let first = self.nodes.get(node.index()).and_then(|n| n.next_edge);

        EdgeIndexWalker {
            edges: &self.edges,
            current_edge: first,
            node: node.index(),
        }
        .map(|e| EdgeId::new(e, self.edges[e].generation))
    }

    /// Returns an iterator over the indices of the labels incident to `node`.
    pub fn incident_label_indices(&self, node: VertexId) -> impl Iterator<Item = usize> + '_ {
        let first = self.nodes.get(node.index()).and_then(|n| n.next_label);

        LabelIndexWalker {
            labels: &self.labels,
            current_label: first,
        }
        .map(|label| label.index())
    }

    /// Returns an iterator over the ids of the labels incident to `node`.
    pub fn incident_label_ids(&self, node: VertexId) -> impl Iterator<Item = LabelId> + '_ {
        let first = self.nodes.get(node.index()).and_then(|n| n.next_label);

        LabelIndexWalker {
            labels: &self.labels,
            current_label: first,
        }
    }

    /// Returns an iterator over the globally unique ids of the labels incident
    /// to `node`.
    pub fn incident_label_keys(&self, node: VertexId) -> impl Iterator<Item = &L> + '_ {
        let first = self.nodes.get(node.index()).and_then(|n| n.next_label);

        LabelWalker {
            labels: &self.labels,
            current: first,
        }
    }

    /// The number of edges and labels incident to `node`.
    pub fn degree(&self, node: VertexId) -> usize {
        self.incident_edge_indices(node).count() + self.incident_label_indices(node).count()
    }

    pub fn is_degree_leq_two(&self, node: VertexId) -> bool {
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

    pub fn remove_node<U>(
        &mut self,
        id: VertexId,
        mut ctx: U,
        mut edge_swapped: impl FnMut(&mut U, &Self, SwapResult<EdgeId>),
        mut label_swapped: impl FnMut(&mut U, &Self, SwapResult<LabelId>),
    ) -> Option<(V, SwapResult<VertexId>)> {
        if self.nodes.get(id.index()).map(|n| n.generation) != Some(id.generation) {
            return None;
        }

        while let Some(next) = self.nodes[id.index()].next_edge {
            let Some((_, swap)) = self.remove_edge_index(next.get()) else {
                panic!("Edge index {} not found in tree", next);
            };

            edge_swapped(&mut ctx, self, swap);
        }

        while let Some(next) = self.nodes[id.index()].next_label {
            let Some((_, swap)) = self.remove_label_index(next.get()) else {
                panic!("Label index {} not found in tree", next);
            };

            label_swapped(&mut ctx, self, swap);
        }

        self.generation.increment();

        let node = self.nodes.swap_remove(id.index());

        if let Some(n) = self.nodes.get_mut(id.index()) {
            let generation = mem::replace(&mut n.generation, self.generation);
            let prev = VertexId::new(self.nodes.len(), generation);
            let current = VertexId::new(id.index(), self.generation);

            // The node formerly at `prev` now lives at `current`, so remap
            // the endpoints of its incident edges and the node of its labels.
            for edge in EdgeWalkerMut::from_edge_and_endpoint(
                &mut self.edges,
                self.nodes[current.index()].next_edge,
                prev.index(),
            ) {
                edge.endpoints.replace(prev.index(), current.index());
            }

            for label in LabelWalkerMut::from_node(self, current.index()) {
                label.node = id.index();
            }

            Some((node.weight, SwapResult::Some { prev, current }))
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
    pub fn add_label(&mut self, node: VertexId, weight: L) -> LabelId {
        let n = self.nodes.get_mut(node.index()).unwrap();
        assert_eq!(
            n.generation,
            node.generation(),
            "Node generation mismatch: expected {:?}, got {:?}",
            n.generation,
            node.generation()
        );

        let first = mem::replace(&mut n.next_label, NonMaxUsize::new(self.labels.len()));

        let generation = self.generation.current();
        let index = self.labels.push_idx(Label {
            weight,
            node: node.index(),
            next: first,
            generation,
        });

        LabelId::new(index, generation)
    }

    // Internal method to remove a label by its index in the `labels` vector.
    pub fn remove_label_index(&mut self, id: usize) -> Option<(L, SwapResult<LabelId>)> {
        let label = self.labels.get(id)?;

        let next = label.next;

        self.fix_label_links(label.node, unsafe { NonMaxUsize::new_unchecked(id) }, next);

        Some(self.swap_remove_label(id))
    }

    /// Removes the label with the given id, returning its weight.
    ///
    /// The ids of all other labels remain valid. Unlinking the label from the
    /// singly linked list of labels at its vertex takes time linear in the
    /// number of labels attached to that vertex.
    pub fn remove_label(&mut self, id: LabelId) -> Option<(L, SwapResult<LabelId>)> {
        if self.labels.get(id.index()).map(|l| l.generation) != Some(id.generation) {
            return None;
        }

        self.generation.increment();

        self.remove_label_index(id.index())
    }

    fn swap_remove_label(&mut self, idx: usize) -> (L, SwapResult<LabelId>) {
        let label = self.labels.swap_remove(idx);

        match self.labels.get_mut(idx) {
            None => (label.weight, SwapResult::None),
            Some(l) => {
                let node = l.node;
                let generation = mem::replace(&mut l.generation, self.generation);

                self.fix_label_links(
                    node,
                    unsafe { NonMaxUsize::new_unchecked(self.labels.len()) },
                    unsafe { Some(NonMaxUsize::new_unchecked(idx)) },
                );

                let prev = LabelId::new(self.labels.len(), generation);
                let current = LabelId::new(idx, self.generation);

                (label.weight, SwapResult::Some { prev, current })
            }
        }
    }

    fn remove_edge_index(&mut self, idx: usize) -> Option<(E, SwapResult<EdgeId>)> {
        let (node, next) = {
            let e = self.edges.get(idx)?;
            (e.endpoints, e.next)
        };

        self.fix_edge_links(node, idx, next);
        Some(self.swap_remove_edge(idx))
    }

    pub fn remove_edge(&mut self, edge: EdgeId) -> Option<(E, SwapResult<EdgeId>)> {
        if self.edges.get(edge.index()).map(|e| e.generation) != Some(edge.generation()) {
            return None;
        }

        self.generation.increment();

        self.remove_edge_index(edge.index())
    }

    /// Removes the edge at `idx` from the `edges` vector by swapping it with
    /// the last edge and popping it off, returning the weight of the removed
    /// edge and fixing any links that referenced the swapped-in edge.
    fn swap_remove_edge(&mut self, idx: usize) -> (E, SwapResult<EdgeId>) {
        let edge = self.edges.swap_remove(idx);

        match self.edges.get_mut(idx) {
            None => (edge.weight, SwapResult::None),
            Some(e) => {
                let idx = unsafe { NonMaxUsize::new_unchecked(idx) };
                let nodes = e.endpoints;
                let generation = mem::replace(&mut e.generation, self.generation);

                self.fix_edge_links(nodes, self.edges.len(), EdgeLinks([Some(idx), Some(idx)]));

                let prev = EdgeId::new(self.edges.len(), generation);
                let current = EdgeId::new(idx.get(), self.generation);

                (edge.weight, SwapResult::Some { prev, current })
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

            if let Some(first) = self
                .nodes
                .get(endpoint)
                .expect("Node index not found in tree")
                .next_edge
            {
                if first.get() == edge {
                    self.nodes
                        .get_mut(endpoint)
                        .expect("Node index not found in tree")
                        .next_edge = replacement;
                } else {
                    for current in EdgeWalkerMut::from_edge_and_endpoint(
                        &mut self.edges,
                        Some(first),
                        endpoint,
                    ) {
                        let current_dir = current
                            .endpoints
                            .direction_of(endpoint)
                            .expect("Node index not found in edge endpoints");
                        if current.next[current_dir] == NonMaxUsize::new(edge) {
                            current.next[current_dir] = replacement;
                            break;
                        }
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
        if let Some(first) = self
            .nodes
            .get(node)
            .expect("Node index not found in tree")
            .next_label
        {
            if first == label_idx {
                self.nodes
                    .get_mut(node)
                    .expect("Node index not found in tree")
                    .next_label = next_idx;
            } else {
                for label in LabelWalkerMut::new(&mut self.labels, first) {
                    if label.next == Some(label_idx) {
                        label.next = next_idx;
                        break;
                    }
                }
            }
        }
    }
}

impl<V, E, L> Default for Tree<V, E, L> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapResult<Id> {
    None,
    Some {
        /// Previous index of the swapped element.
        prev: Id,
        /// New index of the swapped element.
        current: Id,
    },
}

impl<Id> SwapResult<Id> {
    fn try_swap(&self, id: Id) -> Option<Id>
    where
        Id: PartialEq + Copy,
    {
        match self {
            SwapResult::None => None,
            SwapResult::Some { prev, current } => {
                if *prev == id {
                    Some(*current)
                } else {
                    None
                }
            }
        }
    }
}

pub struct LabelWalker<'a, L> {
    labels: &'a [Label<L>],
    current: Option<NonMaxUsize>,
}

impl<'a, L> Iterator for LabelWalker<'a, L> {
    type Item = &'a L;

    fn next(&mut self) -> Option<Self::Item> {
        let label_index = self.current.map(NonMaxUsize::get)?;

        let label = self
            .labels
            .get(label_index)
            .expect("Label index not found in tree");
        self.current = label.next;

        Some(&label.weight)
    }
}

pub struct LabelWalkerMut<'a, L> {
    labels: &'a mut Vec<Label<L>>,
    current_label: Option<NonMaxUsize>,
}

impl<'a, L> LabelWalkerMut<'a, L> {
    fn new(labels: &'a mut Vec<Label<L>>, first_label: NonMaxUsize) -> Self {
        Self {
            labels,
            current_label: Some(first_label),
        }
    }

    fn from_node<V, E>(tree: &'a mut Tree<V, E, L>, node_index: usize) -> Self {
        let node = tree
            .nodes
            .get(node_index)
            .expect("Node index not found in tree");
        Self {
            current_label: node.next_label,
            labels: &mut tree.labels,
        }
    }
}

impl<'a, L> Iterator for LabelWalkerMut<'a, L> {
    type Item = &'a mut Label<L>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current_label.map(NonMaxUsize::get) else {
            return None;
        };

        let label = self
            .labels
            .get_mut(label_index)
            .expect("Label index not found in tree");
        self.current_label = label.next;

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of labels mutable for 'a, and we guarantee that we will not
        // return the same label twice in this iterator.
        let label = unsafe { &mut *(label as *mut Label<L>) };
        Some(label)
    }
}

#[allow(dead_code)]
struct EdgeWalker<'a, E> {
    edges: &'a Vec<Edge<E>>,
    current_edge: Option<NonMaxUsize>,
    node: usize,
}

impl<'a, E> EdgeWalker<'a, E> {
    #[allow(dead_code)]
    fn from_node<V, L>(tree: &'a Tree<V, E, L>, node_index: usize) -> Self {
        let node = tree
            .nodes
            .get(node_index)
            .expect("Node index not found in tree");
        Self {
            edges: &tree.edges,
            current_edge: node.next_edge,
            node: node_index,
        }
    }
}

impl<'a, E> Iterator for EdgeWalker<'a, E> {
    type Item = &'a Edge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(edge_index) = self.current_edge else {
            return None;
        };

        let edge = self
            .edges
            .get(edge_index.get())
            .expect("Edge index not found in tree");

        let direction = edge
            .endpoints
            .direction_of(self.node)
            .expect("Node index not found in edge endpoints");
        self.current_edge = edge.next[direction];

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of edges mutable for 'a, and we guarantee that we will not
        // return the same edge twice in this iterator.
        let edge = unsafe { &*(edge as *const Edge<E>) };
        Some(edge)
    }
}

struct EdgeIndexWalker<'a, W> {
    edges: &'a Vec<Edge<W>>,
    current_edge: Option<NonMaxUsize>,
    node: usize,
}

impl<'a, W> Iterator for EdgeIndexWalker<'a, W> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(edge_index) = self.current_edge else {
            return None;
        };

        let edge = self
            .edges
            .get(edge_index.get())
            .expect("Edge index not found in tree");
        let direction = edge
            .endpoints
            .direction_of(self.node)
            .expect("Node index not found in edge endpoints");
        self.current_edge = edge.next[direction];

        Some(edge_index.get())
    }
}

struct LabelIndexWalker<'a, L> {
    labels: &'a [Label<L>],
    current_label: Option<NonMaxUsize>,
}

impl<'a, L> Iterator for LabelIndexWalker<'a, L> {
    type Item = LabelId;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(label_index) = self.current_label.map(NonMaxUsize::get) else {
            return None;
        };

        let label = self
            .labels
            .get(label_index)
            .expect("Label index not found in tree");
        let id = LabelId::new(label_index, label.generation);
        self.current_label = label.next;

        Some(id)
    }
}

struct EdgeWalkerMut<'a, W> {
    edges: &'a mut Vec<Edge<W>>,
    current_edge: Option<NonMaxUsize>,
    node_index: usize,
}

impl<'a, W> EdgeWalkerMut<'a, W> {
    fn from_edge_and_endpoint(
        edges: &'a mut Vec<Edge<W>>,
        edge_index: Option<NonMaxUsize>,
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
        let Some(edge_index) = self.current_edge else {
            return None;
        };

        let edge = self
            .edges
            .get_mut(edge_index.get())
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
struct EdgeLinks([Option<NonMaxUsize>; 2]);

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
    type Output = Option<NonMaxUsize>;

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
