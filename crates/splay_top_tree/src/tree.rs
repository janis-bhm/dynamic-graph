use std::{
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
    ptr,
};

use crate::{impl_id, index::Generation};

impl_id! {
    pub struct VertexId #v,
    pub struct EdgeId #e,
    pub struct LabelId #l,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapResult<T> {
    None,
    Swapped { prev: T, next: T },
}

pub trait Swappable: Sized {
    fn try_swap(&mut self, other: SwapResult<Self>);
}

impl<T> Swappable for T
where
    T: PartialEq + Copy,
{
    fn try_swap(&mut self, other: SwapResult<Self>) {
        match other {
            SwapResult::None => {}
            SwapResult::Swapped { prev, next } => {
                if *self == prev {
                    *self = next;
                }
            }
        }
    }
}

/// A tree data structure representing a forest of trees.
/// Vertices may have associated labels, which act like leaf edges.
/// Vertices, edges and labels may have an associated weight.
pub struct Tree<V, E, L> {
    pub(crate) vertices: Vec<Vertex<V>>,
    pub(crate) edges: Vec<Edge<E>>,
    pub(crate) labels: Vec<Label<L>>,
    generation: Generation,
}

pub(crate) struct Vertex<V> {
    next_edge: Option<EdgeId>,
    next_label: Option<LabelId>,
    #[cfg(debug_assertions)]
    generation: Generation,
    pub weight: V,
}

pub(crate) struct Edge<E> {
    endpoints: Endpoints,
    next: EdgeLinks,
    #[cfg(debug_assertions)]
    generation: Generation,
    pub weight: E,
}

pub(crate) struct Label<L> {
    vertex: VertexId,
    next: Option<LabelId>,
    #[cfg(debug_assertions)]
    generation: Generation,
    pub weight: L,
}

impl<V, E, L> Default for Tree<V, E, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, E, L> Tree<V, E, L> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            labels: Vec::new(),
            generation: Generation::default(),
        }
    }

    pub fn vertex(&self, id: VertexId) -> &Vertex<V> {
        self.assert_valid_vid(id);
        &self.vertices[id.index()]
    }

    pub fn edge(&self, id: EdgeId) -> &Edge<E> {
        assert!(self.is_valid_edge(id), "EdgeId {:?} is invalid", id);
        &self.edges[id.index()]
    }

    pub fn label(&self, id: LabelId) -> &Label<L> {
        assert!(self.is_valid_label(id), "LabelId {:?} is invalid", id);
        &self.labels[id.index()]
    }

    /// Adds a new vertex to the tree with the given weight and returns its `VertexId`.
    pub fn add_vertex(&mut self, weight: V) -> VertexId {
        let vertex_id = VertexId::new_from_usize(self.vertices.len(), self.generation.current());
        self.vertices.push(Vertex {
            next_edge: None,
            next_label: None,
            #[cfg(debug_assertions)]
            generation: self.generation.current(),
            weight,
        });
        vertex_id
    }

    /// Adds a new edge between the given vertices with the given weight and
    /// returns its `EdgeId`.  The vertices must be distinct and valid, and the
    /// caller is responsible for ensuring that the two vertices are not already
    /// part of the same tree.
    ///
    /// # Panics
    /// Panics if the vertices are the same or if either vertex is invalid.
    pub fn add_edge(&mut self, v: VertexId, w: VertexId, weight: E) -> EdgeId {
        self.assert_valid_vid(v);
        self.assert_valid_vid(w);

        assert_ne!(v, w, "Cannot add an edge between the same vertex");

        let generation = self.generation.current();
        let edge_id = EdgeId::new_from_usize(self.edges.len(), generation);

        let [nu, nv] = self
            .vertices
            .get_disjoint_mut([v.index(), w.index()])
            .expect("Vertices must be distinct");

        let [next_v, next_w] = {
            [
                mem::replace(&mut nu.next_edge, Some(edge_id)),
                mem::replace(&mut nv.next_edge, Some(edge_id)),
            ]
        };

        self.edges.push(Edge {
            endpoints: Endpoints([v, w]),
            next: EdgeLinks([next_v, next_w]),
            #[cfg(debug_assertions)]
            generation,
            weight,
        });

        edge_id
    }

    /// Adds a new label to the given vertex with the given weight and returns its `LabelId`.
    ///
    /// # Panics
    /// Panics if the vertex is invalid.
    pub fn add_label(&mut self, vertex: VertexId, weight: L) -> LabelId {
        self.assert_valid_vid(vertex);

        let generation = self.generation.current();
        let label_id = LabelId::new_from_usize(self.labels.len(), generation);

        let vn = &mut self.vertices[vertex.index()];

        let next = mem::replace(&mut vn.next_label, Some(label_id));

        self.labels.push(Label {
            vertex,
            next,
            #[cfg(debug_assertions)]
            generation,
            weight,
        });

        label_id
    }

    /// Returns the `EdgeId` of the edge connecting the given vertices, if it exists.
    ///
    /// # Panics
    /// Panics if either vertex is invalid.
    pub fn find_edge_with_endpoints(&self, v: VertexId, w: VertexId) -> Option<EdgeId> {
        self.assert_valid_vid(v);
        self.assert_valid_vid(w);

        for (eid, edge) in EdgeWalker::new_from_vertex(self, v) {
            if edge.endpoints.contains(w) {
                return Some(eid);
            }
        }

        None
    }

    /// Returns the endpoints of the edge with the given `EdgeId`.
    ///
    /// # Panics
    /// Panics if `id` is invalid.
    pub fn edge_endpoints(&self, id: EdgeId) -> Endpoints {
        assert!(self.is_valid_edge(id), "EdgeId {:?} is invalid", id);
        self.edges[id.index()].endpoints
    }

    /// Returns the weight of the vertex with the given `VertexId`.
    ///
    /// # Panics
    /// Panics if `id` is invalid.
    pub fn vertex_weight(&self, id: VertexId) -> &V {
        self.assert_valid_vid(id);
        &self.vertices[id.index()].weight
    }

    pub fn vertex_weight_mut(&mut self, id: VertexId) -> &mut V {
        self.assert_valid_vid(id);
        &mut self.vertices[id.index()].weight
    }

    pub fn edge_weight(&self, id: EdgeId) -> &E {
        assert!(self.is_valid_edge(id), "EdgeId {:?} is invalid", id);
        &self.edges[id.index()].weight
    }

    pub fn edge_weight_mut(&mut self, id: EdgeId) -> &mut E {
        assert!(self.is_valid_edge(id), "EdgeId {:?} is invalid", id);
        &mut self.edges[id.index()].weight
    }

    pub fn label_vertex(&self, id: LabelId) -> VertexId {
        assert!(self.is_valid_label(id), "LabelId {:?} is invalid", id);
        self.labels[id.index()].vertex
    }

    pub fn label_weight(&self, id: LabelId) -> &L {
        assert!(self.is_valid_label(id), "LabelId {:?} is invalid", id);
        &self.labels[id.index()].weight
    }

    pub fn label_weight_mut(&mut self, id: LabelId) -> &mut L {
        assert!(self.is_valid_label(id), "LabelId {:?} is invalid", id);
        &mut self.labels[id.index()].weight
    }

    pub fn is_valid_vertex(&self, id: VertexId) -> bool {
        cfg_select! {
            debug_assertions => self
                .vertices
                .get(id.index())
                .is_some_and(|v| v.generation == id.generation()),
            _ => self.vertices.get(id.index()).is_some(),
        }
    }

    pub fn is_valid_edge(&self, id: EdgeId) -> bool {
        cfg_select! {
            debug_assertions => self
                .edges
                .get(id.index())
                .is_some_and(|e| e.generation == id.generation()),
            _ => self.edges.get(id.index()).is_some(),
        }
    }

    pub fn is_valid_label(&self, id: LabelId) -> bool {
        cfg_select! {
            debug_assertions => self
                .labels
                .get(id.index())
                .is_some_and(|l| l.generation == id.generation()),
            _ => self.labels.get(id.index()).is_some(),
        }
    }

    pub fn incident_label_weights(&self, vertex: VertexId) -> LabelWeights<'_, L> {
        self.assert_valid_vid(vertex);
        LabelWeights::new_from_vertex(self, vertex)
    }

    pub fn incident_edge_weights(&self, vertex: VertexId) -> EdgeWeights<'_, E> {
        self.assert_valid_vid(vertex);
        EdgeWeights::new_from_vertex(self, vertex)
    }

    /// Returns the degree of the vertex with the given `VertexId`, counting both incident edges and labels.
    pub fn degree(&self, vertex: VertexId) -> usize {
        self.assert_valid_vid(vertex);

        self.incident_label_weights(vertex).count() + self.incident_edge_weights(vertex).count()
    }

    /// Returns true if the vertex with the given `VertexId` has degree at least `n`, counting both incident edges and labels.
    pub fn is_at_least_degree_n(&self, vertex: VertexId, n: usize) -> bool {
        self.assert_valid_vid(vertex);
        self.incident_label_weights(vertex)
            .map(|_| ())
            .chain(self.incident_edge_weights(vertex).map(|_| ()))
            .take(n)
            .count()
            >= n
    }

    pub fn is_exactly_degree_n(&self, vertex: VertexId, n: usize) -> bool {
        self.assert_valid_vid(vertex);
        self.incident_label_weights(vertex)
            .map(|_| ())
            .chain(self.incident_edge_weights(vertex).map(|_| ()))
            .take(n + 1)
            .count()
            == n
    }

    fn assert_valid_vid(&self, id: VertexId) {
        assert!(
            id.index() < self.vertices.len(),
            "VertexId {:?} is out of bounds",
            id
        );
        #[cfg(debug_assertions)]
        assert!(
            id.generation() == self.vertices[id.index()].generation,
            "VertexId {:?} has invalid generation",
            id
        );
    }

    /// Removes the vertex with the given `VertexId` from the tree, along with
    /// all incident edges and labels.
    pub fn remove_vertex(&mut self, vertex: VertexId) -> VertexRemoval<'_, V, E, L> {
        self.assert_valid_vid(vertex);
        VertexRemoval::new(self, vertex)
    }

    fn remove_vertex_unchecked(&mut self, id: usize) -> (V, SwapResult<VertexId>) {
        let vertex = self.vertices.swap_remove(id);

        assert!(
            vertex.next_edge.is_none(),
            "Cannot remove vertex with incident edges"
        );
        assert!(
            vertex.next_label.is_none(),
            "Cannot remove vertex with incident labels"
        );

        match self.vertices.get_mut(id) {
            None => (vertex.weight, SwapResult::None),
            Some(swapped) => {
                let generation = mem::replace(&mut swapped.generation, self.generation);
                let prev = VertexId::new_from_usize(self.vertices.len(), generation);
                let next = VertexId::new_from_usize(id, self.generation);

                (vertex.weight, SwapResult::Swapped { prev, next })
            }
        }
    }

    pub fn remove_edge(&mut self, id: EdgeId) -> (E, SwapResult<EdgeId>) {
        assert!(self.is_valid_edge(id), "EdgeId {:?} is invalid", id);

        self.generation.increment();

        self.remove_edge_unchecked(id.index())
    }

    fn remove_edge_unchecked(&mut self, id: usize) -> (E, SwapResult<EdgeId>) {
        let (endpoints, links) = {
            let edge = self.edges.get(id).expect("Edge index out of bounds");
            (edge.endpoints, edge.next)
        };

        self.fix_edge_links(endpoints, id, links);

        self.swap_remove_edge(id)
    }

    fn fix_edge_links(&mut self, endpoints: Endpoints, id: usize, links: EdgeLinks) {
        for direction in Directions {
            let endpoint = endpoints[direction];
            let replacement = links[direction];

            if let Some(first) = self.vertices[endpoint.index()].next_edge {
                if first.index() == id {
                    self.vertices[endpoint.index()].next_edge = replacement;
                } else {
                    for edge in EdgeWalkerMut::new(&mut self.edges, first, endpoint) {
                        let dir = edge.endpoints.direction_of(endpoint).unwrap();
                        if edge.next[dir].map(|e| e.index()) == Some(id) {
                            edge.next[dir] = replacement;
                            break;
                        }
                    }
                }
            }
        }
    }

    fn swap_remove_edge(&mut self, id: usize) -> (E, SwapResult<EdgeId>) {
        let edge = self.edges.swap_remove(id);

        match self.edges.get_mut(id) {
            None => (edge.weight, SwapResult::None),
            // the edge `swapped` was previously at the end of the vector and
            // has now moved to index `id`.
            Some(swapped) => {
                let endpoints = swapped.endpoints;
                // the generation of the edge `swapped` is updated to the
                // current generation. The caller of this function must ensure
                // that the generation was incremented before calling this
                // function.
                let generation = mem::replace(&mut swapped.generation, self.generation);

                let prev = EdgeId::new_from_usize(self.edges.len(), generation);
                let next = EdgeId::new_from_usize(id, self.generation);

                self.fix_edge_links(
                    endpoints,
                    self.edges.len(),
                    EdgeLinks([Some(next), Some(next)]),
                );

                (edge.weight, SwapResult::Swapped { prev, next })
            }
        }
    }

    pub fn remove_label(&mut self, id: LabelId) -> (L, SwapResult<LabelId>) {
        assert!(self.is_valid_label(id), "LabelId {:?} is invalid", id);

        self.generation.increment();

        self.remove_label_unchecked(id.index())
    }

    fn remove_label_unchecked(&mut self, id: usize) -> (L, SwapResult<LabelId>) {
        let label = self.labels.get(id).expect("Label index out of bounds");

        let next = label.next;

        self.fix_label_links(label.vertex, id, next);

        self.swap_remove_label(id)
    }

    fn fix_label_links(&mut self, vertex: VertexId, id: usize, next: Option<LabelId>) {
        if let Some(first) = self.vertices[vertex.index()].next_label {
            if first.index() == id {
                self.vertices[vertex.index()].next_label = next;
            } else {
                for label in LabelWalkerMut::new(&mut self.labels, first) {
                    if label.next.map(|l| l.index()) == Some(id) {
                        label.next = next;
                        break;
                    }
                }
            }
        }
    }

    fn swap_remove_label(&mut self, id: usize) -> (L, SwapResult<LabelId>) {
        let label = self.labels.swap_remove(id);

        match self.labels.get_mut(id) {
            None => (label.weight, SwapResult::None),
            Some(swapped) => {
                let vertex = swapped.vertex;
                let generation = mem::replace(&mut swapped.generation, self.generation);

                let prev = LabelId::new_from_usize(self.labels.len(), generation);
                let next = LabelId::new_from_usize(id, self.generation);

                self.fix_label_links(vertex, self.labels.len(), Some(next));

                (label.weight, SwapResult::Swapped { prev, next })
            }
        }
    }
}

pub enum EdgeOrLabelId {
    Edge(SwapResult<EdgeId>),
    Label(SwapResult<LabelId>),
    Vertex(SwapResult<VertexId>),
}

enum Either<A, B> {
    Left(A),
    Right(B),
}

pub struct VertexRemoval<'a, V, E, L> {
    tree: &'a mut Tree<V, E, L>,
    vertex: Either<VertexId, V>,
}

impl<'a, V, E, L> VertexRemoval<'a, V, E, L> {
    pub fn new(tree: &'a mut Tree<V, E, L>, vertex: VertexId) -> Self {
        tree.assert_valid_vid(vertex);
        tree.generation.increment();
        Self {
            tree,
            vertex: Either::Left(vertex),
        }
    }
    pub fn next(&mut self) -> Option<EdgeOrLabelId> {
        match self.vertex {
            Either::Left(vertex) => {
                while let Some(label) = self.tree.vertices[vertex.index()].next_label {
                    if let (_, swap @ SwapResult::Swapped { .. }) =
                        self.tree.remove_label_unchecked(label.index())
                    {
                        return Some(EdgeOrLabelId::Label(swap));
                    }
                }

                while let Some(edge) = self.tree.vertices[vertex.index()].next_edge {
                    if let (_, swap @ SwapResult::Swapped { .. }) =
                        self.tree.remove_edge_unchecked(edge.index())
                    {
                        return Some(EdgeOrLabelId::Edge(swap));
                    }
                }

                let (weight, swap) = self.tree.remove_vertex_unchecked(vertex.index());
                self.vertex = Either::Right(weight);
                Some(EdgeOrLabelId::Vertex(swap))
            }
            Either::Right(_) => None,
        }
    }

    fn into_weight(self) -> V {
        let this = ManuallyDrop::new(self);

        match unsafe { ptr::read(&this.vertex) } {
            Either::Left(_) => panic!("Vertex removal not complete"),
            Either::Right(weight) => weight,
        }
    }
}

impl<'a, V, E, L> Drop for VertexRemoval<'a, V, E, L> {
    fn drop(&mut self) {
        panic!(
            "VertexRemoval must be fully consumed before dropping. Call `next()` until it returns None, then call `into_weight()` to retrieve the vertex weight."
        );
    }
}

struct EdgeWalkerMut<'a, E> {
    edges: &'a mut [Edge<E>],
    current: Option<EdgeId>,
    incident_vertex: VertexId,
}

impl<'a, E> EdgeWalkerMut<'a, E> {
    fn new(edges: &'a mut [Edge<E>], start: EdgeId, incident_vertex: VertexId) -> Self {
        Self {
            edges,
            current: Some(start),
            incident_vertex,
        }
    }
}

impl<'a, E> Iterator for EdgeWalkerMut<'a, E> {
    type Item = &'a mut Edge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(eid) = self.current {
            let edge = &mut self.edges[eid.index()];
            let direction = edge.endpoints.direction_of(self.incident_vertex).unwrap();
            self.current = edge.next[direction];

            unsafe {
                let edge_ptr: *mut Edge<E> = edge;
                Some(&mut *edge_ptr)
            }
        } else {
            None
        }
    }
}

struct LabelWalkerMut<'a, L> {
    labels: &'a mut [Label<L>],
    current: Option<LabelId>,
}

impl<'a, L> LabelWalkerMut<'a, L> {
    fn new(labels: &'a mut [Label<L>], start: LabelId) -> Self {
        Self {
            labels,
            current: Some(start),
        }
    }
}

impl<'a, L> Iterator for LabelWalkerMut<'a, L> {
    type Item = &'a mut Label<L>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lid) = self.current {
            let label = &mut self.labels[lid.index()];
            self.current = label.next;

            unsafe {
                let label_ptr: *mut Label<L> = label;
                Some(&mut *label_ptr)
            }
        } else {
            None
        }
    }
}

pub struct LabelWeights<'a, L> {
    labels: &'a [Label<L>],
    current: Option<LabelId>,
}

impl<'a, L> LabelWeights<'a, L> {
    fn new_from_vertex<V, E>(tree: &'a Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_label;
        Self {
            labels: &tree.labels,
            current,
        }
    }
}

impl<'a, L> Iterator for LabelWeights<'a, L> {
    type Item = &'a L;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lid) = self.current {
            let label = &self.labels[lid.index()];
            self.current = label.next;
            Some(&label.weight)
        } else {
            None
        }
    }
}

pub struct EdgeWeights<'a, E> {
    edges: &'a [Edge<E>],
    current: Option<EdgeId>,
    incident_vertex: VertexId,
}

impl<'a, E> EdgeWeights<'a, E> {
    fn new_from_vertex<V, L>(tree: &'a Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_edge;
        Self {
            edges: &tree.edges,
            current,
            incident_vertex: vertex,
        }
    }
}

impl<'a, E> Iterator for EdgeWeights<'a, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(eid) = self.current {
            let edge = &self.edges[eid.index()];
            let direction = edge.endpoints.direction_of(self.incident_vertex).unwrap();
            self.current = edge.next[direction];
            Some(&edge.weight)
        } else {
            None
        }
    }
}

struct LabelWeightsMut<'a, L> {
    labels: &'a mut [Label<L>],
    current: Option<LabelId>,
}

impl<'a, L> LabelWeightsMut<'a, L> {
    fn new_from_vertex<V, E>(tree: &'a mut Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_label;
        Self {
            labels: &mut tree.labels,
            current,
        }
    }
}

impl<'a, L> Iterator for LabelWeightsMut<'a, L> {
    type Item = &'a mut L;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lid) = self.current {
            let label = &mut self.labels[lid.index()];
            self.current = label.next;

            unsafe {
                let label_ptr: *mut Label<L> = label;
                Some(&mut (*label_ptr).weight)
            }
        } else {
            None
        }
    }
}

struct EdgeWeightsMut<'a, E> {
    edges: &'a mut [Edge<E>],
    current: Option<EdgeId>,
    incident_vertex: VertexId,
}

impl<'a, E> EdgeWeightsMut<'a, E> {
    fn new_from_vertex<V, L>(tree: &'a mut Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_edge;
        Self {
            edges: &mut tree.edges,
            current,
            incident_vertex: vertex,
        }
    }
}

impl<'a, E> Iterator for EdgeWeightsMut<'a, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(eid) = self.current {
            let edge = &mut self.edges[eid.index()];
            let direction = edge.endpoints.direction_of(self.incident_vertex).unwrap();
            self.current = edge.next[direction];

            unsafe {
                let edge_ptr: *mut Edge<E> = edge;
                Some(&mut (*edge_ptr).weight)
            }
        } else {
            None
        }
    }
}

struct LabelWalker<'a, L> {
    labels: &'a [Label<L>],
    current: Option<LabelId>,
}

impl<'a, L> LabelWalker<'a, L> {
    fn new_from_vertex<V, E>(tree: &'a Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_label;
        Self {
            labels: &tree.labels,
            current,
        }
    }
}

impl<'a, L> Iterator for LabelWalker<'a, L> {
    type Item = (LabelId, &'a Label<L>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lid) = self.current {
            let label = &self.labels[lid.index()];
            self.current = label.next;
            Some((lid, label))
        } else {
            None
        }
    }
}

struct EdgeWalker<'a, E> {
    edges: &'a [Edge<E>],
    current: Option<EdgeId>,
    incident_vertex: VertexId,
}

impl<'a, E> EdgeWalker<'a, E> {
    fn new_from_vertex<V, L>(tree: &'a Tree<V, E, L>, vertex: VertexId) -> Self {
        let current = tree.vertices[vertex.index()].next_edge;
        Self {
            edges: &tree.edges,
            current,
            incident_vertex: vertex,
        }
    }
}

impl<'a, E> Iterator for EdgeWalker<'a, E> {
    type Item = (EdgeId, &'a Edge<E>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(eid) = self.current {
            let edge = &self.edges[eid.index()];
            let direction = edge.endpoints.direction_of(self.incident_vertex).unwrap();
            self.current = edge.next[direction];
            Some((eid, edge))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoints(pub(crate) [VertexId; 2]);

struct Directions;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

impl Endpoints {
    pub fn left(&self) -> VertexId {
        self.0[0]
    }

    pub fn right(&self) -> VertexId {
        self.0[1]
    }

    fn direction_of(&self, v: VertexId) -> Option<Direction> {
        if self.0[0] == v {
            Some(Direction::Left)
        } else if self.0[1] == v {
            Some(Direction::Right)
        } else {
            None
        }
    }

    fn replace(&mut self, old: VertexId, new: VertexId) {
        if self.0[0] == old {
            self.0[0] = new;
        } else if self.0[1] == old {
            self.0[1] = new;
        } else {
            panic!(
                "Node index {:?} not found in edge endpoints {:?}",
                old, self
            );
        }
    }

    fn contains(&self, v: VertexId) -> bool {
        self.0[0] == v || self.0[1] == v
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct EdgeLinks([Option<EdgeId>; 2]);

impl Index<bool> for Endpoints {
    type Output = VertexId;

    fn index(&self, index: bool) -> &Self::Output {
        if index { &self.0[1] } else { &self.0[0] }
    }
}

impl Index<Direction> for Endpoints {
    type Output = VertexId;

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
    type Output = Option<EdgeId>;

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
