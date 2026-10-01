//! Differential test foundation for the graph-level and connectivity API of
//! [`FindBridge`].
//!
//! This module contains the shared pieces that the `fixed`, `randomized` and
//! `surface` child modules build their tests on:
//!
//! * [`Naive`], a deliberately straightforward multigraph oracle with real
//!   bridge analysis. It answers every query by BFS over explicit adjacency
//!   maps, so it shares no code or clever invariants with the structure under
//!   test.
//! * [`Graph`], a driver that emulates Appendix A's `Insert`/`Delete` directly
//!   on top of the public [`FindBridge`] API, keeping its own vertex/edge
//!   bookkeeping so it can be compared against [`Naive`] after every mutation.
//! * assertion helpers ([`assert_matches_naive`],
//!   [`assert_internal_invariants`], ...) that compare *every* public query
//!   against the oracle and check the driver's internal bookkeeping.
//!
//! # Vertex handles
//!
//! [`FindBridge::add_vertex`] takes no argument and returns a handle whose
//! [`VertexId::index`] is a stable [`top_tree::ClusterId`] index, which is
//! *not* the same numbering as creation order. The driver therefore owns the
//! mapping: its own vertices are the `usize` indices `0..n` allocated in
//! [`Graph::add_vertex`] order, exactly the numbering [`Naive`] uses. All
//! graph-level driver methods take those `usize` indices, and the driver
//! resolves them to [`VertexId`] handles internally.
//!
//! # Edge handles and slot recycling
//!
//! [`FindBridge::link`] allocates its edge handles from a [`SlotVec`], whose
//! freed slots are *reused*. A deleted [`EdgeId`] is therefore not stable: it
//! may name a different, live edge after a later insertion, and a stale handle
//! passed to [`Graph::delete`] may delete that other edge. The driver keeps its
//! own `EdgeId -> oracle edge index` association ([`Graph::live`]) and
//! validates it in [`assert_internal_invariants`]; tests that care about a
//! specific edge must use the handle returned by [`Graph::insert`] and not
//! assume its index is free of aliasing.
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::*;

mod fixed;
mod randomized;
mod surface;

/// The level used for every inserted non-tree edge, as in Appendix A.
fn non_tree_level() -> Level {
    Level::new(0).expect("level zero is always representable")
}

/// A deliberately straightforward multigraph used as the test oracle.
///
/// The adjacency map stores the multiplicity of each undirected endpoint pair,
/// while `edges` keeps stable handles for the live individual edge copies.
/// Because multiplicities are tracked, an edge is a bridge only when the pair
/// it connects has multiplicity exactly 1 *and* removing that pair disconnects
/// its endpoints; parallel edges are therefore never bridges. Everything is
/// recomputed by BFS on demand, which is what makes this a usable oracle.
#[derive(Debug, Default)]
struct Naive {
    adj: BTreeMap<usize, BTreeMap<usize, usize>>,
    edges: BTreeMap<usize, (usize, usize)>,
    next_edge: usize,
    next_vertex: usize,
}

impl Naive {
    fn new(vertex_count: usize) -> Self {
        let mut graph = Self::default();
        for _ in 0..vertex_count {
            graph.add_vertex();
        }
        graph
    }

    fn add_vertex(&mut self) -> usize {
        let vertex = self.next_vertex;
        self.next_vertex += 1;
        assert!(self.adj.insert(vertex, BTreeMap::new()).is_none());
        vertex
    }

    fn vertex_count(&self) -> usize {
        self.adj.len()
    }

    fn live_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        self.adj.keys().copied()
    }

    fn insert(&mut self, u: usize, v: usize) -> usize {
        assert!(self.adj.contains_key(&u), "unknown vertex {u}");
        assert!(self.adj.contains_key(&v), "unknown vertex {v}");
        assert_ne!(u, v, "the graph-level API rejects self-loops");

        let edge = self.next_edge;
        self.next_edge += 1;
        assert!(self.edges.insert(edge, (u, v)).is_none());
        *self.adj.get_mut(&u).unwrap().entry(v).or_default() += 1;
        *self.adj.get_mut(&v).unwrap().entry(u).or_default() += 1;
        edge
    }

    fn delete(&mut self, edge: usize) -> bool {
        let Some((u, v)) = self.edges.remove(&edge) else {
            return false;
        };
        self.remove_adjacency_copy(u, v);
        self.remove_adjacency_copy(v, u);
        true
    }

    fn remove_vertex(&mut self, vertex: usize) -> bool {
        if !self.adj.contains_key(&vertex) {
            return false;
        }
        let incident_edges = self
            .edges
            .iter()
            .filter_map(|(&edge, &(u, v))| (u == vertex || v == vertex).then_some(edge))
            .collect::<Vec<_>>();
        for edge in incident_edges {
            assert!(self.delete(edge));
        }
        assert!(self.adj[&vertex].is_empty());
        self.adj.remove(&vertex);
        true
    }

    fn remove_adjacency_copy(&mut self, u: usize, v: usize) {
        let neighbors = self.adj.get_mut(&u).expect("live edge endpoint");
        let remove_neighbor = {
            let count = neighbors.get_mut(&v).expect("live adjacency entry");
            *count -= 1;
            *count == 0
        };
        if remove_neighbor {
            neighbors.remove(&v);
        }
    }

    fn connected(&self, u: usize, v: usize) -> bool {
        if !self.adj.contains_key(&u) || !self.adj.contains_key(&v) {
            return false;
        }
        self.reachable_with(u, v, |_, _| false)
    }

    fn reachable_with<F>(&self, start: usize, target: usize, mut blocked: F) -> bool
    where
        F: FnMut(usize, usize) -> bool,
    {
        if !self.adj.contains_key(&start) || !self.adj.contains_key(&target) {
            return false;
        }
        if start == target {
            return true;
        }

        let mut seen = BTreeSet::from([start]);
        let mut pending = VecDeque::from([start]);
        while let Some(vertex) = pending.pop_front() {
            for &neighbor in self.adj[&vertex].keys() {
                if blocked(vertex, neighbor) || !seen.insert(neighbor) {
                    continue;
                }
                if neighbor == target {
                    return true;
                }
                pending.push_back(neighbor);
            }
        }
        false
    }

    fn connected_ignoring_pair(&self, u: usize, v: usize, pair: (usize, usize)) -> bool {
        self.reachable_with(u, v, |a, b| normalized_pair(a, b) == pair)
    }

    fn component_labels(&self, blocked: &BTreeSet<(usize, usize)>) -> Vec<usize> {
        let mut labels = vec![usize::MAX; self.next_vertex];
        for &root in self.adj.keys() {
            if labels[root] != usize::MAX {
                continue;
            }

            labels[root] = root;
            let mut pending = VecDeque::from([root]);
            while let Some(vertex) = pending.pop_front() {
                for &neighbor in self.adj[&vertex].keys() {
                    if blocked.contains(&normalized_pair(vertex, neighbor))
                        || labels[neighbor] != usize::MAX
                    {
                        continue;
                    }
                    labels[neighbor] = root;
                    pending.push_back(neighbor);
                }
            }
        }
        labels
    }

    fn component_size(&self, vertex: usize) -> usize {
        if !self.adj.contains_key(&vertex) {
            return 0;
        }
        let labels = self.component_labels(&BTreeSet::new());
        label_sizes(&labels)[vertex]
    }

    fn is_bridge(&self, edge: usize) -> bool {
        let Some(&(u, v)) = self.edges.get(&edge) else {
            return false;
        };
        if self.adj[&u].get(&v) != Some(&1) {
            return false;
        }
        !self.connected_ignoring_pair(u, v, normalized_pair(u, v))
    }

    fn bridges(&self) -> Vec<usize> {
        self.edges
            .keys()
            .copied()
            .filter(|&edge| self.is_bridge(edge))
            .collect()
    }

    fn bridge_pairs(&self, bridges: &[usize]) -> BTreeSet<(usize, usize)> {
        bridges
            .iter()
            .map(|edge| {
                let &(u, v) = &self.edges[edge];
                normalized_pair(u, v)
            })
            .collect()
    }

    fn two_edge_connected(&self, u: usize, v: usize) -> bool {
        if !self.adj.contains_key(&u) || !self.adj.contains_key(&v) {
            return false;
        }
        if u == v {
            return true;
        }
        if !self.connected(u, v) {
            return false;
        }
        let bridge_pairs = self.bridge_pairs(&self.bridges());
        let labels = self.component_labels(&bridge_pairs);
        labels[u] == labels[v]
    }

    fn two_edge_component_size(&self, vertex: usize) -> usize {
        if !self.adj.contains_key(&vertex) {
            return 0;
        }
        let bridge_pairs = self.bridge_pairs(&self.bridges());
        let labels = self.component_labels(&bridge_pairs);
        label_sizes(&labels)[vertex]
    }

    fn bridge_separating(&self, u: usize, v: usize) -> Option<usize> {
        if u == v || !self.connected(u, v) {
            return None;
        }
        self.bridges().into_iter().find(|&edge| {
            let &(a, b) = &self.edges[&edge];
            !self.connected_ignoring_pair(u, v, normalized_pair(a, b))
        })
    }

    fn bridge_in_component(&self, vertex: usize) -> Option<usize> {
        if !self.adj.contains_key(&vertex) {
            return None;
        }
        self.bridges().into_iter().find(|edge| {
            let &(u, _) = &self.edges[edge];
            self.connected(vertex, u)
        })
    }
}

fn normalized_pair<T: Ord>(u: T, v: T) -> (T, T) {
    if u < v { (u, v) } else { (v, u) }
}

fn label_sizes(labels: &[usize]) -> Vec<usize> {
    let mut sizes = vec![0; labels.len()];
    for &label in labels {
        if label != usize::MAX {
            sizes[label] += 1;
        }
    }
    labels
        .iter()
        .map(|&label| if label == usize::MAX { 0 } else { sizes[label] })
        .collect()
}

/// The driver's own record for one live graph edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveEdge {
    /// The edge's stable index in the [`Naive`] oracle.
    oracle: usize,
    /// The driver's vertex index of the first endpoint.
    u: usize,
    /// The driver's vertex index of the second endpoint.
    v: usize,
}

/// Disjoint borrows of a [`Graph`], so a test can hand the same state to a
/// comparison function that wants both the mutable [`FindBridge`] and a
/// separate oracle.
struct GraphParts<'a> {
    fb: &'a mut FindBridge,
    naive: &'a mut Naive,
    verts: &'a BTreeMap<usize, VertexId>,
    live: &'a BTreeMap<EdgeId, LiveEdge>,
}

/// A graph-level driver over [`FindBridge`], emulating Appendix A's
/// `Insert`/`Delete` on top of the public API and tracking a [`Naive`] oracle.
///
/// The driver's vertices are the `usize` indices allocated by
/// [`Graph::add_vertex`], which is the same numbering the oracle uses; see the
/// module docs. Its edges are the [`EdgeId`] handles returned by
/// [`Graph::insert`], associated with the oracle edge they represent in
/// [`Graph::live`].
struct Graph {
    fb: FindBridge,
    naive: Naive,
    /// Driver vertex index -> the live [`VertexId`] handle for that vertex.
    /// Removed indices are tombstoned by removal from the map.
    verts: BTreeMap<usize, VertexId>,
    /// Live [`EdgeId`] handle -> the oracle edge it stands for.
    live: BTreeMap<EdgeId, LiveEdge>,
}

impl Graph {
    /// Creates a graph with `vertex_count` isolated vertices `0..vertex_count`.
    fn new(vertex_count: usize) -> Self {
        let mut graph = Self {
            fb: FindBridge::new(),
            naive: Naive::new(0),
            verts: BTreeMap::new(),
            live: BTreeMap::new(),
        };
        for _ in 0..vertex_count {
            graph.add_vertex();
        }
        graph
    }

    /// Adds an isolated vertex and returns its driver index.
    fn add_vertex(&mut self) -> usize {
        let index = self.naive.add_vertex();
        let handle = self.fb.add_vertex();
        let previous = self.verts.insert(index, handle);
        assert_eq!(previous, None, "a fresh oracle index is never reused");
        index
    }

    fn parts(&mut self) -> GraphParts<'_> {
        GraphParts {
            fb: &mut self.fb,
            naive: &mut self.naive,
            verts: &self.verts,
            live: &self.live,
        }
    }

    /// The live driver vertices as `(driver index, handle)` pairs, in index
    /// order.
    fn live_vertices(&self) -> impl Iterator<Item = (usize, VertexId)> + '_ {
        self.verts.iter().map(|(&index, &handle)| (index, handle))
    }

    /// The oracle index of a live edge handle, or `None` if the handle is not
    /// one this driver handed out.
    fn naive_edge(&self, edge: EdgeId) -> Option<usize> {
        self.live.get(&edge).map(|record| record.oracle)
    }

    /// The live handle of an oracle edge index, if that edge is still live.
    fn handle_of(&self, oracle: usize) -> Option<EdgeId> {
        self.live
            .iter()
            .find(|(_, record)| record.oracle == oracle)
            .map(|(&handle, _)| handle)
    }

    /// The driver vertex index of a [`VertexId`] handle, if it is live.
    fn index_of(&self, vertex: VertexId) -> Option<usize> {
        self.verts
            .iter()
            .find(|(_, handle)| **handle == vertex)
            .map(|(&index, _)| index)
    }

    /// Removes a vertex and all its incident edges, returning whether the
    /// driver index was live.
    ///
    /// [`FindBridge::remove_vertex`] panics while the vertex still has an
    /// incident forest edge, so every incident edge is removed through
    /// [`FindBridge::remove_edge`] first. The incident edges are visited in
    /// ascending oracle edge index (that is, insertion order) so that a
    /// randomized run is reproducible regardless of the [`SlotVec`]'s free-slot
    /// order. The oracle removes the same edges itself in
    /// [`Naive::remove_vertex`], so they are not deleted through
    /// [`Graph::delete`] here.
    fn remove_vertex(&mut self, u: usize) -> bool {
        let Some(handle) = self.verts.get(&u).copied() else {
            return false;
        };

        let mut incident = self
            .live
            .values()
            .filter(|record| record.u == u || record.v == u)
            .map(|record| record.oracle)
            .collect::<Vec<_>>();
        incident.sort_unstable();
        for oracle in incident {
            let edge = self
                .handle_of(oracle)
                .expect("an oracle index taken from live is live");
            self.fb.remove_edge(edge);
            let record = self.live.remove(&edge).expect("edge {edge:?} is live");
            assert_eq!(record.oracle, oracle, "edge {edge:?} maps back");
        }

        assert!(
            self.naive.remove_vertex(u),
            "the driver and the oracle agree on which vertices are live"
        );
        let removed = self.fb.remove_vertex(handle);
        assert!(removed, "the vertex is isolated after deleting its edges");
        let previous = self.verts.remove(&u);
        assert_eq!(previous, Some(handle), "vertex {u} maps back");
        true
    }

    /// Inserts an undirected edge between the driver vertices `u` and `v`,
    /// returning its [`EdgeId`] handle, or `None` for an unknown vertex or a
    /// self-loop.
    ///
    /// This is Appendix A's `Insert`. An edge between two different trees
    /// becomes a tree edge, which [`FindBridge::link`] already does. An edge
    /// whose endpoints are already connected becomes a level-0 non-tree edge
    /// *and* must cover the tree path between its endpoints:
    /// [`FindBridge::link`] records the non-tree edge but deliberately does not
    /// raise the path cover, so the cover is applied here. Connectivity is
    /// therefore sampled *before* the link.
    fn insert(&mut self, u: usize, v: usize) -> Option<EdgeId> {
        let (u_handle, v_handle) = (*self.verts.get(&u)?, *self.verts.get(&v)?);
        if u == v {
            return None;
        }

        let already_connected = self.fb.connected(u_handle, v_handle);
        let edge = self.fb.link(u_handle, v_handle);
        if already_connected {
            self.fb.cover(u_handle, v_handle, non_tree_level());
        }

        let oracle = self.naive.insert(u, v);
        let previous = self.live.insert(edge, LiveEdge { oracle, u, v });
        assert_eq!(previous, None, "a fresh edge handle is not already live");
        Some(edge)
    }

    /// Deletes a live edge handle, returning whether the handle was live.
    ///
    /// This is Appendix A's `Delete`, which [`FindBridge::remove_edge`]
    /// performs in full: a bridge tree edge is cut, a covered tree edge runs
    /// `Swap` and then `Delete`, and a non-tree edge drops its labels and
    /// uncovers its path.
    fn delete(&mut self, edge: EdgeId) -> bool {
        let Some(record) = self.live.get(&edge).copied() else {
            return false;
        };

        self.fb.remove_edge(edge);
        let removed = self.live.remove(&edge);
        assert_eq!(removed, Some(record), "edge {edge:?} maps back");
        assert!(
            self.naive.delete(record.oracle),
            "the driver and the oracle agree on which edges are live"
        );
        true
    }

    /// The [`VertexId`] endpoints of a live edge handle, delegating to
    /// [`FindBridge::endpoints`]. `None` for a handle this driver does not own.
    fn edge_endpoints(&self, edge: EdgeId) -> Option<(VertexId, VertexId)> {
        if !self.live.contains_key(&edge) {
            return None;
        }
        self.fb.endpoints(edge)
    }

    /// The driver vertex indices of a live edge handle, or `None` for a handle
    /// this driver does not own.
    fn edge_vertices(&self, edge: EdgeId) -> Option<(usize, usize)> {
        let (u, v) = self.fb.endpoints(edge)?;
        Some((self.index_of(u)?, self.index_of(v)?))
    }

    fn vertex_count(&self) -> usize {
        self.verts.len()
    }

    fn edge_count(&self) -> usize {
        self.live.len()
    }

    /// Whether two driver vertices are connected; an unknown vertex returns
    /// `false`.
    fn connected(&mut self, u: usize, v: usize) -> bool {
        connected(&mut self.fb, &self.verts, u, v)
    }

    /// Whether two driver vertices are two-edge-connected, i.e. connected with a
    /// cover level of at least `0` on the path between them. An unknown vertex
    /// returns `false`.
    fn two_edge_connected(&mut self, u: usize, v: usize) -> bool {
        two_edge_connected(&mut self.fb, &self.verts, u, v)
    }

    /// A bridge in `v`'s connected component, or `None` if there is none or `v`
    /// is unknown.
    fn find_bridge(&mut self, v: usize) -> Option<EdgeId> {
        find_bridge(&mut self.fb, &self.verts, v)
    }

    /// A bridge on the `u`-`v` path, or `None` if there is none, if the pair is
    /// not connected, or if a vertex is unknown.
    fn find_bridge_between(&mut self, u: usize, v: usize) -> Option<EdgeId> {
        find_bridge_between(&mut self.fb, &self.verts, u, v)
    }

    fn component_size(&mut self, v: usize) -> u64 {
        component_size(&mut self.fb, &self.verts, v)
    }

    fn two_edge_component_size(&mut self, v: usize) -> u64 {
        two_edge_component_size(&mut self.fb, &self.verts, v)
    }
}

/// The query half of [`Graph`], factored out so the comparison helpers can run
/// against a disjoint borrow of the same state. Every one of these treats an
/// unknown driver vertex as the neutral answer instead of panicking, because
/// the deleted graph module's tests exercised stale-handle paths and so do the
/// randomized runs.
fn connected(fb: &mut FindBridge, verts: &BTreeMap<usize, VertexId>, u: usize, v: usize) -> bool {
    match (verts.get(&u), verts.get(&v)) {
        (Some(u), Some(v)) => fb.connected(*u, *v),
        _ => false,
    }
}

fn two_edge_connected(
    fb: &mut FindBridge,
    verts: &BTreeMap<usize, VertexId>,
    u: usize,
    v: usize,
) -> bool {
    match (verts.get(&u), verts.get(&v)) {
        (Some(u), Some(v)) => fb.connected(*u, *v) && fb.cover_level_between(*u, *v) >= 0,
        _ => false,
    }
}

fn find_bridge(fb: &mut FindBridge, verts: &BTreeMap<usize, VertexId>, v: usize) -> Option<EdgeId> {
    let v = *verts.get(&v)?;
    fb.find_bridge(v)
}

fn find_bridge_between(
    fb: &mut FindBridge,
    verts: &BTreeMap<usize, VertexId>,
    u: usize,
    v: usize,
) -> Option<EdgeId> {
    let (u, v) = (*verts.get(&u)?, *verts.get(&v)?);
    fb.find_bridge_between(u, v)
}

fn component_size(fb: &mut FindBridge, verts: &BTreeMap<usize, VertexId>, v: usize) -> u64 {
    let Some(v) = verts.get(&v) else {
        return 0;
    };
    fb.component_size(*v)
}

fn two_edge_component_size(
    fb: &mut FindBridge,
    verts: &BTreeMap<usize, VertexId>,
    v: usize,
) -> u64 {
    let Some(v) = verts.get(&v) else {
        return 0;
    };
    fb.two_edge_component_size(*v)
}

/// Compares every observable graph query against one BFS-based reference
/// snapshot. Removing each bridge once also gives an independent oracle for
/// `find_bridge_between` without repeatedly recomputing all bridges per pair.
///
/// `naive` is compared against `graph`'s own state, so it must be a second
/// oracle describing the same graph (typically a separate [`Naive`] built by the
/// test). [`Graph::assert_matches`] runs the same comparison against the
/// driver's own oracle.
fn assert_matches_naive(graph: &mut Graph, naive: &Naive, context: &str) {
    let GraphParts {
        fb, verts, live, ..
    } = graph.parts();
    assert_matches_core(fb, naive, verts, live, context);
}

/// The query comparison shared by [`assert_matches_naive`] and
/// [`Graph::assert_matches`].
fn assert_matches_core(
    fb: &mut FindBridge,
    naive: &Naive,
    verts: &BTreeMap<usize, VertexId>,
    live: &BTreeMap<EdgeId, LiveEdge>,
    context: &str,
) {
    assert_eq!(verts.len(), naive.vertex_count(), "{context}: vertex count");
    assert_eq!(live.len(), naive.edges.len(), "{context}: edge count");
    assert_eq!(
        fb.top_tree.forest().node_count(),
        verts.len(),
        "{context}: the forest vertex count matches the live vertex count"
    );

    // A driver vertex index is live exactly when the oracle says so, and the
    // stale ones must answer every query neutrally.
    for index in 0..naive.next_vertex {
        let live_vertex = naive.adj.contains_key(&index);
        assert_eq!(
            verts.contains_key(&index),
            live_vertex,
            "{context}: live status of vertex {index}"
        );
        if !live_vertex {
            assert!(
                !connected(fb, verts, index, index),
                "{context}: stale connected"
            );
            assert_eq!(
                component_size(fb, verts, index),
                0,
                "{context}: stale component size"
            );
            assert_eq!(
                two_edge_component_size(fb, verts, index),
                0,
                "{context}: stale two-edge component size"
            );
            assert!(
                !two_edge_connected(fb, verts, index, index),
                "{context}: stale 2-edge query"
            );
            assert_eq!(
                find_bridge(fb, verts, index),
                None,
                "{context}: stale bridge query"
            );
            assert_eq!(
                find_bridge_between(fb, verts, index, index),
                None,
                "{context}: stale path bridge query"
            );
            assert_eq!(
                find_bridge_between(fb, verts, index, 0),
                None,
                "{context}: stale path bridge query with a live endpoint"
            );
        }
    }

    for (&edge, record) in live {
        assert_eq!(
            naive.edges.get(&record.oracle),
            Some(&(record.u, record.v)),
            "{context}: edge {edge:?} does not match oracle edge {}",
            record.oracle
        );
        let (u, v) = fb
            .endpoints(edge)
            .unwrap_or_else(|| panic!("{context}: edge {edge:?} has no endpoints"));
        assert!(
            u == verts[&record.u] && v == verts[&record.v]
                || u == verts[&record.v] && v == verts[&record.u],
            "{context}: edge {edge:?} endpoints {u:?}, {v:?} do not match oracle vertices {}, {}",
            record.u,
            record.v
        );
    }

    let component_labels = naive.component_labels(&BTreeSet::new());
    let component_sizes = label_sizes(&component_labels);
    let bridges = naive.bridges();
    let bridge_ids = bridges.iter().copied().collect::<BTreeSet<_>>();
    let bridge_pairs = naive.bridge_pairs(&bridges);
    let two_edge_labels = naive.component_labels(&bridge_pairs);
    let two_edge_sizes = label_sizes(&two_edge_labels);
    let bridge_cut_labels = bridges
        .iter()
        .map(|&edge| {
            let &(u, v) = &naive.edges[&edge];
            let blocked = BTreeSet::from([normalized_pair(u, v)]);
            (edge, naive.component_labels(&blocked))
        })
        .collect::<Vec<_>>();

    let live_vertices = naive.live_vertices().collect::<Vec<_>>();
    for &vertex in &live_vertices {
        assert_eq!(
            component_size(fb, verts, vertex) as usize,
            component_sizes[vertex],
            "{context}: component_size({vertex})"
        );
        assert_eq!(
            two_edge_component_size(fb, verts, vertex) as usize,
            two_edge_sizes[vertex],
            "{context}: two_edge_component_size({vertex})"
        );

        let actual_bridge = find_bridge(fb, verts, vertex);
        let expected_bridge = bridges.iter().find(|&&edge| {
            let &(u, _) = &naive.edges[&edge];
            component_labels[u] == component_labels[vertex]
        });
        match (expected_bridge, actual_bridge) {
            (None, None) => {}
            (Some(_), Some(edge)) => {
                assert!(
                    live.contains_key(&edge),
                    "{context}: find_bridge({vertex}) returned unknown edge {edge:?}"
                );
                assert!(
                    bridge_ids.contains(&live[&edge].oracle),
                    "{context}: find_bridge({vertex}) returned non-bridge {edge:?}"
                );
                let &(u, _) = &naive.edges[&live[&edge].oracle];
                assert_eq!(
                    component_labels[u], component_labels[vertex],
                    "{context}: find_bridge({vertex}) returned a bridge in another component"
                );
            }
            (Some(&expected), None) => panic!(
                "{context}: find_bridge({vertex}) returned None, but bridge {expected} is in its component"
            ),
            (None, Some(edge)) => panic!(
                "{context}: find_bridge({vertex}) returned {edge:?}, but its component has no bridge"
            ),
        }
    }

    for &u in &live_vertices {
        for &v in &live_vertices {
            let is_connected = component_labels[u] == component_labels[v];
            assert_eq!(
                connected(fb, verts, u, v),
                is_connected,
                "{context}: connected({u}, {v})"
            );
            assert_eq!(
                two_edge_connected(fb, verts, u, v),
                two_edge_labels[u] == two_edge_labels[v],
                "{context}: two_edge_connected({u}, {v})"
            );

            let expected_bridge = if u == v || !is_connected {
                None
            } else {
                bridge_cut_labels
                    .iter()
                    .find(|(_, labels)| labels[u] != labels[v])
                    .map(|(edge, _)| *edge)
            };
            let actual_bridge = find_bridge_between(fb, verts, u, v);
            match (expected_bridge, actual_bridge) {
                (None, None) => {}
                (Some(_), Some(edge)) => {
                    assert!(
                        live.contains_key(&edge),
                        "{context}: find_bridge_between({u}, {v}) returned unknown edge {edge:?}"
                    );
                    let oracle_edge = live[&edge].oracle;
                    assert!(
                        bridge_ids.contains(&oracle_edge),
                        "{context}: find_bridge_between({u}, {v}) returned non-bridge {edge:?}"
                    );
                    let (_, labels) = bridge_cut_labels
                        .iter()
                        .find(|(candidate, _)| *candidate == oracle_edge)
                        .unwrap_or_else(|| {
                            panic!("{context}: no cut data for returned bridge {edge:?}")
                        });
                    assert_ne!(
                        labels[u], labels[v],
                        "{context}: returned bridge {edge:?} does not separate {u} from {v}"
                    );
                }
                (Some(expected), None) => panic!(
                    "{context}: find_bridge_between({u}, {v}) returned None, but bridge {expected} separates them"
                ),
                (None, Some(edge)) => panic!(
                    "{context}: find_bridge_between({u}, {v}) returned {edge:?}, but no bridge separates them"
                ),
            }
        }
    }
}

impl Graph {
    /// [`assert_matches_naive`] against this driver's own oracle.
    fn assert_matches(&mut self, context: &str) {
        let GraphParts {
            fb,
            naive,
            verts,
            live,
        } = self.parts();
        assert_matches_core(fb, naive, verts, live, context);
    }
}

/// Checks the driver's own bookkeeping against the structure and the oracle,
/// then checks that every edge handle a query can return is live and really is
/// a bridge.
///
/// `context` must describe the state that produced the call, e.g.
/// `"after deleting edge 3"`; it is prefixed to every message.
fn assert_internal_invariants(graph: &mut Graph, context: &str) {
    assert_bookkeeping(graph, context);

    // Every handle a query can return must be live here and a bridge in the
    // oracle. `assert_matches_naive` checks the fine-grained correctness of
    // these queries; this is the internal-consistency backstop.
    let live_vertices = graph
        .live_vertices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for (u, v) in pairs(&live_vertices) {
        if let Some(edge) = graph.find_bridge_between(u, v) {
            let oracle_edge = graph.naive_edge(edge);
            assert_returned_separating_bridge(&graph.naive, u, v, oracle_edge, context);
        }
    }
    for &v in &live_vertices {
        if let Some(edge) = graph.find_bridge(v) {
            let oracle_edge = graph.naive_edge(edge);
            assert_returned_bridge_in_component(&graph.naive, v, oracle_edge, context);
        }
    }
}

/// All unordered pairs `u < v` of the given indices, in a deterministic order.
fn pairs(vertices: &[usize]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (position, &u) in vertices.iter().enumerate() {
        for &v in &vertices[position + 1..] {
            pairs.push((u, v));
        }
    }
    pairs
}

/// The bookkeeping half of [`assert_internal_invariants`]: the driver's vertex
/// and edge maps agree with the oracle, and every edge record in
/// [`FindBridge::edges`] is accounted for.
fn assert_bookkeeping(graph: &Graph, context: &str) {
    let fb = &graph.fb;
    let naive = &graph.naive;
    let verts = &graph.verts;
    let live = &graph.live;

    assert_eq!(
        live.len(),
        naive.edges.len(),
        "{context}: live edge bookkeeping does not match the oracle"
    );
    assert_eq!(
        fb.edges.iter().count(),
        live.len(),
        "{context}: the forest's edge records do not match the live edge bookkeeping"
    );
    assert_eq!(
        verts.len(),
        naive.adj.len(),
        "{context}: live vertex bookkeeping does not match the oracle"
    );
    assert_eq!(
        fb.top_tree.forest().node_count(),
        verts.len(),
        "{context}: the forest vertex count does not match the live vertex count"
    );

    // Every vertex index the driver ever allocated is still accounted for: live
    // in both structures, or tombstoned in both.
    for index in 0..naive.next_vertex {
        assert_eq!(
            verts.contains_key(&index),
            naive.adj.contains_key(&index),
            "{context}: vertex {index} disagrees between driver and oracle"
        );
    }
    for &index in verts.keys() {
        let handle = verts[&index];
        assert!(
            fb.cluster_vertex(handle.0).is_some(),
            "{context}: vertex {index} handle {handle:?} is not a live forest vertex"
        );
    }

    // The live-edge map is a bijection with the oracle's live edges, and every
    // forest edge record is one of ours.
    let mut seen_oracle_edges = BTreeSet::new();
    for (&edge, record) in live {
        assert!(
            seen_oracle_edges.insert(record.oracle),
            "{context}: oracle edge {} is claimed by two handles",
            record.oracle
        );
        assert_eq!(
            naive.edges.get(&record.oracle),
            Some(&(record.u, record.v)),
            "{context}: edge {edge:?} does not match oracle edge {}",
            record.oracle
        );
        assert!(
            verts.contains_key(&record.u) && verts.contains_key(&record.v),
            "{context}: edge {edge:?} has a dead endpoint (vertices {}, {})",
            record.u,
            record.v
        );
    }
    for &oracle in naive.edges.keys() {
        assert!(
            seen_oracle_edges.contains(&oracle),
            "{context}: oracle edge {oracle} has no live handle"
        );
    }

    let mut tree_records = 0;
    let mut non_tree_records = 0;
    for (edge, kind) in fb.edges.iter() {
        let record = live
            .get(&edge)
            .unwrap_or_else(|| panic!("{context}: forest edge {edge:?} is not a live graph edge"));
        match kind {
            Edge::Tree(_) => {
                tree_records += 1;
                // Unordered, like the sibling assertion below: a tree edge's
                // endpoints are resolved through `top_tree::edge_endpoints`,
                // which promises no order, so only the pair is compared here.
                assert!(
                    graph.edge_vertices(edge).is_some_and(|(u, v)| {
                        normalized_pair(u, v) == normalized_pair(record.u, record.v)
                    }),
                    "{context}: tree edge {edge:?} endpoints disagree with its record"
                );
            }
            Edge::NonTree(NonTreeEdge { u, v, level }) => {
                non_tree_records += 1;
                let (u, v) = (VertexId(*u), VertexId(*v));
                assert!(
                    u == verts[&record.u] && v == verts[&record.v]
                        || u == verts[&record.v] && v == verts[&record.u],
                    "{context}: non-tree edge {edge:?} endpoints {u:?}, {v:?} disagree with its record"
                );
                assert!(
                    level.increment().is_some(),
                    "{context}: non-tree edge {edge:?} is at the unrepresentable level {level:?}"
                );
            }
        }
    }
    assert_eq!(
        tree_records + non_tree_records,
        live.len(),
        "{context}: forest edge records split does not add up"
    );
    assert_eq!(
        fb.tree_edge_count(),
        tree_records,
        "{context}: tree_edge_count does not match the live tree edges"
    );
    // The tree edges form a spanning forest, so their number is pinned by the
    // oracle's component count. A tree edge need not be a bridge: a non-tree
    // edge's cover can hide it.
    let component_labels = naive.component_labels(&BTreeSet::new());
    let component_count = naive
        .live_vertices()
        .filter(|&vertex| component_labels[vertex] == vertex)
        .count();
    assert_eq!(
        tree_records,
        verts.len() - component_count,
        "{context}: tree_edge_count is not the spanning forest size"
    );
}

/// Checks that a `find_bridge` result is a bridge in the oracle's component of
/// `vertex`. `result` is an *oracle* edge index, as produced by
/// [`Graph::naive_edge`].
fn assert_returned_bridge_in_component(
    naive: &Naive,
    vertex: usize,
    result: Option<usize>,
    context: &str,
) {
    match result {
        None => assert_eq!(naive.bridge_in_component(vertex), None, "{context}"),
        Some(edge) => {
            assert!(naive.is_bridge(edge), "{context}: {edge} is not a bridge");
            let &(u, _) = naive
                .edges
                .get(&edge)
                .unwrap_or_else(|| panic!("{context}: returned edge {edge} is not live"));
            assert!(
                naive.connected(vertex, u),
                "{context}: bridge {edge} is in another component of {vertex}"
            );
        }
    }
}

/// Checks that a `find_bridge_between` result is a bridge whose removal
/// separates `u` from `v`. `result` is an *oracle* edge index, as produced by
/// [`Graph::naive_edge`].
fn assert_returned_separating_bridge(
    naive: &Naive,
    u: usize,
    v: usize,
    result: Option<usize>,
    context: &str,
) {
    match result {
        None => assert_eq!(naive.bridge_separating(u, v), None, "{context}"),
        Some(edge) => {
            assert!(naive.is_bridge(edge), "{context}: {edge} is not a bridge");
            let &(a, b) = naive
                .edges
                .get(&edge)
                .unwrap_or_else(|| panic!("{context}: returned edge {edge} is not live"));
            assert!(
                !naive.connected_ignoring_pair(u, v, normalized_pair(a, b)),
                "{context}: bridge {edge} does not separate {u} from {v}"
            );
        }
    }
}

/// Checks that a live edge handle's endpoints are exactly the expected pair, in
/// either order. The expected vertices are driver indices.
fn assert_endpoints_match(graph: &Graph, edge: EdgeId, expected_u: usize, expected_v: usize) {
    // The two checks below are not redundant: they read the endpoints through
    // different accessors. `edge_vertices` resolves the pair in one step --
    // `fb.endpoints`, followed by an `index_of` that must succeed for *both*
    // halves -- whereas `edge_endpoints` is guarded by the live-handle map and
    // hands back raw `VertexId`s that are resolved here, one at a time. A stale
    // live map or a broken handle round-trip would be caught by only one of
    // them, and both orderings have to be accepted by each.
    let actual = graph
        .edge_vertices(edge)
        .unwrap_or_else(|| panic!("edge {edge:?} has no endpoints"));
    assert!(
        actual == (expected_u, expected_v) || actual == (expected_v, expected_u),
        "edge {edge:?} endpoints were {actual:?}, expected {expected_u} and {expected_v}"
    );

    let (u, v) = graph
        .edge_endpoints(edge)
        .unwrap_or_else(|| panic!("edge {edge:?} has no FindBridge endpoints"));
    let (got_u, got_v) = (graph.index_of(u), graph.index_of(v));
    assert!(
        (got_u == Some(expected_u) && got_v == Some(expected_v))
            || (got_u == Some(expected_v) && got_v == Some(expected_u)),
        "edge {edge:?} endpoints {u:?}, {v:?} resolved to {got_u:?}, {got_v:?}, expected {expected_u} and {expected_v}"
    );
}

/// The foundation works end to end: the triangle exercises both insert paths
/// (two disconnected tree edges and one already-connected non-tree edge with a
/// cover) and must leave no bridges behind.
#[test]
fn graph_level_triangle_through_the_driver() {
    let mut graph = Graph::new(3);

    // The first two insertions join different trees (tree edges); the third
    // closes the cycle (a level-0 non-tree edge plus a cover of the path).
    let ab = graph.insert(0, 1).expect("a valid non-loop edge");
    let bc = graph.insert(1, 2).expect("a valid non-loop edge");
    let ca = graph.insert(2, 0).expect("a valid non-loop edge");

    assert_eq!(graph.vertex_count(), 3);
    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.fb.tree_edge_count(), 2);
    assert!(graph.naive.bridges().is_empty());

    // A second oracle, built independently of the driver, must agree with it.
    let mut oracle = Naive::new(3);
    oracle.insert(0, 1);
    oracle.insert(1, 2);
    oracle.insert(2, 0);
    assert_matches_naive(&mut graph, &oracle, "triangle");
    graph.assert_matches("triangle");
    assert_internal_invariants(&mut graph, "triangle");

    assert_endpoints_match(&graph, ab, 0, 1);
    assert_endpoints_match(&graph, bc, 1, 2);
    assert_endpoints_match(&graph, ca, 2, 0);
    assert_eq!(graph.insert(1, 1), None, "self-loops are rejected");

    for v in 0..3 {
        assert_eq!(graph.find_bridge(v), None, "vertex {v} in a triangle");
        assert_eq!(graph.component_size(v), 3, "component_size({v})");
        assert_eq!(
            graph.two_edge_component_size(v),
            3,
            "two_edge_component_size({v})"
        );
    }

    for u in 0..3 {
        for v in 0..3 {
            assert!(graph.connected(u, v), "connected({u}, {v})");
            assert!(graph.two_edge_connected(u, v), "two_edge({u}, {v})");
            assert_eq!(
                graph.find_bridge_between(u, v),
                None,
                "find_bridge_between({u}, {v})"
            );
        }
    }

    assert_internal_invariants(&mut graph, "triangle after the query loop");
}
