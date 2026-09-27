use super::{DynamicGraph, EdgeId, EdgeKind, VertexId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A deliberately straightforward multigraph used as the test oracle.
///
/// The adjacency map stores the multiplicity of each undirected endpoint pair,
/// while `edges` keeps stable handles for the live individual edge copies.
#[derive(Debug, Default)]
struct Naive {
    adj: BTreeMap<usize, BTreeMap<usize, usize>>,
    edges: BTreeMap<usize, (usize, usize)>,
    next_edge: usize,
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
        let vertex = self.adj.len();
        assert!(self.adj.insert(vertex, BTreeMap::new()).is_none());
        vertex
    }

    fn vertex_count(&self) -> usize {
        self.adj.len()
    }

    fn insert(&mut self, u: usize, v: usize) -> usize {
        assert!(self.adj.contains_key(&u), "unknown vertex {u}");
        assert!(self.adj.contains_key(&v), "unknown vertex {v}");
        assert_ne!(u, v, "the DynamicGraph API rejects self-loops");

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
        let mut labels = vec![usize::MAX; self.vertex_count()];
        for root in 0..labels.len() {
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

fn normalized_pair(u: usize, v: usize) -> (usize, usize) {
    if u < v { (u, v) } else { (v, u) }
}

fn label_sizes(labels: &[usize]) -> Vec<usize> {
    let mut sizes = vec![0; labels.len()];
    for &label in labels {
        sizes[label] += 1;
    }
    labels.iter().map(|&label| sizes[label]).collect()
}

fn new_graph(vertex_count: usize) -> (DynamicGraph, Naive, Vec<VertexId>) {
    let mut graph = DynamicGraph::new();
    let mut vertices = Vec::with_capacity(vertex_count);
    for expected_index in 0..vertex_count {
        let vertex = graph.add_vertex();
        assert_eq!(vertex.index(), expected_index);
        vertices.push(vertex);
    }
    (graph, Naive::new(vertex_count), vertices)
}

fn insert_both(graph: &mut DynamicGraph, naive: &mut Naive, u: VertexId, v: VertexId) -> EdgeId {
    let edge = graph.insert_edge(u, v).expect("valid non-loop edge");
    let naive_edge = naive.insert(u.index(), v.index());
    assert_eq!(edge.index(), naive_edge);
    edge
}

fn assert_endpoints_match(
    graph: &DynamicGraph,
    edge: EdgeId,
    expected_u: VertexId,
    expected_v: VertexId,
) {
    let actual = graph.edge_endpoints(edge).expect("live edge endpoints");
    assert!(
        actual == (expected_u, expected_v) || actual == (expected_v, expected_u),
        "edge {} endpoints were {actual:?}, expected {expected_u:?} and {expected_v:?}",
        edge.index()
    );
}

fn assert_internal_invariants(graph: &DynamicGraph, context: &str) {
    let live_records = graph.edges.iter().filter(|edge| edge.is_some()).count();
    assert_eq!(
        graph.live_edges, live_records,
        "{context}: live_edges does not match live edge records"
    );
    assert_eq!(
        graph.edge_count(),
        live_records,
        "{context}: public edge count does not match live edge records"
    );

    let mut tree_records = 0;
    let mut non_tree_records = 0;
    for (index, record) in graph.edges.iter().enumerate() {
        let Some(record) = record else {
            continue;
        };
        match record.kind {
            EdgeKind::Tree => {
                tree_records += 1;
                assert_eq!(
                    record.level, graph.l_max,
                    "{context}: tree edge {index} has level {} instead of l_max {}",
                    record.level, graph.l_max
                );
                assert_eq!(
                    graph
                        .tree_edge_at
                        .get(&normalized_pair(record.endpoints.0, record.endpoints.1,)),
                    Some(&EdgeId(index)),
                    "{context}: tree edge {index} is not indexed by its endpoint pair"
                );
            }
            EdgeKind::NonTree { label1, label2 } => {
                non_tree_records += 1;
                assert!(
                    (0..=graph.l_max).contains(&record.level),
                    "{context}: non-tree edge {index} has out-of-range level {}",
                    record.level
                );
                assert_ne!(
                    label1, label2,
                    "{context}: duplicate labels for edge {index}"
                );
                assert_eq!(
                    graph.label_to_edge.get(&label1),
                    Some(&EdgeId(index)),
                    "{context}: first label for edge {index} has the wrong owner"
                );
                assert_eq!(
                    graph.label_to_edge.get(&label2),
                    Some(&EdgeId(index)),
                    "{context}: second label for edge {index} has the wrong owner"
                );
            }
        }
    }

    assert_eq!(
        graph.fb.edge_count(),
        tree_records,
        "{context}: FindBridge tree edge count mismatch"
    );
    assert_eq!(
        graph.tree_edge_at.len(),
        tree_records,
        "{context}: tree_edge_at length mismatch"
    );
    for (&pair, &edge) in &graph.tree_edge_at {
        let record = graph
            .edges
            .get(edge.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{context}: tree_edge_at contains dead edge {edge:?}"));
        assert!(
            matches!(record.kind, EdgeKind::Tree),
            "{context}: tree_edge_at contains non-tree edge {edge:?}"
        );
        assert_eq!(
            pair,
            normalized_pair(record.endpoints.0, record.endpoints.1),
            "{context}: tree_edge_at key does not match edge {edge:?}"
        );
    }

    assert_eq!(
        graph.label_to_edge.len(),
        2 * non_tree_records,
        "{context}: label_to_edge length mismatch"
    );
    for (&label, &edge) in &graph.label_to_edge {
        let record = graph
            .edges
            .get(edge.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{context}: label {label:?} belongs to dead edge {edge:?}"));
        match record.kind {
            EdgeKind::NonTree { label1, label2 } => assert!(
                label == label1 || label == label2,
                "{context}: label {label:?} does not belong to edge {edge:?}"
            ),
            EdgeKind::Tree => panic!("{context}: tree edge {edge:?} owns label {label:?}"),
        }
    }

    assert_eq!(
        graph.vertices.len(),
        graph.vertex_count(),
        "{context}: public-to-internal vertex map length mismatch"
    );
    assert_eq!(
        graph.internal_to_vertex.len(),
        graph.vertex_count(),
        "{context}: internal-to-public vertex map length mismatch"
    );
    for (public_index, &internal_index) in graph.vertices.iter().enumerate() {
        assert_eq!(
            graph.internal_to_vertex.get(internal_index),
            Some(&VertexId(public_index)),
            "{context}: vertex {public_index} does not round-trip through internal index {internal_index}"
        );
    }
    for (internal_index, &public_vertex) in graph.internal_to_vertex.iter().enumerate() {
        assert_eq!(
            graph.vertices.get(public_vertex.index()),
            Some(&internal_index),
            "{context}: internal vertex {internal_index} does not round-trip"
        );
    }
}

/// Compare every observable graph query against one BFS-based reference
/// snapshot. Removing each bridge once also gives an independent oracle for
/// `find_bridge_between` without repeatedly recomputing all bridges per pair.
fn assert_matches_naive(graph: &mut DynamicGraph, naive: &Naive, context: &str) {
    assert_eq!(
        graph.vertex_count(),
        naive.vertex_count(),
        "{context}: vertex count"
    );
    assert_eq!(
        graph.edge_count(),
        naive.edges.len(),
        "{context}: edge count"
    );
    assert_eq!(
        graph.edges.len(),
        naive.next_edge,
        "{context}: stable edge handle count"
    );

    for index in 0..naive.next_edge {
        let edge = EdgeId(index);
        match naive.edges.get(&index) {
            Some(&(u, v)) => assert_endpoints_match(graph, edge, VertexId(u), VertexId(v)),
            None => assert_eq!(
                graph.edge_endpoints(edge),
                None,
                "{context}: deleted edge {index} still has endpoints"
            ),
        }
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

    for vertex in 0..naive.vertex_count() {
        let handle = VertexId(vertex);
        assert_eq!(
            graph.component_size(handle) as usize,
            component_sizes[vertex],
            "{context}: component_size({vertex})"
        );
        assert_eq!(
            graph.two_edge_component_size(handle) as usize,
            two_edge_sizes[vertex],
            "{context}: two_edge_component_size({vertex})"
        );

        let actual_bridge = graph.find_bridge(handle);
        let expected_bridge = bridges.iter().find(|&&edge| {
            let &(u, _) = &naive.edges[&edge];
            component_labels[u] == component_labels[vertex]
        });
        match (expected_bridge, actual_bridge) {
            (None, None) => {}
            (Some(_), Some(edge)) => {
                assert!(
                    bridge_ids.contains(&edge.index()),
                    "{context}: find_bridge({vertex}) returned non-bridge {edge:?}"
                );
                let &(u, _) = naive.edges.get(&edge.index()).unwrap_or_else(|| {
                    panic!("{context}: find_bridge returned dead edge {edge:?}")
                });
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

    for u in 0..naive.vertex_count() {
        for v in 0..naive.vertex_count() {
            let connected = component_labels[u] == component_labels[v];
            assert_eq!(
                graph.connected(VertexId(u), VertexId(v)),
                connected,
                "{context}: connected({u}, {v})"
            );
            let two_edge_connected = two_edge_labels[u] == two_edge_labels[v];
            assert_eq!(
                graph.two_edge_connected(VertexId(u), VertexId(v)),
                two_edge_connected,
                "{context}: two_edge_connected({u}, {v})"
            );

            let expected_bridge = if u == v || !connected {
                None
            } else {
                bridge_cut_labels
                    .iter()
                    .find(|(_, labels)| labels[u] != labels[v])
                    .map(|(edge, _)| *edge)
            };
            let actual_bridge = graph.find_bridge_between(VertexId(u), VertexId(v));
            match (expected_bridge, actual_bridge) {
                (None, None) => {}
                (Some(_), Some(edge)) => {
                    assert!(
                        bridge_ids.contains(&edge.index()),
                        "{context}: find_bridge_between({u}, {v}) returned non-bridge {edge:?}"
                    );
                    let (_, labels) = bridge_cut_labels
                        .iter()
                        .find(|(candidate, _)| *candidate == edge.index())
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

fn assert_returned_bridge_in_component(naive: &Naive, vertex: usize, result: Option<EdgeId>) {
    match result {
        None => assert_eq!(naive.bridge_in_component(vertex), None),
        Some(edge) => {
            assert!(naive.is_bridge(edge.index()));
            let &(u, _) = naive
                .edges
                .get(&edge.index())
                .expect("returned edge is live");
            assert!(naive.connected(vertex, u));
        }
    }
}

fn assert_returned_separating_bridge(naive: &Naive, u: usize, v: usize, result: Option<EdgeId>) {
    match result {
        None => assert_eq!(naive.bridge_separating(u, v), None),
        Some(edge) => {
            assert!(naive.is_bridge(edge.index()));
            let &(a, b) = naive
                .edges
                .get(&edge.index())
                .expect("returned edge is live");
            assert!(!naive.connected_ignoring_pair(u, v, normalized_pair(a, b)));
        }
    }
}

#[test]
fn single_edge_is_bridge() {
    let (mut graph, mut naive, vertices) = new_graph(2);
    let [a, b] = [vertices[0], vertices[1]];
    let edge = insert_both(&mut graph, &mut naive, a, b);

    assert_eq!(graph.find_bridge(a), Some(edge));
    assert_eq!(graph.find_bridge_between(a, b), Some(edge));
    assert!(!graph.two_edge_connected(a, b));
    assert_eq!(graph.two_edge_component_size(a), 1);
    assert_eq!(graph.component_size(a), 2);
    assert!(naive.is_bridge(edge.index()));
    assert_eq!(naive.bridge_in_component(a.index()), Some(edge.index()));
    assert_eq!(
        naive.bridge_separating(a.index(), b.index()),
        Some(edge.index())
    );
}

#[test]
fn triangle_has_no_bridges() {
    let (mut graph, mut naive, vertices) = new_graph(3);
    insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    insert_both(&mut graph, &mut naive, vertices[1], vertices[2]);
    insert_both(&mut graph, &mut naive, vertices[2], vertices[0]);

    assert!(naive.bridges().is_empty());
    for &u in &vertices {
        assert_eq!(graph.find_bridge(u), None);
        assert_eq!(graph.two_edge_component_size(u), 3);
        for &v in &vertices {
            assert!(graph.two_edge_connected(u, v));
            assert_eq!(graph.find_bridge_between(u, v), None);
        }
    }
}

#[test]
fn parallel_edges_remove_bridge() {
    let (mut graph, mut naive, vertices) = new_graph(2);
    let a = vertices[0];
    let b = vertices[1];
    let first = insert_both(&mut graph, &mut naive, a, b);
    let second = insert_both(&mut graph, &mut naive, a, b);

    assert_eq!(graph.edge_count(), 2);
    assert!(naive.bridges().is_empty());
    assert_eq!(graph.find_bridge(a), None);
    assert!(graph.two_edge_connected(a, b));
    assert_eq!(graph.two_edge_component_size(a), 2);

    assert!(graph.delete_edge(second));
    assert!(naive.delete(second.index()));
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.find_bridge(a), Some(first));
    assert_eq!(graph.find_bridge_between(a, b), Some(first));
    assert!(!graph.two_edge_connected(a, b));
    assert!(naive.is_bridge(first.index()));
}

#[test]
fn self_loop_rejected() {
    let (mut graph, _naive, vertices) = new_graph(1);
    let a = vertices[0];
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.insert_edge(a, a), None);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.find_bridge(a), None);
    assert_eq!(graph.component_size(a), 1);
}

fn assert_tree_properties(
    graph: &mut DynamicGraph,
    naive: &Naive,
    vertices: &[VertexId],
    edges: &[EdgeId],
    description: &str,
) {
    assert_eq!(graph.edge_count(), edges.len(), "{description}: edge count");
    assert_eq!(
        naive.bridges().len(),
        edges.len(),
        "{description}: bridge count"
    );
    for &edge in edges {
        assert!(
            naive.is_bridge(edge.index()),
            "{description}: edge {edge:?}"
        );
    }

    for &u in vertices {
        assert_eq!(
            graph.component_size(u),
            vertices.len() as u64,
            "{description}"
        );
        assert_eq!(graph.two_edge_component_size(u), 1, "{description}");
        assert_returned_bridge_in_component(naive, u.index(), graph.find_bridge(u));
        for &v in vertices {
            assert_eq!(graph.connected(u, v), true, "{description}");
            assert_eq!(
                graph.two_edge_connected(u, v),
                u == v,
                "{description}: two_edge_connected({}, {})",
                u.index(),
                v.index()
            );
            if u != v {
                let bridge = graph
                    .find_bridge_between(u, v)
                    .expect("distinct vertices in a tree are separated by a bridge");
                assert_returned_separating_bridge(naive, u.index(), v.index(), Some(bridge));
            }
        }
    }
}

#[test]
fn tree_all_edges_bridges() {
    let (mut path, mut path_naive, path_vertices) = new_graph(6);
    let mut path_edges = Vec::new();
    for index in 0..path_vertices.len() - 1 {
        path_edges.push(insert_both(
            &mut path,
            &mut path_naive,
            path_vertices[index],
            path_vertices[index + 1],
        ));
    }
    assert_tree_properties(&mut path, &path_naive, &path_vertices, &path_edges, "path");

    let (mut star, mut star_naive, star_vertices) = new_graph(6);
    let mut star_edges = Vec::new();
    for &leaf in &star_vertices[1..] {
        star_edges.push(insert_both(
            &mut star,
            &mut star_naive,
            star_vertices[0],
            leaf,
        ));
    }
    assert_tree_properties(&mut star, &star_naive, &star_vertices, &star_edges, "star");
}

#[test]
fn disconnected_components() {
    let (mut graph, mut naive, vertices) = new_graph(7);
    insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    insert_both(&mut graph, &mut naive, vertices[1], vertices[2]);
    insert_both(&mut graph, &mut naive, vertices[3], vertices[4]);
    insert_both(&mut graph, &mut naive, vertices[4], vertices[5]);
    insert_both(&mut graph, &mut naive, vertices[5], vertices[3]);

    assert!(!graph.connected(vertices[0], vertices[3]));
    assert!(!graph.connected(vertices[0], vertices[6]));
    assert_eq!(graph.component_size(vertices[0]), 3);
    assert_eq!(graph.component_size(vertices[4]), 3);
    assert_eq!(graph.component_size(vertices[6]), 1);
    assert_eq!(graph.two_edge_component_size(vertices[0]), 1);
    assert_eq!(graph.two_edge_component_size(vertices[4]), 3);
    assert_eq!(graph.two_edge_component_size(vertices[6]), 1);
    assert!(graph.find_bridge(vertices[0]).is_some());
    assert_eq!(graph.find_bridge(vertices[4]), None);
    assert_eq!(graph.find_bridge(vertices[6]), None);
    assert_eq!(graph.find_bridge_between(vertices[0], vertices[3]), None);
    assert_returned_bridge_in_component(
        &naive,
        vertices[0].index(),
        graph.find_bridge(vertices[0]),
    );
}

#[test]
fn delete_bridge_disconnects() {
    let (mut graph, mut naive, vertices) = new_graph(4);
    insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    let deleted = insert_both(&mut graph, &mut naive, vertices[1], vertices[2]);
    insert_both(&mut graph, &mut naive, vertices[2], vertices[3]);

    assert!(graph.delete_edge(deleted));
    assert!(naive.delete(deleted.index()));
    assert_eq!(graph.edge_count(), 2);
    assert!(!graph.connected(vertices[1], vertices[2]));
    assert_eq!(graph.component_size(vertices[0]), 2);
    assert_eq!(graph.component_size(vertices[1]), 2);
    assert_eq!(graph.component_size(vertices[2]), 2);
    assert_eq!(graph.component_size(vertices[3]), 2);
    assert!(graph.find_bridge(vertices[0]).is_some());
    assert!(graph.find_bridge(vertices[3]).is_some());
    assert_eq!(graph.find_bridge_between(vertices[0], vertices[3]), None);
}

#[test]
fn delete_non_bridge_uses_replacement() {
    let (mut graph, mut naive, vertices) = new_graph(4);
    // K4 has overlapping cycles. Deleting AB leaves a bridgeless graph; a
    // single cycle would instead open into a path and create bridges.
    let ab = insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    insert_both(&mut graph, &mut naive, vertices[1], vertices[2]);
    insert_both(&mut graph, &mut naive, vertices[2], vertices[3]);
    insert_both(&mut graph, &mut naive, vertices[3], vertices[0]);
    insert_both(&mut graph, &mut naive, vertices[0], vertices[2]);
    insert_both(&mut graph, &mut naive, vertices[1], vertices[3]);

    assert!(naive.bridges().is_empty());
    assert!(graph.delete_edge(ab));
    assert!(naive.delete(ab.index()));
    assert_eq!(graph.edge_count(), 5);
    assert!(naive.bridges().is_empty());
    for &u in &vertices {
        assert_eq!(graph.find_bridge(u), None);
        assert_eq!(graph.two_edge_component_size(u), 4);
        for &v in &vertices {
            assert!(graph.connected(u, v));
            assert!(graph.two_edge_connected(u, v));
            assert_eq!(graph.find_bridge_between(u, v), None);
        }
    }
}

#[test]
fn delete_parallel_edge_keeps_bridge_status() {
    let (mut graph, mut naive, vertices) = new_graph(2);
    let tree_edge = insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    let replacement = insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);

    assert_eq!(graph.find_bridge(vertices[0]), None);
    assert!(graph.delete_edge(tree_edge));
    assert!(naive.delete(tree_edge.index()));

    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.edge_endpoints(tree_edge), None);
    assert_endpoints_match(&graph, replacement, vertices[0], vertices[1]);
    assert_eq!(graph.find_bridge(vertices[0]), Some(replacement));
    assert_eq!(
        graph.find_bridge_between(vertices[0], vertices[1]),
        Some(replacement)
    );
    assert!(!graph.two_edge_connected(vertices[0], vertices[1]));
    assert!(naive.is_bridge(replacement.index()));
}

#[test]
fn invalid_handles_return_none() {
    let (mut graph, _naive, vertices) = new_graph(2);
    let a = vertices[0];
    let b = vertices[1];
    let live_edge = graph.insert_edge(a, b).unwrap();
    let invalid_vertex = VertexId(usize::MAX);
    let invalid_edge = EdgeId(usize::MAX);

    assert_eq!(graph.insert_edge(invalid_vertex, a), None);
    assert_eq!(graph.insert_edge(a, invalid_vertex), None);
    assert_eq!(graph.insert_edge(invalid_vertex, invalid_vertex), None);
    assert_eq!(graph.edge_count(), 1);
    assert!(!graph.connected(a, invalid_vertex));
    assert!(!graph.connected(invalid_vertex, invalid_vertex));
    assert_eq!(graph.find_bridge(invalid_vertex), None);
    assert_eq!(graph.find_bridge_between(a, invalid_vertex), None);
    assert_eq!(graph.find_bridge_between(invalid_vertex, a), None);
    assert_eq!(graph.component_size(invalid_vertex), 0);
    assert_eq!(graph.two_edge_component_size(invalid_vertex), 0);
    assert!(!graph.two_edge_connected(a, invalid_vertex));
    assert!(!graph.two_edge_connected(invalid_vertex, invalid_vertex));
    assert_eq!(graph.edge_endpoints(invalid_edge), None);
    assert_eq!(graph.edge_endpoints(EdgeId(1)), None);
    assert!(!graph.delete_edge(invalid_edge));
    assert!(!graph.delete_edge(EdgeId(1)));
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.edge_endpoints(live_edge), Some((a, b)));
}

#[test]
fn delete_twice_returns_false() {
    let (mut graph, mut naive, vertices) = new_graph(2);
    let edge = insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);

    assert!(graph.delete_edge(edge));
    assert!(naive.delete(edge.index()));
    assert!(!graph.delete_edge(edge));
    assert!(!naive.delete(edge.index()));
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.edge_endpoints(edge), None);
    assert!(!graph.connected(vertices[0], vertices[1]));
}

#[test]
fn edge_endpoints_roundtrip() {
    let (mut graph, mut naive, vertices) = new_graph(3);
    let first = insert_both(&mut graph, &mut naive, vertices[0], vertices[1]);
    let second = insert_both(&mut graph, &mut naive, vertices[2], vertices[1]);
    let parallel = insert_both(&mut graph, &mut naive, vertices[1], vertices[0]);

    assert_endpoints_match(&graph, first, vertices[0], vertices[1]);
    assert_endpoints_match(&graph, second, vertices[2], vertices[1]);
    assert_endpoints_match(&graph, parallel, vertices[1], vertices[0]);
    assert_eq!(first.index(), 0);
    assert_eq!(second.index(), 1);
    assert_eq!(parallel.index(), 2);

    assert!(graph.delete_edge(second));
    assert!(naive.delete(second.index()));
    assert_eq!(graph.edge_endpoints(second), None);
}

fn fixed_graphs() -> Vec<(usize, Vec<(usize, usize)>)> {
    vec![
        (6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        (
            7,
            vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)],
        ),
        (6, vec![(0, 1), (0, 1), (1, 2), (2, 3), (3, 4), (4, 2)]),
        (5, vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]),
    ]
}

#[test]
fn component_size_matches_naive() {
    for (case, (vertex_count, edges)) in fixed_graphs().into_iter().enumerate() {
        let (mut graph, mut naive, vertices) = new_graph(vertex_count);
        for (u, v) in edges {
            insert_both(&mut graph, &mut naive, vertices[u], vertices[v]);
        }
        for (vertex, &handle) in vertices.iter().enumerate() {
            assert_eq!(
                graph.component_size(handle) as usize,
                naive.component_size(vertex),
                "fixture {case}, vertex {vertex}"
            );
        }
    }
}

#[test]
fn two_edge_component_size_matches_naive() {
    for (case, (vertex_count, edges)) in fixed_graphs().into_iter().enumerate() {
        let (mut graph, mut naive, vertices) = new_graph(vertex_count);
        for (u, v) in edges {
            insert_both(&mut graph, &mut naive, vertices[u], vertices[v]);
        }
        for (vertex, &handle) in vertices.iter().enumerate() {
            assert_eq!(
                graph.two_edge_component_size(handle) as usize,
                naive.two_edge_component_size(vertex),
                "fixture {case}, vertex {vertex}"
            );
        }
    }
}

#[test]
fn two_edge_connected_matches_naive() {
    for (case, (vertex_count, edges)) in fixed_graphs().into_iter().enumerate() {
        let (mut graph, mut naive, vertices) = new_graph(vertex_count);
        for (u, v) in edges {
            insert_both(&mut graph, &mut naive, vertices[u], vertices[v]);
        }
        for (u, &u_handle) in vertices.iter().enumerate() {
            for (v, &v_handle) in vertices.iter().enumerate() {
                assert_eq!(
                    graph.two_edge_connected(u_handle, v_handle),
                    naive.two_edge_connected(u, v),
                    "fixture {case}, pair ({u}, {v})"
                );
                assert_returned_separating_bridge(
                    &naive,
                    u,
                    v,
                    graph.find_bridge_between(u_handle, v_handle),
                );
            }
        }
        for (vertex, &handle) in vertices.iter().enumerate() {
            assert_returned_bridge_in_component(&naive, vertex, graph.find_bridge(handle));
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn index(&mut self, length: usize) -> usize {
        assert!(length > 0);
        (self.next() as usize) % length
    }
}

fn random_distinct_pair(rng: &mut Rng, vertex_count: usize) -> (usize, usize) {
    assert!(vertex_count > 1);
    let u = rng.index(vertex_count);
    let mut v = rng.index(vertex_count - 1);
    if v >= u {
        v += 1;
    }
    (u, v)
}

fn choose_insert_pair(rng: &mut Rng, naive: &Naive, parallel_bias: u64) -> (usize, usize) {
    if !naive.edges.is_empty() && rng.next() % 100 < parallel_bias {
        let edge_index = rng.index(naive.edges.len());
        *naive.edges.values().nth(edge_index).unwrap()
    } else {
        random_distinct_pair(rng, naive.vertex_count())
    }
}

const RANDOMIZED_STEPS: usize = 180;
const EXHAUSTIVE_CHECK_INTERVAL: usize = 3;

fn assert_randomized_sample_matches_naive(
    graph: &mut DynamicGraph,
    naive: &Naive,
    case: usize,
    step: usize,
    context: &str,
) {
    assert_eq!(
        graph.vertex_count(),
        naive.vertex_count(),
        "{context}: vertex count"
    );
    assert_eq!(
        graph.edge_count(),
        naive.edges.len(),
        "{context}: edge count"
    );
    assert_eq!(
        graph.edges.len(),
        naive.next_edge,
        "{context}: edge handle count"
    );

    let vertex_count = naive.vertex_count();
    let first = (step.wrapping_mul(7) + case) % vertex_count;
    let second = (step.wrapping_mul(13) + case + 1) % vertex_count;
    let third = (step.wrapping_mul(19) + case + 2) % vertex_count;

    for vertex in [first, second] {
        assert_eq!(
            graph.component_size(VertexId(vertex)) as usize,
            naive.component_size(vertex),
            "{context}: sampled component_size({vertex})"
        );
    }
    for (u, v) in [(first, second), (second, third)] {
        assert_eq!(
            graph.connected(VertexId(u), VertexId(v)),
            naive.connected(u, v),
            "{context}: sampled connected({u}, {v})"
        );
    }
}

fn run_randomized_case(seed: u64, parallel_edges_bias: bool, case: usize) {
    let mut rng = Rng(seed);
    let initial_vertices = 8 + rng.index(5);
    let (mut graph, mut naive, _) = new_graph(initial_vertices);

    for step in 0..RANDOMIZED_STEPS {
        let roll = rng.next() % 100;
        let operation = if step == RANDOMIZED_STEPS / 3 || step == 2 * RANDOMIZED_STEPS / 3 {
            let graph_vertex = graph.add_vertex();
            let naive_vertex = naive.add_vertex();
            assert_eq!(graph_vertex.index(), naive_vertex);
            format!("scheduled add_vertex -> {naive_vertex}")
        } else {
            match roll {
                0 => {
                    let graph_vertex = graph.add_vertex();
                    let naive_vertex = naive.add_vertex();
                    assert_eq!(graph_vertex.index(), naive_vertex);
                    format!("add_vertex -> {naive_vertex}")
                }
                1..=40 => {
                    let parallel_bias = if parallel_edges_bias { 75 } else { 12 };
                    let (u, v) = choose_insert_pair(&mut rng, &naive, parallel_bias);
                    let edge = insert_both(&mut graph, &mut naive, VertexId(u), VertexId(v));
                    format!("insert_edge({u}, {v}) -> {}", edge.index())
                }
                41..=74 => {
                    let live_edges = naive.edges.keys().copied().collect::<Vec<_>>();
                    if live_edges.is_empty() {
                        let unused = EdgeId(naive.next_edge);
                        assert!(!graph.delete_edge(unused));
                        assert!(!naive.delete(unused.index()));
                        format!("delete_edge({}) -> false", unused.index())
                    } else {
                        let index = live_edges[rng.index(live_edges.len())];
                        let graph_deleted = graph.delete_edge(EdgeId(index));
                        let naive_deleted = naive.delete(index);
                        assert_eq!(graph_deleted, naive_deleted);
                        assert!(graph_deleted);
                        format!("delete_edge({index}) -> true")
                    }
                }
                _ => {
                    let u = rng.index(naive.vertex_count());
                    let v = rng.index(naive.vertex_count());
                    match rng.next() % 5 {
                        0 => assert_eq!(
                            graph.connected(VertexId(u), VertexId(v)),
                            naive.connected(u, v)
                        ),
                        1 => assert_eq!(
                            graph.two_edge_connected(VertexId(u), VertexId(v)),
                            naive.two_edge_connected(u, v)
                        ),
                        2 => assert_eq!(
                            graph.component_size(VertexId(u)) as usize,
                            naive.component_size(u)
                        ),
                        3 => {
                            let result = graph.find_bridge(VertexId(u));
                            assert_returned_bridge_in_component(&naive, u, result);
                        }
                        _ => {
                            let result = graph.find_bridge_between(VertexId(u), VertexId(v));
                            assert_returned_separating_bridge(&naive, u, v, result);
                        }
                    }
                    format!("query({u}, {v})")
                }
            }
        };

        let context = format!(
            "seed {seed:#x}, case {case}, step {step}, {operation}; live edges {:?}",
            naive.edges
        );
        assert_internal_invariants(&graph, &context);
        if (step + 1) % EXHAUSTIVE_CHECK_INTERVAL == 0 || step + 1 == RANDOMIZED_STEPS {
            assert_matches_naive(&mut graph, &naive, &context);
        } else {
            assert_randomized_sample_matches_naive(&mut graph, &naive, case, step, &context);
        }
    }
}

#[test]
fn randomized_differential() {
    for (case, seed) in [0x1234_5678, 0x5eed_cafe, 0xdead_beef]
        .into_iter()
        .enumerate()
    {
        run_randomized_case(seed, false, case);
    }
}

#[test]
fn randomized_differential_with_parallel_edges() {
    for (case, seed) in [0xa5a5_0123, 0xc001_d00d, 0xfade_9876]
        .into_iter()
        .enumerate()
    {
        run_randomized_case(seed, true, case);
    }
}
