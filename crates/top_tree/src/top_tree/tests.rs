use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use super::*;

/// The top tree no longer carries user supplied edge weights: the payload of a
/// forest edge is the index of its top tree leaf. To keep the summaries below
/// exercising non-trivial per-edge values, we derive a deterministic "weight"
/// from an edge's endpoints. Along the path `0-1-2-...` this yields `1, 2, ...`,
/// matching the weights the old weight-carrying tests used.
fn edge_weight(u: usize, v: usize) -> u64 {
    (u.min(v) + 1) as u64
}

/// A monoid summary over the whole cluster: number of tree edges and xor of
/// their weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Agg {
    edges: u32,
    xor: u64,
}

impl Summary for Agg {
    type Tag = ();

    fn tree_edge(u: usize, v: usize) -> Self {
        Agg {
            edges: 1,
            xor: edge_weight(u, v),
        }
    }

    fn label(_v: usize) -> Self {
        Agg { edges: 0, xor: 0 }
    }

    fn combine(left: &Self, right: &Self, _ctx: &MergeContext) -> Self {
        Agg {
            edges: left.edges + right.edges,
            xor: left.xor ^ right.xor,
        }
    }
}

/// A boundary aware summary over the cluster path: the number of edges on the
/// cluster path. This exercises the `is_compress`/`is_rake` distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathLen {
    len: u32,
}

impl Summary for PathLen {
    type Tag = ();

    fn tree_edge(_u: usize, _v: usize) -> Self {
        PathLen { len: 1 }
    }

    fn label(_v: usize) -> Self {
        PathLen { len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            PathLen {
                len: u32::from(ctx.left_is_path_child()) * left.len
                    + u32::from(ctx.right_is_path_child()) * right.len,
            }
        } else {
            PathLen { len: 0 }
        }
    }
}

/// An aggregate that includes every leaf, so a label summary mutation is
/// visible from an exposed path containing that label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LabelValue {
    value: u64,
}

impl Summary for LabelValue {
    type Tag = ();

    fn tree_edge(_u: usize, _v: usize) -> Self {
        LabelValue { value: 0 }
    }

    fn label(_v: usize) -> Self {
        LabelValue { value: 1 }
    }

    fn combine(left: &Self, right: &Self, _ctx: &MergeContext) -> Self {
        LabelValue {
            value: left.value + right.value,
        }
    }
}

/// The sum of the edge weights on the cluster path, with a lazy "add `x` to
/// every edge on the path" tag. This is the mechanism used by cover-level
/// style algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathSum {
    sum: i64,
    len: u32,
}

impl Summary for PathSum {
    type Tag = i64;

    fn tree_edge(u: usize, v: usize) -> Self {
        PathSum {
            sum: edge_weight(u, v) as i64,
            len: 1,
        }
    }

    fn label(_v: usize) -> Self {
        PathSum { sum: 0, len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            let mut sum = 0;
            let mut len = 0;
            if ctx.left_is_path_child() {
                sum += left.sum;
                len += left.len;
            }
            if ctx.right_is_path_child() {
                sum += right.sum;
                len += right.len;
            }
            PathSum { sum, len }
        } else {
            PathSum { sum: 0, len: 0 }
        }
    }

    fn apply(&mut self, tag: &i64) {
        self.sum += tag * self.len as i64;
    }

    fn compose(tag: &mut i64, parent: &i64) {
        *tag += *parent;
    }
}

/// The maximum edge weight on the cluster path (`0` for point clusters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathMax {
    max: u64,
}

impl Summary for PathMax {
    type Tag = ();

    fn tree_edge(u: usize, v: usize) -> Self {
        PathMax {
            max: edge_weight(u, v) * 10,
        }
    }

    fn label(_v: usize) -> Self {
        PathMax { max: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            let mut max = 0;
            if ctx.left_is_path_child() {
                max = max.max(left.max);
            }
            if ctx.right_is_path_child() {
                max = max.max(right.max);
            }
            PathMax { max }
        } else {
            PathMax { max: 0 }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectedPath {
    edges: Vec<(usize, usize)>,
}

impl Summary for DirectedPath {
    type Tag = ();

    fn tree_edge(u: usize, v: usize) -> Self {
        DirectedPath {
            edges: vec![(u, v)],
        }
    }

    fn label(_v: usize) -> Self {
        DirectedPath { edges: Vec::new() }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        let mut edges = Vec::new();
        if ctx.left_is_path_child() {
            edges.extend_from_slice(&left.edges);
        }
        if ctx.right_is_path_child() {
            edges.extend_from_slice(&right.edges);
        }
        DirectedPath { edges }
    }

    fn flip(&mut self) {
        self.edges = self.edges.drain(..).rev().map(|(u, v)| (v, u)).collect();
    }
}

struct Harness<S: Summary> {
    tt: TopTree<S>,
    adj: HashMap<tree::VertexId, HashSet<tree::VertexId>>,
}

impl<S: Summary> Harness<S> {
    fn new(n: usize) -> Self {
        let mut tt = TopTree::new();
        let mut adj = HashMap::new();
        for i in 0..n {
            let id = tt.add_vertex();
            adj.insert(id, Default::default());
        }
        Self { tt, adj }
    }

    fn link(&mut self, u: tree::VertexId, v: tree::VertexId) {
        self.tt.link(u, v);
        self.adj.get_mut(&u).unwrap().insert(v);
        self.adj.get_mut(&v).unwrap().insert(u);
    }

    fn cut(&mut self, u: tree::VertexId, v: tree::VertexId) {
        self.tt.cut(u, v);
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.adj.get_mut(&v).unwrap().remove(&u);
    }

    fn components(&self) -> Vec<Vec<tree::VertexId>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for &start in self.adj.keys() {
            if !seen.insert(start) {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(v) = queue.pop_front() {
                comp.push(v);
                for &w in &self.adj[&v] {
                    if seen.insert(w) {
                        queue.push_back(w);
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    fn component_edges(&self, v: tree::VertexId) -> Vec<(tree::VertexId, tree::VertexId)> {
        let comp = self
            .components()
            .into_iter()
            .find(|c| c.contains(&v))
            .unwrap();
        let mut edges: Vec<_> = comp
            .iter()
            .flat_map(|&u| {
                self.adj[&u]
                    .iter()
                    .filter(move |&&w| u < w)
                    .map(move |&w| (u, w))
            })
            .collect();
        edges.sort();
        edges
    }
}

fn live_roots<S: Summary>(tt: &TopTree<S>) -> Vec<ClusterId> {
    let clusters = (0..tt.tree.nodes.len())
        .map(|i| tt.tree.vertex_id_from_index(i))
        .filter_map(|l| tt.find_root(l))
        .collect::<HashSet<_>>();
    clusters.into_iter().collect()
}

fn collect_leaves<S: Summary>(
    tt: &TopTree<S>,
    node: ClusterId,
    leaves: &mut Vec<ClusterId>,
    nodes: &mut usize,
) {
    *nodes += 1;
    let cluster = tt.cl(node);
    if let Some(Children { left, right }) = cluster.children {
        assert_eq!(tt.cl(left).parent, Some(node), "left child parent mismatch");
        assert_eq!(
            tt.cl(right).parent,
            Some(node),
            "right child parent mismatch"
        );
        collect_leaves(tt, left, leaves, nodes);
        collect_leaves(tt, right, leaves, nodes);
    } else {
        leaves.push(node);
    }
}

/// The boundary vertices of a node in the three "slots" used by the classic
/// splay top tree: a left boundary, a (possibly shared) middle boundary, and a
/// right boundary. At most two slots are ever populated.
#[derive(Clone, Copy, Debug, Default)]
struct Boundaries {
    left: Option<tree::VertexId>,
    mid: Option<tree::VertexId>,
    right: Option<tree::VertexId>,
}

impl Boundaries {
    fn count(&self) -> usize {
        usize::from(self.left.is_some())
            + usize::from(self.mid.is_some())
            + usize::from(self.right.is_some())
    }

    fn leftmost(&self) -> Option<tree::VertexId> {
        self.left.or(self.mid)
    }

    fn rightmost(&self) -> Option<tree::VertexId> {
        self.right.or(self.mid)
    }

    fn set(&self) -> BTreeSet<tree::VertexId> {
        [self.left, self.mid, self.right]
            .into_iter()
            .flatten()
            .collect()
    }

    fn flipped(&self) -> Self {
        Self {
            left: self.right,
            mid: self.mid,
            right: self.left,
        }
    }
}

/// Counts the leaves below `node` incident to `vertex`.
fn count_incident_leaves<S: Summary>(
    tt: &TopTree<S>,
    node: ClusterId,
    vertex: tree::VertexId,
) -> usize {
    match tt.cl(node).data {
        ClusterData::Edge(edge) => {
            let (u, v) = tt.tree.edge_endpoints(edge).expect("edge must exist");
            usize::from(u == vertex || v == vertex)
        }
        ClusterData::Node(label) => {
            usize::from(tt.tree.label_vertex(label).expect("label must exist") == vertex)
        }
        ClusterData::Internal => {
            let Children { left, right } = tt.cl(node).children.expect("internal has children");
            count_incident_leaves(tt, left, vertex) + count_incident_leaves(tt, right, vertex)
        }
    }
}

/// Recomputes the boundary vertices of `node` from scratch (from the underlying
/// forest and the exposed bits) and checks them against the stored value.
///
/// The orientation invariant from the paper is a property of the *materialized*
/// tree, i.e. the frame obtained by pushing every pending flip from the root
/// down. `parity` accumulates the flip bits of the proper ancestors of `node`,
/// so the node's effective flip is `parity ^ cluster.flipped`. The returned
/// `Boundaries` are expressed in that materialized frame; the stored value is in
/// the node's local frame and must equal the returned value flipped by the
/// effective flip.
///
/// The invariant checked for every internal node is: the rightmost boundary of
/// the materialized left child and the leftmost boundary of the materialized
/// right child must both exist and equal the central vertex.
fn check_node_boundaries<S: Summary>(tt: &TopTree<S>, node: ClusterId) -> Boundaries {
    check_node_boundaries_in(tt, node, false)
}

fn check_node_boundaries_in<S: Summary>(
    tt: &TopTree<S>,
    node: ClusterId,
    parity: bool,
) -> Boundaries {
    let cluster = tt.cl(node);
    let effective_flip = parity ^ cluster.flipped;

    let materialized = match cluster.data {
        ClusterData::Edge(edge) => {
            let (left, right) = tt.tree.edge_endpoints(edge).expect("edge must exist");
            // Materialized left endpoint is `endpoints[effective_flip]`.
            let (left, right) = if effective_flip {
                (right, left)
            } else {
                (left, right)
            };
            let mut c = Boundaries::default();
            if tt.is_boundary_vertex(left) {
                c.left = Some(left);
            }
            if tt.is_boundary_vertex(right) {
                c.right = Some(right);
            }
            c
        }
        ClusterData::Node(label) => {
            let vertex = tt.tree.label_vertex(label).expect("label must exist");
            let mut c = Boundaries::default();
            if tt.is_boundary_vertex(vertex) {
                c.mid = Some(vertex);
            }
            c
        }
        ClusterData::Internal => {
            let Children { left, right } = cluster.children.expect("internal has children");
            // Materialized left/right child of the node.
            let (materialized_left, materialized_right) = if effective_flip {
                (right, left)
            } else {
                (left, right)
            };

            let bl = check_node_boundaries_in(tt, materialized_left, effective_flip);
            let br = check_node_boundaries_in(tt, materialized_right, effective_flip);

            assert_eq!(
                bl.rightmost(),
                br.leftmost(),
                "children of internal node {node:?} must share a central boundary vertex"
            );
            let central = bl.rightmost().unwrap_or_else(|| {
                panic!("left child must have a rightmost boundary at node {node:?}")
            });

            let mut c = Boundaries::default();
            let inside = count_incident_leaves(tt, node, central);
            if tt.exposed.get(central.index()) || inside < tt.tree.degree(central) {
                c.mid = Some(central);
            }
            if bl.leftmost() != bl.rightmost() {
                c.left = bl.leftmost();
            }
            if br.leftmost() != br.rightmost() {
                c.right = br.rightmost();
            }
            c
        }
    };

    // The stored value lives in the node's local frame.
    let expected = if effective_flip {
        materialized.flipped()
    } else {
        materialized
    };
    let stored = cluster.boundary_vertices;
    assert_eq!(
        expected.count() as u8,
        stored.count(),
        "boundary count mismatch at node {node:?} ({:?})",
        cluster.data
    );

    let expected_set: BTreeSet<tree::VertexId> = expected.set();
    let stored_set: BTreeSet<tree::VertexId> = match stored {
        BoundaryVertices::None => BTreeSet::new(),
        BoundaryVertices::One(v) => BTreeSet::from([v]),
        BoundaryVertices::Two { left, right } => BTreeSet::from([left, right]),
    };
    assert_eq!(
        expected_set, stored_set,
        "boundary vertices mismatch at node {node} ({:?})",
        cluster.data
    );

    let mapped = match [expected.left, expected.mid, expected.right]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => BoundaryVertices::None,
        [v] => BoundaryVertices::One(*v),
        [l, r] => BoundaryVertices::Two {
            left: *l,
            right: *r,
        },
        other => panic!("more than two boundary vertices {other:?} at node {node:?}"),
    };
    assert_eq!(
        stored, mapped,
        "boundary orientation mismatch at node {node} ({:?})",
        cluster.data
    );

    if matches!(cluster.data, ClusterData::Internal) {
        assert_eq!(
            tt.has_middle_boundary(node),
            materialized.mid.is_some(),
            "has_middle_boundary mismatch at node {node}"
        );
    }

    materialized
}

fn check_invariants<S: Summary>(h: &Harness<S>) {
    let tt = &h.tt;

    for (i, node) in tt.clusters.iter().enumerate() {
        if let Some(cluster) = node
            && let Some(parent) = cluster.parent.map(NonMaxUsize::get)
        {
            let p = tt.clusters[parent].as_ref().expect("parent must be live");
            let is_child = p
                .children
                .is_some_and(|c| c.left.get() == i || c.right.get() == i);
            assert!(is_child, "node {i} is not a child of its parent {parent}");
        }
    }

    // Every live node's stored boundary vertices must match the forest.
    for root in live_roots(tt) {
        check_node_boundaries(tt, root);
    }

    let mut total_nodes = 0;
    let mut covered_edges = BTreeSet::new();
    let mut covered_labels = BTreeSet::new();

    for root in live_roots(tt) {
        let mut leaves = Vec::new();
        let mut nodes = 0;
        collect_leaves(tt, root, &mut leaves, &mut nodes);
        assert_eq!(
            nodes,
            leaves.len() * 2 - 1,
            "a top tree with k leaves must have 2k-1 nodes"
        );
        total_nodes += nodes;

        // The boundaries of a component root are exactly its exposed vertices.
        for vertex in check_node_boundaries(tt, root).set() {
            assert!(
                tt.exposed.get(vertex),
                "root boundary vertex {vertex} must be exposed"
            );
        }

        for leaf in leaves {
            match tt.cl(leaf).data {
                ClusterData::Edge(edge) => {
                    assert!(covered_edges.insert(edge), "edge leaf appears twice");
                }
                ClusterData::Node(label) => {
                    assert!(covered_labels.insert(label), "label leaf appears twice");
                }
                ClusterData::Internal => panic!("leaf cannot be internal"),
            }
        }
    }

    let live_nodes = tt.clusters.iter().filter(|n| n.is_some()).count();
    assert_eq!(total_nodes, live_nodes, "all live nodes must be reachable");

    assert_eq!(
        covered_edges.len(),
        tt.tree.edge_count(),
        "every tree edge must occur exactly once"
    );
    assert_eq!(
        covered_labels.len(),
        tt.tree.label_count(),
        "every label must occur exactly once"
    );

    // Verify that the edges of each top tree component match the mirror graph.
    for root in live_roots(tt) {
        let mut leaves = Vec::new();
        let mut nodes = 0;
        collect_leaves(tt, root, &mut leaves, &mut nodes);

        let mut top_edges = BTreeSet::new();
        let mut sample = None;
        for leaf in leaves {
            if let ClusterData::Edge(edge) = tt.cl(leaf).data {
                let (u, v) = tt.tree.edge_endpoints(edge).unwrap();
                top_edges.insert((u.min(v), u.max(v)));
                sample = Some(u);
            }
        }
        if let Some(sample) = sample {
            let expected: BTreeSet<_> = h.component_edges(sample).into_iter().collect();
            assert_eq!(top_edges, expected, "top tree component edges mismatch");
        }
    }
}

fn path_edges(h: &Harness<impl Summary>, u: usize, v: usize) -> Option<Vec<(usize, usize)>> {
    let mut prev: BTreeMap<usize, usize> = BTreeMap::new();
    let mut seen = BTreeSet::from([u]);
    let mut queue = VecDeque::from([u]);
    while let Some(x) = queue.pop_front() {
        if x == v {
            let mut edges = Vec::new();
            let mut cur = v;
            while cur != u {
                let p = prev[&cur];
                edges.push((p.min(cur), p.max(cur)));
                cur = p;
            }
            edges.reverse();
            return Some(edges);
        }
        for &w in &h.adj[&x] {
            if seen.insert(w) {
                prev.insert(w, x);
                queue.push_back(w);
            }
        }
    }
    None
}

#[test]
fn single_edge_summary() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);

    assert_eq!(h.tt.expose(0), Some(PathLen { len: 1 }));
    check_invariants(&h);
    assert_eq!(h.tt.expose(1), Some(PathLen { len: 1 }));
    check_invariants(&h);

    h.tt.cut(0, 1);
    assert_eq!(h.tt.edge_count(), 0);
    check_invariants(&h);
}

#[test]
fn path_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
        check_invariants(&h);
    }

    for (u, v, expected) in [(0, 5, 5), (0, 3, 3), (2, 5, 3), (4, 4, 0)] {
        assert_eq!(h.tt.expose_path(u, v), Some(PathLen { len: expected }));
        check_invariants(&h);
        h.tt.deexpose(v);
        h.tt.deexpose(u);
        check_invariants(&h);
    }

    for i in (1..6).rev() {
        h.cut(i - 1, i);
        check_invariants(&h);
    }
}

#[test]
fn update_label_summary_recomputes_ancestors_without_relinking() {
    let mut h = Harness::<LabelValue>::new(2);
    h.link(0, 1);
    h.tt.attach(0, 7);

    let roots_before = live_roots(&h.tt);
    let structure_before: Vec<_> =
        h.tt.clusters
            .iter()
            .map(|node| {
                node.as_ref()
                    .map(|cluster| (cluster.parent, cluster.children, cluster.data))
            })
            .collect();

    h.tt.update_label_summary(&7, |summary| summary.value = 7);

    assert_eq!(live_roots(&h.tt), roots_before);
    let structure_after: Vec<_> =
        h.tt.clusters
            .iter()
            .map(|node| {
                node.as_ref()
                    .map(|cluster| (cluster.parent, cluster.children, cluster.data))
            })
            .collect();
    assert_eq!(structure_after, structure_before);
    check_invariants(&h);

    assert_eq!(
        h.tt.expose_path(0, 1),
        Some(LabelValue { value: 7 }),
        "the exposed path aggregate must include the updated label summary"
    );
    check_invariants(&h);
}

#[test]
fn orientation_sensitive_summary_flips_with_cluster() {
    let mut tt: TopTree<u32, u32, DirectedPath> = TopTree::new();
    let u = tt.add_vertex(0);
    let v = tt.add_vertex(1);
    tt.link(u, v);

    // The payload of the forest edge is the index of its top tree leaf.
    let leaf = *tt.tree.edge_weight(0).expect("edge must exist");
    tt.toggle_flipped(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(1, 0)]);

    // Materializing the lazy flip must not reverse the already-flipped sum again.
    tt.push_flip(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(1, 0)]);

    tt.toggle_flipped(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(0, 1)]);
}

#[test]
fn star_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(0, i);
        check_invariants(&h);
    }

    for i in 1..6 {
        assert_eq!(h.tt.expose_path(0, i), Some(PathLen { len: 1 }));
        h.tt.deexpose(i);
        h.tt.deexpose(0);
        assert_eq!(h.tt.expose_path(i, 0), Some(PathLen { len: 1 }));
        h.tt.deexpose(0);
        h.tt.deexpose(i);
        check_invariants(&h);
    }
}

#[test]
fn component_aggregate_is_whole_component() {
    let mut h = Harness::<Agg>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
    }

    let expected_xor = (1..6u64).fold(0, |a, b| a ^ b);
    let summary = h.tt.expose_path(0, 5).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.tt.deexpose(5);
    h.tt.deexpose(0);

    // Even exposing a sub-path returns the whole component's summary, since
    // the root cluster still contains the whole tree.
    let summary = h.tt.expose_path(1, 3).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.tt.deexpose(3);
    h.tt.deexpose(1);

    assert_eq!(
        h.tt.component_summary(4).unwrap(),
        Agg {
            edges: 5,
            xor: expected_xor
        }
    );
}

#[test]
fn path_max_summary() {
    let mut h = Harness::<PathMax>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
    }
    for i in 1..6 {
        for j in i + 1..6 {
            let max = (i..j).map(|k| (k as u64 + 1) * 10).max().unwrap();
            assert_eq!(h.tt.expose_path(i, j), Some(PathMax { max }));
            h.tt.deexpose(j);
            h.tt.deexpose(i);
        }
    }
}

#[test]
fn lazy_path_tag_propagates() {
    let mut tt: TopTree<u32, u32, PathSum> = TopTree::new();
    for i in 0..5 {
        tt.add_vertex(i as u32);
    }
    for i in 1..5 {
        tt.link(i - 1, i);
    }

    // Path 1..3 has edges 2 and 3.
    assert_eq!(tt.expose_path(1, 3), Some(PathSum { sum: 5, len: 2 }));
    tt.deexpose(3);
    tt.deexpose(1);

    // Add 10 to every edge of path 1..3.
    let tagged = tt.expose_path_tagged(1, 3, 10);
    assert_eq!(tagged, Some(PathSum { sum: 25, len: 2 }));
    tt.deexpose(3);
    tt.deexpose(1);

    // The tag must be visible from a different path covering the same edges.
    assert_eq!(
        tt.expose_path(0, 4),
        Some(PathSum {
            sum: 1 + 12 + 13 + 4,
            len: 4
        })
    );
    tt.deexpose(4);
    tt.deexpose(0);

    assert_eq!(tt.expose_path(1, 2), Some(PathSum { sum: 12, len: 1 }));
    tt.deexpose(2);
    tt.deexpose(1);
}

#[test]
fn labels_count_towards_component() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);
    h.tt.attach(0, 9);
    check_invariants(&h);

    assert_eq!(h.tt.label_count(), 1);
    // With no exposed boundary vertices the root is a point cluster, so the
    // path length is zero; exposing the two endpoints gives the path.
    assert_eq!(h.tt.component_summary(0), Some(PathLen { len: 0 }));
    assert_eq!(h.tt.expose_path(0, 1), Some(PathLen { len: 1 }));
    h.tt.deexpose(1);
    h.tt.deexpose(0);

    h.tt.detach(&9);
    assert_eq!(h.tt.label_count(), 0);
    check_invariants(&h);
}

#[test]
fn label_on_isolated_vertex() {
    let mut h = Harness::<PathLen>::new(2);
    h.tt.attach(0, 5);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 1);

    h.tt.detach(&5);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 0);
}

#[test]
fn stable_label_handles() {
    let mut h = Harness::<PathLen>::new(3);
    h.link(0, 1);

    h.tt.attach(0, 1);
    h.tt.attach(0, 2);
    h.tt.attach(1, 3);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 3);

    // Removing `2` must not invalidate the other keys even though the
    // underlying label storage swap-removes.
    h.tt.detach(&2);
    assert_eq!(h.tt.label_count(), 2);
    check_invariants(&h);

    h.tt.detach(&1);
    check_invariants(&h);
    h.tt.detach(&3);
    assert_eq!(h.tt.label_count(), 0);
    check_invariants(&h);
}

#[test]
fn link_cut_and_expose_paths() {
    let mut h = Harness::<PathLen>::new(10);
    for i in 1..10 {
        h.link(i - 1, i);
    }
    for u in 0..10 {
        for v in u + 1..10 {
            let expected = (v - u) as u32;
            assert_eq!(
                h.tt.expose_path(u, v),
                Some(PathLen { len: expected }),
                "path length mismatch for {u}..{v}"
            );
            check_invariants(&h);
            h.tt.deexpose(v);
            h.tt.deexpose(u);
            check_invariants(&h);
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
}

#[test]
fn randomized_link_cut() {
    for seed in 0..25 {
        let mut rng = Rng(seed);
        let mut h = Harness::<PathLen>::new(9);

        for _ in 0..120 {
            let comps = h.components();
            let edges: Vec<(usize, usize)> = h
                .adj
                .iter()
                .flat_map(|(&u, nbrs)| nbrs.iter().map(move |&v| (u, v)))
                .filter(|(u, v)| u < v)
                .collect();

            match rng.next() % 3 {
                0 if comps.len() >= 2 => {
                    let a = (rng.next() as usize) % comps.len();
                    let mut b = (rng.next() as usize) % (comps.len() - 1);
                    if b >= a {
                        b += 1;
                    }
                    let u = comps[a][(rng.next() as usize) % comps[a].len()];
                    let v = comps[b][(rng.next() as usize) % comps[b].len()];
                    h.link(u, v);
                }
                1 if !edges.is_empty() => {
                    let (u, v) = edges[(rng.next() as usize) % edges.len()];
                    h.cut(u, v);
                }
                _ => {
                    let a = (rng.next() as usize) % 9;
                    let b = (rng.next() as usize) % 9;
                    if a != b
                        && let Some(path) = path_edges(&h, a, b)
                    {
                        assert_eq!(
                            h.tt.expose_path(a, b),
                            Some(PathLen {
                                len: path.len() as u32
                            })
                        );
                        h.tt.deexpose(b);
                        h.tt.deexpose(a);
                    }
                }
            }

            check_invariants(&h);
        }
    }
}

#[test]
fn bit_vec_packs_bits() {
    let mut bits = BitVec::new();
    assert!(!bits.get(0));

    bits.grow_to(200);
    bits.set(0, true);
    bits.set(64, true);
    bits.set(65, true);
    bits.set(199, true);

    assert!(bits.get(0));
    assert!(!bits.get(1));
    assert!(bits.get(64));
    assert!(bits.get(65));
    assert!(bits.get(199));

    bits.set(64, false);
    assert!(!bits.get(64));
    assert!(bits.get(65));

    assert_eq!(bits.blocks.len(), 4);
}

/// Regression test: removing a label must undo the boundary vertices that
/// attaching it added, even though the boundaries are maintained relative to
/// each cluster's parent context.
#[test]
fn detach_restores_boundary_count() {
    // A vertex whose degree drops from 2 to 1 stops being a boundary.
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);
    h.tt.attach(0, 9);
    h.tt.detach(&9);
    check_invariants(&h);
    for root in live_roots(&h.tt) {
        assert_eq!(
            h.tt.cl(root).boundary_vertices,
            BoundaryVertices::None,
            "stale boundary after detach"
        );
    }

    // A vertex that remains a boundary must not be over-decremented.
    let mut h = Harness::<PathLen>::new(3);
    h.link(0, 1);
    h.link(0, 2);
    h.tt.attach(0, 9);
    h.tt.detach(&9);
    check_invariants(&h);
    for root in live_roots(&h.tt) {
        assert_eq!(
            h.tt.cl(root).boundary_vertices,
            BoundaryVertices::None,
            "vertex 0 is internal to the merged edge cluster"
        );
    }
}
