#![cfg(test)]

use std::collections::{BTreeMap, BTreeSet, VecDeque, btree_map::Entry};

use crate::{LabelId, Summary, TopTree, VertexId, tests::Xor};

/// Weight space shared with the tests: fold u64 weights into the summary type.
pub(crate) trait OracleAgg: Copy + core::fmt::Debug + PartialEq {
    /// Identity of the fold.
    const EMPTY: Self;
    /// Lifts a user-supplied u64 weight into the summary type.
    fn of(weight: u64) -> Self;
    /// Associative fold of two already-folded weights.
    fn combine(a: Self, b: Self) -> Self;
}

/// Distinct powers of two, so an xor-fold over any subset of leaves is injective.
#[allow(dead_code)]
pub(crate) fn distinct_bit(i: usize) -> u64 {
    1u64 << (i & 63)
}

/// Distinct odd values for additive summaries.
pub(crate) fn distinct_value(i: usize) -> u64 {
    (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
}

/// Mirror harness: drives a TopTree and independently records the forest.
pub(crate) struct Oracle<W> {
    tree: TopTree<W>,
    adj: BTreeMap<VertexId, BTreeMap<VertexId, u64>>,
    labels: BTreeMap<LabelId, (VertexId, u64)>,
}

impl<W: Summary + Clone> Oracle<W> {
    pub(crate) fn new(num_vertices: usize) -> Self {
        let mut tree = TopTree::new();
        let mut adj = BTreeMap::new();
        for _ in 0..num_vertices {
            let vertex = tree.add_vertex();
            adj.insert(vertex, BTreeMap::new());
        }
        Self {
            tree,
            adj,
            labels: BTreeMap::new(),
        }
    }

    pub(crate) fn tree(&self) -> &TopTree<W> {
        &self.tree
    }

    pub(crate) fn tree_mut(&mut self) -> &mut TopTree<W> {
        &mut self.tree
    }

    pub(crate) fn vertices(&self) -> Vec<VertexId> {
        self.adj.keys().copied().collect()
    }

    pub(crate) fn add_vertex(&mut self) -> VertexId {
        let vertex = self.tree.add_vertex();
        assert!(self.adj.insert(vertex, BTreeMap::new()).is_none());
        vertex
    }
}

impl<W: Summary + Clone + OracleAgg> Oracle<W> {
    pub(crate) fn link(&mut self, u: VertexId, v: VertexId, weight: u64) {
        assert!(self.adj.contains_key(&u), "vertex {u} must exist in oracle");
        assert!(self.adj.contains_key(&v), "vertex {v} must exist in oracle");
        assert!(
            !self.adj[&u].contains_key(&v),
            "edge {u}-{v} already exists in oracle"
        );
        self.tree.link_with(u, v, |_| W::of(weight));
        self.adj.get_mut(&u).unwrap().insert(v, weight);
        self.adj.get_mut(&v).unwrap().insert(u, weight);
    }

    pub(crate) fn cut(&mut self, u: VertexId, v: VertexId) {
        let weight = self
            .adj
            .get(&u)
            .and_then(|neighbors| neighbors.get(&v))
            .copied()
            .expect("edge must exist in oracle before cut");
        assert_eq!(self.adj.get_mut(&u).unwrap().remove(&v), Some(weight));
        assert_eq!(self.adj.get_mut(&v).unwrap().remove(&u), Some(weight));
        self.tree.cut(u, v);
    }

    pub(crate) fn attach(&mut self, v: VertexId, weight: u64) -> LabelId {
        assert!(self.adj.contains_key(&v), "vertex {v} must exist in oracle");
        let label = self.tree.attach_with(v, |_| W::of(weight));
        assert!(self.labels.insert(label, (v, weight)).is_none());
        label
    }

    pub(crate) fn detach(&mut self, l: LabelId) {
        assert!(
            self.labels.contains_key(&l),
            "label {l} must exist in oracle"
        );
        self.tree.detach(l);
        self.labels.remove(&l).unwrap();
    }

    pub(crate) fn update_label(&mut self, l: LabelId, weight: u64) {
        let (_, mirrored_weight) = self.labels.get_mut(&l).expect("label must exist in oracle");
        self.tree.update_label(l, W::of(weight));
        *mirrored_weight = weight;
    }

    /// u64 weight of edge (u, v), for assertion messages.
    pub(crate) fn edge_weight(&self, u: VertexId, v: VertexId) -> u64 {
        *self
            .adj
            .get(&u)
            .and_then(|neighbors| neighbors.get(&v))
            .expect("edge must exist in oracle")
    }

    pub(crate) fn label_weight(&self, l: LabelId) -> u64 {
        self.labels.get(&l).expect("label must exist in oracle").1
    }
}

impl<W: Summary + Clone + OracleAgg> Oracle<W> {
    /// Reference value: fold of every edge weight and every label weight in
    /// v's component, computed ONLY from the mirror graph by BFS.
    pub(crate) fn component_oracle(&self, v: VertexId) -> W {
        let component = self.component_vertices(v);

        // Check the TopTree's connectivity against the independently discovered
        // component for every known vertex; no TopTree structure informs the fold.
        for &other in self.adj.keys() {
            assert_eq!(
                self.tree().is_connected(v, other),
                component.contains(&other),
                "TopTree/mirror connectivity disagreement for {v} and {other}; mirror component {component:?}"
            );
        }

        let mut aggregate = W::EMPTY;
        for &vertex in &component {
            for (&neighbor, &weight) in &self.adj[&vertex] {
                if vertex < neighbor {
                    aggregate = W::combine(aggregate, W::of(weight));
                }
            }
        }
        for &(vertex, weight) in self.labels.values() {
            if component.contains(&vertex) {
                aggregate = W::combine(aggregate, W::of(weight));
            }
        }
        aggregate
    }

    /// What the top tree actually returns for an exposed path v..w.
    pub(crate) fn exposed_path(&mut self, v: VertexId, w: VertexId) -> W {
        self.tree.with_exposed_path(v, w, |_, weight| weight)
    }

    /// What the top tree actually returns for an exposed vertex (None if v has no leaves).
    pub(crate) fn exposed_vertex(&mut self, v: VertexId) -> Option<W> {
        self.tree.try_with_exposed_vertex(v, |_, weight| weight)
    }

    /// Layer-1 check via the vertex query form.
    pub(crate) fn check_vertex_component_fold(&mut self, v: VertexId) {
        let expected = self.component_oracle(v);
        let has_leaves =
            !self.adj[&v].is_empty() || self.labels.values().any(|&(vertex, _)| vertex == v);
        let actual = self.exposed_vertex(v);
        assert_eq!(
            actual,
            has_leaves.then_some(expected),
            "vertex component-fold mismatch at {v}: query={actual:?}, oracle={expected:?}, vertices={:?}, edges={:?}, labels={:?}",
            self.component_vertices(v),
            self.component_edges(v),
            self.component_labels(v),
        );
    }

    /// Layer-1 check via the path query form (requires the pair to be connected).
    pub(crate) fn check_path_component_fold(&mut self, v: VertexId, w: VertexId) {
        let expected = self.component_oracle(v);
        let actual = self.exposed_path(v, w);
        assert_eq!(
            actual,
            expected,
            "path component-fold mismatch for {v}..{w}: query={actual:?}, oracle={expected:?}, vertices={:?}, edges={:?}, labels={:?}",
            self.component_vertices(v),
            self.component_edges(v),
            self.component_labels(v),
        );
    }

    fn component_vertices(&self, v: VertexId) -> BTreeSet<VertexId> {
        let mut component = BTreeSet::from([v]);
        let mut queue = VecDeque::from([v]);
        while let Some(vertex) = queue.pop_front() {
            for &neighbor in self.adj[&vertex].keys() {
                if component.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        component
    }

    fn component_edges(&self, v: VertexId) -> Vec<(VertexId, VertexId, u64)> {
        self.component_vertices(v)
            .into_iter()
            .flat_map(|vertex| {
                self.adj[&vertex]
                    .iter()
                    .filter(move |(neighbor, _)| vertex < **neighbor)
                    .map(move |(&neighbor, &weight)| (vertex, neighbor, weight))
            })
            .collect()
    }

    fn component_labels(&self, v: VertexId) -> Vec<(LabelId, VertexId, u64)> {
        let component = self.component_vertices(v);
        self.labels
            .iter()
            .filter_map(|(&label, &(vertex, weight))| {
                component
                    .contains(&vertex)
                    .then_some((label, vertex, weight))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Sum(pub(crate) u64);

impl Summary for Sum {
    fn create_edge(_e: crate::index::EdgeId) -> Self {
        todo!()
    }

    fn create_label(_l: crate::index::LabelId) -> Self {
        todo!()
    }

    fn merge(left: &Self, right: &Self, _ctx: &crate::MergeContext) -> Self {
        Self(left.0 + right.0)
    }

    fn split(&self, left: &mut Self, right: &mut Self, _ctx: &crate::MergeContext) {
        assert_eq!(left.0 + right.0, self.0);
    }
}

impl OracleAgg for Sum {
    const EMPTY: Self = Sum(0);

    fn of(weight: u64) -> Self {
        Sum(weight)
    }

    fn combine(a: Self, b: Self) -> Self {
        Sum(a.0 + b.0)
    }
}

impl OracleAgg for Xor {
    const EMPTY: Self = Xor(0);

    fn of(weight: u64) -> Self {
        Xor(weight)
    }

    fn combine(a: Self, b: Self) -> Self {
        Xor(a.0 ^ b.0)
    }
}

trait FixtureWeights: OracleAgg {
    fn fixture_weight(index: usize) -> u64;
}

impl FixtureWeights for Xor {
    fn fixture_weight(index: usize) -> u64 {
        distinct_bit(index)
    }
}

impl FixtureWeights for Sum {
    fn fixture_weight(index: usize) -> u64 {
        (distinct_value(index) & 0x3FF) | 1
    }
}

fn assert_all_queries<W: Summary + Clone + OracleAgg>(oracle: &mut Oracle<W>) {
    let vertices = oracle.vertices();
    for &vertex in &vertices {
        oracle.check_vertex_component_fold(vertex);
    }
    for (index, &u) in vertices.iter().enumerate() {
        for &v in &vertices[index + 1..] {
            if oracle.component_vertices(u).contains(&v) {
                oracle.check_path_component_fold(u, v);
            }
        }
    }
}

fn check_path_with_labels<W: Summary + Clone + FixtureWeights>() {
    let mut oracle = Oracle::<W>::new(5);
    let vertices = oracle.vertices();
    for i in 0..4 {
        oracle.link(vertices[i], vertices[i + 1], W::fixture_weight(i));
    }
    for (i, &vertex) in vertices.iter().enumerate() {
        oracle.attach(vertex, W::fixture_weight(i + 4));
    }
    assert_all_queries(&mut oracle);
}

fn check_star_with_labels<W: Summary + Clone + FixtureWeights>() {
    let mut oracle = Oracle::<W>::new(5);
    let vertices = oracle.vertices();
    for (i, &leaf) in vertices[1..].iter().enumerate() {
        oracle.link(vertices[0], leaf, W::fixture_weight(i));
    }
    for (i, &vertex) in vertices.iter().enumerate() {
        oracle.attach(vertex, W::fixture_weight(i + 4));
    }
    assert_all_queries(&mut oracle);
}

fn check_branched_tree<W: Summary + Clone + FixtureWeights>() {
    let mut oracle = Oracle::<W>::new(6);
    let vertices = oracle.vertices();
    for i in 0..4 {
        oracle.link(vertices[i], vertices[i + 1], W::fixture_weight(i));
    }
    oracle.link(vertices[2], vertices[5], W::fixture_weight(4));
    oracle.attach(vertices[5], W::fixture_weight(5));
    oracle.attach(vertices[2], W::fixture_weight(6));
    assert_all_queries(&mut oracle);
    // Endpoints on the a..e path still aggregate the branch and both labels.
    oracle.check_path_component_fold(vertices[0], vertices[4]);
}

fn check_forest<W: Summary + Clone + FixtureWeights>() {
    let mut oracle = Oracle::<W>::new(8);
    let extra = oracle.add_vertex();
    let vertices = oracle.vertices();
    oracle.link(vertices[0], vertices[1], W::fixture_weight(0));
    oracle.link(vertices[1], vertices[2], W::fixture_weight(1));
    oracle.link(vertices[3], vertices[4], W::fixture_weight(2));
    oracle.link(vertices[3], vertices[5], W::fixture_weight(3));
    oracle.attach(vertices[6], W::fixture_weight(4));
    assert_eq!(oracle.vertices().last(), Some(&extra));
    assert_eq!(oracle.exposed_vertex(vertices[7]), None);
    assert_eq!(
        oracle.exposed_vertex(vertices[6]),
        Some(W::of(W::fixture_weight(4)))
    );
    assert_all_queries(&mut oracle);
}

fn check_label_lifecycle<W: Summary + Clone + FixtureWeights>() {
    let mut oracle = Oracle::<W>::new(4);
    let vertices = oracle.vertices();
    oracle.link(vertices[0], vertices[1], W::fixture_weight(0));
    oracle.link(vertices[1], vertices[2], W::fixture_weight(1));
    let first = oracle.attach(vertices[0], W::fixture_weight(2));
    let second = oracle.attach(vertices[1], W::fixture_weight(3));
    let third = oracle.attach(vertices[3], W::fixture_weight(4));
    assert_eq!(
        oracle.edge_weight(vertices[0], vertices[1]),
        W::fixture_weight(0)
    );
    assert_eq!(oracle.label_weight(first), W::fixture_weight(2));
    assert_all_queries(&mut oracle);

    oracle.update_label(first, W::fixture_weight(5));
    assert_all_queries(&mut oracle);
    oracle.detach(second);
    assert_all_queries(&mut oracle);
    oracle.update_label(third, W::fixture_weight(6));
    assert_all_queries(&mut oracle);
    oracle.detach(first);
    assert_all_queries(&mut oracle);

    oracle.cut(vertices[1], vertices[2]);
    assert_all_queries(&mut oracle);
    oracle.link(vertices[2], vertices[3], W::fixture_weight(7));
    assert_all_queries(&mut oracle);
}

#[test]
fn oracle_component_fold_on_a_path_with_labels() {
    check_path_with_labels::<Xor>();
    check_path_with_labels::<Sum>();
}

#[test]
fn oracle_component_fold_on_a_star_with_labels() {
    check_star_with_labels::<Xor>();
    check_star_with_labels::<Sum>();
}

#[test]
fn oracle_component_fold_on_a_branched_tree() {
    check_branched_tree::<Xor>();
    check_branched_tree::<Sum>();
}

#[test]
fn oracle_component_fold_on_a_forest() {
    check_forest::<Xor>();
    check_forest::<Sum>();
}

#[test]
fn oracle_component_fold_follows_label_updates() {
    check_label_lifecycle::<Xor>();
    check_label_lifecycle::<Sum>();
}

#[test]
#[should_panic(expected = "vertices must be connected")]
fn oracle_reports_disconnected_pair() {
    let mut oracle = Oracle::<Xor>::new(2);
    let vertices = oracle.vertices();
    oracle.exposed_path(vertices[0], vertices[1]);
}

#[test]
#[should_panic(expected = "vertex must not be exposed")]
fn oracle_rejects_same_vertex_path() {
    let mut oracle = Oracle::<Xor>::new(2);
    let vertices = oracle.vertices();
    oracle.link(vertices[0], vertices[1], 1);
    oracle.exposed_path(vertices[0], vertices[0]);
}

#[test]
#[should_panic(expected = "vertex must have a consuming node")]
fn oracle_reports_missing_consuming_node() {
    let mut oracle = Oracle::<Xor>::new(1);
    let vertex = oracle.vertices()[0];
    assert_eq!(oracle.exposed_vertex(vertex), None);
    oracle.tree_mut().with_exposed_vertex(vertex, |_, _| ());
}

// Layer 2: summaries that distinguish compresses (path + path) from rakes
// (path + off-path subtree), with the mirror graph as a path-value oracle.
/// A rake-aware summary: the value of a path cluster is the sum of the weights
/// on its path, and subtrees raked off the path contribute nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PathSum(pub(crate) u64);

impl Summary for PathSum {
    fn create_edge(_e: crate::EdgeId) -> Self {
        todo!()
    }

    fn create_label(_l: LabelId) -> Self {
        todo!()
    }

    fn merge(left: &Self, right: &Self, ctx: &crate::MergeContext) -> Self {
        if ctx.is_compress() {
            Self(left.0 + right.0)
        } else if ctx.left_is_path_child() {
            *left
        } else if ctx.right_is_path_child() {
            *right
        } else {
            // Point cluster: neither child is a path child, so this cluster
            // lies off every exposed path and its value never reaches a query.
            Self::default()
        }
    }
}

/// Component folds are not meaningful for PathSum; use the path-value oracle
/// instead.
impl OracleAgg for PathSum {
    const EMPTY: Self = Self(0);

    fn of(weight: u64) -> Self {
        Self(weight)
    }

    fn combine(_a: Self, _b: Self) -> Self {
        unreachable!("component folds are not meaningful for PathSum; use the path-value oracle")
    }
}

impl Oracle<PathSum> {
    /// Reference: sum of the edge weights on the unique v..w path, computed
    /// ONLY from the mirror graph by BFS (record parents from v, then walk back
    /// from w). Labels contribute nothing: a label is always attached to a
    /// vertex via a point-cluster (rake) child, so it never lands on the path.
    pub(crate) fn path_value_oracle(&self, v: VertexId, w: VertexId) -> PathSum {
        PathSum(
            self.path_edges(v, w)
                .into_iter()
                .map(|(_, _, weight)| weight)
                .sum(),
        )
    }

    /// Asserts that the top tree's exposed-path query for v..w equals the
    /// oracle. Requires v != w and v, w connected.
    pub(crate) fn check_path_value(&mut self, v: VertexId, w: VertexId) {
        assert_ne!(v, w, "vertex must not be exposed");
        let component = self.component_vertices(v);
        assert!(
            component.contains(&w),
            "vertices must be connected: {v} and {w}; mirror component {component:?}"
        );
        let expected = self.path_value_oracle(v, w);
        let actual = self.exposed_path(v, w);
        assert_eq!(
            actual,
            expected,
            "path-value mismatch for {v}..{w}: query={actual:?}, oracle={expected:?}, vertices={component:?}, edges={:?}, labels={:?}",
            self.component_edges(v),
            self.component_labels(v),
        );
    }

    /// Mirror path edges in v-to-w order, found by BFS and parent backtracking.
    fn path_edges(&self, v: VertexId, w: VertexId) -> Vec<(VertexId, VertexId, u64)> {
        let mut parents = BTreeMap::from([(v, None)]);
        let mut queue = VecDeque::from([v]);
        while let Some(vertex) = queue.pop_front() {
            if vertex == w {
                break;
            }
            for (&neighbor, &weight) in &self.adj[&vertex] {
                if let Entry::Vacant(entry) = parents.entry(neighbor) {
                    entry.insert(Some((vertex, weight)));
                    queue.push_back(neighbor);
                }
            }
        }
        assert!(
            parents.contains_key(&w),
            "vertices must be connected in mirror graph: {v} and {w}"
        );

        let mut edges = Vec::new();
        let mut current = w;
        while current != v {
            let (parent, weight) =
                parents[&current].expect("non-root path vertex must have parent");
            edges.push((parent, current, weight));
            current = parent;
        }
        edges.reverse();
        edges
    }
}

fn assert_all_path_values(oracle: &mut Oracle<PathSum>) {
    let vertices = oracle.vertices();
    for &v in &vertices {
        let component = oracle.component_vertices(v);
        for &w in &vertices {
            if v != w && component.contains(&w) {
                oracle.check_path_value(v, w);
            }
        }
    }
}

fn path_fixture_weight(index: usize) -> u64 {
    index as u64 + 1
}

impl FixtureWeights for PathSum {
    fn fixture_weight(index: usize) -> u64 {
        index as u64 + 1
    }
}

trait RandomFixtureWeights: FixtureWeights {
    fn random_fixture_weight(index: usize) -> u64;
    fn has_fresh_random_weight(index: usize) -> bool;
}

impl RandomFixtureWeights for Xor {
    fn random_fixture_weight(index: usize) -> u64 {
        distinct_bit(index)
    }

    // Xor-fold injectivity over a subset requires one distinct bit position per leaf,
    // so the xor summary runs out after 64 fresh weights while additive summaries are unbounded.
    fn has_fresh_random_weight(index: usize) -> bool {
        index < 64
    }
}

impl RandomFixtureWeights for Sum {
    fn random_fixture_weight(index: usize) -> u64 {
        index as u64 + 1
    }

    fn has_fresh_random_weight(_index: usize) -> bool {
        true
    }
}

impl RandomFixtureWeights for PathSum {
    fn random_fixture_weight(index: usize) -> u64 {
        index as u64 + 1
    }

    fn has_fresh_random_weight(_index: usize) -> bool {
        true
    }
}

fn branched_path_sum_fixture(with_labels: bool) -> (Oracle<PathSum>, Vec<VertexId>) {
    let mut oracle = Oracle::<PathSum>::new(6);
    let vertices = oracle.vertices();
    for i in 0..4 {
        oracle.link(vertices[i], vertices[i + 1], path_fixture_weight(i));
    }
    oracle.link(vertices[2], vertices[5], path_fixture_weight(4));
    if with_labels {
        for (i, &vertex) in vertices.iter().enumerate() {
            oracle.attach(vertex, 20 + i as u64);
        }
    }
    (oracle, vertices)
}

#[test]
fn oracle_path_value_on_a_simple_path() {
    let mut oracle = Oracle::<PathSum>::new(5);
    let vertices = oracle.vertices();
    for i in 0..4 {
        oracle.link(vertices[i], vertices[i + 1], path_fixture_weight(i));
    }
    assert_all_path_values(&mut oracle);
}

#[test]
fn oracle_path_value_on_a_star() {
    let mut oracle = Oracle::<PathSum>::new(5);
    let vertices = oracle.vertices();
    for (i, &leaf) in vertices[1..].iter().enumerate() {
        oracle.link(vertices[0], leaf, path_fixture_weight(i));
    }
    assert_all_path_values(&mut oracle);
}

#[test]
fn oracle_path_value_on_a_branched_tree() {
    let (mut oracle, vertices) = branched_path_sum_fixture(false);
    assert_all_path_values(&mut oracle);

    let queried_value = oracle.path_value_oracle(vertices[0], vertices[4]).0;
    let path_edge_sum: u64 = oracle
        .path_edges(vertices[0], vertices[4])
        .into_iter()
        .map(|(_, _, weight)| weight)
        .sum();
    let whole_component_edge_sum: u64 = oracle
        .component_edges(vertices[0])
        .into_iter()
        .map(|(_, _, weight)| weight)
        .sum();
    assert_eq!(queried_value, path_edge_sum);
    assert_ne!(queried_value, whole_component_edge_sum);
    let actual = oracle.exposed_path(vertices[0], vertices[4]);
    assert_eq!(actual, PathSum(queried_value));
    oracle.check_path_value(vertices[0], vertices[4]);
}

#[test]
fn oracle_path_value_ignores_labels() {
    let (mut bare, vertices) = branched_path_sum_fixture(false);
    let (mut labeled, labeled_vertices) = branched_path_sum_fixture(true);
    assert_eq!(vertices, labeled_vertices);
    // This pins that a label never contributes to a path value.
    for &v in &vertices {
        for &w in &vertices {
            if v != w {
                bare.check_path_value(v, w);
                labeled.check_path_value(v, w);
                assert_eq!(
                    bare.exposed_path(v, w),
                    labeled.exposed_path(v, w),
                    "labels changed path value for {v}..{w}"
                );
            }
        }
    }
}

#[test]
fn oracle_path_value_after_mutations() {
    let mut oracle = Oracle::<PathSum>::new(6);
    let vertices = oracle.vertices();
    let [a, b, c, d, e, f]: [VertexId; 6] = vertices
        .clone()
        .try_into()
        .expect("fixture must have six vertices");

    oracle.link(a, b, 1);
    assert_all_path_values(&mut oracle);
    oracle.link(b, c, 2);
    assert_all_path_values(&mut oracle);
    oracle.link(d, e, 3);
    assert_all_path_values(&mut oracle);
    oracle.link(e, f, 4);
    assert_all_path_values(&mut oracle);
    let label = oracle.attach(b, 30);
    assert_all_path_values(&mut oracle);
    oracle.link(c, d, 5);
    assert_all_path_values(&mut oracle);
    oracle.update_label(label, 31);
    assert_all_path_values(&mut oracle);
    oracle.detach(label);
    assert_all_path_values(&mut oracle);
    oracle.cut(b, c);
    assert_all_path_values(&mut oracle);
    oracle.link(a, c, 6);
    assert_all_path_values(&mut oracle);
    let second_label = oracle.attach(f, 32);
    assert_all_path_values(&mut oracle);
    oracle.update_label(second_label, 33);
    assert_all_path_values(&mut oracle);
    oracle.detach(second_label);
    assert_all_path_values(&mut oracle);
    oracle.cut(d, e);
    assert_all_path_values(&mut oracle);
    oracle.link(d, f, 7);
    assert_all_path_values(&mut oracle);
}

#[test]
fn oracle_path_value_on_a_random_static_tree() {
    let mut rng = crate::tests::Rng(0x5EED);
    let mut oracle = Oracle::<PathSum>::new(12);
    let vertices = oracle.vertices();
    for i in 1..vertices.len() {
        let parent = (rng.next() as usize) % i;
        oracle.link(vertices[parent], vertices[i], path_fixture_weight(i - 1));
    }
    assert_all_path_values(&mut oracle);
}

fn random_component_pairs<W: Summary + Clone + OracleAgg>(
    oracle: &Oracle<W>,
) -> Vec<(VertexId, VertexId)> {
    let vertices = oracle.vertices();
    let mut pairs = Vec::new();
    for (index, &u) in vertices.iter().enumerate() {
        let component = oracle.component_vertices(u);
        for &v in &vertices[index + 1..] {
            if component.contains(&v) {
                pairs.push((u, v));
            }
        }
    }
    pairs
}

fn random_edges<W>(oracle: &Oracle<W>) -> Vec<(VertexId, VertexId)> {
    oracle
        .adj
        .iter()
        .flat_map(|(&u, neighbors)| {
            neighbors
                .keys()
                .filter(move |&&v| u < v)
                .map(move |&v| (u, v))
        })
        .collect()
}

fn fresh_weight<W: RandomFixtureWeights>(counter: &mut usize) -> u64 {
    assert!(W::has_fresh_random_weight(*counter));
    let weight = W::random_fixture_weight(*counter);
    *counter += 1;
    weight
}

/// Renders a caught panic payload so oracle failures keep their diagnostics.
fn panic_payload(payload: &(dyn core::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

fn random_link_pairs<W: Summary + Clone + OracleAgg>(
    oracle: &Oracle<W>,
) -> Vec<(VertexId, VertexId)> {
    let vertices = oracle.vertices();
    let components: BTreeMap<_, _> = vertices
        .iter()
        .map(|&vertex| (vertex, oracle.component_vertices(vertex)))
        .collect();
    let mut pairs = Vec::new();
    for (index, &u) in vertices.iter().enumerate() {
        for &v in &vertices[index + 1..] {
            if !components[&u].contains(&v) {
                pairs.push((u, v));
            }
        }
    }
    pairs
}

fn random_index(rng: &mut crate::tests::Rng, len: usize) -> usize {
    assert!(len > 0);
    const RANGE: u64 = 1 << 31;
    let limit = RANGE - RANGE % len as u64;
    loop {
        let value = rng.next();
        if value < limit {
            return (value % len as u64) as usize;
        }
    }
}

fn run_random_component_fold_ops<W>(seed: u64, num_vertices: usize, num_ops: usize)
where
    W: Summary + Clone + FixtureWeights + RandomFixtureWeights,
{
    let mut rng = crate::tests::Rng(seed);
    let mut oracle = Oracle::<W>::new(num_vertices);
    let mut weight_index = 0;

    for op_index in 0..num_ops {
        let first_kind = (rng.next() % 5) as usize;
        let mut op_kind = "skip";
        for offset in 0..5 {
            match (first_kind + offset) % 5 {
                0 => {
                    let pairs = random_link_pairs(&oracle);
                    if !pairs.is_empty() && W::has_fresh_random_weight(weight_index) {
                        let (u, v) = pairs[random_index(&mut rng, pairs.len())];
                        let weight = fresh_weight::<W>(&mut weight_index);
                        oracle.link(u, v, weight);
                        op_kind = "link";
                        break;
                    }
                }
                1 => {
                    let edges = random_edges(&oracle);
                    if !edges.is_empty() {
                        let (u, v) = edges[random_index(&mut rng, edges.len())];
                        oracle.cut(u, v);
                        op_kind = "cut";
                        break;
                    }
                }
                2 => {
                    let vertices = oracle.vertices();
                    if !vertices.is_empty() && W::has_fresh_random_weight(weight_index) {
                        let vertex = vertices[random_index(&mut rng, vertices.len())];
                        let weight = fresh_weight::<W>(&mut weight_index);
                        oracle.attach(vertex, weight);
                        op_kind = "attach";
                        break;
                    }
                }
                3 => {
                    let labels: Vec<_> = oracle.labels.keys().copied().collect();
                    if !labels.is_empty() {
                        let label = labels[random_index(&mut rng, labels.len())];
                        oracle.detach(label);
                        op_kind = "detach";
                        break;
                    }
                }
                4 => {
                    let labels: Vec<_> = oracle.labels.keys().copied().collect();
                    if !labels.is_empty() && W::has_fresh_random_weight(weight_index) {
                        let label = labels[random_index(&mut rng, labels.len())];
                        let weight = fresh_weight::<W>(&mut weight_index);
                        oracle.update_label(label, weight);
                        op_kind = "update_label";
                        break;
                    }
                }
                _ => unreachable!(),
            }
        }

        let check_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let vertices = oracle.vertices();
            if !vertices.is_empty() {
                let vertex = vertices[random_index(&mut rng, vertices.len())];
                oracle.check_vertex_component_fold(vertex);
            }
            let pairs = random_component_pairs(&oracle);
            if !pairs.is_empty() {
                let (u, v) = pairs[random_index(&mut rng, pairs.len())];
                oracle.check_path_component_fold(u, v);
            }
        }));
        if let Err(payload) = check_result {
            panic!(
                "random component-fold oracle failed: seed {seed}, op {op_index}, kind {op_kind}\n{}",
                panic_payload(payload.as_ref()),
            );
        }
    }
}

#[test]
fn randomized_component_fold_oracle_xor() {
    for seed in [1, 7, 19, 0x5EED, 0xC0FFEE, 0xBAD5EED, 0x1234_5678, u64::MAX] {
        run_random_component_fold_ops::<Xor>(seed, 9, 180);
    }
}

#[test]
fn randomized_component_fold_oracle_sum() {
    for seed in [
        2,
        11,
        23,
        0xA11CE,
        0xDEADBEEF,
        0xF00D,
        0x8765_4321,
        u64::MAX - 1,
    ] {
        run_random_component_fold_ops::<Sum>(seed, 9, 180);
    }
}

fn run_random_path_value_ops(seed: u64, num_vertices: usize, num_ops: usize) {
    let mut rng = crate::tests::Rng(seed);
    let mut oracle = Oracle::<PathSum>::new(num_vertices);
    let mut weight_index = 0;

    for op_index in 0..num_ops {
        let first_kind = (rng.next() % 5) as usize;
        let mut op_kind = "skip";
        for offset in 0..5 {
            match (first_kind + offset) % 5 {
                0 => {
                    let pairs = random_link_pairs(&oracle);
                    if !pairs.is_empty() {
                        let (u, v) = pairs[random_index(&mut rng, pairs.len())];
                        let weight = fresh_weight::<PathSum>(&mut weight_index);
                        oracle.link(u, v, weight);
                        op_kind = "link";
                        break;
                    }
                }
                1 => {
                    let edges = random_edges(&oracle);
                    if !edges.is_empty() {
                        let (u, v) = edges[random_index(&mut rng, edges.len())];
                        oracle.cut(u, v);
                        op_kind = "cut";
                        break;
                    }
                }
                2 => {
                    let vertices = oracle.vertices();
                    if !vertices.is_empty() {
                        let vertex = vertices[random_index(&mut rng, vertices.len())];
                        let weight = fresh_weight::<PathSum>(&mut weight_index);
                        oracle.attach(vertex, weight);
                        op_kind = "attach";
                        break;
                    }
                }
                3 => {
                    let labels: Vec<_> = oracle.labels.keys().copied().collect();
                    if !labels.is_empty() {
                        let label = labels[random_index(&mut rng, labels.len())];
                        oracle.detach(label);
                        op_kind = "detach";
                        break;
                    }
                }
                4 => {
                    let labels: Vec<_> = oracle.labels.keys().copied().collect();
                    if !labels.is_empty()
                        && <PathSum as RandomFixtureWeights>::has_fresh_random_weight(weight_index)
                    {
                        let label = labels[random_index(&mut rng, labels.len())];
                        let weight = fresh_weight::<PathSum>(&mut weight_index);
                        oracle.update_label(label, weight);
                        op_kind = "update_label";
                        break;
                    }
                }
                _ => unreachable!(),
            }
        }

        let check_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pairs = random_component_pairs(&oracle);
            if !pairs.is_empty() {
                let (u, v) = pairs[random_index(&mut rng, pairs.len())];
                oracle.check_path_value(u, v);
            }
        }));
        if let Err(payload) = check_result {
            panic!(
                "random path-value oracle failed: seed {seed}, op {op_index}, kind {op_kind}\n{}",
                panic_payload(payload.as_ref()),
            );
        }
    }
}

#[test]
fn randomized_path_value_oracle() {
    for seed in [
        3,
        13,
        29,
        0x51DE,
        0xBADC0DE,
        0xFACE,
        0x3141_5926,
        u64::MAX - 2,
    ] {
        run_random_path_value_ops(seed, 9, 250);
    }
}

#[test]
fn randomized_path_value_oracle_deep_tree() {
    for seed in [
        5,
        17,
        31,
        0xD33F,
        0xC0FFEE,
        0xABCDEF,
        0x2718_2818,
        u64::MAX - 3,
    ] {
        let mut rng = crate::tests::Rng(seed);
        let mut oracle = Oracle::<PathSum>::new(1);
        let mut weight_index = 0;
        let mut previous = oracle.vertices()[0];
        let root = previous;

        for step in 0..39 {
            // Linking each new vertex to the previous one deliberately grows
            // a deep path; random queries then exercise both long and short
            // paths through the resulting tree.
            let parent = previous;
            let fresh = oracle.add_vertex();
            let weight = fresh_weight::<PathSum>(&mut weight_index);
            oracle.link(parent, fresh, weight);
            previous = fresh;

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let pairs = random_component_pairs(&oracle);
                for _ in 0..2 {
                    let (u, v) = pairs[random_index(&mut rng, pairs.len())];
                    oracle.check_path_value(u, v);
                }
                // Always exercise a long path as the random recursive growth proceeds.
                oracle.check_path_value(root, fresh);
            }));
            if let Err(payload) = result {
                panic!(
                    "random deep-tree path-value oracle failed: seed {seed}, op {step}, kind link\n{}",
                    panic_payload(payload.as_ref()),
                );
            }
        }
    }
}
