//! Randomized differential runs against the [`Naive`] oracle.
//!
//! This is the port of the randomized half of the deleted
//! `src/graph/tests.rs`. A run drives [`Graph`] with a long pseudo-random mix of
//! the operations the paper's Appendix A has to survive -- vertex addition and
//! removal, edge insertion (with a bias towards parallel edges) and deletion --
//! and after *every* step compares [`FindBridge`] against the oracle:
//!
//! * [`assert_internal_invariants`] after every step: the driver's bookkeeping
//!   plus a check that every handle a query can return really is a bridge.
//! * [`Graph::assert_matches`] every [`EXHAUSTIVE_CHECK_INTERVAL`] steps: every
//!   query, for every pair of live vertices.
//! * [`assert_randomized_sample_matches_naive`] on the other steps: the same
//!   comparison for a deterministic handful of vertices and pairs, which keeps
//!   the per-step cost low enough to check something every time.
//!
//! Two suites run, differing only in how strongly they favour parallel edges
//! when choosing the endpoints of a new edge. See
//! [`PARALLEL_BIAS_SPARSE`] and [`PARALLEL_BIAS_DENSE`].
//!
//! Everything is driven by a fixed [`Rng`] seeded from the `#[test]`s, so a
//! failure reproduces exactly from the `context` string in the panic message.

use super::*;

/// Number of operations in one randomized run.
///
/// Unchanged from the deleted test file. At 8..=12 starting vertices the
/// ~30% insert / ~24% delete mix builds up, tears down and rebuilds overlapping
/// cycles many times over within 180 steps, which is what reaches the `Swap` and
/// `Recover` paths; the [`EXHAUSTIVE_CHECK_INTERVAL`] below is what had to be
/// re-tuned for the new driver, not this one.
const RANDOMIZED_STEPS: usize = 180;

/// How often a step is followed by the *exhaustive* comparison against the
/// oracle.
///
/// Unchanged from the deleted test file. Every third step means each run spends
/// about a third of its oracle budget proving that all O(V^2) vertex pairs agree,
/// which is what makes these runs differential rather than smoke tests.
const EXHAUSTIVE_CHECK_INTERVAL: usize = 3;

/// Base number of vertices a run starts with; the actual count is this plus a
/// pseudo-random `0..INITIAL_VERTICES_SPREAD`.
///
/// Unchanged from the deleted test file. `8 + rng.index(5)` gives 8..=12
/// vertices, which is large enough for overlapping cycles (so a single deletion
/// can leave a replacement edge) and small enough that the O(V^2) comparisons
/// stay affordable.
const INITIAL_VERTICES_BASE: usize = 8;

/// Spread added to [`INITIAL_VERTICES_BASE`]; see there.
const INITIAL_VERTICES_SPREAD: usize = 5;

/// Percentage chance that an insertion reuses the endpoint pair of an existing
/// live edge, for `randomized_differential`.
///
/// Unchanged from the deleted test file's `false` arm. 12% is enough to make
/// parallel edges (and therefore multiplicity-aware bridge detection) common
/// without collapsing the graph into a few heavily doubled pairs.
const PARALLEL_BIAS_SPARSE: u64 = 12;

/// Percentage chance that an insertion reuses the endpoint pair of an existing
/// live edge, for `randomized_differential_with_parallel_edges`.
///
/// Unchanged from the deleted test file's `true` arm. 75% keeps the graph mostly
/// a handful of doubled pairs, which stresses the `cover`/`uncover` of a path
/// whose endpoints already share an edge and the level bookkeeping around it.
const PARALLEL_BIAS_DENSE: u64 = 75;

/// A tiny deterministic linear congruential generator.
///
/// The run must be reproducible from a seed alone -- no `rand`, no thread races
/// and no wall clock -- so that a failure can be replayed by rerunning the same
/// `#[test]`. The constants are the ones from the deleted test file, so the seeds
/// kept below still drive the same operation streams; that matters only for
/// reproducing an old report, not for the coverage of this module.
struct Rng(u64);

impl Rng {
    /// Advances the generator and returns the next 31-bit word.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// A uniform-enough index in `0..length`; `length` must be non-zero.
    fn index(&mut self, length: usize) -> usize {
        assert!(length > 0);
        (self.next() as usize) % length
    }
}

/// Two distinct vertices drawn from `vertices`, which must hold at least two.
fn random_distinct_pair(rng: &mut Rng, vertices: &[usize]) -> (usize, usize) {
    assert!(vertices.len() > 1);
    let u_index = rng.index(vertices.len());
    let mut v_index = rng.index(vertices.len() - 1);
    if v_index >= u_index {
        v_index += 1;
    }
    (vertices[u_index], vertices[v_index])
}

/// The endpoints of the next edge to insert, or `None` if the graph has fewer
/// than two live vertices and therefore cannot host an edge.
///
/// With probability `parallel_bias` (in percent) an existing *live* edge's
/// endpoint pair is reused, which makes a parallel edge; otherwise two distinct
/// live vertices are drawn uniformly. Reusing a live edge's pair keeps both
/// endpoints alive by construction, so the returned pair is always insertable.
fn choose_insert_pair(rng: &mut Rng, naive: &Naive, parallel_bias: u64) -> Option<(usize, usize)> {
    if naive.vertex_count() < 2 {
        return None;
    }
    if !naive.edges.is_empty() && rng.next() % 100 < parallel_bias {
        let edge_index = rng.index(naive.edges.len());
        Some(*naive.edges.values().nth(edge_index).unwrap())
    } else {
        Some(random_distinct_pair(
            rng,
            &naive.live_vertices().collect::<Vec<_>>(),
        ))
    }
}

/// The live edges as `(oracle edge, first endpoint, second endpoint)` triples.
///
/// This is what the per-step `context` prints: the whole edge set is small (see
/// [`INITIAL_VERTICES_BASE`]), and naming it is what turns a panic in some step
/// into a state that can be replayed by hand.
fn live_edges_snapshot(graph: &Graph) -> Vec<(usize, usize, usize)> {
    graph
        .naive
        .edges
        .iter()
        .map(|(&edge, &(u, v))| (edge, u, v))
        .collect()
}

/// The diagnostic prefix of every assertion in a step.
///
/// `live_edges` is the edge set *before* `operation` when it is used to
/// justify the operation's own assertions, and the edge set *after* it when it
/// is used to justify the post-step comparison.
fn step_context(
    seed: u64,
    case: usize,
    step: usize,
    operation: &str,
    live_edges: &[(usize, usize, usize)],
) -> String {
    format!("seed {seed:#x}, case {case}, step {step}, {operation}; live edges {live_edges:?}")
}

/// The oracle edge index of a handle a query returned, panicking if the handle
/// is not one the driver ever handed out.
///
/// The assertion helpers take an *oracle* edge index, so a returned [`EdgeId`]
/// has to be translated first. Bailing out here rather than yielding `None` is
/// essential: a dead handle would otherwise turn the comparison into a vacuous
/// `None == None` and hide exactly the bug this module exists to find.
fn returned_oracle(live: &BTreeMap<EdgeId, LiveEdge>, edge: EdgeId, context: &str) -> usize {
    live.get(&edge)
        .unwrap_or_else(|| panic!("{context}: a query returned dead handle {edge:?}"))
        .oracle
}

/// A handle for a [`top_tree::slot::SlotVec`] slot the driver has never issued.
///
/// [`EdgeId`] is a `SlotVec` index whose freed slots are recycled, so a handle below the oracle's `next_edge` may well name a
/// different, live edge by now, and deleting it twice must *not* be expected to
/// return `false`. A handle at or above `next_edge`, on the other hand, has
/// never been handed out and is therefore inert; the delete arm uses one of
/// these rather than trying to prove a particular live edge is absent.
fn unused_edge_handle(index: usize) -> EdgeId {
    EdgeId(NonMaxUsize::new(index).expect("a fabricated handle index is representable"))
}

/// The cheap per-step comparison against the oracle: counts, the spanning-forest
/// invariant, and a deterministic handful of vertices and pairs.
///
/// This is the port of the deleted `assert_randomized_sample_matches_naive`,
/// with two changes. First, the oracle is a second [`Naive`] the run maintains
/// independently of the driver's own, and the graph is borrowed through
/// [`Graph::parts`] so both can be read at once. Second, the two bridge checks
/// pipe the returned handle through [`returned_oracle`], because the helpers in
/// this module speak oracle edge indices.
///
/// The sampled vertices rotate with `step` and are offset by `case`, so two
/// cases with the same seed still compare different vertices at the same step.
fn assert_randomized_sample_matches_naive(
    graph: &mut Graph,
    naive: &Naive,
    case: usize,
    step: usize,
    context: &str,
) {
    let GraphParts {
        fb, verts, live, ..
    } = graph.parts();

    assert_eq!(verts.len(), naive.vertex_count(), "{context}: vertex count");
    assert_eq!(live.len(), naive.edges.len(), "{context}: edge count");

    // The forest edges always form a spanning forest, so their count is pinned by
    // the oracle's component count alone. That is a real constraint on
    // `FindBridge` (it says nothing may be linked into a cycle) and it costs one
    // BFS pass.
    let component_labels = naive.component_labels(&BTreeSet::new());
    let components = naive
        .live_vertices()
        .filter(|&vertex| component_labels[vertex] == vertex)
        .count();
    assert_eq!(
        fb.tree_edge_count(),
        verts.len() - components,
        "{context}: tree_edge_count is not the spanning forest size"
    );

    let live_vertices = naive.live_vertices().collect::<Vec<_>>();
    if live_vertices.is_empty() {
        return;
    }
    let first = live_vertices[(step.wrapping_mul(7) + case) % live_vertices.len()];
    let second = live_vertices[(step.wrapping_mul(13) + case + 1) % live_vertices.len()];
    let third = live_vertices[(step.wrapping_mul(19) + case + 2) % live_vertices.len()];

    for vertex in [first, second] {
        assert_eq!(
            component_size(fb, verts, vertex) as usize,
            naive.component_size(vertex),
            "{context}: sampled component_size({vertex})"
        );
        assert_eq!(
            two_edge_component_size(fb, verts, vertex) as usize,
            naive.two_edge_component_size(vertex),
            "{context}: sampled two_edge_component_size({vertex})"
        );
        let bridge =
            find_bridge(fb, verts, vertex).map(|edge| returned_oracle(live, edge, context));
        assert_returned_bridge_in_component(naive, vertex, bridge, context);
    }

    for (u, v) in [(first, second), (second, third)] {
        assert_eq!(
            connected(fb, verts, u, v),
            naive.connected(u, v),
            "{context}: sampled connected({u}, {v})"
        );
        assert_eq!(
            two_edge_connected(fb, verts, u, v),
            naive.two_edge_connected(u, v),
            "{context}: sampled two_edge_connected({u}, {v})"
        );
        let bridge =
            find_bridge_between(fb, verts, u, v).map(|edge| returned_oracle(live, edge, context));
        assert_returned_separating_bridge(naive, u, v, bridge, context);
    }
}

/// What one run actually exercised.
///
/// The public API cannot distinguish the interesting shapes of a run: whether a
/// deleted tree edge was covered by a non-tree edge (Appendix A's `Swap`, then
/// `Delete` and `Recover`) or was a real bridge (a plain cut), or whether a
/// non-tree edge ever got promoted above level 0. Those live in
/// `FindBridge`'s private edge records, so the run peeks at them for counting
/// only -- every correctness claim still goes through the oracle. Every counter
/// is printed at the end of a run, and the ones that guard a code path are
/// asserted by [`assert_coverage`], so a tuned-down parameter set cannot silently
/// stop reaching one; see that function for which those are.
#[derive(Debug, Default)]
struct Coverage {
    /// Highest level any live non-tree edge reached; above 0 means `Recover`
    /// promoted a label.
    max_non_tree_level: i32,
    /// Deletions of a tree edge that some non-tree edge covered, i.e. `Swap`.
    covered_tree_edge_deletions: usize,
    /// Deletions of an uncovered tree edge, i.e. a real bridge.
    bridge_deletions: usize,
    /// Deletions of a non-tree edge, which uncovers its path and recovers.
    non_tree_deletions: usize,
    /// Steps whose graph had vertices but no bridge at all.
    bridgeless_steps: usize,
    /// Steps that had a pair of parallel edges.
    parallel_edge_steps: usize,
    /// Steps with vertices but no edge.
    edgeless_steps: usize,
    /// Steps that ended with no vertex at all.
    ///
    /// Always zero at the deleted parameters: 13% removals against ~9%
    /// additions is a net drift of only about -5 vertices over
    /// [`RANDOMIZED_STEPS`] steps from [`INITIAL_VERTICES_BASE`], and the
    /// lowest live vertex count any of the six seeds reaches is 3 (sparse
    /// suite) or 4 (dense suite). The counter is kept so that a future change to
    /// the mix which *does* empty the graph shows up in the report instead of
    /// being silently absorbed.
    empty_steps: usize,
    /// Steps that had to add a vertex because the graph had no vertex at all.
    /// Always zero for the same reason as [`Coverage::empty_steps`]; the branch
    /// is kept because it is what stops the run from doing nothing at all once
    /// the removals have eaten the additions.
    empty_refills: usize,
    /// Insertion arms that found fewer than two live vertices and had to add one
    /// before an edge could be inserted. Always zero for the same reason as
    /// [`Coverage::empty_steps`].
    starved_inserts: usize,
    /// Largest live edge count seen.
    max_edges: usize,
    /// Largest live vertex count seen.
    max_vertices: usize,
    /// Smallest live vertex count seen.
    min_vertices: usize,
}

impl Coverage {
    /// Fresh counters. `min_vertices` starts above any reachable count so that
    /// [`Coverage::observe`] can fold in the minimum with a plain `min`.
    fn new() -> Self {
        Self {
            min_vertices: usize::MAX,
            ..Self::default()
        }
    }

    /// Folds the state after one step into the running counters.
    fn observe(&mut self, graph: &Graph) {
        for (_, edge) in graph.fb.edges.iter() {
            if let Edge::NonTree(non_tree) = edge {
                self.max_non_tree_level = self.max_non_tree_level.max(i32::from(non_tree.level));
            }
        }

        let vertices = graph.vertex_count();
        let edges = graph.edge_count();
        self.max_edges = self.max_edges.max(edges);
        self.max_vertices = self.max_vertices.max(vertices);
        self.min_vertices = self.min_vertices.min(vertices);
        match (vertices, edges) {
            (0, _) => self.empty_steps += 1,
            (_, 0) => self.edgeless_steps += 1,
            _ => {}
        }
        if vertices > 0 && graph.naive.bridges().is_empty() {
            self.bridgeless_steps += 1;
        }
        if graph
            .naive
            .adj
            .values()
            .flat_map(|neighbors| neighbors.values())
            .any(|&multiplicity| multiplicity > 1)
        {
            self.parallel_edge_steps += 1;
        }
    }
}

/// Fails a run that never reached a shape the suites are supposed to cover.
///
/// These are assertions about the *test*, not about `FindBridge`: a green suite
/// that never deletes a covered tree edge proves nothing about `Swap`. The
/// thresholds are the minimum the seeds are known to exceed, so they catch a
/// regression that quietly disables an operation arm.
///
/// Only the counters that guard a code path are enforced, and all six are
/// required to be positive: `covered_tree_edge_deletions` (`Swap`/`Recover`),
/// `bridge_deletions` (a plain cut), `non_tree_deletions`,
/// `parallel_edge_steps`, `edgeless_steps` and `bridgeless_steps`. On top of
/// those, `max_non_tree_level` must be positive when the caller claims this seed
/// reaches `RecoverPhase`'s promotion branch, and the dense suite does not claim
/// it.
///
/// The remaining counters are informational only -- they are printed for the
/// reader and nothing asserts them: `empty_steps`, `empty_refills`,
/// `starved_inserts`, `max_edges`, `max_vertices` and `min_vertices`. Three of
/// them cannot be reached at the current parameters: no seed ever empties the
/// graph, since the lowest live vertex count any of them reaches is 3 (sparse
/// suite) or 4 (dense suite), so the `(0, _)` state -- and with it the 0-vertex
/// refill branch that would be the only way to reach `empty_refills` -- is dead
/// code in practice.
fn assert_coverage(coverage: &Coverage, seed: u64, case: usize, expect_promotion: bool) {
    let context = format!("seed {seed:#x}, case {case}");
    assert!(
        coverage.covered_tree_edge_deletions > 0,
        "{context}: the run never deleted a covered tree edge, so Swap/Recover is untested: {coverage:?}"
    );
    assert!(
        coverage.bridge_deletions > 0,
        "{context}: the run never deleted a bridge, so a plain cut is untested: {coverage:?}"
    );
    assert!(
        coverage.non_tree_deletions > 0,
        "{context}: the run never deleted a non-tree edge: {coverage:?}"
    );
    assert!(
        coverage.parallel_edge_steps > 0,
        "{context}: the run never held parallel edges: {coverage:?}"
    );
    assert!(
        coverage.edgeless_steps > 0,
        "{context}: the run never held an edgeless graph: {coverage:?}"
    );
    assert!(
        coverage.bridgeless_steps > 0,
        "{context}: the run never held a bridgeless graph, so the `no bridge in this component` answer is untested: {coverage:?}"
    );
    // Only the sparse suite is expected to promote a label. A graph that is
    // mostly doubled pairs has one-edge paths to cover, and `RecoverPhase` only
    // promotes when the level-`i+1` size of the candidate's path fits in half
    // the level-`i` size of the path being repaired -- which a two-vertex cycle
    // never satisfies. See the report on `randomized_differential`; asserting it
    // for the dense suite would be asserting something the bias prevents.
    if expect_promotion {
        assert!(
            coverage.max_non_tree_level > 0,
            "{context}: no non-tree edge was promoted above level 0, so label promotion is untested: {coverage:?}"
        );
    }
}

/// Drives one randomized run: `RANDOMIZED_STEPS` operations, each followed by
/// the internal-invariant check and either the exhaustive or the sampled oracle
/// comparison.
///
/// The operation mix is the deleted driver's, and its comments explain why the
/// weights are what they are. Two oracles are kept: the driver's own, which
/// [`Graph`] maintains inside every mutation, and a second one this function
/// mirrors by hand. They must describe the same multigraph at every step; that
/// is asserted below, and it is what makes `Graph::assert_matches` a genuine
/// comparison against an oracle rather than against the driver's own bookkeeping.
///
/// `expect_promotion` is the caller's claim that this seed reaches
/// `RecoverPhase`'s promotion branch; see [`assert_coverage`].
fn run_randomized_case(seed: u64, parallel_bias: u64, case: usize, expect_promotion: bool) {
    let mut rng = Rng(seed);
    let initial_vertices = INITIAL_VERTICES_BASE + rng.index(INITIAL_VERTICES_SPREAD);
    let mut graph = Graph::new(initial_vertices);
    let mut naive = Naive::new(initial_vertices);
    let mut coverage = Coverage::new();

    for step in 0..RANDOMIZED_STEPS {
        let roll = rng.next() % 100;
        let before = live_edges_snapshot(&graph);
        let live_vertices = naive.live_vertices().collect::<Vec<_>>();

        // The deleted driver's mix, unchanged: the two scheduled additions
        // guarantee the graph keeps growing even under an unlucky roll stream;
        // an empty graph must be refilled before anything else can happen; then
        // 8% additions, 13% removals, 30% insertions, 24% deletions and 25%
        // queries.
        let operation = if step == RANDOMIZED_STEPS / 3 || step == 2 * RANDOMIZED_STEPS / 3 {
            let vertex = graph.add_vertex();
            assert_eq!(vertex, naive.add_vertex(), "add_vertex disagrees");
            format!("scheduled add_vertex -> {vertex}")
        } else if naive.vertex_count() == 0 {
            // Nothing can be done to a graph with no vertices, so refill it
            // before rolling the mix; this is the branch that keeps the run from
            // spinning once the 13% removals have eaten the additions.
            coverage.empty_refills += 1;
            let vertex = graph.add_vertex();
            assert_eq!(vertex, naive.add_vertex(), "add_vertex disagrees");
            format!("add_vertex (empty graph) -> {vertex}")
        } else {
            // The operation is only known after the arm picks it, so the assertions an arm
            // makes about its own result are labelled with the state it started
            // from; the post-step checks below get the full description.
            let context = step_context(seed, case, step, "about to perform an operation", &before);
            match roll {
                0..=7 => {
                    let vertex = graph.add_vertex();
                    assert_eq!(
                        vertex,
                        naive.add_vertex(),
                        "{context}: add_vertex disagrees"
                    );
                    format!("add_vertex -> {vertex}")
                }
                8..=20 => {
                    let index = live_vertices[rng.index(live_vertices.len())];
                    assert!(
                        graph.remove_vertex(index),
                        "{context}: remove_vertex({index})"
                    );
                    assert!(
                        naive.remove_vertex(index),
                        "{context}: oracle remove_vertex"
                    );
                    // A removed vertex is inert: it cannot be removed again,
                    // cannot host an edge, and is not reused as an index.
                    assert!(
                        !graph.remove_vertex(index),
                        "{context}: removing vertex {index} twice succeeded"
                    );
                    assert_eq!(
                        graph.insert(index, index),
                        None,
                        "{context}: a self-loop on the removed vertex {index} was accepted"
                    );
                    if let Some(live) = naive.live_vertices().next() {
                        assert_eq!(
                            graph.insert(index, live),
                            None,
                            "{context}: an edge from the removed vertex {index} to {live} was accepted"
                        );
                    }
                    format!("remove_vertex({index})")
                }
                21..=50 => {
                    match choose_insert_pair(&mut rng, &naive, parallel_bias) {
                        Some((u, v)) => {
                            let edge = graph
                                .insert(u, v)
                                .expect("a chosen pair is live and distinct");
                            let oracle_edge = naive.insert(u, v);
                            assert_eq!(
                                graph.naive_edge(edge),
                                Some(oracle_edge),
                                "{context}: inserted handle {edge:?} does not map to the new oracle edge"
                            );
                            format!(
                                "insert_edge({u}, {v}) -> handle {} / oracle edge {oracle_edge}",
                                edge.index()
                            )
                        }
                        None => {
                            // Fewer than two live vertices: add one, so the run
                            // can get back to inserting edges.
                            coverage.starved_inserts += 1;
                            let vertex = graph.add_vertex();
                            assert_eq!(
                                vertex,
                                naive.add_vertex(),
                                "{context}: add_vertex disagrees"
                            );
                            format!("add_vertex (need endpoints) -> {vertex}")
                        }
                    }
                }
                51..=74 => {
                    let edges = naive.edges.keys().copied().collect::<Vec<_>>();
                    if edges.is_empty() {
                        // No edge is live, so *no* handle is live. Delete a
                        // slot the driver never issued rather than a recycled
                        // one: expecting `false` from a second delete of a real
                        // handle would be wrong, because its slot may since
                        // have been handed to a brand-new edge.
                        let unused = unused_edge_handle(naive.next_edge);
                        assert_eq!(
                            graph.naive_edge(unused),
                            None,
                            "{context}: the fabricated handle is live"
                        );
                        assert!(
                            !graph.delete(unused),
                            "{context}: deleting an unissued handle succeeded"
                        );
                        assert!(
                            !naive.delete(naive.next_edge),
                            "{context}: the oracle accepted an unissued edge"
                        );
                        format!("delete_edge(unissued handle {}) -> false", unused.index())
                    } else {
                        let index = edges[rng.index(edges.len())];
                        let handle = graph
                            .handle_of(index)
                            .expect("a live oracle edge has a live handle");

                        // Classify the deletion for the coverage counters before
                        // it changes anything: this is the only place that can
                        // tell Appendix A's `Swap` from a plain cut.
                        let endpoints = graph
                            .edge_vertices(handle)
                            .expect("a live handle has endpoints");
                        let (a, b) = (graph.verts[&endpoints.0], graph.verts[&endpoints.1]);
                        let covered = graph.fb.cover_level_between(a, b) >= 0;
                        match graph.fb.edges.get(handle) {
                            Some(Edge::Tree(_)) if covered => {
                                coverage.covered_tree_edge_deletions += 1
                            }
                            Some(Edge::Tree(_)) => coverage.bridge_deletions += 1,
                            Some(Edge::NonTree(_)) => coverage.non_tree_deletions += 1,
                            None => panic!("{context}: live handle {handle:?} has no edge record"),
                        }

                        let graph_deleted = graph.delete(handle);
                        let naive_deleted = naive.delete(index);
                        assert_eq!(
                            graph_deleted, naive_deleted,
                            "{context}: delete_edge({index}) disagrees with the oracle"
                        );
                        assert!(graph_deleted, "{context}: delete_edge({index}) failed");
                        assert_eq!(
                            graph.naive_edge(handle),
                            None,
                            "{context}: handle {handle:?} is still live after deletion"
                        );
                        format!("delete_edge({index}) via handle {} -> true", handle.index())
                    }
                }
                _ => {
                    let u = live_vertices[rng.index(live_vertices.len())];
                    let v = live_vertices[rng.index(live_vertices.len())];
                    let query = rng.next() % 5;
                    let operation = format!("query({u}, {v}) variant {query}");
                    let context = step_context(seed, case, step, &operation, &before);
                    match query {
                        0 => assert_eq!(
                            graph.connected(u, v),
                            naive.connected(u, v),
                            "{context}: connected({u}, {v})"
                        ),
                        1 => assert_eq!(
                            graph.two_edge_connected(u, v),
                            naive.two_edge_connected(u, v),
                            "{context}: two_edge_connected({u}, {v})"
                        ),
                        2 => assert_eq!(
                            graph.component_size(u) as usize,
                            naive.component_size(u),
                            "{context}: component_size({u})"
                        ),
                        3 => {
                            let bridge = match graph.find_bridge(u) {
                                None => None,
                                Some(edge) => Some(returned_oracle(&graph.live, edge, &context)),
                            };
                            assert_returned_bridge_in_component(&naive, u, bridge, &context);
                        }
                        _ => {
                            let bridge = match graph.find_bridge_between(u, v) {
                                None => None,
                                Some(edge) => Some(returned_oracle(&graph.live, edge, &context)),
                            };
                            assert_returned_separating_bridge(&naive, u, v, bridge, &context);
                        }
                    }
                    operation
                }
            }
        };

        // The two oracles must describe the same multigraph. `adj` is derived
        // from `edges` by the oracle itself, so comparing the edge maps is
        // enough; comparing them every step means a differential failure below
        // is about `FindBridge`, never about this test's bookkeeping.
        let context = step_context(seed, case, step, &operation, &before);
        assert_eq!(
            graph.naive.edges, naive.edges,
            "{context}: the driver's oracle and the test oracle disagree"
        );

        let context = step_context(seed, case, step, &operation, &live_edges_snapshot(&graph));
        // Unconditional, as in the deleted driver, which called its equivalent
        // after every step too. This check is not as cheap as the old one: it
        // re-runs `find_bridge_between` for every vertex pair and re-derives
        // each candidate bridge's separating cut from the oracle, i.e. O(V^2)
        // oracle BFS on top of O(V^2) top-tree queries, which dominates the run
        // time. Every step is still affordable at the vertex counts below --
        // this module's two tests take ~3.2 s in total, measured with
        // `cargo test -p find_bridge --lib -- cover_level::graph_api::randomized`
        // -- so the every-step cadence is kept rather than being throttled.
        assert_internal_invariants(&mut graph, &context);
        if (step + 1) % EXHAUSTIVE_CHECK_INTERVAL == 0 || step + 1 == RANDOMIZED_STEPS {
            graph.assert_matches(&context);
        } else {
            assert_randomized_sample_matches_naive(&mut graph, &naive, case, step, &context);
        }

        coverage.observe(&graph);
    }

    eprintln!("seed {seed:#x}, case {case}: {coverage:?}");
    assert_coverage(&coverage, seed, case, expect_promotion);
}

/// The base suite: mostly trees and cycles with an occasional parallel edge.
///
/// All three seeds promote a non-tree edge above level 0, so
/// `randomized_differential` is the suite that guards `RecoverPhase`'s promotion
/// branch; see [`assert_coverage`].
#[test]
fn randomized_differential() {
    for (case, seed) in [0x1234_5678, 0x5eed_cafe, 0xdead_beef]
        .into_iter()
        .enumerate()
    {
        run_randomized_case(seed, PARALLEL_BIAS_SPARSE, case, true);
    }
}

/// The parallel-edge-heavy suite, which mostly inserts a second copy of a pair
/// that already has an edge.
///
/// Its graphs are a handful of doubled pairs, whose paths are one edge long, so
/// none of these three seeds reaches `RecoverPhase`'s promotion branch; the
/// `expect_promotion` flag is therefore off. What this suite buys is parallel
/// edges in bulk: multiplicity-aware bridge detection, `cover`/`uncover` of a
/// path whose endpoints already share an edge, and `uncover` when the last copy
/// of a pair is deleted.
#[test]
fn randomized_differential_with_parallel_edges() {
    for (case, seed) in [0xa5a5_0123, 0xc001_d00d, 0xfade_9876]
        .into_iter()
        .enumerate()
    {
        run_randomized_case(seed, PARALLEL_BIAS_DENSE, case, false);
    }
}
