//! Fixed (hand-built) graph-level cases against the `Graph` driver.
//!
//! Every test here is deterministic: a hand-built graph, a handful of
//! hand-chosen mutations, and an explicit expectation for each query. The point
//! is not to explore the state space — that is `randomized`'s job — but to pin
//! the *specific* behaviours of the graph-level reduction that a randomized
//! run would only cover in aggregate: that a lone edge is a bridge, that closing
//! a cycle uncovers nothing and leaves no bridge behind, that a covered tree
//! edge is replaced when it is deleted, that a deleted handle cannot be
//! confused for a live one, and that every mutation keeps the driver's
//! bookkeeping in step with the structure.

use super::*;

/// The oracle edge index of a query result.
///
/// The bridge assertions take an *oracle* edge index, while every query returns
/// an [`EdgeId`], so the result has to be translated through the driver's
/// handle map. A `Some` result that the driver does not own would be a bug in
/// the structure or the driver, not a neutral answer, so it panics here.
fn oracle_result(graph: &Graph, result: Option<EdgeId>) -> Option<usize> {
    result.map(|edge| {
        graph
            .naive_edge(edge)
            .expect("a handle returned by a query is one the driver handed out")
    })
}

/// Asserts that `edges` is a spanning tree of `vertices`: every edge is a
/// bridge, so each vertex is alone in its two-edge component, every pair is
/// connected, only a vertex is two-edge-connected to itself, and every distinct
/// pair is separated by a real bridge.
///
/// This is the strongest structural statement the driver can make, and it is
/// the base case the rest of the graph-level reduction is measured against: a
/// tree has no non-tree edge at all, so every query degenerates to "the forest
/// itself", and any level bookkeeping mistake would show up as a spurious
/// cover.
fn assert_tree_properties(
    graph: &mut Graph,
    vertices: &[usize],
    edges: &[EdgeId],
    description: &str,
) {
    assert_eq!(graph.edge_count(), edges.len(), "{description}: edge count");
    assert_eq!(
        graph.fb.tree_edge_count(),
        edges.len(),
        "{description}: every edge of a tree is a tree edge"
    );
    assert_eq!(
        graph.naive.bridges().len(),
        edges.len(),
        "{description}: every edge of a tree is a bridge"
    );
    for &edge in edges {
        let oracle = oracle_result(graph, Some(edge)).expect("a live tree edge handle");
        assert!(
            graph.naive.is_bridge(oracle),
            "{description}: edge {edge:?} is a bridge"
        );
    }

    for &u in vertices {
        assert_eq!(
            graph.component_size(u),
            vertices.len() as u64,
            "{description}: component_size({u})"
        );
        assert_eq!(
            graph.two_edge_component_size(u),
            1,
            "{description}: two_edge_component_size({u})"
        );

        let found = graph.find_bridge(u);
        assert_returned_bridge_in_component(
            &graph.naive,
            u,
            oracle_result(graph, found),
            &format!("{description}: vertex {u}"),
        );

        for &v in vertices {
            assert!(graph.connected(u, v), "{description}: connected({u}, {v})");
            assert_eq!(
                graph.two_edge_connected(u, v),
                u == v,
                "{description}: two_edge_connected({u}, {v})"
            );
        }
        for (u, v) in pairs(vertices) {
            let bridge = graph
                .find_bridge_between(u, v)
                .expect("distinct vertices of a tree are separated by a bridge");
            assert_returned_separating_bridge(
                &graph.naive,
                u,
                v,
                oracle_result(graph, Some(bridge)),
                &format!("{description}: pair ({u}, {v})"),
            );
        }
    }
}

/// The hand-built graphs the oracle comparison is run over.
///
/// Each entry is a vertex count and the edges to insert, in order. Between them
/// they cover a bare path (every edge a bridge), two triangles joined by a
/// bridge (a two-edge component next to a one-edge component), a parallel edge
/// (a two-vertex two-edge component that no single tree edge can express), and
/// a chorded cycle (every edge covered by an overlapping cycle).
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

/// The smallest interesting graph: a single uncovered tree edge is a bridge for
/// both of its endpoints, is reported by both bridge queries, and is the only
/// thing preventing the two vertices from being two-edge-connected.
///
/// This pins the degenerate case of Appendix A's `Delete` and of the paper's
/// `FindBridge`: an edge with cover level `-1` is a bridge, and the bridge is
/// simultaneously the one in each endpoint's component and the one on the path
/// between them.
#[test]
fn single_edge_is_a_bridge() {
    let mut graph = Graph::new(2);
    let edge = graph
        .insert(0, 1)
        .expect("two distinct live vertices accept an edge");

    assert_eq!(graph.find_bridge(0), Some(edge), "find_bridge(0)");
    assert_eq!(graph.find_bridge(1), Some(edge), "find_bridge(1)");
    assert_eq!(
        graph.find_bridge_between(0, 1),
        Some(edge),
        "find_bridge_between(0, 1)"
    );
    assert_eq!(
        graph.find_bridge_between(1, 0),
        Some(edge),
        "find_bridge_between(1, 0) is the same edge as (0, 1)"
    );
    assert!(
        !graph.two_edge_connected(0, 1),
        "a single edge does not make 0 and 1 two-edge-connected"
    );
    assert_eq!(
        graph.two_edge_component_size(0),
        1,
        "two_edge_component_size(0)"
    );
    assert_eq!(
        graph.two_edge_component_size(1),
        1,
        "two_edge_component_size(1)"
    );
    assert_eq!(graph.component_size(0), 2, "component_size(0)");
    assert_eq!(graph.component_size(1), 2, "component_size(1)");

    let oracle = oracle_result(&graph, Some(edge)).expect("a live edge handle");
    assert!(graph.naive.is_bridge(oracle), "oracle edge {oracle}");
    assert_eq!(
        graph.naive.bridge_in_component(0),
        Some(oracle),
        "the oracle's only bridge lies in 0's component"
    );
    assert_eq!(
        graph.naive.bridge_separating(0, 1),
        Some(oracle),
        "the oracle's only bridge separates 0 from 1"
    );

    graph.assert_matches("single edge");
    assert_internal_invariants(&mut graph, "single edge, after the query sweep");
}

/// A cycle must leave no bridge behind: the edge that closes the cycle covers
/// the tree path between its endpoints, so none of the three edges of a
/// triangle is a bridge and the whole triangle is one two-edge component.
///
/// This is the smallest end-to-end exercise of the *non-tree* insert path of
/// Appendix A's `Insert` (link, then cover), and it is the load-bearing
/// property of the whole reduction: if a cover is forgotten or applied at the
/// wrong level, a bridge appears in a graph that has none.
#[test]
fn triangle_has_no_bridges() {
    let mut graph = Graph::new(3);
    graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    graph.insert(1, 2).expect("1-2 is a valid non-loop edge");
    graph.insert(2, 0).expect("2-0 is a valid non-loop edge");

    assert_eq!(
        graph.fb.tree_edge_count(),
        2,
        "the first two insertions are tree edges and the third is not"
    );
    assert!(
        graph.naive.bridges().is_empty(),
        "the oracle agrees a triangle has no bridge"
    );
    for v in 0..3 {
        assert_eq!(graph.find_bridge(v), None, "find_bridge({v})");
        assert_eq!(
            graph.two_edge_component_size(v),
            3,
            "two_edge_component_size({v})"
        );
        for u in 0..3 {
            assert!(
                graph.two_edge_connected(u, v),
                "two_edge_connected({u}, {v}) in a triangle"
            );
            assert_eq!(
                graph.find_bridge_between(u, v),
                None,
                "find_bridge_between({u}, {v}) in a triangle"
            );
        }
    }

    graph.assert_matches("triangle");
    assert_internal_invariants(&mut graph, "triangle, after the query grid");
}

/// Two parallel copies of the same pair are never a bridge: an edge is a
/// bridge only if the endpoint *pair* is a cut, so multiplicity alone removes
/// the bridge without any cycle being formed. Deleting one copy restores the
/// bridge, which pins the transition in both directions.
///
/// The `graph` side of this is that the second insertion must cover the
/// one-edge path between the already-connected endpoints, and that `Delete`
/// must uncover exactly that path again.
#[test]
fn parallel_edges_remove_bridge() {
    let mut graph = Graph::new(2);
    let first = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    let second = graph
        .insert(0, 1)
        .expect("a parallel copy of 0-1 is accepted");

    assert_eq!(graph.edge_count(), 2, "both copies are live");
    assert_eq!(
        graph.fb.tree_edge_count(),
        1,
        "only the first copy is a tree edge"
    );
    assert!(
        graph.naive.bridges().is_empty(),
        "the oracle agrees a parallel pair has no bridge"
    );
    assert_eq!(graph.find_bridge(0), None, "find_bridge(0)");
    assert_eq!(graph.find_bridge(1), None, "find_bridge(1)");
    assert!(
        graph.two_edge_connected(0, 1),
        "a parallel pair is two-edge-connected"
    );
    assert_eq!(
        graph.two_edge_component_size(0),
        2,
        "two_edge_component_size(0)"
    );

    assert!(graph.delete(second), "deleting the second copy succeeds");
    assert_eq!(graph.edge_count(), 1, "one copy is left");
    assert_eq!(
        graph.find_bridge(0),
        Some(first),
        "the remaining copy is a bridge again"
    );
    assert_eq!(
        graph.find_bridge_between(0, 1),
        Some(first),
        "the remaining copy separates 0 from 1"
    );
    assert!(
        !graph.two_edge_connected(0, 1),
        "a single copy is no longer two-edge-connected"
    );
    let oracle = oracle_result(&graph, Some(first)).expect("a live edge handle");
    assert!(graph.naive.is_bridge(oracle), "oracle edge {oracle}");

    graph.assert_matches("parallel edges, after deleting the second copy");
    assert_internal_invariants(&mut graph, "parallel edges, after the deletion");
}

/// A self-loop is rejected without mutating anything, and an isolated vertex
/// has no bridge and a one-vertex component.
///
/// `insert(v, v)` is the one insert that is refused for a *valid* vertex, so it
/// is the sharpest test that a refused insert leaves no trace: a half-applied
/// link or cover would change the edge count, the component size, or the
/// bookkeeping asserted by `assert_internal_invariants`.
#[test]
fn self_loop_rejected() {
    let mut graph = Graph::new(1);
    assert_eq!(graph.edge_count(), 0, "a fresh graph has no edge");
    assert_eq!(graph.vertex_count(), 1, "a fresh graph has one vertex");

    assert_eq!(graph.insert(0, 0), None, "a self-loop is refused");
    assert_eq!(graph.edge_count(), 0, "the refused self-loop added no edge");
    assert_eq!(
        graph.find_bridge(0),
        None,
        "an isolated vertex has no bridge"
    );
    assert_eq!(
        graph.component_size(0),
        1,
        "an isolated vertex is a one-vertex component"
    );
    assert_eq!(
        graph.two_edge_component_size(0),
        1,
        "an isolated vertex is its own two-edge component"
    );
    assert!(graph.connected(0, 0), "a vertex is connected to itself");

    graph.assert_matches("self loop rejected");
    assert_internal_invariants(&mut graph, "self loop rejected");
}

/// A path and a star have no non-tree edge, so every edge of both is a bridge
/// and every vertex is alone in its two-edge component.
///
/// Path and star are the two extremes of tree shape: the path maximises the
/// diameter a `find_bridge_between` query has to walk, the star maximises the
/// number of edges incident to one vertex. Together they pin the claim that the
/// level bookkeeping of a bare spanning forest is trivial and correct.
#[test]
fn tree_all_edges_bridges() {
    let mut path = Graph::new(6);
    let mut path_edges = Vec::new();
    for index in 0..5 {
        path_edges.push(
            path.insert(index, index + 1)
                .expect("a valid non-loop edge"),
        );
    }
    assert_tree_properties(&mut path, &(0..6).collect::<Vec<_>>(), &path_edges, "path");
    path.assert_matches("path");
    assert_internal_invariants(&mut path, "path, after the property sweep");

    let mut star = Graph::new(6);
    let mut star_edges = Vec::new();
    for leaf in 1..6 {
        star_edges.push(star.insert(0, leaf).expect("a valid non-loop edge"));
    }
    assert_tree_properties(&mut star, &(0..6).collect::<Vec<_>>(), &star_edges, "star");
    star.assert_matches("star");
    assert_internal_invariants(&mut star, "star, after the property sweep");
}

/// Components are independent: a query must never leak a bridge, a component
/// size, or a path across a component boundary.
///
/// The graph holds a path (a one-edge two-edge component), a triangle (a
/// three-edge two-edge component), and an isolated vertex, so all three answers
/// — one, three, and one vertex — are present at once, and the isolated vertex
/// pins the "component with no bridge" case.
#[test]
fn disconnected_components() {
    let mut graph = Graph::new(7);
    for &(u, v) in &[(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)] {
        graph.insert(u, v).expect("a valid non-loop edge");
    }

    assert!(
        !graph.connected(0, 3),
        "0 and 3 are in different components"
    );
    assert!(
        !graph.connected(0, 6),
        "0 and 6 are in different components"
    );
    assert_eq!(graph.component_size(0), 3, "component_size(0)");
    assert_eq!(graph.component_size(4), 3, "component_size(4)");
    assert_eq!(graph.component_size(6), 1, "component_size(6)");
    assert_eq!(
        graph.two_edge_component_size(0),
        1,
        "the path is a chain of one-edge two-edge components"
    );
    assert_eq!(
        graph.two_edge_component_size(4),
        3,
        "the whole triangle is one two-edge component"
    );
    assert_eq!(
        graph.two_edge_component_size(6),
        1,
        "two_edge_component_size(6)"
    );

    assert!(graph.find_bridge(0).is_some(), "the path has a bridge");
    assert_eq!(graph.find_bridge(4), None, "the triangle has no bridge");
    assert_eq!(
        graph.find_bridge(6),
        None,
        "an isolated vertex has no bridge"
    );
    assert_eq!(
        graph.find_bridge_between(0, 3),
        None,
        "no path, so no separating bridge, between 0 and 3"
    );
    assert_eq!(
        graph.find_bridge_between(6, 2),
        None,
        "no path, so no separating bridge, between 6 and 2"
    );

    let found = graph.find_bridge(0);
    assert_returned_bridge_in_component(
        &graph.naive,
        0,
        oracle_result(&graph, found),
        "the bridge reported for 0 is inside 0's own component",
    );

    graph.assert_matches("three disconnected components");
    assert_internal_invariants(&mut graph, "three disconnected components");
}

/// Deleting a bridge tree edge cuts the forest instead of running `Swap`:
/// the cover level is `-1`, so there is no replacement to find and the graph
/// splits into two components.
///
/// This is the *other* branch of Appendix A's `Delete` from
/// `delete_non_bridge_uses_replacement`, and it is the only branch that leaves
/// `FindBridge`'s level structure untouched, which makes it the case where a
/// bug in the swap path cannot hide.
#[test]
fn delete_bridge_disconnects() {
    let mut graph = Graph::new(4);
    graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    let deleted = graph.insert(1, 2).expect("1-2 is a valid non-loop edge");
    graph.insert(2, 3).expect("2-3 is a valid non-loop edge");

    assert!(graph.delete(deleted), "deleting the middle edge succeeds");
    assert_eq!(graph.edge_count(), 2, "two of the three edges are left");
    assert!(
        !graph.connected(1, 2),
        "removing a bridge disconnects its endpoints"
    );
    for v in 0..4 {
        assert_eq!(graph.component_size(v), 2, "component_size({v})");
    }
    assert!(graph.find_bridge(0).is_some(), "0-1 is still a bridge");
    assert!(graph.find_bridge(3).is_some(), "2-3 is still a bridge");
    assert_eq!(
        graph.find_bridge_between(0, 3),
        None,
        "no path remains between the two components"
    );

    graph.assert_matches("bridge deleted out of a path");
    assert_internal_invariants(&mut graph, "bridge deleted out of a path");
}

/// Deleting a *covered* tree edge runs Appendix A's `Swap` and `Recover`: the
/// deleted edge is turned into a non-tree edge at the path's minimum cover
/// level, a replacement non-tree edge is promoted to tree edge, and the
/// affected levels are recovered.
///
/// K4 is the right shape for this: its triangles overlap, so each tree edge is
/// covered twice and deleting one of them leaves a bridgeless graph, whereas a
/// single cycle would instead open into a path and grow bridges. The deleted
/// test file only ever deleted a bridge, a parallel copy, or a lone non-bridge;
/// this pins the replacement path itself.
#[test]
fn delete_non_bridge_uses_replacement() {
    let mut graph = Graph::new(4);
    for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)] {
        graph.insert(u, v).expect("a valid non-loop edge");
    }

    // The K4 edges 0-1, 1-2 and 2-3 are tree edges; the rest are non-tree.
    let ab = graph.handle_of(0).expect("oracle edge 0 is live");
    assert!(
        graph.naive.bridges().is_empty(),
        "K4 has no bridge, so 0-1 must go through Swap and not through Cut"
    );
    assert!(
        graph.delete(ab),
        "deleting the covered tree edge 0-1 succeeds"
    );
    assert_eq!(graph.edge_count(), 5, "one of K4's six edges is gone");
    assert!(
        graph.naive.bridges().is_empty(),
        "K4 minus one edge is still bridgeless"
    );
    assert_eq!(graph.fb.tree_edge_count(), 3, "the forest still spans");

    for u in 0..4 {
        assert_eq!(graph.find_bridge(u), None, "find_bridge({u})");
        assert_eq!(
            graph.two_edge_component_size(u),
            4,
            "two_edge_component_size({u})"
        );
        for v in 0..4 {
            assert!(graph.connected(u, v), "connected({u}, {v})");
            assert!(
                graph.two_edge_connected(u, v),
                "two_edge_connected({u}, {v}) after the replacement"
            );
            assert_eq!(
                graph.find_bridge_between(u, v),
                None,
                "find_bridge_between({u}, {v}) after the replacement"
            );
        }
    }

    graph.assert_matches("covered tree edge deleted out of K4");
    assert_internal_invariants(&mut graph, "covered tree edge deleted out of K4");
}

/// Deleting the *tree* copy of a parallel pair promotes the non-tree copy to
/// tree edge, so the pair keeps exactly one bridge and the surviving handle is
/// the one a query returns.
///
/// Together with `parallel_edges_remove_bridge` (which deletes the non-tree
/// copy) this pins both orders of the same `Swap`, and it is the case where
/// the two handles are interchangeable by endpoint pair, so a query that
/// reports "a bridge of this pair" cannot tell them apart — the handle
/// identity must come out right anyway.
#[test]
fn delete_parallel_edge_keeps_bridge_status() {
    let mut graph = Graph::new(2);
    let tree_edge = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    let replacement = graph
        .insert(0, 1)
        .expect("a parallel copy of 0-1 is accepted");

    assert_eq!(
        graph.find_bridge(0),
        None,
        "find_bridge(0) before the delete"
    );
    assert!(graph.delete(tree_edge), "deleting the tree copy succeeds");

    assert_eq!(graph.edge_count(), 1, "one copy is left");
    assert!(
        graph.naive_edge(tree_edge).is_none(),
        "the deleted handle is no longer live"
    );
    assert!(
        graph.edge_vertices(tree_edge).is_none(),
        "the deleted handle has no endpoints"
    );
    assert_endpoints_match(&graph, replacement, 0, 1);
    assert_eq!(
        graph.find_bridge(0),
        Some(replacement),
        "the promoted copy is the reported bridge"
    );
    assert_eq!(
        graph.find_bridge_between(0, 1),
        Some(replacement),
        "the promoted copy separates 0 from 1"
    );
    assert!(
        !graph.two_edge_connected(0, 1),
        "a single surviving copy is not two-edge-connected"
    );
    let oracle = oracle_result(&graph, Some(replacement)).expect("a live edge handle");
    assert!(graph.naive.is_bridge(oracle), "oracle edge {oracle}");

    graph.assert_matches("tree copy of a parallel pair deleted");
    assert_internal_invariants(&mut graph, "tree copy of a parallel pair deleted");
}

/// Removing a vertex deletes every incident edge through the ordinary delete
/// path, so replacement search and level recovery run as often as the vertex is
/// incident, and the edges between the *surviving* vertices must come through
/// unchanged.
///
/// The removed vertex has two incident tree edges and the survivor is a third
/// edge between the other two vertices, so the deletion has to promote the
/// survivor to keep the forest connected. Removing the index again must be
/// refused, and a later `add_vertex` must get a fresh index rather than
/// resurrecting the tombstone.
#[test]
fn remove_vertex_deletes_incident_edges_and_keeps_survivors() {
    let mut graph = Graph::new(4);
    let removed = 0;
    let a = 1;
    let isolated = 2;
    let moved = 3;

    let first = graph
        .insert(removed, a)
        .expect("0-1 is a valid non-loop edge");
    let second = graph
        .insert(removed, moved)
        .expect("0-3 is a valid non-loop edge");
    let surviving = graph
        .insert(moved, a)
        .expect("3-1 is a valid non-loop edge");

    // Deleting the first tree edge promotes the surviving edge as its
    // replacement; the second incident tree edge is deleted afterwards.
    assert!(
        graph.remove_vertex(removed),
        "removing a live vertex succeeds"
    );
    assert_eq!(graph.vertex_count(), 3, "one vertex is gone");
    assert_eq!(
        graph.edge_count(),
        1,
        "only the edge between the survivors is left"
    );
    assert!(
        graph.naive_edge(first).is_none(),
        "the first incident edge is gone"
    );
    assert!(
        graph.naive_edge(second).is_none(),
        "the second incident edge is gone"
    );
    assert_endpoints_match(&graph, surviving, moved, a);
    assert!(graph.connected(a, moved), "a and moved are still connected");
    assert_eq!(graph.component_size(a), 2, "component_size(a)");
    assert_eq!(
        graph.find_bridge(a),
        Some(surviving),
        "the surviving edge is the only bridge left"
    );
    assert_eq!(
        graph.component_size(isolated),
        1,
        "the vertex that was always isolated is untouched"
    );

    assert!(
        !graph.remove_vertex(removed),
        "a tombstoned index is refused"
    );
    let added = graph.add_vertex();
    assert_eq!(added, 4, "a new vertex gets a fresh index");
    assert_ne!(added, removed, "a removed index is never handed out again");

    assert_internal_invariants(&mut graph, "vertex removed, one added");
    graph.assert_matches("vertex removed, one added");

    assert!(
        graph.remove_vertex(added),
        "removing the new vertex succeeds"
    );
    assert!(
        !graph.connected(added, added),
        "the new index is a tombstone"
    );
    assert_eq!(graph.vertex_count(), 3, "back to three vertices");
    assert_internal_invariants(&mut graph, "the new vertex removed again");
    graph.assert_matches("the new vertex removed again");
}

/// A removed vertex index stays a tombstone forever, and a vertex added after
/// the removal is a fully working vertex: it can take part in inserts and in
/// every query, and the removed index is inert next to it.
///
/// The deleted `remove_then_add_vertex_keeps_handles_valid` checked the
/// driver's cluster tables directly, which the new driver replaces with a
/// `BTreeMap`; the same contract is pinned from the outside here, because what
/// matters is that a stale index and a fresh index never get confused.
#[test]
fn remove_then_add_vertex_keeps_handles_valid() {
    let mut graph = Graph::new(3);
    let left = 0;
    let middle = 1;
    let right = 2;

    let incident = graph
        .insert(middle, left)
        .expect("1-0 is a valid non-loop edge");
    let surviving = graph
        .insert(left, right)
        .expect("0-2 is a valid non-loop edge");

    assert!(
        graph.remove_vertex(middle),
        "removing the middle vertex succeeds"
    );
    assert_eq!(graph.vertex_count(), 2, "only the two ends are left");
    assert!(
        graph.naive_edge(incident).is_none(),
        "the removed vertex's only edge is gone"
    );
    assert_endpoints_match(&graph, surviving, left, right);

    let added = graph.add_vertex();
    assert_eq!(added, 3, "the new vertex gets a fresh index");
    assert_ne!(added, middle, "a removed index is never handed out again");

    assert!(
        !graph.remove_vertex(middle),
        "a tombstoned index is refused"
    );
    assert!(
        !graph.connected(middle, added),
        "a tombstone is not connected"
    );
    assert_eq!(
        graph.component_size(middle),
        0,
        "a tombstoned index has no component"
    );
    assert_eq!(
        graph.find_bridge(middle),
        None,
        "a tombstoned index has no bridge"
    );
    assert_eq!(
        graph.insert(middle, added),
        None,
        "a tombstone is not a usable endpoint"
    );

    let added_edge = graph
        .insert(added, left)
        .expect("3-0 is a valid non-loop edge");
    assert_endpoints_match(&graph, added_edge, added, left);
    let size = graph.component_size(added);
    let expected = graph.naive.component_size(added);
    assert_eq!(
        size as usize, expected,
        "the new vertex joins the existing component"
    );
    assert_eq!(
        size, 3,
        "the joined component holds all three live vertices"
    );

    let between = graph.find_bridge_between(added, right);
    assert_returned_separating_bridge(
        &graph.naive,
        added,
        right,
        oracle_result(&graph, between),
        "the new vertex's only bridge separates it from the far end",
    );
    let two_edge = graph.two_edge_connected(added, right);
    let expected_two_edge = graph.naive.two_edge_connected(added, right);
    assert_eq!(
        two_edge, expected_two_edge,
        "two_edge_connected({added}, {right}) matches the oracle"
    );
    assert!(
        !two_edge,
        "the new vertex is only attached through a bridge"
    );
    assert!(
        graph.two_edge_connected(added, added),
        "the new vertex is two-edge-connected to itself"
    );

    assert_internal_invariants(&mut graph, "remove-then-add vertex");
    graph.assert_matches("remove-then-add vertex");
}

/// A deleted handle is *not* guaranteed to stay dead.
///
/// [`FindBridge::link`] allocates [`EdgeId`]s from a [`SlotVec`], and a
/// `SlotVec` hands a freed slot back to the next insertion, so after a deletion
/// a stale handle may name an unrelated live edge and calling `Graph::delete`
/// on it again would delete *that* edge. The deleted
/// `delete_twice_returns_false` relied on stable handles and could not be
/// ported as written.
///
/// Liveness is therefore checked through the driver's own bookkeeping, and the
/// "a second delete returns `false`" half is asserted only after verifying that
/// the forest really has nothing in that slot any more.
#[test]
fn delete_twice_returns_false() {
    let mut graph = Graph::new(2);
    let edge = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");

    assert!(
        graph.delete(edge),
        "the first delete of a live edge succeeds"
    );
    assert!(
        graph.naive_edge(edge).is_none(),
        "the handle is no longer live"
    );
    assert!(
        graph.edge_vertices(edge).is_none(),
        "the handle no longer has driver endpoints"
    );
    assert!(
        graph.edge_endpoints(edge).is_none(),
        "the handle no longer has FindBridge endpoints"
    );
    assert_eq!(graph.edge_count(), 0, "the graph has no edge left");
    assert!(!graph.connected(0, 1), "deleting the only edge disconnects");
    assert_eq!(graph.component_size(0), 1, "component_size(0)");
    assert_eq!(graph.component_size(1), 1, "component_size(1)");

    // `remove_edge` allocates no new handle and nothing was inserted since, so
    // the slot is genuinely unoccupied; only then does a second `delete` have to
    // report "not live" instead of removing somebody else's edge.
    assert!(
        graph.fb.edges.get(edge).is_none(),
        "the forest's slot for the handle is free"
    );
    assert!(!graph.delete(edge), "a delete of a free handle is refused");
    assert_eq!(graph.edge_count(), 0, "the refused delete removed nothing");

    // The caveat is not hypothetical: re-inserting may well hand the same
    // handle out again, and a test may not assume otherwise.
    let again = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    assert_eq!(
        oracle_result(&graph, Some(again)),
        Some(1),
        "the re-inserted edge is the oracle's second edge"
    );
    if let Some(recycled) = graph.naive_edge(edge) {
        assert_eq!(
            recycled, 1,
            "a recycled handle names the new edge, not the deleted one"
        );
    }

    graph.assert_matches("edge deleted, deleted again, then re-inserted");
    assert_internal_invariants(&mut graph, "edge deleted, deleted again, then re-inserted");
}

/// `edge_endpoints` must report the endpoints of a *live* handle and nothing at
/// all for a deleted one, for tree edges and for the non-tree copy of a
/// parallel pair alike.
///
/// Endpoint recovery is the driver's only reverse mapping from a handle to the
/// graph, so a stale or mis-resolved endpoint would silently corrupt every
/// downstream comparison. The order of the pair is not part of the contract:
/// `link(a, b)` and `link(b, a)` record the same undirected edge.
#[test]
fn edge_endpoints_roundtrip() {
    let mut graph = Graph::new(3);
    let first = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    let second = graph.insert(2, 1).expect("2-1 is a valid non-loop edge");
    let parallel = graph
        .insert(1, 0)
        .expect("a parallel copy of 0-1 is accepted");

    assert_endpoints_match(&graph, first, 0, 1);
    assert_endpoints_match(&graph, second, 2, 1);
    assert_endpoints_match(&graph, parallel, 1, 0);
    let (u, v) = graph
        .edge_endpoints(first)
        .expect("a live edge has endpoints");
    let resolved = [graph.index_of(u), graph.index_of(v)];
    assert!(
        resolved.contains(&Some(0)) && resolved.contains(&Some(1)),
        "edge_endpoints round-trips the handles of 0 and 1, in some order (got {resolved:?})"
    );
    assert_ne!(first, second, "each insertion hands out a distinct handle");
    assert_ne!(
        first, parallel,
        "each insertion hands out a distinct handle"
    );
    assert_ne!(
        second, parallel,
        "each insertion hands out a distinct handle"
    );

    assert!(graph.delete(second), "deleting 2-1 succeeds");
    assert_eq!(
        graph.edge_endpoints(second),
        None,
        "a deleted handle has no endpoints"
    );
    assert_eq!(
        graph.edge_vertices(second),
        None,
        "a deleted handle has no driver endpoints"
    );

    graph.assert_matches("edge endpoints, before and after a delete");
    assert_internal_invariants(&mut graph, "edge endpoints, after the delete");
}

/// `assert_endpoints_match` documents that the expected pair may be given "in
/// either order", so a caller is entitled to pass the reverse of the order the
/// edge was inserted in. A helper that only accepted one orientation would
/// silently spare the callers that happened to guess right and fail the rest.
///
/// Every case is called in *both* orders, so whichever orientation
/// `FindBridge::endpoints` happens to report, one of the two calls exercises the
/// reversed path. The three cases are the three ways the driver stores an edge:
/// a tree edge resolved from the forest, the non-tree copy of a parallel pair
/// resolved from its non-tree record, and that same copy after a swap has
/// promoted it into the forest.
#[test]
fn assert_endpoints_match_accepts_either_order() {
    let mut graph = Graph::new(3);
    let tree = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");
    let non_tree = graph
        .insert(1, 0)
        .expect("a parallel copy of 0-1 is accepted");

    // A tree edge, called in insertion order and in the reverse.
    assert_endpoints_match(&graph, tree, 0, 1);
    assert_endpoints_match(&graph, tree, 1, 0);
    // The non-tree copy of the parallel pair, likewise both ways round.
    assert_endpoints_match(&graph, non_tree, 1, 0);
    assert_endpoints_match(&graph, non_tree, 0, 1);

    // Deleting the tree copy of a parallel pair swaps the surviving copy into
    // the forest, so the one handle now reports its endpoints from the other
    // structure entirely.
    assert!(graph.delete(tree), "deleting the tree copy succeeds");
    assert!(
        graph.edge_vertices(tree).is_none(),
        "the deleted handle has no endpoints"
    );
    assert_endpoints_match(&graph, non_tree, 0, 1);
    assert_endpoints_match(&graph, non_tree, 1, 0);

    graph.assert_matches("assert_endpoints_match, both endpoint orders");
    assert_internal_invariants(&mut graph, "assert_endpoints_match, both endpoint orders");
}

/// A driver index that was never allocated, and a driver index whose vertex has
/// been removed, are inert: every query answers neutrally and the mutating
/// operations refuse them without changing the graph.
///
/// This is the driver-level replacement for the deleted
/// `invalid_handles_return_none`, which passed `VertexId(usize::MAX)` and
/// `EdgeId(usize::MAX)` to `DynamicGraph`. The raw `FindBridge` handle
/// behaviour is covered in `surface`; here the contract is only that the
/// driver never resolves such an index to a live vertex, so a stale index
/// cannot be confused with a recycled slot the way an `EdgeId` can.
#[test]
fn stale_and_unknown_handles_are_inert() {
    let mut graph = Graph::new(2);
    let edge = graph.insert(0, 1).expect("0-1 is a valid non-loop edge");

    // 2 is one past the end of a two-vertex graph, 7 is far past it, and
    // `usize::MAX` is the value the deleted tests used for an impossible
    // handle.
    for unknown in [2, 7, usize::MAX] {
        assert_eq!(
            graph.insert(unknown, 0),
            None,
            "insert({unknown}, 0) is refused"
        );
        assert_eq!(
            graph.insert(0, unknown),
            None,
            "insert(0, {unknown}) is refused"
        );
        assert_eq!(
            graph.insert(unknown, unknown),
            None,
            "insert({unknown}, {unknown}) is refused"
        );
        assert!(
            !graph.connected(unknown, 0),
            "connected({unknown}, 0) is false"
        );
        assert!(
            !graph.connected(0, unknown),
            "connected(0, {unknown}) is false"
        );
        assert!(
            !graph.connected(unknown, unknown),
            "an unknown vertex is not even connected to itself"
        );
        assert!(
            !graph.two_edge_connected(unknown, 0),
            "two_edge_connected({unknown}, 0) is false"
        );
        assert!(
            !graph.two_edge_connected(0, unknown),
            "two_edge_connected(0, {unknown}) is false"
        );
        assert!(
            !graph.two_edge_connected(unknown, unknown),
            "two_edge_connected({unknown}, {unknown}) is false"
        );
        assert_eq!(
            graph.find_bridge(unknown),
            None,
            "find_bridge({unknown}) is None"
        );
        assert_eq!(
            graph.find_bridge_between(unknown, 0),
            None,
            "find_bridge_between({unknown}, 0) is None"
        );
        assert_eq!(
            graph.find_bridge_between(0, unknown),
            None,
            "find_bridge_between(0, {unknown}) is None"
        );
        assert_eq!(
            graph.find_bridge_between(unknown, unknown),
            None,
            "find_bridge_between({unknown}, {unknown}) is None"
        );
        assert_eq!(
            graph.component_size(unknown),
            0,
            "component_size({unknown}) is 0"
        );
        assert_eq!(
            graph.two_edge_component_size(unknown),
            0,
            "two_edge_component_size({unknown}) is 0"
        );
        assert!(
            !graph.remove_vertex(unknown),
            "remove_vertex({unknown}) is refused"
        );
    }
    assert_eq!(
        graph.vertex_count(),
        2,
        "no refused operation added a vertex"
    );
    assert_eq!(graph.edge_count(), 1, "no refused operation added an edge");
    graph.assert_matches("unknown driver indices");
    assert_internal_invariants(&mut graph, "unknown driver indices");

    // A deleted edge handle is inert on every accessor, whether or not its slot
    // would later be recycled: the driver answers from its own map, not from the
    // forest.
    assert!(graph.delete(edge), "deleting 0-1 succeeds");
    assert!(
        graph.naive_edge(edge).is_none(),
        "the deleted handle is no longer live"
    );
    assert!(
        graph.edge_endpoints(edge).is_none(),
        "a deleted handle has no endpoints"
    );
    assert!(
        graph.edge_vertices(edge).is_none(),
        "a deleted handle has no driver endpoints"
    );

    // A removed vertex index is a tombstone, not a free index, and must answer
    // exactly like an index that was never allocated.
    assert!(graph.remove_vertex(1), "removing 1 succeeds");
    assert!(!graph.remove_vertex(1), "removing 1 again is refused");
    for stale in [1, 9] {
        assert_eq!(
            graph.insert(stale, 0),
            None,
            "insert({stale}, 0) is refused"
        );
        assert_eq!(
            graph.insert(0, stale),
            None,
            "insert(0, {stale}) is refused"
        );
        assert!(!graph.connected(stale, 0), "connected({stale}, 0) is false");
        assert!(
            !graph.connected(stale, stale),
            "a tombstone is not connected to itself"
        );
        assert!(
            !graph.two_edge_connected(stale, 0),
            "two_edge_connected({stale}, 0) is false"
        );
        assert_eq!(
            graph.find_bridge(stale),
            None,
            "find_bridge({stale}) is None"
        );
        assert_eq!(
            graph.find_bridge_between(stale, 0),
            None,
            "find_bridge_between({stale}, 0) is None"
        );
        assert_eq!(
            graph.find_bridge_between(0, stale),
            None,
            "find_bridge_between(0, {stale}) is None"
        );
        assert_eq!(
            graph.component_size(stale),
            0,
            "component_size({stale}) is 0"
        );
        assert_eq!(
            graph.two_edge_component_size(stale),
            0,
            "two_edge_component_size({stale}) is 0"
        );
        assert!(
            !graph.remove_vertex(stale),
            "remove_vertex({stale}) is refused"
        );
    }

    graph.assert_matches("stale driver indices and a deleted handle");
    assert_internal_invariants(&mut graph, "stale driver indices and a deleted handle");
}

/// Every query of the graph-level API must agree with the BFS oracle on a table
/// of hand-built graphs: per-vertex component and two-edge component sizes, the
/// full grid of `two_edge_connected`, and a bridge for every vertex and every
/// pair.
///
/// This folds the deleted `component_size_matches_naive`,
/// `two_edge_component_size_matches_naive` and `two_edge_connected_matches_naive`
/// into one test. Between them the four fixtures cover a bare path, a bridged
/// pair of triangles, a parallel edge, and a chorded cycle, i.e. every
/// combination of "component is a tree", "two-edge component is a cycle" and
/// "two endpoints are joined twice".
#[test]
fn component_sizes_and_two_edge_connectivity_match_the_oracle() {
    for (case, (vertex_count, edges)) in fixed_graphs().into_iter().enumerate() {
        let mut graph = Graph::new(vertex_count);
        for &(u, v) in &edges {
            graph.insert(u, v).expect("a valid non-loop edge");
        }
        assert_eq!(
            graph.edge_count(),
            edges.len(),
            "fixture {case}: every edge was inserted"
        );

        for v in 0..vertex_count {
            let size = graph.component_size(v);
            assert_eq!(
                size as usize,
                graph.naive.component_size(v),
                "fixture {case}, vertex {v}: component_size"
            );
            let two_edge_size = graph.two_edge_component_size(v);
            assert_eq!(
                two_edge_size as usize,
                graph.naive.two_edge_component_size(v),
                "fixture {case}, vertex {v}: two_edge_component_size"
            );

            let found = graph.find_bridge(v);
            assert_returned_bridge_in_component(
                &graph.naive,
                v,
                oracle_result(&graph, found),
                &format!("fixture {case}, vertex {v}"),
            );
        }

        for u in 0..vertex_count {
            for v in 0..vertex_count {
                let is_connected = graph.connected(u, v);
                assert_eq!(
                    is_connected,
                    graph.naive.connected(u, v),
                    "fixture {case}, pair ({u}, {v}): connected"
                );
                let is_two_edge_connected = graph.two_edge_connected(u, v);
                assert_eq!(
                    is_two_edge_connected,
                    graph.naive.two_edge_connected(u, v),
                    "fixture {case}, pair ({u}, {v}): two_edge_connected"
                );

                let between = graph.find_bridge_between(u, v);
                assert_returned_separating_bridge(
                    &graph.naive,
                    u,
                    v,
                    oracle_result(&graph, between),
                    &format!("fixture {case}, pair ({u}, {v})"),
                );
            }
        }

        let context = format!("fixture {case}");
        assert_internal_invariants(&mut graph, &context);
        graph.assert_matches(&context);
    }
}

/// Deleting a non-tree edge that three cycles share keeps the whole graph
/// two-edge-connected.
///
/// The deleted file only ever deleted a bridge, a parallel copy, or a covered
/// *tree* edge, so the pure `Delete` branch — drop the edge's two labels, then
/// `Uncover` the path and `Recover` the affected levels — was never exercised
/// without `Swap`. Here the three triangles `0-1-2`, `0-2-3` and `0-2-4` all
/// share the single edge `0-2`, so uncovering its path must leave every
/// surviving edge covered by one of the other two triangles, at whatever level
/// recovery decided.
#[test]
fn delete_shared_non_tree_edge_keeps_overlapping_cycles() {
    let mut graph = Graph::new(5);
    for &(u, v) in &[(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (2, 4), (0, 4)] {
        graph.insert(u, v).expect("a valid non-loop edge");
    }
    let shared = graph.handle_of(2).expect("oracle edge 2 is live");
    assert_endpoints_match(&graph, shared, 0, 2);
    assert!(
        graph.naive.bridges().is_empty(),
        "the three overlapping triangles leave no bridge"
    );
    for v in 0..5 {
        assert_eq!(
            graph.two_edge_component_size(v),
            5,
            "two_edge_component_size({v}) before the delete"
        );
    }

    assert!(
        graph.delete(shared),
        "deleting the shared non-tree edge succeeds"
    );
    assert_eq!(graph.edge_count(), 6, "one of the seven edges is gone");
    assert!(
        graph.naive.bridges().is_empty(),
        "the two remaining triangles still overlap, so nothing is a bridge"
    );
    for u in 0..5 {
        assert_eq!(graph.find_bridge(u), None, "find_bridge({u})");
        assert_eq!(
            graph.two_edge_component_size(u),
            5,
            "two_edge_component_size({u}) after the delete"
        );
        for v in 0..5 {
            assert!(graph.connected(u, v), "connected({u}, {v})");
            assert!(
                graph.two_edge_connected(u, v),
                "two_edge_connected({u}, {v}) after the delete"
            );
            assert_eq!(
                graph.find_bridge_between(u, v),
                None,
                "find_bridge_between({u}, {v}) after the delete"
            );
        }
    }

    graph.assert_matches("shared non-tree edge deleted");
    assert_internal_invariants(&mut graph, "shared non-tree edge deleted");
}

/// Removing a vertex that has both tree and parallel (non-tree) incident edges
/// must leave a consistent forest behind.
///
/// The deleted file only removed a vertex whose incident edges were all tree
/// edges. Here the removed vertex carries a parallel pair, so one of the two
/// `remove_edge` calls operates on an edge that the previous one may already
/// have promoted to a tree edge, and the only edge left over has to end up
/// carrying the forest on its own: two vertices, one edge, one bridge.
#[test]
fn remove_vertex_with_parallel_and_tree_incident_edges() {
    let mut graph = Graph::new(3);
    let removed = 0;
    let first_copy = graph
        .insert(removed, 1)
        .expect("0-1 is a valid non-loop edge");
    let second_copy = graph.insert(removed, 1).expect("a parallel copy of 0-1");
    let to_two = graph
        .insert(removed, 2)
        .expect("0-2 is a valid non-loop edge");
    let survivor = graph.insert(2, 1).expect("2-1 is a valid non-loop edge");

    assert_eq!(graph.edge_count(), 4, "all four edges are live");
    assert_eq!(
        graph.fb.tree_edge_count(),
        2,
        "the parallel pair contributes one tree edge and one non-tree edge"
    );
    assert!(
        graph.naive.bridges().is_empty(),
        "the triangle with a doubled side has no bridge"
    );
    for v in 0..3 {
        assert_eq!(
            graph.two_edge_component_size(v),
            3,
            "two_edge_component_size({v}) before the removal"
        );
    }

    assert!(graph.remove_vertex(removed), "removing 0 succeeds");
    assert_eq!(graph.vertex_count(), 2, "only 1 and 2 are left");
    assert_eq!(graph.edge_count(), 1, "only the 2-1 edge is left");
    for handle in [first_copy, second_copy, to_two] {
        assert!(
            graph.naive_edge(handle).is_none(),
            "every edge incident to the removed vertex is gone"
        );
    }
    // The driver records an edge's endpoints in the order it was inserted, which
    // is the order `assert_bookkeeping` requires of a tree edge.
    assert_endpoints_match(&graph, survivor, 2, 1);
    assert_eq!(graph.fb.tree_edge_count(), 1, "one edge holds the forest");
    assert!(graph.connected(1, 2), "1 and 2 are still connected");
    assert_eq!(graph.component_size(1), 2, "component_size(1)");
    assert_eq!(graph.component_size(2), 2, "component_size(2)");
    assert_eq!(
        graph.two_edge_component_size(1),
        1,
        "the survivors are no longer two-edge-connected"
    );
    assert_eq!(
        graph.find_bridge(1),
        Some(survivor),
        "the surviving edge is the only bridge left"
    );
    assert_eq!(
        graph.find_bridge_between(1, 2),
        Some(survivor),
        "the surviving edge separates 1 from 2"
    );

    graph.assert_matches("vertex with a parallel incident edge removed");
    assert_internal_invariants(&mut graph, "vertex with a parallel incident edge removed");
}

/// Linking two trees at interior vertices must produce a tree, and the join
/// must be visible on the handle it was given.
///
/// Every other fixed case inserts a tree edge with at least one endpoint that
/// is still isolated, so the forest is only ever grown at its fringe. Joining
/// two non-trivial trees at interior vertices instead splits both of them, and
/// a mistake there (a lost subtree, a mis-resolved cluster, a stale endpoint on
/// the join) would show up as a wrong component size or a missing bridge.
#[test]
fn insert_joins_two_trees_at_interior_vertices() {
    let mut graph = Graph::new(8);
    let mut edges = Vec::new();
    for index in 0..3 {
        edges.push(
            graph
                .insert(index, index + 1)
                .expect("a valid non-loop edge"),
        );
    }
    for index in 4..7 {
        edges.push(
            graph
                .insert(index, index + 1)
                .expect("a valid non-loop edge"),
        );
    }

    // Both endpoints of the future join are interior vertices: 1 hangs off 0 and
    // 2, and 5 hangs off 4, 6 and 7. Every other vertex is a leaf of one of the
    // two trees.
    let pre_join = [(0, 4u64), (1, 4), (5, 4), (7, 4)];
    for (vertex, expected) in pre_join {
        let size = graph.component_size(vertex);
        assert_eq!(size, expected, "component_size({vertex}) before the join");
    }
    assert_eq!(graph.fb.tree_edge_count(), 6, "two separate trees");
    assert_eq!(graph.naive.bridges().len(), 6, "both trees are all bridge");

    let join = graph.insert(1, 5).expect("1-5 is a valid non-loop edge");
    edges.push(join);
    assert_eq!(
        graph.edge_vertices(join),
        Some((1, 5)),
        "the join edge joins the two interior vertices"
    );
    assert_eq!(
        graph.fb.tree_edge_count(),
        7,
        "the join is a tree edge, so the forest is one tree"
    );

    assert_tree_properties(
        &mut graph,
        &(0..8).collect::<Vec<_>>(),
        &edges,
        "two paths joined",
    );
    graph.assert_matches("two paths joined at interior vertices");
    assert_internal_invariants(&mut graph, "two paths joined at interior vertices");
}
