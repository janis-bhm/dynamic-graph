//! Contract tests for the public surface of [`FindBridge`] that nothing else in
//! the crate's test suite exercises.
//!
//! Where the neighbouring modules do the job, these tests do not: `fixed` and
//! `randomized` drive the [`Graph`] driver and compare whole snapshots against
//! the [`Naive`] oracle, and the sibling `cover_level::tests` module covers the
//! `SizeVector`/`PartTree` internals, `find_first_label` and `find_size_internal`.
//! This module instead pins one documented promise of the public API per test,
//! including the corners that the oracle comparison cannot reach: dead and
//! unissued [`EdgeId`] handles, the numbering spaces of the two handle types,
//! [`CoverTag`] in isolation, and the graph-level effects of [`FindBridge::remove_edge`].
//!
//! The oracle-based modules never call `FindBridge::endpoints` on a handle the
//! driver does not own, and never pin the `Level`/`CoverTag` boundary tables, so
//! those live here.
//!
//! Reached-for internals: the public API cannot say whether an [`EdgeId`] is a
//! tree edge or a non-tree record, so a few tests read `fb.edges`. The
//! `graph_api` module owns the `Graph` driver, which is a *differential* tool
//! and not the subject of these tests, so the shared helper types of
//! `cover_level::tests` are re-implemented locally where needed.

use top_tree::Summary;
use top_tree::slot::Indexing;

use super::*;

/// [`Level`] at `value`, panicking for a value outside the level domain.
fn lvl(value: i32) -> Level {
    Level::new(value).expect("a level inside the represented domain")
}

/// The cover level that a non-tree edge label carries, read back through
/// `find_first_label`.
///
/// The public API has no "which labels are attached to this vertex" query, so
/// this reaches for the private Section 6 entry point the way `cover_level::tests`
/// does: a non-tree edge *is* a label, and `find_first_label(v, v, level)`
/// reports the smallest one attached at `vertex`.
fn label_at(fb: &mut FindBridge, vertex: VertexId, level: Level) -> Option<EdgeId> {
    let internal = fb
        .cluster_vertex(vertex.0)
        .expect("a live vertex has a stable cluster");
    fb.find_first_label(internal, internal, level)
}

/// The model of a path graph `0 - 1 - ... - n - 1` used by the
/// `min_covered_edge_between` contract test.
///
/// Model edge `k` joins model vertices `k` and `k + 1`, so the `u`-`v` path of
/// `u < v` is exactly the model edges `u..v` and there is nothing to search
/// for. `covers[k]` is the cover level the model believes model edge `k` has.
struct PathModel {
    covers: Vec<i32>,
}

impl PathModel {
    /// A path of `vertices` vertices whose edges all start at cover level -1.
    fn new(vertices: usize) -> Self {
        PathModel {
            covers: vec![-1; vertices - 1],
        }
    }

    /// `fb.cover`-equivalent: raise every model edge of the `u`-`v` path to
    /// `level`.
    fn cover(&mut self, u: usize, v: usize, level: i32) {
        let (lo, hi) = ordered(u, v);
        for cover in &mut self.covers[lo..hi] {
            *cover = (*cover).max(level);
        }
    }

    /// The minimum cover level of the `u`-`v` path, for `u != v`.
    fn path_minimum(&self, u: usize, v: usize) -> i32 {
        let (lo, hi) = ordered(u, v);
        self.covers[lo..hi]
            .iter()
            .copied()
            .min()
            .expect("a non-empty path has a minimum")
    }
}

/// `(min, max)` of two model vertices.
fn ordered(u: usize, v: usize) -> (usize, usize) {
    if u <= v { (u, v) } else { (v, u) }
}

/// Unordered endpoint pair, because `FindBridge::endpoints` does not promise an
/// order for a tree edge (see `endpoints_resolves_tree_and_non_tree_handles`).
fn unordered((u, v): (VertexId, VertexId)) -> (VertexId, VertexId) {
    if u <= v { (u, v) } else { (v, u) }
}

// ---------------------------------------------------------------------------
// Gap 1: `FindBridge::endpoints`.
// ---------------------------------------------------------------------------

/// Pins that [`FindBridge::endpoints`] answers for both kinds of live record.
///
/// `endpoints` is re-exported from `lib.rs` and documented as returning
/// `Option<(VertexId, VertexId)>`, but nothing states *which* endpoints or in
/// what order, so the order is pinned here from the implementation:
///
/// * a non-tree record stores its endpoint clusters at insertion time, so it
///   returns them in the order the `link` call passed them;
/// * a tree edge is resolved through `top_tree::edge_endpoints`, so only the
///   unordered pair is promised.
///
/// The `cover` in the middle is the graph-level step the `Graph` driver performs
/// after `link` on an already-connected pair (`graph_api::Graph::insert`); it
/// must not disturb any record.
#[test]
fn endpoints_resolves_tree_and_non_tree_handles() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let c = fb.add_vertex();

    let ab = fb.link(a, b);
    let bc = fb.link(b, c);
    assert_eq!(
        fb.tree_edge_count(),
        2,
        "two links on disconnected pairs are tree edges"
    );

    assert_eq!(
        unordered(fb.endpoints(ab).expect("a live tree edge has endpoints")),
        unordered((a, b)),
        "a tree edge resolves to the pair that was linked"
    );
    assert_eq!(
        unordered(fb.endpoints(bc).expect("a live tree edge has endpoints")),
        unordered((b, c)),
        "a tree edge resolves to the pair that was linked"
    );

    // A tree edge and a non-tree edge may share the same endpoint pair. The
    // second `link` on an already-connected pair is a non-tree record: it leaves
    // the forest alone, so `tree_edge_count` is the observable difference
    // (the record kind itself is private).
    let ab_again = fb.link(b, a);
    assert_eq!(
        fb.tree_edge_count(),
        2,
        "a link on an already-connected pair is not a tree edge"
    );
    assert!(fb.connected(a, b), "and it does not connect anything new");
    assert_ne!(ab_again, ab, "the non-tree record is a distinct handle");
    assert_eq!(
        unordered(fb.endpoints(ab_again).expect("a live non-tree record")),
        unordered((a, b)),
        "a non-tree record resolves to the same unordered pair as the tree edge"
    );
    assert_eq!(
        unordered(fb.endpoints(ab).expect("a live tree edge")),
        unordered((a, b)),
        "and the tree edge still resolves to it"
    );

    // A non-tree record keeps the argument order of the `link` that created it.
    // `link(b, a)` was passed in that order and `b > a` in cluster order, so
    // the assertion below is not trivially satisfied by a sorted pair.
    assert!(
        b > a,
        "b's cluster sorts after a's, so (b, a) is not sorted"
    );
    assert_eq!(
        fb.endpoints(ab_again),
        Some((b, a)),
        "a non-tree record stores its endpoints in link order, not sorted order"
    );

    // A non-tree record answers identically before and after the path it closes
    // is covered. The cover is the graph-level step the `Graph` driver performs
    // after `link` on a connected pair (`graph_api::Graph::insert`).
    let ca = fb.link(c, a);
    let before = fb.endpoints(ca).expect("a live non-tree record");
    assert_eq!(before, (c, a), "recorded in link order");
    fb.cover(c, a, lvl(0));
    assert_eq!(
        fb.endpoints(ca),
        Some(before),
        "covering the path must not move or reorder a record"
    );
    assert_eq!(
        fb.endpoints(ab_again),
        Some((b, a)),
        "nor the earlier record"
    );
}

// ---------------------------------------------------------------------------

/// Pins that `FindBridge::endpoints` answers `None` for a handle that names no
/// live record, rather than panicking or returning a stale pair.
///
/// No doc comment on `endpoints` states this, so the test documents reality:
/// the method is a lookup in the private `edges: SlotVec`, whose
/// `SlotVec::get` returns `None` for a free or out-of-range slot. That is what
/// makes the method safe for a handle whose slot was recycled by a later
/// insertion, and it is why the `graph_api` module documents a deleted `EdgeId`
/// as "not stable" rather than "dangling".
#[test]
fn endpoints_of_a_dead_or_unissued_handle_is_none() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let c = fb.add_vertex();
    let ab = fb.link(a, b);
    fb.link(b, c);
    fb.link(c, a);
    fb.cover(c, a, lvl(0));
    assert!(
        fb.endpoints(ab).is_some(),
        "the handle is live before anything is deleted"
    );

    // A handle that was never issued: `EdgeId::new` is reachable through the
    // `top_tree::slot::Indexing` impl for `EdgeId`, so a caller can fabricate
    // one. `index` 7 is past the end of the arena and must not panic.
    assert_eq!(ab.index(), 0, "the arena hands out index 0 first");
    assert_eq!(
        fb.endpoints(EdgeId::new(7)),
        None,
        "an index past the end of the edge arena names nothing"
    );
    assert_eq!(
        fb.endpoints(EdgeId::new(usize::MAX - 1)),
        None,
        "an astronomically out-of-range index names nothing"
    );
    assert_eq!(
        EdgeId::new(4).index(),
        4,
        "the `Indexing` impl agrees with the public `index` accessor"
    );

    // A slot freed by `remove_edge` on a *covered* tree edge. The Appendix-A
    // `Swap` writes a temporary non-tree record into the slot and `remove_edge`
    // then frees it again, so the handle is dead at the end.
    fb.remove_edge(ab);
    assert_eq!(
        fb.endpoints(ab),
        None,
        "the slot freed by remove_edge on a covered tree edge is dead"
    );

    // A bridge tree edge and a non-tree record both become dead after removal.
    let mut fb2 = FindBridge::new();
    let (x, y) = (fb2.add_vertex(), fb2.add_vertex());
    let tree = fb2.link(x, y);
    let non_tree = fb2.link(x, y);
    fb2.remove_edge(tree);
    fb2.remove_edge(non_tree);
    assert_eq!(fb2.endpoints(tree), None, "a cut tree edge slot is dead");
    assert_eq!(
        fb2.endpoints(non_tree),
        None,
        "a cut non-tree edge slot is dead"
    );
    assert_eq!(fb2.tree_edge_count(), 0, "both records are gone");
}

// ---------------------------------------------------------------------------

/// Pins the `VertexId` stability claim that `FindBridge::endpoints` depends on.
///
/// `FindBridge::vertex_cluster` documents that "its `top_tree::ClusterId` does
/// not change when other vertices are removed", and `FindBridge::remove_vertex`
/// documents that a holder of a stable `ClusterId` should re-resolve the
/// volatile internal handle instead of applying a swap. `endpoints` is the
/// public consumer of that promise: it re-resolves both endpoints of a live
/// tree edge through `vertex_cluster`, so a wrong cluster would surface here as
/// a wrong `VertexId` rather than as a panic.
#[test]
fn endpoints_survive_an_unrelated_vertex_removal() {
    let mut fb = FindBridge::new();
    let s = fb.add_vertex();
    let t = fb.add_vertex();
    let spare = fb.add_vertex();

    let st = fb.link(s, t);
    let parallel = fb.link(s, t);
    let before_tree = fb.endpoints(st);
    let before_non_tree = fb.endpoints(parallel);
    assert!(before_tree.is_some(), "a live tree edge has endpoints");
    assert!(
        before_non_tree.is_some(),
        "a live non-tree record has endpoints"
    );

    // `spare` is isolated, so it can be removed, and removing it recycles the
    // last forest slot -- the volatile handle of `s` or `t` moves.
    assert_eq!(spare.index(), 2, "the third vertex added carries cluster 2");
    assert!(fb.remove_vertex(spare), "an isolated vertex is removable");
    assert!(
        fb.cluster_vertex(spare.0).is_none(),
        "the removed vertex's cluster no longer resolves"
    );

    assert_eq!(
        fb.endpoints(st),
        before_tree,
        "a live tree edge keeps its endpoint handles across another vertex's removal"
    );
    assert_eq!(
        fb.endpoints(parallel),
        before_non_tree,
        "a live non-tree record keeps its endpoint handles too"
    );
    assert_eq!(
        before_non_tree,
        Some((s, t)),
        "a non-tree record stores clusters, so it needs no re-resolution at all"
    );

    fb.remove_edge(parallel);
    // The other queries that take `VertexId` still work on the moved vertices.
    assert!(fb.connected(s, t), "s and t are still joined");
    assert_eq!(fb.component_size(s), 2, "the component is unchanged");
    assert_eq!(fb.two_edge_component_size(s), 1, "s-t is a single bridge");
    assert_eq!(fb.find_bridge(s), Some(st), "s-t is still a bridge");
}

// ---------------------------------------------------------------------------
// Gap 2: `min_covered_edge_between`.
// ---------------------------------------------------------------------------

/// Pins the documented contract of [`FindBridge::min_covered_edge_between`]:
/// "an edge on the `u`-`v` path attaining `cover_level_between(u, v)`".
///
/// `cover_level::tests::check_against_naive` compares the same pair against an
/// oracle, but only as a by-product of a whole-snapshot comparison; this test
/// states the contract directly and deliberately builds a path with *mixed*
/// cover levels so that the minimum moves around the path. The model is a bare
/// line, so "the returned edge is on the path" needs no search: model edge `k`
/// joins model vertices `k` and `k + 1`, and the assertions compare
/// `cover_level_between` against the model first, which validates the model.
#[test]
fn min_covered_edge_between_attains_the_path_minimum() {
    let n = 7;
    let mut fb = FindBridge::new();
    let verts: Vec<VertexId> = (0..n).map(|_| fb.add_vertex()).collect();
    let edges: Vec<EdgeId> = (1..n).map(|i| fb.link(verts[i - 1], verts[i])).collect();
    let mut model = PathModel::new(n);

    // Three overlapping sub-paths at three different levels, so that neither
    // endpoint of a queried path nor its minimum is at a fixed place.
    for &(u, v, level) in &[(1, 2, 3), (2, 4, 0), (3, 5, 1)] {
        fb.cover(verts[u], verts[v], lvl(level));
        model.cover(u, v, level);
    }

    for u in 0..n {
        for v in (u + 1)..n {
            let minimum = model.path_minimum(u, v);
            assert_eq!(
                fb.cover_level_between(verts[u], verts[v]),
                minimum,
                "cover_level_between({u}, {v}) must equal the model minimum of the model edges {u}..{v}"
            );

            let chosen = fb
                .min_covered_edge_between(verts[u], verts[v])
                .unwrap_or_else(|| {
                    panic!("the {u}-{v} path is non-empty and has a finite minimum {minimum}")
                });

            let (a, b) = fb.endpoints(chosen).unwrap_or_else(|| {
                panic!("min_covered_edge_between({u}, {v}) returned a dead handle")
            });

            // Locate the returned edge in the model: its endpoints must be two
            // model vertices `k`, `k + 1` with `k` on the `u`-`v` path.
            let index_of = |vertex: VertexId| verts.iter().position(|&x| x == vertex);
            let (a_model, b_model) = (
                index_of(a).expect("an endpoint is one of the modelled vertices"),
                index_of(b).expect("an endpoint is one of the modelled vertices"),
            );
            let (k, k1) = ordered(a_model, b_model);
            assert_eq!(
                k1,
                k + 1,
                "the returned edge must be one of the model's path edges, not a chord {a_model}-{b_model}"
            );
            let (lo, hi) = ordered(u, v);
            assert!(
                k >= lo && k < hi,
                "min_covered_edge_between({u}, {v}) returned model edge {k}, which is off the path {u}..{v}"
            );
            assert_eq!(
                model.covers[k], minimum,
                "min_covered_edge_between({u}, {v}) returned model edge {k} at cover {} but the path minimum is {minimum}",
                model.covers[k]
            );
            assert_eq!(
                chosen, edges[k],
                "the returned handle is the one `link` issued for model edge {k}"
            );
        }
    }

    // Where the path minimum is attained by exactly one edge, the result is
    // pinned down rather than merely "some edge at the minimum". Three
    // single-edge covers at three levels leave the path 0-1-2-3 with cover
    // levels 2, 0, 1, and every sub-path of it then has a unique minimum.
    let mut fb2 = FindBridge::new();
    let v: Vec<VertexId> = (0..4).map(|_| fb2.add_vertex()).collect();
    let e: Vec<EdgeId> = (1..4).map(|i| fb2.link(v[i - 1], v[i])).collect();
    let mut model2 = PathModel::new(4);
    for (k, level) in [2, 0, 1].into_iter().enumerate() {
        fb2.cover(v[k], v[k + 1], lvl(level));
        model2.cover(k, k + 1, level);
    }
    assert_eq!(
        model2.covers,
        vec![2, 0, 1],
        "the model has a unique minimum on every queried path"
    );
    for (u, w) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let (lo, hi) = ordered(u, w);
        let minima: Vec<usize> = (lo..hi)
            .filter(|&k| model2.covers[k] == model2.path_minimum(u, w))
            .collect();
        assert_eq!(
            minima.len(),
            1,
            "the model path {u}-{w} must have a unique minimum edge to make this check meaningful"
        );
        assert_eq!(
            fb2.cover_level_between(v[u], v[w]),
            model2.path_minimum(u, w),
            "cover_level_between({u}, {w}) must agree with the model"
        );
        assert_eq!(
            fb2.min_covered_edge_between(v[u], v[w]),
            Some(e[minima[0]]),
            "with a unique minimum, the returned handle is that edge"
        );
    }
}

// ---------------------------------------------------------------------------

/// Pins the `None` half of `min_covered_edge_between`'s contract.
///
/// The method documents its return as `Option<EdgeId>` but says nothing about
/// `None`; the implementation returns `None` for the two cases where there is
/// no path (`u == v`, or the pair is in different trees) and for a path whose
/// exposed root carries no `min_path_edge`. The first two are pinned here; the
/// third is unreachable from the public API because every non-trivial path has
/// an edge leaf.
#[test]
fn min_covered_edge_between_is_none_without_a_path() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let lonely = fb.add_vertex();
    fb.link(a, b);

    assert_eq!(
        fb.min_covered_edge_between(a, a),
        None,
        "a trivial path has no edge, hence no minimum edge"
    );
    assert_eq!(
        fb.min_covered_edge_between(b, b),
        None,
        "a trivial path has no edge, hence no minimum edge"
    );
    assert_eq!(
        fb.min_covered_edge_between(a, lonely),
        None,
        "a disconnected pair has no path, hence no edge"
    );
    assert_eq!(
        fb.min_covered_edge_between(lonely, a),
        None,
        "the disconnected case is symmetric"
    );
    assert_eq!(
        fb.min_covered_edge_between(lonely, lonely),
        None,
        "an isolated vertex against itself has no path either"
    );

    // A non-trivial path is still answered after the negative queries, so they
    // do not disturb the exposed structure.
    assert!(
        fb.min_covered_edge_between(a, b).is_some(),
        "a-b is a real edge and must still be reported"
    );
}

// ---------------------------------------------------------------------------
// Gap 3: `cover_level_between` degenerate inputs.
// ---------------------------------------------------------------------------

/// Pins the three documented degenerate cases of `cover_level_between`.
///
/// The method documents: "If `u == v` (or the two are not connected and there
/// is no path) this is `NO_COVER`", and `NO_COVER` is documented as "strictly
/// larger than every real cover level". The third case (an isolated vertex
/// against itself) is the intersection of the first two and is pinned because
/// the `FindBridge` doctest in `lib.rs` never exercises it.
#[test]
fn cover_level_between_degenerate_inputs_are_no_cover() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let lonely = fb.add_vertex();
    fb.link(a, b);

    assert_eq!(
        fb.cover_level_between(a, a),
        NO_COVER,
        "u == v is the trivial path and has no edges"
    );
    assert_eq!(
        fb.cover_level_between(lonely, lonely),
        NO_COVER,
        "an isolated vertex against itself is still the trivial path"
    );
    assert_eq!(
        fb.cover_level_between(a, lonely),
        NO_COVER,
        "a disconnected pair has no path"
    );
    assert_eq!(
        fb.cover_level_between(lonely, a),
        NO_COVER,
        "the disconnected case is symmetric"
    );

    fb.cover(a, lonely, lvl(2));
    assert_eq!(fb.cover_level_between(a, b), -1);
    fb.uncover(a, lonely, lvl(2));
    assert_eq!(
        fb.cover_level_between(a, b),
        -1,
        "cover and uncover across components leave the existing tree unchanged"
    );

    // The sentinel really is above every real level, which is what makes it
    // usable as the degenerate answer without being a valid `Level`.
    assert_eq!(NO_COVER, i32::MAX, "NO_COVER is the i32::MAX sentinel");
    assert_eq!(
        Level::new(NO_COVER),
        None,
        "and the sentinel is not a representable Level"
    );
    for value in -1..=LEVEL_CAP {
        assert!(
            value < NO_COVER,
            "the sentinel must be strictly above the real level {value}"
        );
    }

    // A trivial `cover`/`uncover` is a no-op, so it cannot turn the sentinel
    // into a real level: `with_path_tag` returns early when `u == v`.
    fb.cover(a, a, lvl(3));
    fb.uncover(a, a, lvl(3));
    assert_eq!(
        fb.cover_level_between(a, a),
        NO_COVER,
        "cover and uncover of a trivial path must not raise anything"
    );
    assert_eq!(
        fb.cover_level_between(a, b),
        -1,
        "and they must not disturb the real path either"
    );
}

// ---------------------------------------------------------------------------
// Gap 4: `Level::increment`.
// ---------------------------------------------------------------------------

/// Pins `Level::increment` as a boundary table.
///
/// `Level::increment` is public with no doc comment; the implementation is
/// `Self::new(self.0 + 1)`, i.e. it is a partial successor that fails exactly
/// at the top of the domain. `cover_level::tests::level_bounds_and_conversions`
/// already pins `Level::new`, `TryFrom` and `From` exhaustively, so this test
/// only pins the successor. The last row matters because `FindBridge`'s
/// Appendix-A promotion loop calls `level.increment()` and takes the `None`
/// branch when a non-tree edge sits at the cap.
#[test]
fn level_increment_boundary_table() {
    assert_eq!(
        Level::MIN.increment(),
        Some(Level::new(0).expect("0 is representable")),
        "the smallest level has a successor"
    );
    assert_eq!(
        Level::MAX.increment(),
        None,
        "the largest representable level has no successor"
    );

    // The whole domain round-trips, and only the top fails.
    for value in -1..=LEVEL_CAP {
        let level = Level::new(value).expect("the loop covers the whole domain");
        assert_eq!(
            level.increment(),
            Level::new(value + 1),
            "increment of {value} must be Level::new({})",
            value + 1
        );
    }

    // `increment` is strictly monotone and only fails at the very top.
    let failures: Vec<i32> = (-1..=LEVEL_CAP)
        .filter(|&value| Level::new(value).expect("in domain").increment().is_none())
        .collect();
    assert_eq!(
        failures,
        vec![LEVEL_CAP],
        "exactly Level::MAX fails to increment"
    );

    // Repeated increments walk to the cap and stop there.
    let mut level = Level::MIN;
    for expected in -1..=LEVEL_CAP {
        assert_eq!(i32::from(level), expected, "walking up from MIN");
        if expected < LEVEL_CAP {
            level = level.increment().expect("not at the cap yet");
        }
    }
    assert_eq!(level, Level::MAX, "walking up from MIN ends at MAX");
    assert_eq!(level.increment(), None, "and stops there");
}

// ---------------------------------------------------------------------------
// Gap 5: `CoverTag`.
// ---------------------------------------------------------------------------

/// Pins `CoverTag` as exactly the two functions of Section 4 that its doc
/// comment claims: `cover(level)` is `x -> max(x, level)` and `uncover(level)`
/// is `x -> if x <= level then -1 else x`.
///
/// Nothing else in the crate constructs a `CoverTag`, so the whole point of the
/// type -- that a lazy tag is a *composable* function on cover levels -- is
/// untested. The probe table runs from below the domain to above it, which also
/// pins the two extremes the doc comment of `NO_COVER` depends on: `-1` is the
/// identity's fixed point and `NO_COVER` is a fixed point of *every* tag, which
/// is why a point cluster keeps the sentinel as its cover.
#[test]
fn cover_tag_is_the_cover_and_uncover_functions() {
    const PROBES: [i32; 9] = [
        -2,
        -1,
        0,
        1,
        2,
        LEVEL_CAP - 1,
        LEVEL_CAP,
        LEVEL_CAP + 1,
        NO_COVER,
    ];

    // `Default` is documented as "the identity on valid cover levels (-1 is the
    // smallest possible cover level)"; `identity()` is documented as "the
    // identity tag" and is defined as `Self::default()`. The qualifier matters:
    // the tag is `{threshold: -1, constant: -1}`, so it is the identity on
    // `-1..=LEVEL_CAP` and *clamps* anything below -1 up to -1. That clamping is
    // not stated anywhere, so it is pinned here as reality; it is harmless
    // because no cover level below -1 is representable.
    assert_eq!(
        CoverTag::default(),
        CoverTag::identity(),
        "identity() is default()"
    );
    for value in -1..=LEVEL_CAP {
        assert_eq!(
            CoverTag::identity().apply_to(value),
            value,
            "the identity tag leaves the valid level {value} alone"
        );
    }
    assert_eq!(
        CoverTag::identity().apply_to(-2),
        -1,
        "below the level domain the identity tag clamps to -1 rather than passing the value through"
    );
    for value in [LEVEL_CAP + 1, NO_COVER] {
        assert_eq!(
            CoverTag::identity().apply_to(value),
            value,
            "above the level domain the identity tag still passes {value} through"
        );
    }

    for level in [-1, 0, 1, 3, LEVEL_CAP] {
        let level = lvl(level);
        let raw = i32::from(level);
        let cover = CoverTag::cover(level);
        let uncover = CoverTag::uncover(level);

        for value in PROBES {
            assert_eq!(
                cover.apply_to(value),
                value.max(raw),
                "CoverTag::cover({raw}) must be x -> max(x, {raw}); it gave {} for x = {value}",
                cover.apply_to(value)
            );
            assert_eq!(
                uncover.apply_to(value),
                if value <= raw { -1 } else { value },
                "CoverTag::uncover({raw}) must be x -> if x <= {raw} then -1 else x; it gave {} for x = {value}",
                uncover.apply_to(value)
            );
        }
    }

    // The sentinel is a fixed point of every tag, which is what lets a point
    // cluster's `cover` stay `NO_COVER` under any pending tag.
    for level in [-1, 0, LEVEL_CAP] {
        let level = lvl(level);
        for tag in [
            CoverTag::identity(),
            CoverTag::cover(level),
            CoverTag::uncover(level),
        ] {
            assert_eq!(
                tag.apply_to(NO_COVER),
                NO_COVER,
                "NO_COVER must survive a tag built from level {level:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------

/// Pins that `Summary::compose` and `Summary::apply` implement function
/// composition, so the lazy path tag in a top-tree cluster means what
/// `CoverTag`'s doc comment says it means.
///
/// `CoverTag::after` and the `Summary::apply`/`Summary::compose` pair are
/// private and untested: `apply` folds the tag into the summary's own `pending`
/// field and `compose(tag, parent)` must produce "the operation that applies
/// `tag` first and then `parent`", i.e. `parent . tag`. The top-tree contract in
/// `top_tree`'s crate guide is that `compose` is called as
/// `S::compose(&mut child.tag, &parent_tag)`, so the order is what makes a
/// pushed-down stack of tags come out right; getting it backwards would silently
/// apply covers before the uncovers that preceded them.
///
/// This test states reality: the private `apply_to` is the only way to observe
/// a tag, and the private `pending` field is the only way to observe that
/// `apply` actually folded the tag rather than dropping it.
#[test]
fn cover_tag_compose_and_apply_agree_with_hand_composition() {
    const PROBES: [i32; 7] = [-1, 0, 1, 2, 3, LEVEL_CAP, NO_COVER];

    let levels = [-1, 0, 3, LEVEL_CAP];
    for &older_level in &levels {
        for &newer_level in &levels {
            for older in [
                CoverTag::cover(lvl(older_level)),
                CoverTag::uncover(lvl(older_level)),
                CoverTag::identity(),
            ] {
                for newer in [
                    CoverTag::cover(lvl(newer_level)),
                    CoverTag::uncover(lvl(newer_level)),
                    CoverTag::identity(),
                ] {
                    // `compose(tag, parent)` is documented as producing
                    // `parent . tag`: the older tag runs first.
                    let mut composed = older;
                    CoverLevel::compose(&mut composed, &newer);

                    for &x in &PROBES {
                        assert_eq!(
                            composed.apply_to(x),
                            newer.apply_to(older.apply_to(x)),
                            "compose(cover|uncover({older_level}), cover|uncover({newer_level})) must be the hand composition `newer . older`; at x = {x}"
                        );
                    }
                }
            }
        }
    }

    // `apply` folds the tag into `pending`, so a second tag composes with the
    // first instead of overwriting it. An edge leaf is the only leaf that
    // carries an edge, so it is the only one whose `global_cover` moves.
    let mut edge = CoverLevel::tree_edge(top_tree::ClusterId::new(0));
    assert_eq!(edge.cover, -1, "a fresh tree edge starts uncovered");
    assert_eq!(edge.global_cover, -1, "including off-path");
    assert_eq!(edge.pending, CoverTag::identity(), "with no pending tag");

    let first = CoverTag::cover(lvl(0));
    let second = CoverTag::uncover(lvl(0));
    edge.apply(&first);
    assert_eq!(edge.cover, first.apply_to(-1), "apply moved the path cover");
    assert_eq!(
        edge.global_cover,
        first.apply_to(-1),
        "an edge leaf's off-path cover is the same edge"
    );
    assert_eq!(
        edge.pending, first,
        "apply folded the tag into pending rather than dropping it"
    );
    edge.apply(&second);
    assert_eq!(edge.cover, -1, "the two tags cancel on a fresh edge");
    assert_eq!(edge.global_cover, -1, "including off-path");
    assert_eq!(
        edge.pending.apply_to(-1),
        second.apply_to(first.apply_to(-1)),
        "pending is the composition of the applied tags"
    );

    // A label leaf is a point cluster: both of its cover fields are `NO_COVER`,
    // which every tag leaves alone, so a pending tag cannot give a point cluster
    // a finite cover.
    let mut label = CoverLevel::label(top_tree::ClusterId::new(0));
    assert_eq!(label.cover, NO_COVER, "a point cluster has no path cover");
    assert_eq!(label.global_cover, NO_COVER, "nor an off-path cover");
    label.apply(&CoverTag::cover(lvl(3)));
    assert_eq!(label.cover, NO_COVER, "the sentinel survives a cover tag");
    assert_eq!(
        label.global_cover, NO_COVER,
        "the sentinel survives a cover tag"
    );
    label.apply(&CoverTag::uncover(lvl(3)));
    assert_eq!(
        label.cover, NO_COVER,
        "the sentinel survives an uncover tag"
    );
    assert_eq!(
        label.global_cover, NO_COVER,
        "the sentinel survives an uncover tag"
    );
}

// ---------------------------------------------------------------------------
// Gap 6: `FindBridge::default`.
// ---------------------------------------------------------------------------

/// Pins `Default for FindBridge` as "an empty, usable structure, exactly like
/// `new`".
///
/// `Default` is a required trait impl with no doc comment, and nothing in the
/// crate calls it. The contract checked here is the only one that can be
/// stated: the two constructors must produce indistinguishable structures, so
/// the same sequence of public calls gives the same answers.
#[test]
fn find_bridge_default_is_an_empty_usable_structure() {
    let mut default = FindBridge::default();
    let mut fresh = FindBridge::new();

    assert_eq!(
        default.tree_edge_count(),
        0,
        "a default FindBridge has no forest edges"
    );
    let lone = default.add_vertex();
    assert_eq!(
        default.component_size(lone),
        1,
        "a fresh vertex is a component of one"
    );
    assert_eq!(
        default.two_edge_component_size(lone),
        1,
        "and a two-edge component of one"
    );
    assert_eq!(default.find_bridge(lone), None, "with no bridge to find");
    assert_eq!(default.tree_edge_count(), 0, "still no forest edges");

    // Drive both through the same public sequence and check the same answers.
    for fb in [&mut default, &mut fresh] {
        let a = fb.add_vertex();
        let b = fb.add_vertex();
        let c = fb.add_vertex();
        let ab = fb.link(a, b);
        let bc = fb.link(b, c);
        let ca = fb.link(c, a);
        fb.cover(c, a, lvl(0));
        assert_eq!(fb.find_bridge(a), None, "a covered cycle has no bridge");
        assert_eq!(fb.component_size(a), 3, "the triangle is one component");
        assert_eq!(
            fb.two_edge_component_size(a),
            3,
            "and one two-edge component"
        );
        assert_eq!(fb.cover_level_between(a, b), 0, "the covered path");
        assert_eq!(fb.cover_level_between(b, c), 0, "the covered path");
        assert_eq!(fb.tree_edge_count(), 2, "two of the three edges are trees");

        // `remove_edge` of the non-tree record is the documented graph-level
        // delete: it drops the record and uncovers the path it covered.
        fb.remove_edge(ca);
        assert_eq!(fb.endpoints(ca), None, "the non-tree record is gone");
        assert_eq!(fb.cover_level_between(a, b), -1, "the path is uncovered");
        assert_eq!(fb.find_bridge(a), Some(ab), "a-b is a bridge again");
        // `find_bridge` answers with a bridge of v's *component*, not
        // necessarily one incident to `v`, so only membership is promised.
        assert!(
            matches!(fb.find_bridge(b), Some(edge) if edge == ab || edge == bc),
            "some bridge of the component is reported for b"
        );
        assert_eq!(fb.two_edge_component_size(b), 1, "and it is alone again");
    }
    assert_eq!(
        default.tree_edge_count(),
        fresh.tree_edge_count(),
        "default() and new() stay in step"
    );
}

// ---------------------------------------------------------------------------
// Gap 7: handle numbering spaces.
// ---------------------------------------------------------------------------

/// Pins the two facts about [`VertexId::index`] and [`EdgeId::index`] that the
/// `graph_api` module's docs warn callers about but nothing asserts.
///
/// 1. The numbering spaces are *independent*: the first vertex and the first
///    edge both live in slot 0 of their own arena, so a `VertexId` and an
///    `EdgeId` sharing an index is normal and not a bug. They are also different
///    types, so they cannot be compared to each other at all.
/// 2. `VertexId::index()` is *not* the `add_vertex` order. `VertexId` wraps a
///    `top_tree::ClusterId`, and clusters -- labels, edge leaves and internal
///    unions alike -- share one arena in the top tree, so linking vertices
///    between two `add_vertex` calls consumes indices in between. This is the
///    trap documented in the `graph_api` module docs; the existing suite avoids
///    it by keeping its own index map.
///
/// Neither `index` accessor has a doc comment, so this test documents reality
/// rather than a promise. It is worth pinning because both numbers are public
/// and a caller could reasonably assume they are creation counters.
#[test]
fn handle_indices_are_per_type_and_not_a_creation_counter() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let ab = fb.link(a, b);
    let c = fb.add_vertex();

    // (2) The third vertex added does not carry index 2.
    assert_eq!(a.index(), 0, "the first vertex added carries cluster 0");
    assert_eq!(b.index(), 1, "the second vertex added carries cluster 1");
    assert_ne!(
        c.index(),
        2,
        "the third vertex added must not carry index 2: VertexId::index() is a top-tree cluster index, not the add_vertex order"
    );
    assert!(
        c.index() > b.index() + 1,
        "the link between a and b allocated clusters in between, so the next label cluster lands past index {} (it landed at {})",
        b.index() + 1,
        c.index()
    );
    let indices = [a.index(), b.index(), c.index()];
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        3,
        "distinct vertices have distinct cluster indices, got {indices:?}"
    );

    // (1) The spaces are independent: vertex 0 and edge 0 coexist.
    assert_eq!(ab.index(), 0, "the first edge also starts at index 0");
    assert_eq!(
        a.index(),
        ab.index(),
        "a VertexId and an EdgeId may share an index; they are different arenas"
    );
    assert_eq!(
        EdgeId::new(3).index(),
        3,
        "the `Indexing` impl and the public accessor agree for EdgeId"
    );

    // Edge indices are a dense counter over a run of insertions with no
    // removals, which is the sense in which they are "dense per type".
    let mut fb2 = FindBridge::new();
    let v: Vec<VertexId> = (0..4).map(|_| fb2.add_vertex()).collect();
    let edges: Vec<EdgeId> = (1..4).map(|i| fb2.link(v[i - 1], v[i])).collect();
    assert_eq!(
        edges.iter().map(EdgeId::index).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "three links with no removals allocate edge indices 0..3"
    );

    // ... and a removal recycles the freed index, so density does not survive
    // one. This is the "a deleted EdgeId is not stable" caveat of the
    // `graph_api` module docs, pinned on the public accessor.
    fb2.remove_edge(edges[1]);
    assert_eq!(
        fb2.tree_edge_count(),
        2,
        "one of three tree edges is removed"
    );
    let recycled = fb2.link(v[0], v[3]);
    assert_eq!(
        recycled.index(),
        edges[1].index(),
        "the freed edge index is handed out again, so a stale handle can name a different live edge"
    );
    assert_eq!(
        unordered(fb2.endpoints(recycled).expect("a live tree edge")),
        unordered((v[0], v[3])),
        "and it now names the new edge"
    );
}

/// Pins `remove_edge`'s Appendix A reduction for a covered tree edge.
///
/// `remove_edge` runs Appendix A's `Delete`: for a covered tree edge it runs
/// `Swap` (promote a replacement non-tree edge to a tree edge, re-record the
/// deleted handle as a non-tree edge) and then uncovers the path and runs
/// `Recover` for every level up to the deleted edge's cover level. The
/// observable consequences are that the forest stays connected and the
/// covering is restored so the remaining path is a bridge again.
///
/// The assertions cover the connected component, the promoted replacement, the
/// freed handle, and the recovered cover level. The public API cannot express
/// "this handle used to be a non-tree edge and is now a tree edge", so that
/// assertion reads the private `fb.edges`.
#[test]
fn remove_edge_on_a_covered_tree_edge_runs_the_swap_and_recover_reduction() {
    let mut fb = FindBridge::new();
    let p = fb.add_vertex();
    let q = fb.add_vertex();
    let r = fb.add_vertex();
    let pq = fb.link(p, q);
    let qr = fb.link(q, r);
    let closing = fb.link(r, p);
    assert!(
        matches!(fb.edges[closing], Edge::NonTree(_)),
        "the edge closing the cycle starts out as a non-tree record"
    );
    fb.cover(r, p, lvl(0));
    assert_eq!(fb.cover_level_between(p, q), 0);
    assert_eq!(fb.find_bridge(p), None);
    assert_eq!(fb.two_edge_component_size(p), 3);

    fb.remove_edge(pq);

    // The forest did *not* split: `Swap` found a replacement crossing the cut
    // and promoted it, so the tree edge count is unchanged.
    assert_eq!(
        fb.tree_edge_count(),
        2,
        "Swap promotes a replacement, so no tree edge is lost"
    );
    assert!(fb.connected(p, q), "the component stays connected");
    assert!(fb.connected(q, r), "the component stays connected");
    assert_eq!(fb.component_size(p), 3, "all three vertices stay together");
    assert_eq!(fb.component_size(q), 3, "all three vertices stay together");
    assert_eq!(fb.component_size(r), 3, "all three vertices stay together");

    // The closing record is now the tree edge that spans the cut, and the
    // deleted handle's slot is free. The public API cannot distinguish a tree
    // edge from a non-tree record, so the promotion is asserted through the
    // private `fb.edges`, the way `cover_level::tests` does.
    assert_eq!(fb.endpoints(pq), None, "the deleted handle's slot is freed");
    assert!(
        matches!(fb.edges[closing], Edge::Tree(_)),
        "Swap promoted the record that crossed the cut to a tree edge"
    );
    assert_eq!(
        unordered(fb.endpoints(closing).expect("a live tree edge")),
        unordered((r, p)),
        "and the promoted edge keeps its endpoints"
    );
    assert!(
        matches!(fb.edges[qr], Edge::Tree(_)),
        "the tree edge that did not cross the cut is untouched"
    );

    // `Recover` restored the invariant: the surviving path is uncovered and a
    // bridge is exposed again.
    assert_eq!(
        fb.cover_level_between(p, r),
        -1,
        "the Delete reduction recovered the cover levels"
    );
    assert_eq!(
        fb.cover_level_between(p, q),
        -1,
        "including the path through the deleted edge's replacement"
    );
    assert_eq!(
        fb.two_edge_component_size(p),
        1,
        "every vertex is alone in its two-edge component again"
    );

    assert_eq!(fb.cover_level_between(q, r), -1);
    assert!(fb.find_bridge(p).is_some());
}

// ---------------------------------------------------------------------------
// Closely related edge cases found while reading.
// ---------------------------------------------------------------------------

/// Pins every query on a vertex that has no incident edge at all, plus the
/// trivial-path guards they all share.
///
/// The `lib.rs` doctests only ever query a vertex inside a component, and
/// `cover_level::tests::disconnected_vertices` covers a *pair* across
/// components. An isolated vertex is the case where every "no such path" branch
/// is taken at once, so it is the cheapest place to pin that the neutral answers
/// are all consistent with each other and that the structure stays usable.
#[test]
fn queries_on_an_isolated_vertex() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let lonely = fb.add_vertex();
    let ab = fb.link(a, b);

    assert_eq!(fb.tree_edge_count(), 1, "only a-b exists");
    assert!(
        fb.connected(lonely, lonely),
        "a vertex is connected to itself"
    );
    assert!(!fb.connected(lonely, a), "but not to anyone else");
    assert_eq!(
        fb.cover_level_between(lonely, lonely),
        NO_COVER,
        "the trivial path has no cover"
    );
    assert_eq!(
        fb.min_covered_edge_between(lonely, lonely),
        None,
        "and no minimum edge"
    );
    assert_eq!(
        fb.find_bridge(lonely),
        None,
        "an isolated vertex is in no bridge"
    );
    assert_eq!(
        fb.find_bridge_between(lonely, lonely),
        None,
        "a trivial path has no bridge"
    );
    assert_eq!(
        fb.component_size(lonely),
        1,
        "an isolated vertex is a component of one"
    );
    assert_eq!(
        fb.two_edge_component_size(lonely),
        1,
        "and a two-edge component of one"
    );

    // The neutral answers did not disturb the rest of the structure.
    assert!(fb.connected(a, b), "a and b are still joined");
    assert_eq!(fb.component_size(a), 2, "and still a component of two");
    assert_eq!(fb.find_bridge(a), Some(ab), "and a-b is still a bridge");
    assert_eq!(
        fb.endpoints(ab),
        Some((a, b)),
        "and the record is untouched"
    );
}

/// Pins that `Level::MAX` is usable as an ordinary cover level, and that
/// `uncover` at `Level::MAX` undoes it.
///
/// No existing test uses a `Level` above 5 (`cover_level::tests` tops out at
/// `lvl(5)`; its `LEVEL_CAP + 1` arguments are raw `i32`s for `find_size`, not
/// levels), and `Level::MAX` is the one level at which `recover_phase`'s
/// promotion loop takes its `level.increment() == None` branch, so it is the
/// most interesting member of the domain. Nothing in the doc comments promises
/// that a level above the others behaves differently, so this test documents
/// reality.
#[test]
fn cover_and_uncover_at_level_max() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let ab = fb.link(a, b);

    assert_eq!(fb.cover_level_between(a, b), -1, "a fresh edge is a bridge");
    assert_eq!(fb.find_bridge(a), Some(ab));

    fb.cover(a, b, Level::MAX);
    assert_eq!(
        fb.cover_level_between(a, b),
        i32::from(Level::MAX),
        "covering at Level::MAX raises the path to the cap"
    );
    assert_eq!(fb.find_bridge(a), None, "so the edge is no longer a bridge");
    assert_eq!(
        fb.component_size(a),
        2,
        "component_size does not depend on the cover level"
    );
    assert_eq!(
        fb.two_edge_component_size(a),
        2,
        "and a-b covered at the cap puts both vertices in one two-edge component"
    );

    // An uncover at a *smaller* level must not touch a level-capped edge.
    fb.uncover(a, b, lvl(0));
    assert_eq!(
        fb.cover_level_between(a, b),
        i32::from(Level::MAX),
        "uncovering at 0 leaves an edge at Level::MAX alone"
    );

    fb.uncover(a, b, Level::MAX);
    assert_eq!(
        fb.cover_level_between(a, b),
        -1,
        "uncovering at Level::MAX restores the bridge level"
    );
    assert_eq!(
        fb.find_bridge(a),
        Some(ab),
        "and the bridge is exposed again"
    );
    assert_eq!(
        fb.two_edge_component_size(a),
        1,
        "and the two-edge component is a single vertex again"
    );
}

/// Pins that a vertex is removable once every record mentioning it is gone, in
/// particular once its *only* incident non-tree record has been removed.
///
/// `remove_vertex` documents that it "Removes an isolated-in-the-forest vertex,
/// its structural label leaf, and all user labels attached to it", and panics
/// while a forest edge is still incident. `cover_level::tests` removes labels
/// through its own `remove_label` helper rather than through the public API, so
/// this combination is tested here.
#[test]
fn remove_vertex_after_its_only_non_tree_record_was_removed() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let tree = fb.link(a, b);
    let parallel = fb.link(a, b);
    assert!(
        matches!(fb.edges[parallel], Edge::NonTree(_)),
        "the second link on a connected pair is a non-tree record"
    );
    assert_eq!(
        label_at(&mut fb, a, lvl(0)),
        Some(parallel),
        "the non-tree record labels a at level 0"
    );

    fb.remove_edge(parallel);
    assert_eq!(
        label_at(&mut fb, a, lvl(0)),
        None,
        "removing the non-tree record drops the label from both endpoints"
    );
    assert_eq!(fb.tree_edge_count(), 1, "the forest is untouched");

    fb.remove_edge(tree);
    assert_eq!(fb.tree_edge_count(), 0, "both records are gone");
    assert_eq!(fb.component_size(a), 1, "a is now isolated");
    assert_eq!(fb.component_size(b), 1, "b is now isolated");

    assert!(fb.remove_vertex(a), "a has no incident record left");
    assert!(
        fb.cluster_vertex(a.0).is_none(),
        "and its cluster no longer resolves"
    );
    assert_eq!(fb.component_size(b), 1, "b is unaffected");
    assert!(
        fb.remove_vertex(b),
        "and b is removable too: the structure is still usable"
    );
    assert_eq!(fb.tree_edge_count(), 0, "an empty forest at the end");
}

/// Removing one endpoint of a non-tree record removes the record from the edge
/// arena and from the surviving endpoint's labels, leaving the survivor usable.
#[test]
fn remove_vertex_removes_a_non_tree_record_from_both_endpoints() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex();
    let b = fb.add_vertex();
    let tree = fb.link(a, b);
    let parallel = fb.link(a, b);

    // Remove only the *tree* edge. The non-tree record stays live and still
    // names `a`, but `a` has no incident forest edge, so the panic does not fire.
    fb.remove_edge(tree);
    assert_eq!(
        fb.component_size(a),
        2,
        "a is connected to b through `parallel`"
    );

    fb.remove_edge(parallel);
    assert_eq!(fb.component_size(a), 1, "a is now forest-isolated");

    assert!(fb.connected(a, a), "and still a live vertex");

    assert!(
        fb.remove_vertex(a),
        "the panic is only about forest edges, so a vertex with a live non-tree record is accepted"
    );
    assert_eq!(
        fb.endpoints(parallel),
        None,
        "the record is dropped from the edge arena"
    );
    assert_eq!(
        label_at(&mut fb, b, lvl(0)),
        None,
        "the surviving endpoint no longer lists the removed record"
    );
    let internal_b = fb
        .cluster_vertex(b.0)
        .expect("the surviving vertex still has a cluster");
    assert_eq!(
        fb.debug_root_incident(internal_b, internal_b) & level_bit(lvl(0)),
        0,
        "the surviving endpoint's incident mask has no stale bit at the record's level"
    );
    assert_eq!(
        label_at(&mut fb, b, lvl(1)),
        None,
        "the surviving endpoint has no label at another level"
    );

    // Re-link b into a component and check that the bridge and size queries
    // still work after removing the non-tree record.
    let c = fb.add_vertex();
    let bc = fb.link(b, c);
    assert_eq!(fb.tree_edge_count(), 1, "only b-c is a tree edge now");
    assert_eq!(fb.component_size(b), 2, "b and c form one component");
    assert_eq!(fb.component_size(c), 2, "b and c form one component");
    assert_eq!(
        fb.two_edge_component_size(b),
        1,
        "the removed record does not make b two-edge connected to c"
    );
    assert_eq!(fb.two_edge_component_size(c), 1, "and c is alone as well");
    assert_eq!(
        fb.cover_level_between(b, c),
        -1,
        "and the uncovered b-c edge still reports -1"
    );
    assert_eq!(
        fb.find_bridge(b),
        Some(bc),
        "so b-c is still reported as a bridge"
    );
    assert_eq!(
        fb.find_bridge_between(b, c),
        Some(bc),
        "and the path query agrees"
    );
}
