# `find_bridge` crate guide

`find_bridge` implements dynamic bridge finding on top of `top_tree`. Everything that compiles lives in one module: `FindBridge` owns a dynamic forest whose edge weights are `EdgeId` handles and whose node weights are per-vertex `VertexLabels`, and it applies the paper's graph-level reduction itself.

The public API is therefore the single module's worth of items re-exported from `lib.rs`: `CoverLevel`, `CoverTag`, `FindBridge`, `Level`, `NO_COVER`, `UserLabel`, and the `VertexId`/`EdgeId` handles.

## Structure

- `src/lib.rs` is 72 lines of crate docs, then `mod cover_level;` and one `pub use`. Those crate docs are stale — see *Known gaps*.
- `src/cover_level.rs` (1679 lines) holds everything that is compiled:
  - `PartTree = BTree<i32, PartEntry>` with `SizeVector`, `PartEntry`, and the FindSize helpers (`along_path_find_size`, `off_path_find_size`, `total_sum`, `range_sum`, `restrict`, `diagonal_sum`, ...). `augmented_tree` supplies the aggregated balanced `BTree`.
  - `CoverLevel`, the `top_tree::Summary` impl with `type Tag = CoverTag`; `CoverTag`; `NO_COVER`; `Level` over `-1..=LEVEL_CAP` with `LEVEL_CAP = 32` and `SLOTS = LEVEL_CAP + 2`.
  - Handles `VertexId(top_tree::ClusterId)`, `EdgeId(NonMaxUsize)`, `UserLabel(NonMaxUsize)`.
  - `FindBridge`, owning `top_tree::TopTree<CoverLevel, VertexLabels, EdgeId>` and `edges: SlotVec<Edge, EdgeId>`.
  - The Appendix A reduction: `add_non_tree_edge`, `find_first_label`, `smallest_label_at`, `first_path`, `find_label_vertex`, `remove_edge`, `swap_edge_for_delete`, `recover`, `find_repalcement` (typo is in the source, `cover_level.rs:1265`), `recover_phase`, `with_path_tag`, `with_vertex_id_path_tag`.
- `FindBridge`'s public methods: `new`, `add_vertex() -> VertexId` (takes no arguments), `remove_vertex`, `edge_count`, `link -> EdgeId`, `cut_edge -> bool`, `connected`, `remove_edge`, `cover`, `uncover`, `cover_level_between -> i32`, `min_covered_edge_between -> Option<EdgeId>`, `find_bridge -> Option<EdgeId>`, `find_bridge_between -> Option<EdgeId>`, `component_size -> u64`, `two_edge_component_size -> u64`.
- `src/graph.rs` and `src/graph/tests.rs` are on disk but outside the build — see *Known gaps*.
- `top_tree` supplies the forest and summary callbacks; it needs nightly (`#![feature(pattern_types, pattern_type_macro, structural_match)]` in `crates/top_tree/src/lib.rs`).

## Important constraints and maintenance notes

- `Level` spans `-1..=32`; `NO_COVER` is the separate sentinel above every real level. The dense `SizeVector`s and the `u64` incident masks are sized to that fixed cap. The graph-level `< 2^31` vertex caveat now lives in `graph.rs`'s `DynamicGraph` docs (a WIP module): the paper's `l_max = floor(log2 n)` can exceed the cap, and hitting it can invalidate the size invariant used to guarantee a replacement edge. Revisit the level representation and the promotion logic together if that limit changes.
- `FindBridge` attaches exactly one label leaf per vertex in `add_vertex`; that leaf's `top_tree::ClusterId` is the stable identity wrapped by `VertexId`. Internal `top_tree::VertexId`/`EdgeId` are compaction-sensitive and must be re-resolved through the private `vertex_cluster`/`cluster_vertex` after a removal. `top_tree::Summary` requires only `tree_edge`, `label`, `combine`, `flip`, `apply`, and `compose` — there is no `remap_vertex`, so nothing remaps cluster data for you.
- `find_bridge` and `find_bridge_between` return a single `Option<EdgeId>`, not a vertex pair; `link` returns the new `EdgeId`.
- `remove_vertex` is only valid for a vertex with no incident forest edge and panics otherwise; cut or remove those edges first.
- `CoverLevel`'s `combine`, `flip`, `apply`, and `compose` must agree with `top_tree::Summary` semantics. Boundary-indexed `PartTree` entries must follow the logical boundary order when a cluster flips; tags must affect path children only.
- The Section 5/6 entry points are private: `find_size_internal`, `find_first_label`, `smallest_label_at`, `first_path`. Only `component_size` and `two_edge_component_size` are public, and there is no public label add/remove operation on `FindBridge`.

## Known gaps / WIP state

- `mod graph;` is absent from `lib.rs` (commit `953782c`, "disable graph module"), so `src/graph.rs`'s `DynamicGraph` and `src/graph/tests.rs` are dead code. `DynamicGraph` is **not** part of the public API; its docs are the only place the graph-level `< 2^31` limit is written down.
- `graph.rs` has uncommitted WIP on `main`: a half-finished migration from `EdgeRecord`/`EdgeKind` to a new `Edge`/`EdgeLabels` pair plus a `Vertex { cluster, generation }` record using `top_tree::Generation`. The field is already `edges: Vec<Option<Edge>>` while most of the file still constructs `EdgeRecord`, so the module would not compile if re-enabled. Do not treat it as a description of the current API.
- The last two lines of `cover_level.rs` are `// #[cfg(test)]` / `// mod tests;`, so `src/cover_level/tests.rs` is not compiled.
- `lib.rs`'s crate docs still describe `DynamicGraph`, `FindBridge::find_size`, `add_label`, `remove_label`, and `find_first_label`, none of which are public today.

## Paper references

The primary reference is [*Dynamic Bridge-Finding in O(log² n) Amortized Time* by Holm, Rotenberg, and Thorup (2017)](../../papers/Dynamic%20Bridge-Finding%20in%20O%28log2%20n%29%20Amortized%20Time%20%28holm%2Crotenberg%2Cthorup%29.txt):

- Section 2 gives the dynamic-tree operation interface and the level/cover invariants.
- Section 4 is `CoverLevel`/`CoverTag`.
- Section 5 is `FindSize`: `along_path_find_size`, `off_path_find_size`, `find_size_internal`.
- Section 6 is `FindFirstLabel`: `add_non_tree_edge`, `find_first_label`, `smallest_label_at`, `first_path`, `find_label_vertex`.
- Appendix A is the graph-level `Insert`, `Delete`, `Swap`, `FindReplacement`, `Recover`, `RecoverPhase` reduction. Those internals now live inside `FindBridge` in `cover_level.rs`; `graph.rs` is the WIP graph-level wrapper.

Use those sections for the algorithmic invariants, but check the Rust before attributing the paper's full asymptotic bounds to a code path: this implementation has a fixed level cap and exact `u64` size vectors.

## Tests

`cargo test -p find_bridge` currently **fails**: the only two tests are the stale doctests in `lib.rs`, and both fail to compile — the one at `lib.rs:22` asserts `graph.find_bridge(a) == Some((a, b))` (`E0308`, `find_bridge` now returns `Option<EdgeId>`) and the one at `lib.rs:48` does `use find_bridge::DynamicGraph;` (`E0432`).

`cargo test -p find_bridge --lib` runs **0 tests**, because `mod tests;` is commented out in `cover_level.rs`. `cargo build -p find_bridge` is clean; the lib-test build warns about the unused private `FindBridge::debug_root_incident`.

Intended test locations once re-enabled: `src/cover_level/tests.rs` for the tree-level operations against a simple forest model, and `src/graph/tests.rs` for differential/randomized dynamic-graph behavior against a naive oracle (it still imports the old `EdgeKind`, so it needs porting too).