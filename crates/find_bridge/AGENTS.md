# `find_bridge` crate guide

`find_bridge` implements dynamic bridge finding on top of `top_tree`. It has two layers: `FindBridge` exposes the paper's dynamic-forest operations, while `DynamicGraph` manages graph edges and applies the insertion/deletion reduction.

## Structure

- `src/lib.rs` documents and re-exports the public API: `FindBridge`, `Level`, `CoverLevel`, `CoverTag`, and `DynamicGraph` with its graph-level IDs.
- `src/cover_level.rs` contains the tree-level implementation. `CoverLevel` is the `top_tree::Summary`; it stores path and off-path minimum cover levels and their edges, plus the data used for `FindSize` and `FindFirstLabel`. `FindBridge` owns `TopTree<CoverLevel>` and implements `Link`, `Cut`, `Connected`, `Cover`, `Uncover`, cover/min-edge queries, `FindSize`, label operations, and removal of a vertex with no incident forest edges.
- `src/graph.rs` contains `DynamicGraph`, the Appendix A reduction. It translates stable public IDs to `top_tree::VertexId`, records whether each graph edge is a tree or non-tree edge, and implements insert/delete, vertex removal, replacement-edge search, and recovery/promotion.
- `src/cover_level/tests.rs` checks the tree-level operations against a simple forest model. `src/graph/tests.rs` checks dynamic graph behavior against a naive graph implementation, including randomized update sequences.
- `augmented_tree` supplies the aggregated balanced `BTree` used by `PartTree` in `cover_level.rs`; `top_tree` supplies the dynamic forest and summary callbacks.

## How the layers fit together

1. `DynamicGraph` keeps a spanning forest in `FindBridge`. An inserted edge joining two components becomes a tree edge. An edge whose endpoints are already connected becomes a non-tree edge: its two endpoints get user labels at the edge's level, and that level covers the tree path between them.
2. `FindBridge` maintains one structural label leaf per vertex. The leaf makes each vertex count once in `FindSize` and stores a bit mask of levels that have user labels at that vertex. Actual user-label handles and their `(vertex, level)` buckets are indexed separately; they are not additional top-tree leaves.
3. `CoverLevel` is recomputed by `TopTree` whenever clusters are merged or rebalanced. `cover`/`min_path_edge` describe the cluster path; `global_cover`/`min_global_edge` describe edges off that path. A cover level of `-1` identifies a bridge among the forest edges.
4. Path `Cover` and `Uncover` operations use `CoverTag`, the composed monotone function `g(x) = if x > threshold { x } else { constant }`, and are applied lazily to the exposed path. `PartTree` stores per-cover-level size vectors and incident-level masks, supporting `FindSize` and the guided descent for `FindFirstLabel`.
5. On deletion of a covered tree edge, `DynamicGraph` swaps in a replacement non-tree edge. `FindReplacement`, `Recover`, and `RecoverPhase` search labels and promote eligible edges/labels to higher levels while restoring path covers.

## Important constraints and maintenance notes

- `Level` represents `-1..=32`; `NO_COVER` is the separate sentinel above real cover levels. The dense size vectors and incident masks are tied to this fixed cap. `DynamicGraph` documents a supported range of fewer than `2^31` vertices; revisit the level representation and promotion logic together if changing that limit.
- `DynamicGraph::VertexId` and `DynamicGraph::EdgeId` are public stable handles, distinct from the internal `top_tree` IDs. Removed public vertex IDs are tombstoned and never reused. Keep the translation tables and `tree_edge_at`/`label_to_edge` ownership maps consistent when changing swap, recovery, or deletion behavior; vertex removal must delete incident edges through `delete_edge` first.
- `FindBridge::remove_vertex` is only valid after a vertex has no incident forest edges; it removes the structural leaf and attached user labels. Its `SwapResult` must be applied to any held internal handle for the moved last vertex.
- `CoverLevel::combine`, `flip`, `apply`, and `compose` must agree with `top_tree::Summary` semantics. Boundary-indexed `PartTree` entries must follow the logical boundary order when a cluster flips; tags must affect path children only.
- Keep graph-edge policy in `graph.rs` and tree-level summary/query logic in `cover_level.rs`. The former should use `FindBridge` operations rather than editing top-tree summaries directly.

## Paper references

The primary reference is [*Dynamic Bridge-Finding in O(log² n) Amortized Time* by Holm, Rotenberg, and Thorup (2017)](../../papers/Dynamic%20Bridge-Finding%20in%20O%28log2%20n%29%20Amortized%20Time%20%28holm%2Crotenberg%2Cthorup%29.txt):

- Section 2 gives the dynamic-tree operation interface and the level/cover invariants.
- Sections 4–6 describe `CoverLevel`, `FindSize`, and `FindFirstLabel`, respectively; these are combined in `cover_level.rs`.
- Appendix A gives the graph-level `Insert`, `Delete`, `Swap`, `FindReplacement`, `Recover`, and `RecoverPhase` reduction implemented in `graph.rs`.

Use those sections to understand the algorithmic invariants, but check the Rust implementation before attributing the paper's full asymptotic bounds to a code path: this implementation uses a fixed level cap and exact `u64` size vectors.

## Tests

From the workspace root, run `cargo test -p find_bridge` after changes. For tree-summary changes, start with `src/cover_level/tests.rs`; for graph insertion/deletion or recovery changes, use `src/graph/tests.rs` and its differential/randomized cases.
