# `top_tree` crate guide

`top_tree` is the generic dynamic-forest layer used by `find_bridge`. It maintains a binary top tree over a forest and lets an application attach a summary and optional lazy tag to every cluster. It implements the forest and cluster machinery; bridge, cover-level, and graph-update policy live in `find_bridge`.

## Structure

- `src/lib.rs` declares the modules and public exports (`TopTree`, `Summary`, `Boundary`, IDs, and forest types). Its crate docs include a small example of implementing a summary.
- `src/summary.rs` defines the application contract. `Summary::tree_edge` and `Summary::label` initialize leaf values; `Summary::combine` builds a parent value from its children and a `MergeContext`; `flip`, `apply`, and `compose` handle orientation and lazy updates.
- `src/top_tree.rs` implements `TopTree<S>` and the cluster tree. A cluster is a leaf for a forest edge or attached label, or an internal union of two clusters sharing a vertex. This file owns cluster boundaries, splay/rotation and expose logic, link/cut/attach/detach, summary recomputation, and lazy-tag propagation.
- `src/tree/mod.rs` implements the underlying forest storage: vertices, tree edges, attached labels, incidence lists, handles, and removal/remapping. It does not maintain the top-tree summaries.
- `src/top_tree/tests.rs` exercises cluster invariants and operations, including randomized link/cut/expose sequences and custom summaries. `src/tree/tests.rs` covers forest storage and handle-remapping behavior.

## Data model and invariants

- A cluster has zero, one, or two boundary vertices. Two boundaries make it a *path cluster*; zero or one make it a *point cluster*. The parent is the union of its children at their shared central vertex. These shapes determine whether a child contributes its path to the parent's path.
- Tree-edge leaves represent the dynamic forest. Label leaves are attached to one vertex and can represent vertex-local data or non-tree edges. `find_bridge` uses a label leaf per vertex for its summaries.
- `TopTree<S>` stores the forest, an arena of cluster nodes, and a bit vector of currently exposed vertices. Cluster nodes retain parent/child links, logical orientation, boundaries, the `S` summary, and a pending `S::Tag`.
- `MergeContext` is the authority for boundary-aware combination. In particular, path aggregates should include only path children; a raked point child is off the parent path. Tags are propagated only to path children, not raked children.
- Reversing a cluster changes the logical left/right order. If `S` depends on path direction, implement `Summary::flip` consistently with that reversal. For lazy updates, `apply` must update a cluster summary and `compose` must preserve the order of pending operations.
- Structural changes and exposes can change cluster boundaries and orientation. Recompute summaries through `Summary::combine` rather than assuming a cluster's old shape or child order is stable.

## Working in this crate

For a new application aggregate, implement `Summary` and keep application-specific state in that summary and its tag. Use `MergeContext` to distinguish path and point contributions. The bridge-specific example is `find_bridge::CoverLevel`; avoid putting bridge policy into the generic forest storage.

The public `TopTree` methods provide the forest operations and expose summaries/boundaries as needed. The lower-level node-inspection methods in `top_tree.rs` exist for algorithms such as `find_first_label` that must descend an exposed cluster; use them with the current logical orientation and push pending tags before inspecting child summaries.

## Paper references

The algorithm built on this crate is described in [*Dynamic Bridge-Finding in O(log² n) Amortized Time* by Holm, Rotenberg, and Thorup (2017)](../../papers/Dynamic%20Bridge-Finding%20in%20O%28log2%20n%29%20Amortized%20Time%20%28holm%2Crotenberg%2Cthorup%29.txt). Section 2 specifies the dynamic-tree operations used by the reduction, and Section 3 defines top-tree clusters, boundaries, and expose. The `Summary` interface is the application's way to maintain the Section 4 cover-level information over those clusters; that specialization is in `find_bridge/src/cover_level.rs`.

The bridge paper treats top trees as an abstract data structure. The concrete splay/rotation mechanics are implemented in `src/top_tree.rs`; the companion `papers/Splay Top Trees (Holm, Rotenberg, Ryhl).txt` gives additional background on that implementation style.

## Tests

From the workspace root, run `cargo test -p top_tree` after changes to this crate. Prefer the focused tests under `src/top_tree/tests.rs` or `src/tree/tests.rs` while iterating, then run the crate suite.
