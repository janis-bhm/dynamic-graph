//! The CoverLevel structure of Section 4, and the [`FindBridge`] wrapper.
//!
//! For every cluster `C` we maintain
//!
//! * `cover(C)`, the minimum cover level of an edge on the cluster path
//!   `pi(C)` (or the sentinel [`NO_COVER`] if `C` is a point cluster),
//! * `global_cover(C)`, the minimum cover level of an edge of `C` that is not
//!   on the cluster path,
//! * `min_path_edge(C)`, an edge of `pi(C)` attaining `cover(C)`,
//! * `min_global_edge(C)`, an edge of `C \ pi(C)` attaining `global_cover(C)`.
//!
//! Covering and uncovering a path does not touch every edge individually.
//! Instead the whole exposed path is tagged lazily. Because the cover levels
//! of a path are updated by the operations `x -> max(x, i)` (a *cover*) and
//! `x -> if x <= i then -1 else x` (an *uncover*), any sequence of them
//! composes to a function of the form
//!
//! ```text
//! g(x) = if x > t then x else c
//! ```
//!
//! and `c <= t` is an invariant. Such a function is exactly [`CoverTag`]. The
//! minimum of the path is updated by simply evaluating `g` on the previous
//! minimum, so the edge attaining it never changes: `g` is monotone.
//!
//! # FindSize and FindFirstLabel
//!
//! [`FindBridge::find_size`] and [`FindBridge::find_first_label`] implement
//! the queries of Sections 5 and 6. The paper maintains them as part of the
//! top tree summary using vectors indexed by the *identity* of the cluster's
//! boundary vertices. The safe [`top_tree`] abstraction only exposes the
//! *number* of boundary vertices to [`Summary::combine`], not which vertices
//! they are (and flipping a cluster does not reorient its summary), so those
//! boundary-indexed vectors cannot be expressed without widening that
//! abstraction. These two queries are therefore answered directly from the
//! maintained cover levels, which keeps them correct at the cost of the
//! paper's polylogarithmic bound.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use top_tree::{MergeContext, Summary};

/// Sentinel used for "there is no such edge" and for the cover level of a
/// point cluster. It is strictly larger than every real cover level.
pub const NO_COVER: i32 = i32::MAX;

/// A stable handle to a user label added with [`FindBridge::add_label`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(pub usize);

/// An undirected edge, stored as the pair of endpoints passed to
/// [`FindBridge::link`].
pub type Edge = (usize, usize);

/// A lazily pending pair of `Cover`/`Uncover` operations, represented as the
/// monotone function `g(x) = if x > threshold { x } else { constant }`.
///
/// The default is the identity on valid cover levels (`-1` is the smallest
/// possible cover level).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverTag {
    threshold: i32,
    constant: i32,
}

impl Default for CoverTag {
    fn default() -> Self {
        CoverTag {
            threshold: -1,
            constant: -1,
        }
    }
}

impl CoverTag {
    /// The function `x -> max(x, level)` implementing a `Cover`.
    pub fn cover(level: i32) -> Self {
        CoverTag {
            threshold: level,
            constant: level,
        }
    }

    /// The function `x -> if x <= level then -1 else x` implementing an
    /// `Uncover`.
    pub fn uncover(level: i32) -> Self {
        CoverTag {
            threshold: level,
            constant: -1,
        }
    }

    /// The identity tag.
    pub fn identity() -> Self {
        Self::default()
    }

    fn apply_to(&self, value: i32) -> i32 {
        if value > self.threshold {
            value
        } else {
            self.constant
        }
    }

    /// Returns the composition `self ∘ older`, i.e. the operation that applies
    /// `older` first and then `self`.
    fn after(self, older: Self) -> Self {
        if self.threshold <= older.threshold {
            CoverTag {
                threshold: older.threshold,
                constant: if older.constant > self.threshold {
                    older.constant
                } else {
                    self.constant
                },
            }
        } else {
            // `older.constant <= older.threshold < self.threshold`, so the
            // older operation is masked out for every value it could change.
            self
        }
    }
}

/// What a cluster is made of, which is needed to interpret its summary when it
/// is a leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeafKind {
    /// A single tree edge.
    Edge,
    /// A single vertex label.
    Label,
    /// The union of two children.
    Internal,
}

/// Per-cluster summary of the CoverLevel structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverLevel {
    /// Minimum cover level on the cluster path, or [`NO_COVER`].
    pub cover: i32,
    /// Minimum cover level off the cluster path, or [`NO_COVER`].
    pub global_cover: i32,
    /// An edge of the cluster path attaining `cover`.
    pub min_path_edge: Option<Edge>,
    /// An edge off the cluster path attaining `global_cover`.
    pub min_global_edge: Option<Edge>,
    leaf: LeafKind,
}

/// Picks the smaller of two `(level, edge)` candidates, preferring the first
/// on ties.
fn min_candidate(left: (i32, Option<Edge>), right: (i32, Option<Edge>)) -> (i32, Option<Edge>) {
    if right.0 < left.0 { right } else { left }
}

impl Summary<()> for CoverLevel {
    type Tag = CoverTag;

    fn tree_edge(_weight: &(), u: usize, v: usize) -> Self {
        // A freshly linked tree edge is a bridge: its cover level is -1. The
        // same value is stored as the global cover so that an edge leaf can
        // serve as the root of an `expose`; whether the edge is on or off the
        // cluster path is decided by the parent (or by the root query).
        CoverLevel {
            cover: -1,
            global_cover: -1,
            min_path_edge: Some((u, v)),
            min_global_edge: Some((u, v)),
            leaf: LeafKind::Edge,
        }
    }

    fn label(_weight: &(), _v: usize) -> Self {
        CoverLevel {
            cover: NO_COVER,
            global_cover: NO_COVER,
            min_path_edge: None,
            min_global_edge: None,
            leaf: LeafKind::Label,
        }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        let parent_is_path = ctx.boundary == 2;
        let left_is_path = ctx.left_boundary == 2;
        let right_is_path = ctx.right_boundary == 2;

        // The cluster path of the parent is the concatenation of the paths of
        // its path children.
        let (cover, min_path_edge) = if parent_is_path {
            let mut best = (NO_COVER, None);
            if left_is_path {
                best = min_candidate(best, (left.cover, left.min_path_edge));
            }
            if right_is_path {
                best = min_candidate(best, (right.cover, right.min_path_edge));
            }
            best
        } else {
            (NO_COVER, None)
        };

        // For the off-path minimum, an edge that ends up on the parent's path
        // must not be counted again, while an edge of a child that is raked in
        // wholesale counts even if it was that child's path edge. An edge leaf
        // is special: it carries a single edge, which is on the parent's path
        // exactly when the leaf is a path child of a path parent.
        let candidate = |child: &Self, child_is_path: bool| -> (i32, Option<Edge>) {
            if child.leaf == LeafKind::Edge {
                if parent_is_path && child_is_path {
                    (NO_COVER, None)
                } else {
                    (child.cover, child.min_path_edge)
                }
            } else if (parent_is_path && child_is_path) || child.global_cover <= child.cover {
                (child.global_cover, child.min_global_edge)
            } else {
                (child.cover, child.min_path_edge)
            }
        };
        let (global_cover, min_global_edge) = min_candidate(
            candidate(left, left_is_path),
            candidate(right, right_is_path),
        );

        CoverLevel {
            cover,
            global_cover,
            min_path_edge,
            min_global_edge,
            leaf: LeafKind::Internal,
        }
    }

    fn apply(&mut self, tag: &Self::Tag) {
        self.cover = tag.apply_to(self.cover);
        if self.leaf == LeafKind::Edge {
            self.global_cover = tag.apply_to(self.global_cover);
        }
    }

    fn compose(tag: &mut Self::Tag, parent: &Self::Tag) {
        *tag = parent.after(*tag);
    }
}

/// A dynamic forest supporting the CoverLevel operations and bridge queries of
/// Section 4.
///
/// The forest is maintained in a [`top_tree::TopTree`]; tree edges start with
/// cover level `-1` (they are bridges). [`FindBridge::cover`] and
/// [`FindBridge::uncover`] update the cover levels of a whole path lazily.
pub struct FindBridge {
    top_tree: top_tree::TopTree<usize, CoverLevel, (), ()>,
    /// Adjacency of the forest, used by the FindSize and FindFirstLabel
    /// queries.
    adj: Vec<BTreeSet<usize>>,
    /// The user labels of the FindFirstLabel structure, keyed by handle.
    labels: BTreeMap<LabelId, (usize, i32)>,
    next_label: usize,
}

impl Default for FindBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl FindBridge {
    /// Creates an empty forest.
    pub fn new() -> Self {
        FindBridge {
            top_tree: top_tree::TopTree::new(),
            adj: Vec::new(),
            labels: BTreeMap::new(),
            next_label: 0,
        }
    }

    /// Adds a vertex keyed by `key` and returns its index.
    pub fn add_vertex(&mut self, key: usize) -> usize {
        let index = self.top_tree.add_vertex(key, ());
        while self.adj.len() <= index {
            self.adj.push(BTreeSet::new());
        }
        index
    }

    /// Returns the index of the vertex with the given key, if it exists.
    pub fn vertex_index(&self, key: usize) -> Option<usize> {
        self.top_tree.vertex_index(&key)
    }

    /// The number of tree edges.
    pub fn edge_count(&self) -> usize {
        self.top_tree.edge_count()
    }

    /// Links two vertices that are in different trees with a new tree edge.
    pub fn link(&mut self, u: usize, v: usize) {
        self.top_tree.link(u, v, ());
        self.adj[u].insert(v);
        self.adj[v].insert(u);
    }

    /// Cuts the tree edge between `u` and `v`, returning whether it existed.
    pub fn cut(&mut self, u: usize, v: usize) -> bool {
        if self.top_tree.cut(u, v).is_none() {
            return false;
        }
        self.adj[u].remove(&v);
        self.adj[v].remove(&u);
        true
    }

    /// Returns whether `u` and `v` are in the same tree.
    pub fn connected(&mut self, u: usize, v: usize) -> bool {
        self.top_tree.connected(u, v)
    }

    /// Applies `Cover(u, v, level)`: every edge on the `u`-`v` path whose
    /// cover level is below `level` is raised to `level`.
    pub fn cover(&mut self, u: usize, v: usize, level: i32) {
        self.with_path_tag(u, v, CoverTag::cover(level));
    }

    /// Applies `Uncover(u, v, level)`: every edge on the `u`-`v` path whose
    /// cover level is at most `level` gets cover level `-1`.
    pub fn uncover(&mut self, u: usize, v: usize, level: i32) {
        self.with_path_tag(u, v, CoverTag::uncover(level));
    }

    fn with_path_tag(&mut self, u: usize, v: usize, tag: CoverTag) {
        if u == v {
            // A trivial path has no edges, so there is nothing to tag.
            return;
        }
        self.top_tree.expose_path_tagged(u, v, tag);
        self.top_tree.deexpose(v);
        self.top_tree.deexpose(u);
    }

    /// `CoverLevel(v)`: the minimum cover level of any edge in `v`'s tree, or
    /// [`NO_COVER`] if there are none.
    pub fn cover_level(&mut self, v: usize) -> i32 {
        let summary = self.top_tree.expose(v);
        self.top_tree.deexpose(v);
        summary.map_or(NO_COVER, |node| node.global_cover)
    }

    /// `MinCoveredEdge(v)`: an edge attaining [`FindBridge::cover_level`].
    pub fn min_covered_edge(&mut self, v: usize) -> Option<Edge> {
        let summary = self.top_tree.expose(v);
        self.top_tree.deexpose(v);
        summary.and_then(|node| node.min_global_edge)
    }

    /// `CoverLevel(u, v)`: the minimum cover level on the `u`-`v` path. If
    /// `u == v` (or the two are not connected and there is no path) this is
    /// [`NO_COVER`].
    pub fn cover_level_between(&mut self, u: usize, v: usize) -> i32 {
        if u == v || !self.connected(u, v) {
            return NO_COVER;
        }
        let summary = self.top_tree.expose_path(u, v);
        self.top_tree.deexpose(v);
        self.top_tree.deexpose(u);
        summary.map_or(NO_COVER, |node| node.cover)
    }

    /// `MinCoveredEdge(u, v)`: an edge on the `u`-`v` path attaining
    /// [`FindBridge::cover_level_between`].
    pub fn min_covered_edge_between(&mut self, u: usize, v: usize) -> Option<Edge> {
        if u == v || !self.connected(u, v) {
            return None;
        }
        let summary = self.top_tree.expose_path(u, v);
        self.top_tree.deexpose(v);
        self.top_tree.deexpose(u);
        summary.and_then(|node| node.min_path_edge)
    }

    /// `FindBridge(v)`: a bridge in `v`'s tree, if one exists.
    pub fn find_bridge(&mut self, v: usize) -> Option<Edge> {
        if self.cover_level(v) == -1 {
            self.min_covered_edge(v)
        } else {
            None
        }
    }

    /// `FindBridge(u, v)`: a bridge on the `u`-`v` path, if one exists.
    pub fn find_bridge_between(&mut self, u: usize, v: usize) -> Option<Edge> {
        if self.cover_level_between(u, v) == -1 {
            self.min_covered_edge_between(u, v)
        } else {
            None
        }
    }

    /// The vertices on the tree path from `u` to `v`, in order, or `None` if
    /// the two are not connected.
    fn path_vertices(&self, u: usize, v: usize) -> Option<Vec<usize>> {
        if u == v {
            return Some(vec![u]);
        }
        let mut prev: BTreeMap<usize, usize> = BTreeMap::new();
        let mut seen = BTreeSet::from([u]);
        let mut queue = VecDeque::from([u]);
        while let Some(x) = queue.pop_front() {
            if x == v {
                let mut path = vec![v];
                let mut cur = v;
                while cur != u {
                    cur = prev[&cur];
                    path.push(cur);
                }
                path.reverse();
                return Some(path);
            }
            for &y in &self.adj[x] {
                if seen.insert(y) {
                    prev.insert(y, x);
                    queue.push_back(y);
                }
            }
        }
        None
    }

    /// For every vertex in the component of the vertices of `path`, its
    /// projection (nearest vertex) onto `path`.
    fn projections(&self, path: &[usize]) -> BTreeMap<usize, usize> {
        let mut projection: BTreeMap<usize, usize> = BTreeMap::new();
        let mut queue = VecDeque::new();
        for &m in path {
            projection.insert(m, m);
            queue.push_back(m);
        }
        while let Some(x) = queue.pop_front() {
            let px = projection[&x];
            for &y in &self.adj[x] {
                if let std::collections::btree_map::Entry::Vacant(slot) = projection.entry(y) {
                    slot.insert(px);
                    queue.push_back(y);
                }
            }
        }
        projection
    }

    /// `FindSize(v, w, i)`: the number of vertices `u` with
    /// `CoverLevel(u, meet(u, v, w)) >= i`.
    ///
    /// In particular `find_size(v, v, -1)` is the size of `v`'s component.
    pub fn find_size(&mut self, v: usize, w: usize, i: i32) -> u64 {
        if !self.connected(v, w) {
            return 0;
        }
        let Some(path) = self.path_vertices(v, w) else {
            return 0;
        };
        let projection = self.projections(&path);
        let mut count = 0;
        for (&u, &m) in &projection {
            if self.cover_level_between(u, m) >= i {
                count += 1;
            }
        }
        count
    }

    /// `AddLabel(v, i)`: attaches a user label at `v` with level `i`.
    pub fn add_label(&mut self, v: usize, level: i32) -> LabelId {
        let id = LabelId(self.next_label);
        self.next_label += 1;
        self.labels.insert(id, (v, level));
        id
    }

    /// `RemoveLabel(l)`: removes a user label.
    pub fn remove_label(&mut self, label: LabelId) -> Option<(usize, i32)> {
        self.labels.remove(&label)
    }

    /// `FindFirstLabel(v, w, i)`: a label at level `i` whose vertex `u`
    /// satisfies `CoverLevel(u, meet(u, v, w)) >= i`, minimizing the distance
    /// from `v` to `meet(u, v, w)`.
    pub fn find_first_label(&mut self, v: usize, w: usize, i: i32) -> Option<LabelId> {
        if !self.connected(v, w) {
            return None;
        }
        let path = self.path_vertices(v, w)?;
        let projection = self.projections(&path);
        for &m in &path {
            let candidates: Vec<LabelId> = self
                .labels
                .iter()
                .filter(|&(_, &(u, level))| level == i && projection.get(&u) == Some(&m))
                .map(|(&id, _)| id)
                .collect();
            for id in candidates {
                let (u, _) = self.labels[&id];
                if self.cover_level_between(u, m) >= i {
                    return Some(id);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
