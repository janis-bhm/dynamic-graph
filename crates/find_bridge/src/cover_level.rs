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
//! [`FindBridge::find_size`] uses the FindSize vectors from Section 5 as part
//! of each top-tree summary. [`FindBridge::find_first_label`] remains a direct
//! search over the forest and user labels.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use top_tree::{MergeContext, Summary};

/// Sentinel used for "there is no such edge" and for the cover level of a
/// point cluster. It is strictly larger than every real cover level.
pub const NO_COVER: i32 = i32::MAX;

/// Highest cover level represented explicitly by the dense FindSize vectors.
const LEVEL_CAP: i32 = 32;
/// Slots for levels -1 through [`LEVEL_CAP`], inclusive.
const SLOTS: usize = LEVEL_CAP as usize + 2;

type SizeVector = Vec<u64>;
/// A dense map from cover-level keys -1..=`LEVEL_CAP` to size vectors.
type PartTree = Vec<SizeVector>;

/// Clamps a cover level to the FindSize key/level domain. The no-cover
/// sentinel is above every real level and therefore shares the cap slot.
fn find_size_level(level: i32) -> i32 {
    if level == NO_COVER {
        LEVEL_CAP
    } else {
        level.clamp(-1, LEVEL_CAP)
    }
}

fn level_slot(level: i32) -> usize {
    (find_size_level(level) + 1) as usize
}

fn zero_vector() -> SizeVector {
    vec![0; SLOTS]
}

fn zero_tree() -> PartTree {
    (0..SLOTS).map(|_| zero_vector()).collect()
}

fn single_key_tree(key: i32, value: &[u64]) -> PartTree {
    let mut tree = zero_tree();
    add_at(&mut tree, key, value);
    tree
}

/// Sum every key's vector in a dense part tree.
fn total_sum(tree: &[SizeVector]) -> SizeVector {
    let mut sum = zero_vector();
    for value in tree {
        add_vec(&mut sum, value);
    }
    sum
}

fn key_range(klo: i32, khi: i32) -> Option<std::ops::RangeInclusive<usize>> {
    if khi < -1 || klo > LEVEL_CAP {
        return None;
    }
    let low = level_slot(klo);
    let high = level_slot(khi);
    (low <= high).then_some(low..=high)
}

/// Sum the vectors whose keys lie in the inclusive range `[klo, khi]`.
fn range_sum(tree: &[SizeVector], klo: i32, khi: i32) -> SizeVector {
    let mut sum = zero_vector();
    if let Some(range) = key_range(klo, khi) {
        for key in range {
            if let Some(value) = tree.get(key) {
                add_vec(&mut sum, value);
            }
        }
    }
    sum
}

/// Keep only keys in the inclusive range `[klo, khi]`.
fn restrict(tree: &[SizeVector], klo: i32, khi: i32) -> PartTree {
    let mut restricted = zero_tree();
    if let Some(range) = key_range(klo, khi) {
        for key in range {
            if let (Some(dst), Some(src)) = (restricted.get_mut(key), tree.get(key)) {
                dst.clone_from(src);
            }
        }
    }
    restricted
}

fn add_at(tree: &mut [SizeVector], key: i32, value: &[u64]) {
    if let Some(slot) = tree.get_mut(level_slot(key)) {
        add_vec(slot, value);
    }
}

fn add_vec(dst: &mut [u64], src: &[u64]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
}

/// Apply the diagonal mask `M(key)`: keep level slots `j` with `j <= key`.
fn m_apply(key: i32, value: &[u64]) -> SizeVector {
    let key = find_size_level(key);
    value
        .iter()
        .enumerate()
        .map(|(slot, &entry)| {
            let level = slot as i32 - 1;
            if level <= key { entry } else { 0 }
        })
        .collect()
}

fn diagonal_sum(tree: &[SizeVector], klo: i32, khi: i32) -> SizeVector {
    let mut sum = zero_vector();
    if let Some(range) = key_range(klo, khi) {
        for key_slot in range {
            if let Some(value) = tree.get(key_slot) {
                let key = key_slot as i32 - 1;
                add_vec(&mut sum, &m_apply(key, value));
            }
        }
    }
    sum
}

/// Materialize the effect of a child's pending cover tag on the keys of its
/// boundary part tree. The unmodified (raw) tree remains stored in the child.
fn clean_tree(tree: &[SizeVector], pending: CoverTag) -> PartTree {
    let threshold = find_size_level(pending.threshold);
    let constant = find_size_level(pending.constant);
    let low = threshold.max(constant);
    // Only the prefix at keys <= low is collapsed to the pending constant;
    // higher keys retain their original cover value.
    let low_keys = restrict(tree, -1, low);
    let sum = total_sum(&low_keys);
    let mut cleaned = restrict(tree, low + 1, LEVEL_CAP);
    add_at(&mut cleaned, constant, &sum);
    cleaned
}

fn boundary_contains(boundary: top_tree::Boundary, vertex: usize) -> bool {
    boundary.slots().contains(&Some(vertex))
}

/// Select the child whose path/boundary owns a parent's outer boundary. At a
/// central endpoint of a rake, prefer the point child; this is the `x = c`
/// case in FS.Merge.
fn endpoint_child(
    left: top_tree::Boundary,
    right: top_tree::Boundary,
    vertex: usize,
    central: usize,
) -> usize {
    let in_left = boundary_contains(left, vertex);
    let in_right = boundary_contains(right, vertex);
    if vertex == central {
        if in_left && left.count() == 1 && !(in_right && right.count() == 1) {
            return 0;
        }
        if in_right && right.count() == 1 {
            return 1;
        }
    }
    match (in_left, in_right) {
        (true, false) => 0,
        (false, true) => 1,
        (true, true) => 0,
        (false, false) => panic!(
            "parent boundary {vertex} must belong to a child (left: {left:?}, right: {right:?})"
        ),
    }
}

/// Get the per-boundary tree for `vertex`. A single-boundary cluster's tree
/// can be in either slot after a lazy orientation flip, so use its populated
/// slot rather than assuming it is still slot zero.
fn boundary_tree<'a>(
    parts: &'a [PartTree; 2],
    boundary: top_tree::Boundary,
    vertex: usize,
) -> &'a PartTree {
    let slot = match boundary {
        top_tree::Boundary::Two { left, .. } if vertex == left => 0,
        top_tree::Boundary::Two { left: _, right } if vertex == right => 1,
        top_tree::Boundary::One(v) if vertex == v => {
            if parts[0].is_empty() {
                1
            } else {
                0
            }
        }
        top_tree::Boundary::None => {
            if parts[0].is_empty() {
                1
            } else {
                0
            }
        }
        _ => panic!("vertex {vertex} is not a boundary of {boundary:?}"),
    };
    &parts[slot]
}

fn set_boundary_tree(
    parts: &mut [PartTree; 2],
    boundary: top_tree::Boundary,
    vertex: usize,
    tree: PartTree,
) {
    match boundary {
        top_tree::Boundary::Two { left, .. } if vertex == left => parts[0] = tree,
        top_tree::Boundary::Two { right, .. } if vertex == right => parts[1] = tree,
        top_tree::Boundary::One(v) if vertex == v => {
            parts[0] = tree;
            parts[1] = Vec::new();
        }
        top_tree::Boundary::None => parts[0] = tree,
        _ => panic!("vertex {vertex} is not a boundary of {boundary:?}"),
    }
}

fn along_path_find_size(
    left: &CoverLevel,
    right: &CoverLevel,
    ctx: &MergeContext,
) -> (SizeVector, u64, [PartTree; 2]) {
    let mut size = left.size.clone();
    add_vec(&mut size, &right.size);

    let mut parts = [Vec::new(), Vec::new()];
    let mut children_parts = [left.part.clone(), right.part.clone()];
    let child_boundaries = [ctx.left_vertices, ctx.right_vertices];
    let child_summaries = [left, right];

    let (outer, path_vertices) = match ctx.parent_vertices {
        top_tree::Boundary::Two {
            left: left_vertex,
            right: right_vertex,
        } => {
            let path_vertices = match (ctx.left_boundary == 2, ctx.right_boundary == 2) {
                // The two child paths meet at `ctx.central`.
                (true, true) => left.path_vertices + right.path_vertices - 1,
                // A point child is raked onto the path at an existing vertex;
                // it does not extend the parent's cluster path.
                (true, false) => left.path_vertices,
                (false, true) => right.path_vertices,
                (false, false) => unreachable!("a two-boundary parent needs a path child"),
            };
            ([left_vertex, right_vertex], path_vertices)
        }
        // This is unreachable for the current caller (a parent is along-path
        // exactly when it has two boundaries), but mirrors FS.Merge's
        // degenerate `a = b = c` case.
        _ => (
            [ctx.central, ctx.central],
            left.path_vertices + right.path_vertices,
        ),
    };
    let outer_children = outer
        .map(|vertex| endpoint_child(ctx.left_vertices, ctx.right_vertices, vertex, ctx.central));

    // Step 1 of FS.Merge: clean each outer-boundary tree and its central tree
    // when that outer boundary is different from the central vertex.
    for (&vertex, &child) in outer.iter().zip(&outer_children) {
        if vertex != ctx.central {
            for boundary_vertex in [vertex, ctx.central] {
                let raw = boundary_tree(
                    &children_parts[child],
                    child_boundaries[child],
                    boundary_vertex,
                );
                let cleaned = clean_tree(raw, child_summaries[child].pending);
                set_boundary_tree(
                    &mut children_parts[child],
                    child_boundaries[child],
                    boundary_vertex,
                    cleaned,
                );
            }
        }
    }

    // Step 2 of FS.Merge: build the parent tree at each outer boundary.
    for endpoint_slot in 0..2 {
        let x = outer[endpoint_slot];
        let x_child = outer_children[endpoint_slot];
        let y = outer[1 - endpoint_slot];
        let y_child = 1 - x_child;
        let x_tree = boundary_tree(&children_parts[x_child], child_boundaries[x_child], x);
        let y_tree = boundary_tree(
            &children_parts[y_child],
            child_boundaries[y_child],
            ctx.central,
        );
        let cover_x = find_size_level(child_summaries[x_child].cover);
        let suffix = range_sum(y_tree, cover_x, LEVEL_CAP);
        let at_cover = x_tree[level_slot(cover_x)].clone();
        let mut part_at_cover = at_cover.clone();
        add_vec(&mut part_at_cover, &suffix);
        let mut diag_at_cover = m_apply(cover_x, &at_cover);
        add_vec(&mut diag_at_cover, &m_apply(cover_x, &suffix));

        let mut result = if x == ctx.central {
            // The `part_at_cover` entry below already includes the point
            // child's contribution at key LEVEL_CAP. Starting from an empty
            // tree here avoids counting that point twice when NO_COVER is
            // represented by the cap key.
            zero_tree()
        } else {
            restrict(x_tree, cover_x + 1, LEVEL_CAP)
        };
        if y != ctx.central {
            let prefix = restrict(y_tree, -1, cover_x - 1);
            for (dst, src) in result.iter_mut().zip(prefix) {
                add_vec(dst, &src);
            }
        }
        add_at(&mut result, cover_x, &part_at_cover);
        // `diag_at_cover` is represented implicitly as M(key) * partsize.
        // Keep the calculation explicit here to follow the paper's recurrence
        // and to make its invariant easy to audit.
        debug_assert_eq!(diag_at_cover, m_apply(cover_x, &part_at_cover));
        set_boundary_tree(&mut parts, ctx.parent_vertices, x, result);
    }

    (size, path_vertices, parts)
}

fn off_path_find_size(
    left: &CoverLevel,
    right: &CoverLevel,
    ctx: &MergeContext,
) -> (SizeVector, [PartTree; 2]) {
    let mut a = match ctx.parent_vertices {
        top_tree::Boundary::One(vertex) if vertex != ctx.central => Some(vertex),
        top_tree::Boundary::Two { left, right } => {
            // Defensive only: the caller classifies two-boundary parents as
            // along-path.
            Some(if left != ctx.central { left } else { right })
        }
        _ => None,
    };

    let mut a_child = if let Some(vertex) = a {
        endpoint_child(ctx.left_vertices, ctx.right_vertices, vertex, ctx.central)
    } else if ctx.left_boundary == 2 && ctx.right_boundary != 2 {
        0
    } else if ctx.right_boundary == 2 && ctx.left_boundary != 2 {
        1
    } else {
        0
    };

    if a.is_none() {
        a = Some(ctx.central);
    }
    let a = a.expect("off-path merge must have a representative boundary");
    if !boundary_contains(ctx.left_vertices, a) {
        a_child = 1;
    } else if !boundary_contains(ctx.right_vertices, a) {
        a_child = 0;
    }
    let b_child = 1 - a_child;
    let summaries = [left, right];
    let boundaries = [ctx.left_vertices, ctx.right_vertices];
    let a_tree = boundary_tree(&summaries[a_child].part, boundaries[a_child], a);

    let low = find_size_level(
        summaries[a_child]
            .pending
            .threshold
            .max(summaries[a_child].pending.constant),
    );
    let diagonal = diagonal_sum(a_tree, low + 1, LEVEL_CAP);
    let prefix = range_sum(a_tree, -1, low);
    let mut size = diagonal;
    add_vec(
        &mut size,
        &m_apply(summaries[a_child].pending.constant, &prefix),
    );
    add_vec(
        &mut size,
        &m_apply(summaries[a_child].cover, &summaries[b_child].size),
    );

    let mut parts = [Vec::new(), Vec::new()];
    set_boundary_tree(
        &mut parts,
        ctx.parent_vertices,
        a,
        single_key_tree(LEVEL_CAP, &size),
    );
    (size, parts)
}

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverLevel {
    /// Minimum cover level on the cluster path, or [`NO_COVER`].
    pub cover: i32,
    /// Minimum cover level off the cluster path, or [`NO_COVER`].
    pub global_cover: i32,
    /// An edge of the cluster path attaining `cover`.
    pub min_path_edge: Option<Edge>,
    /// An edge off the cluster path attaining `global_cover`.
    pub min_global_edge: Option<Edge>,
    /// The composed lazy cover operation. The vectors below remain raw while
    /// this operation is pending.
    pending: CoverTag,
    /// `size_C[i]` at slot `i + 1`, for levels -1 through LEVEL_CAP.
    size: SizeVector,
    /// The number of labels on the cluster path, used above LEVEL_CAP.
    path_vertices: u64,
    /// Boundary-indexed partsize trees; slot 0 is the first boundary and slot
    /// 1 is the second boundary.
    part: [PartTree; 2],
    leaf: LeafKind,
}

/// Picks the smaller of two `(level, edge)` candidates, preferring the first
/// on ties.
fn min_candidate(left: (i32, Option<Edge>), right: (i32, Option<Edge>)) -> (i32, Option<Edge>) {
    if right.0 < left.0 { right } else { left }
}

impl Summary<()> for CoverLevel {
    type Tag = CoverTag;
    type LabelKey = usize;

    fn tree_edge(_weight: &(), u: usize, v: usize) -> Self {
        // A freshly linked tree edge is a bridge: its cover level is -1. The
        // same value is stored as the global cover so that an edge leaf can
        // serve as the root of an `expose`; whether the edge is on or off the
        // cluster path is decided by the parent (or by the root query).
        let zero_tree = single_key_tree(LEVEL_CAP, &zero_vector());
        CoverLevel {
            cover: -1,
            global_cover: -1,
            min_path_edge: Some((u, v)),
            min_global_edge: Some((u, v)),
            pending: CoverTag::identity(),
            size: zero_vector(),
            // An edge path contains both endpoint vertices.
            path_vertices: 2,
            part: [zero_tree.clone(), zero_tree],
            leaf: LeafKind::Edge,
        }
    }

    fn label(_key: &usize, _v: usize) -> Self {
        let size = vec![1; SLOTS];
        let tree = single_key_tree(LEVEL_CAP, &size);
        CoverLevel {
            cover: NO_COVER,
            global_cover: NO_COVER,
            min_path_edge: None,
            min_global_edge: None,
            pending: CoverTag::identity(),
            size: size.clone(),
            path_vertices: 1,
            part: [tree, Vec::new()],
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

        let (size, path_vertices, part) = if parent_is_path {
            along_path_find_size(left, right, ctx)
        } else {
            let (size, part) = off_path_find_size(left, right, ctx);
            (size, ctx.parent_vertices.count() as u64, part)
        };
        CoverLevel {
            cover,
            global_cover,
            min_path_edge,
            min_global_edge,
            pending: CoverTag::identity(),
            size,
            path_vertices,
            part,
            leaf: LeafKind::Internal,
        }
    }

    fn flip(&mut self) {
        self.part.swap(0, 1);
    }

    fn apply(&mut self, tag: &Self::Tag) {
        self.cover = tag.apply_to(self.cover);
        if self.leaf == LeafKind::Edge {
            self.global_cover = tag.apply_to(self.global_cover);
        }
        self.pending = tag.after(self.pending);
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
    top_tree: top_tree::TopTree<usize, usize, CoverLevel>,
    /// Adjacency of the forest, used by the brute-force FindFirstLabel query.
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
        if let Some(index) = self.top_tree.vertex_index(&key) {
            return index;
        }
        let index = self.top_tree.add_vertex(key);
        while self.adj.len() <= index {
            self.adj.push(BTreeSet::new());
        }
        // One internal label per vertex makes every vertex contribute exactly
        // once to FindSize and supplies the point cluster for that vertex.
        self.top_tree.attach(index, index);
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
        self.top_tree.link(u, v);
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
        let summary = self.top_tree.expose_path(v, w);
        self.top_tree.deexpose(w);
        self.top_tree.deexpose(v);
        match summary {
            Some(summary) if i <= -1 => summary.size[0],
            Some(summary) if i <= LEVEL_CAP => summary.size[(i + 1) as usize],
            Some(_) if i > LEVEL_CAP && v == w => {
                // Only v itself has the sentinel cover level; all real edge covers are capped.
                1
            }
            Some(summary) => summary.path_vertices,
            None if v == w => 1,
            None => 0,
        }
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
