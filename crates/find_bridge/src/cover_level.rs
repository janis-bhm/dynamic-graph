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
//! of each top-tree summary. [`FindBridge::find_first_label`] descends the
//! exposed path cluster using the per-cluster incident masks from Section 6.

use std::collections::{BTreeMap, BTreeSet};

use augmented_tree::{Aggregate, BTree};
use top_tree::{MergeContext, Summary};

/// Sentinel used for "there is no such edge" and for the cover level of a
/// point cluster. It is strictly larger than every real cover level.
pub const NO_COVER: i32 = i32::MAX;

/// Highest cover level represented explicitly by the dense FindSize vectors.
/// This represents a node-count of 2^LEVEL_CAP, which is more than enough for any practical graph.
const LEVEL_CAP: i32 = 32;
/// Slots for levels -1 through [`LEVEL_CAP`], inclusive.
const SLOTS: usize = LEVEL_CAP as usize + 2;

/// A cover level in the domain represented by the dense vectors/masks,
/// `-1..=LEVEL_CAP`. Construction is fallible; it never clamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Level(i32);

impl Level {
    pub const MIN: Self = Self(-1);
    pub const MAX: Self = Self(LEVEL_CAP);

    pub const fn new(value: i32) -> Option<Self> {
        if value >= -1 && value <= LEVEL_CAP {
            Some(Self(value))
        } else {
            None
        }
    }
}

impl TryFrom<i32> for Level {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, ()> {
        Level::new(v).ok_or(())
    }
}

impl From<Level> for i32 {
    fn from(l: Level) -> i32 {
        l.0
    }
}

/// A size vector over slots `0..SLOTS` (slot = level + 1). Trailing zero slots
/// are implicit: the backing slice ends at the last nonzero slot and `get`
/// returns 0 beyond it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SizeVector(Box<[u64]>);

impl SizeVector {
    /// Canonical form: drop trailing zeros.
    fn from_vec(mut data: Vec<u64>) -> Self {
        while data.last() == Some(&0) {
            data.pop();
        }
        SizeVector(data.into_boxed_slice())
    }

    fn empty() -> Self {
        Self(Box::new([]))
    }

    fn get(&self, slot: usize) -> u64 {
        self.0.get(slot).copied().unwrap_or(0)
    }

    fn add(&self, other: &Self) -> Self {
        let new_len = self.0.len().max(other.0.len());
        let mut out = Box::new_uninit_slice(new_len);
        for slot in 0..new_len {
            out[slot].write(self.get(slot) + other.get(slot));
        }
        Self(unsafe { out.assume_init() })
    }

    fn add_assign(&mut self, other: &Self) {
        if self.0.len() < other.0.len() {
            *self = self.add(other);
        } else {
            for slot in 0..other.0.len() {
                self.0[slot] += other.0[slot];
            }
        }
    }

    /// Keep slots `0..=key+1` (`m_apply`), then re-trim.
    fn masked(&self, key: i32) -> Self {
        let key = cover_key(key);
        let max_slots = (key + 2).clamp(0, SLOTS as i32) as usize;
        let n = self.0.len().min(max_slots);
        Self(Box::from(&self.0[..n]))
    }
}

/// One entry of a boundary part tree: the raw size vector stored at a cover
/// level, and the OR of its parts' incident level masks.
///
/// The diagonal `M(key) * raw` and its incident analogue are not stored: a
/// part tree has at most `SLOTS` distinct (clamped) cover-level keys, so those
/// aggregates are recomputed on demand by a bounded scan.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PartEntry {
    raw: SizeVector,
    inc: u64,
}

impl Aggregate for PartEntry {
    fn reduce(&self, other: &Self) -> Self {
        PartEntry {
            raw: add_vectors(&self.raw, &other.raw),
            inc: self.inc | other.inc,
        }
    }

    fn identity() -> Self {
        PartEntry {
            raw: zero_vector(),
            inc: 0,
        }
    }
}

/// A map from cover-level keys to size vectors, augmented with the sum of its
/// values so that whole-tree and range sums are cheap.
type PartTree = BTree<i32, PartEntry>;

/// Clamps a cover value to the FindSize key/level domain. The no-cover
/// sentinel is above every real level and therefore shares the cap slot.
fn cover_key(cover: i32) -> i32 {
    if cover == NO_COVER {
        LEVEL_CAP
    } else {
        cover.clamp(-1, LEVEL_CAP)
    }
}

fn cover_slot(cover: i32) -> u32 {
    (cover_key(cover) + 1) as u32
}

/// Slot index (0..=LEVEL_CAP+1) used for a user `level` in an incident mask.
fn level_slot(level: Level) -> u32 {
    (i32::from(level) + 1) as u32
}

/// A mask with the single bit for the valid user `level` set.
fn level_bit(level: Level) -> u64 {
    1u64 << level_slot(level)
}

/// Keep the incident bits for cover values `<= cover` (bitwise analogue of
/// `m_apply`). Cover values can include [`NO_COVER`] and are clamped to the
/// represented cover-key domain.
fn inc_m_apply(cover: i32, bits: u64) -> u64 {
    let slot = cover_slot(cover);
    if slot >= u64::BITS - 1 {
        bits
    } else {
        bits & ((1u64 << (slot + 1)) - 1)
    }
}

fn zero_vector() -> SizeVector {
    SizeVector::empty()
}

fn add_vectors(left: &SizeVector, right: &SizeVector) -> SizeVector {
    left.add(right)
}

fn zero_tree() -> PartTree {
    PartTree::new()
}

fn single_key_tree(key: i32, value: &SizeVector, inc: u64) -> PartTree {
    let mut tree = zero_tree();
    add_at(&mut tree, key, value, inc);
    tree
}

/// Sum every key's vector in a part tree.
fn total_sum(tree: &PartTree) -> SizeVector {
    tree.aggregate().raw.clone()
}

/// Sum the vectors whose keys lie in the inclusive range `[klo, khi]`.
fn range_sum(tree: &PartTree, klo: i32, khi: i32) -> SizeVector {
    tree.range_aggregate(klo..=khi).raw
}

/// Keep only keys in the inclusive range `[klo, khi]`.
fn restrict(tree: &PartTree, klo: i32, khi: i32) -> PartTree {
    let mut restricted = PartTree::new();
    for (key, entry) in tree.iter() {
        if *key >= klo && *key <= khi {
            restricted.insert(*key, entry.clone());
        }
    }
    restricted
}

fn add_at(tree: &mut PartTree, key: i32, value: &SizeVector, inc: u64) {
    let key = cover_key(key);
    tree.entry(key).update_or_insert_with(
        |entry| {
            entry.raw = add_vectors(&entry.raw, value);
            entry.inc |= inc;
        },
        || PartEntry {
            raw: value.clone(),
            inc,
        },
    );
}

fn add_vec(dst: &mut SizeVector, src: &SizeVector) {
    dst.add_assign(src);
}

/// Apply the diagonal mask `M(key)`: keep level slots `j` with `j <= key`.
fn m_apply(key: i32, value: &SizeVector) -> SizeVector {
    value.masked(key)
}

fn total_inc(tree: &PartTree) -> u64 {
    tree.aggregate().inc
}

fn total_inc_diag(tree: &PartTree) -> u64 {
    tree.iter()
        .fold(0, |acc, (key, entry)| acc | inc_m_apply(*key, entry.inc))
}

fn range_inc(tree: &PartTree, klo: i32, khi: i32) -> u64 {
    tree.range_aggregate(klo..=khi).inc
}

fn range_inc_diag(tree: &PartTree, klo: i32, khi: i32) -> u64 {
    tree.iter()
        .filter(|(key, _)| **key >= klo && **key <= khi)
        .fold(0, |acc, (key, entry)| acc | inc_m_apply(*key, entry.inc))
}

fn diagonal_sum(tree: &PartTree, klo: i32, khi: i32) -> SizeVector {
    let mut sum = zero_vector();
    for (key, entry) in tree.iter() {
        if *key >= klo && *key <= khi {
            sum.add_assign(&entry.raw.masked(*key));
        }
    }
    sum
}

/// Materialize the effect of a child's pending cover tag on the keys of its
/// boundary part tree. The unmodified (raw) tree remains stored in the child.
fn clean_tree(tree: &PartTree, pending: CoverTag) -> PartTree {
    let threshold = cover_key(pending.threshold);
    let constant = cover_key(pending.constant);
    let low = threshold.max(constant);
    // Only the prefix at keys <= low is collapsed to the pending constant;
    // higher keys retain their original cover value.
    let low_keys = restrict(tree, -1, low);
    let sum = total_sum(&low_keys);
    let inc = total_inc(&low_keys);
    let mut cleaned = restrict(tree, low + 1, LEVEL_CAP);
    add_at(&mut cleaned, constant, &sum, inc);
    cleaned
}

fn boundary_contains(boundary: top_tree::Boundary, vertex: top_tree::VertexId) -> bool {
    boundary.slots().contains(&Some(vertex))
}

/// Select the child whose path/boundary owns a parent's outer boundary. At a
/// central endpoint of a rake, prefer the point child; this is the `x = c`
/// case in FS.Merge.
fn endpoint_child(
    left: top_tree::Boundary,
    right: top_tree::Boundary,
    vertex: top_tree::VertexId,
    central: top_tree::VertexId,
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
/// The unused slot is empty, while every stored boundary tree must be
/// non-empty so this test continues to identify the unused slot.
fn boundary_tree(
    parts: &[PartTree; 2],
    boundary: top_tree::Boundary,
    vertex: top_tree::VertexId,
) -> &PartTree {
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
    vertex: top_tree::VertexId,
    tree: PartTree,
) {
    match boundary {
        top_tree::Boundary::Two { left, .. } if vertex == left => parts[0] = tree,
        top_tree::Boundary::Two { right, .. } if vertex == right => parts[1] = tree,
        top_tree::Boundary::One(v) if vertex == v => {
            parts[0] = tree;
            parts[1] = PartTree::new();
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
    size.add_assign(&right.size);

    let mut parts = [PartTree::new(), PartTree::new()];
    let mut children_parts = [left.part.clone(), right.part.clone()];
    let child_boundaries = [ctx.left_boundary, ctx.right_boundary];
    let child_summaries = [left, right];

    let (outer, path_vertices) = match ctx.boundary {
        top_tree::Boundary::Two {
            left: left_vertex,
            right: right_vertex,
        } => {
            let path_vertices = match (ctx.left_boundary.is_path(), ctx.right_boundary.is_path()) {
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
        .map(|vertex| endpoint_child(ctx.left_boundary, ctx.right_boundary, vertex, ctx.central));

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
        let cover_x = cover_key(child_summaries[x_child].cover);
        let suffix = range_sum(y_tree, cover_x, LEVEL_CAP);
        let suffix_inc = range_inc(y_tree, cover_x, LEVEL_CAP);
        let at_cover = x_tree
            .get(&cover_x)
            .map(|entry| entry.raw.clone())
            .unwrap_or_else(zero_vector);
        let at_cover_inc = x_tree.get(&cover_x).map(|entry| entry.inc).unwrap_or(0);
        let mut part_at_cover = at_cover.clone();
        part_at_cover.add_assign(&suffix);
        let part_at_cover_inc = at_cover_inc | suffix_inc;
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
            for (key, entry) in prefix.iter() {
                add_at(&mut result, *key, &entry.raw, entry.inc);
            }
        }
        add_at(&mut result, cover_x, &part_at_cover, part_at_cover_inc);
        // `diag_at_cover` is represented implicitly as M(key) * partsize.
        // Keep the calculation explicit here to follow the paper's recurrence
        // and to make its invariant easy to audit.
        debug_assert_eq!(diag_at_cover, m_apply(cover_x, &part_at_cover));
        set_boundary_tree(&mut parts, ctx.boundary, x, result);
    }

    (size, path_vertices, parts)
}

fn off_path_find_size(
    left: &CoverLevel,
    right: &CoverLevel,
    ctx: &MergeContext,
) -> (SizeVector, u64, [PartTree; 2]) {
    let mut a = match ctx.boundary {
        top_tree::Boundary::One(vertex) if vertex != ctx.central => Some(vertex),
        top_tree::Boundary::Two { left, right } => {
            // Defensive only: the caller classifies two-boundary parents as
            // along-path.
            Some(if left != ctx.central { left } else { right })
        }
        _ => None,
    };

    let mut a_child = if let Some(vertex) = a {
        endpoint_child(ctx.left_boundary, ctx.right_boundary, vertex, ctx.central)
    } else if ctx.left_boundary.is_path() && !ctx.right_boundary.is_path() {
        0
    } else if ctx.right_boundary.is_path() && !ctx.left_boundary.is_path() {
        1
    } else {
        0
    };

    if a.is_none() {
        a = Some(ctx.central);
    }
    let a = a.expect("off-path merge must have a representative boundary");
    if !boundary_contains(ctx.left_boundary, a) {
        a_child = 1;
    } else if !boundary_contains(ctx.right_boundary, a) {
        a_child = 0;
    }
    let b_child = 1 - a_child;
    let summaries = [left, right];
    let boundaries = [ctx.left_boundary, ctx.right_boundary];
    let a_tree = boundary_tree(&summaries[a_child].part, boundaries[a_child], a);

    let low = cover_key(
        summaries[a_child]
            .pending
            .threshold
            .max(summaries[a_child].pending.constant),
    );
    let diagonal = diagonal_sum(a_tree, low + 1, LEVEL_CAP);
    let prefix = range_sum(a_tree, -1, low);
    let mut size = diagonal;
    size.add_assign(&m_apply(summaries[a_child].pending.constant, &prefix));
    size.add_assign(&m_apply(summaries[a_child].cover, &summaries[b_child].size));

    let diag_inc = range_inc_diag(a_tree, low + 1, LEVEL_CAP);
    let prefix_inc = range_inc(a_tree, -1, low);
    let incident = diag_inc
        | inc_m_apply(summaries[a_child].pending.constant, prefix_inc)
        | inc_m_apply(summaries[a_child].cover, summaries[b_child].incident);

    let mut parts = [PartTree::new(), PartTree::new()];
    set_boundary_tree(
        &mut parts,
        ctx.boundary,
        a,
        single_key_tree(LEVEL_CAP, &size, incident),
    );
    (size, incident, parts)
}

/// A stable handle to a user label added with [`FindBridge::add_label`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UserLabel(pub usize);

/// An undirected edge, stored as the pair of endpoints passed to
/// [`FindBridge::link`].
pub type Edge = (top_tree::VertexId, top_tree::VertexId);

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
    pub fn cover(level: Level) -> Self {
        let level = i32::from(level);
        CoverTag {
            threshold: level,
            constant: level,
        }
    }

    /// The function `x -> if x <= level then -1 else x` implementing an
    /// `Uncover`.
    pub fn uncover(level: Level) -> Self {
        CoverTag {
            threshold: i32::from(level),
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
    /// User-label levels incident to this cluster's path or point boundary.
    pub incident: u64,
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

impl CoverLevel {
    /// Removes all user-levels in `levels` from the levels stored on this vertex label.
    fn remove_vertex_levels(&mut self, levels: u64) {
        let levels = self.incident & !levels;
        self.set_vertex_levels(levels);
    }

    /// Adds all user-levels in `levels` to the levels stored on this vertex label.
    fn add_vertex_levels(&mut self, levels: u64) {
        let levels = self.incident | levels;
        self.set_vertex_levels(levels);
    }

    /// Sets the bitmask of user-label levels stored on this vertex label.
    fn set_vertex_levels(&mut self, levels: u64) {
        // incident_C = ([v has assoc label at level j])_{j = 0..LEVEL_CAP}
        self.incident = levels;
        // Note that if \partial{C} = {v} then pointsize_{C,v} = size_C.
        let size = self.size.clone();
        // partsize_{C,v,i} = \sum_{m \in \pi(C) where CoverLevel(v,m)=i} pointsize_{C,m}
        self.part = [single_key_tree(LEVEL_CAP, &size, levels), PartTree::new()];
    }
}

/// Whether `summary.incident` has the bit for `level`.
fn incident_has(summary: &CoverLevel, level: Level) -> bool {
    summary.incident & level_bit(level) != 0
}

/// `pointincident_{C,v}[level]` computed on demand from the part tree and
/// pending tag.
fn pointincident_has(
    summary: &CoverLevel,
    boundary: top_tree::Boundary,
    vertex: top_tree::VertexId,
    level: Level,
) -> bool {
    let raw = boundary_tree(&summary.part, boundary, vertex);
    total_inc_diag(&clean_tree(raw, summary.pending)) & level_bit(level) != 0
}

/// Picks the smaller of two `(level, edge)` candidates, preferring the first
/// on ties.
fn min_candidate(left: (i32, Option<Edge>), right: (i32, Option<Edge>)) -> (i32, Option<Edge>) {
    if right.0 < left.0 { right } else { left }
}

impl Summary for CoverLevel {
    type Tag = CoverTag;

    fn tree_edge(u: top_tree::VertexId, v: top_tree::VertexId) -> Self {
        // A freshly linked tree edge is a bridge: its cover level is -1. The
        // same value is stored as the global cover so that an edge leaf can
        // serve as the root of an `expose`; whether the edge is on or off the
        // cluster path is decided by the parent (or by the root query).
        let zero_tree = single_key_tree(LEVEL_CAP, &zero_vector(), 0);
        CoverLevel {
            cover: -1,
            global_cover: -1,
            min_path_edge: Some((u, v)),
            min_global_edge: Some((u, v)),
            incident: 0,
            pending: CoverTag::identity(),
            size: zero_vector(),
            // An edge path contains both endpoint vertices.
            path_vertices: 2,
            part: [zero_tree.clone(), zero_tree],
            leaf: LeafKind::Edge,
        }
    }

    fn label(_v: top_tree::VertexId) -> Self {
        let size = SizeVector::from_vec(vec![1; SLOTS]);
        let tree = single_key_tree(LEVEL_CAP, &size, 0);
        CoverLevel {
            cover: NO_COVER,
            global_cover: NO_COVER,
            min_path_edge: None,
            min_global_edge: None,
            incident: 0,
            pending: CoverTag::identity(),
            size: size.clone(),
            path_vertices: 1,
            part: [tree, PartTree::new()],
            leaf: LeafKind::Label,
        }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        let parent_is_path = ctx.boundary.is_path();
        let left_is_path = ctx.left_boundary.is_path();
        let right_is_path = ctx.right_boundary.is_path();

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

        let (size, path_vertices, incident, part) = if parent_is_path {
            let (size, path_vertices, part) = along_path_find_size(left, right, ctx);

            (size, path_vertices, left.incident | right.incident, part)
        } else {
            let (size, incident, part) = off_path_find_size(left, right, ctx);

            (size, ctx.boundary.count() as u64, incident, part)
        };

        CoverLevel {
            cover,
            global_cover,
            min_path_edge,
            min_global_edge,
            incident,
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

    fn remap_vertex(&mut self, old: top_tree::VertexId, new: top_tree::VertexId) {
        let remap = |(u, v): Edge| {
            (
                if u == old { new } else { u },
                if v == old { new } else { v },
            )
        };
        self.min_path_edge = self.min_path_edge.map(remap);
        self.min_global_edge = self.min_global_edge.map(remap);
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
    top_tree: top_tree::TopTree<CoverLevel>,
    /// The user labels of the FindFirstLabel structure, keyed by handle.
    labels: BTreeMap<UserLabel, (top_tree::VertexId, Level)>,
    /// Live label ids grouped by their (vertex, level), ordered by id.
    labels_at: BTreeMap<(top_tree::VertexId, Level), BTreeSet<UserLabel>>,
    /// The next label handle to allocate.
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
            labels: BTreeMap::new(),
            labels_at: BTreeMap::new(),
            next_label: 0,
        }
    }

    /// Adds a vertex and returns its internal forest handle.
    pub fn add_vertex(&mut self) -> top_tree::VertexId {
        let index = self.top_tree.add_vertex();

        // One label per vertex makes every vertex contribute exactly once to
        // FindSize and supplies the point cluster for that vertex.
        self.top_tree.attach(index);
        index
    }

    /// Removes an isolated-in-the-forest vertex, its structural label leaf,
    /// and all user labels attached to it.
    ///
    /// If the underlying vertex storage compacts another vertex into this
    /// vertex's slot, the returned [`top_tree::SwapResult`] describes that
    /// handle remapping; callers holding the moved handle should apply
    /// [`top_tree::VertexId::swap`]. Returns `None` for a stale vertex handle.
    ///
    /// # Panics
    ///
    /// Panics if a live vertex has any incident forest edge. Graph-level
    /// callers should first delete its incident graph edges.
    pub fn remove_vertex(
        &mut self,
        vertex: top_tree::VertexId,
    ) -> Option<top_tree::SwapResult<top_tree::VertexId>> {
        self.top_tree.forest().node(vertex)?;

        assert_eq!(
            self.top_tree.forest().incident_edge_indices(vertex).count(),
            0,
            "FindBridge::remove_vertex requires no incident forest edges"
        );

        let user_labels = self
            .labels_at
            .extract_if((vertex, Level::MIN)..=(vertex, Level::MAX), |_, _| true)
            .flat_map(|((_, _), labels)| labels)
            .collect::<Vec<_>>();

        for label in user_labels {
            self.labels.remove(&label);
        }

        let swap = self
            .top_tree
            .remove_vertex(vertex)
            .expect("the live vertex was validated before removal");

        if let top_tree::SwapResult::Some { prev, current } = swap {
            debug_assert_eq!(current.index(), vertex.index());

            // remove range of labels_at for the moved vertex, then reinsert
            // them under the new vertex handle
            let prev_labels = self
                .labels_at
                .extract_if((prev, Level::MIN)..=(prev, Level::MAX), |_, _| true)
                .collect::<Vec<_>>();

            for ((_, level), labels) in prev_labels {
                for label in &labels {
                    self.labels.get_mut(label).expect("asdf").0 = current;
                }
                assert!(self.labels_at.insert((current, level), labels).is_none());
            }
        }

        Some(swap)
    }

    /// The number of tree edges.
    pub fn edge_count(&self) -> usize {
        self.top_tree.edge_count()
    }

    /// Links two vertices that are in different trees with a new tree edge.
    pub fn link(&mut self, u: top_tree::VertexId, v: top_tree::VertexId) {
        self.top_tree.link(u, v);
    }

    /// Cuts the tree edge between `u` and `v`, returning whether it existed.
    pub fn cut(&mut self, u: top_tree::VertexId, v: top_tree::VertexId) -> bool {
        self.top_tree.cut(u, v).is_some()
    }

    /// Returns whether `u` and `v` are in the same tree.
    pub fn connected(&mut self, u: top_tree::VertexId, v: top_tree::VertexId) -> bool {
        self.top_tree.connected(u, v)
    }

    #[cfg(test)]
    fn debug_root_incident(&mut self, v: top_tree::VertexId, w: top_tree::VertexId) -> u64 {
        let summary = self.top_tree.expose_path(v, w);
        self.top_tree.deexpose(w);
        self.top_tree.deexpose(v);
        summary.map_or(0, |s| s.incident)
    }

    /// Applies `Cover(u, v, level)`: every edge on the `u`-`v` path whose
    /// cover level is below `level` is raised to `level`.
    pub fn cover(&mut self, u: top_tree::VertexId, v: top_tree::VertexId, level: Level) {
        self.with_path_tag(u, v, CoverTag::cover(level));
    }

    /// Applies `Uncover(u, v, level)`: every edge on the `u`-`v` path whose
    /// cover level is at most `level` gets cover level `-1`.
    pub fn uncover(&mut self, u: top_tree::VertexId, v: top_tree::VertexId, level: Level) {
        self.with_path_tag(u, v, CoverTag::uncover(level));
    }

    fn with_path_tag(&mut self, u: top_tree::VertexId, v: top_tree::VertexId, tag: CoverTag) {
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
    pub fn cover_level(&mut self, v: top_tree::VertexId) -> i32 {
        let summary = self.top_tree.expose(v);
        self.top_tree.deexpose(v);
        summary.map_or(NO_COVER, |node| node.global_cover)
    }

    /// `MinCoveredEdge(v)`: an edge attaining [`FindBridge::cover_level`].
    pub fn min_covered_edge(&mut self, v: top_tree::VertexId) -> Option<Edge> {
        let summary = self.top_tree.expose(v);
        self.top_tree.deexpose(v);
        summary.and_then(|node| node.min_global_edge)
    }

    /// `CoverLevel(u, v)`: the minimum cover level on the `u`-`v` path. If
    /// `u == v` (or the two are not connected and there is no path) this is
    /// [`NO_COVER`].
    pub fn cover_level_between(&mut self, u: top_tree::VertexId, v: top_tree::VertexId) -> i32 {
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
    pub fn min_covered_edge_between(
        &mut self,
        u: top_tree::VertexId,
        v: top_tree::VertexId,
    ) -> Option<Edge> {
        if u == v || !self.connected(u, v) {
            return None;
        }
        let summary = self.top_tree.expose_path(u, v);
        self.top_tree.deexpose(v);
        self.top_tree.deexpose(u);
        summary.and_then(|node| node.min_path_edge)
    }

    /// `FindBridge(v)`: a bridge in `v`'s tree, if one exists.
    pub fn find_bridge(&mut self, v: top_tree::VertexId) -> Option<Edge> {
        if self.cover_level(v) == -1 {
            self.min_covered_edge(v)
        } else {
            None
        }
    }

    /// `FindBridge(u, v)`: a bridge on the `u`-`v` path, if one exists.
    pub fn find_bridge_between(
        &mut self,
        u: top_tree::VertexId,
        v: top_tree::VertexId,
    ) -> Option<Edge> {
        if self.cover_level_between(u, v) == -1 {
            self.min_covered_edge_between(u, v)
        } else {
            None
        }
    }

    /// `FindSize(v, w, i)`: the number of vertices `u` with
    /// `CoverLevel(u, meet(u, v, w)) >= i`.
    ///
    /// In particular `find_size(v, v, -1)` is the size of `v`'s component.
    pub fn find_size(&mut self, v: top_tree::VertexId, w: top_tree::VertexId, i: i32) -> u64 {
        if !self.connected(v, w) {
            return 0;
        }
        let summary = self.top_tree.expose_path(v, w);
        self.top_tree.deexpose(w);
        self.top_tree.deexpose(v);
        match summary {
            Some(summary) if i <= -1 => summary.size.get(0),
            Some(summary) if i <= LEVEL_CAP => summary.size.get((i + 1) as usize),
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
    pub fn add_label(&mut self, v: top_tree::VertexId, level: Level) -> UserLabel {
        let id = UserLabel(self.next_label);
        self.next_label += 1;
        self.labels.insert(id, (v, level));

        let first_at_level = {
            // labels are stored in a set addressed by (vertex, level).
            let labels = self.labels_at.entry((v, level)).or_default();
            let first_at_level = labels.is_empty();
            labels.insert(id);
            first_at_level
        };

        if first_at_level {
            // if our label is the first for this vertex at this level we have
            // to update the vertex's summary to reflect the new incident level.
            let label = self.top_tree.first_label(v).unwrap();
            let label = self.top_tree.node_label_key(label).unwrap();

            // updates incident_C and the part tree for this vertex's label cluster.
            self.top_tree
                .update_label_summary(label, |sum| sum.add_vertex_levels(level_bit(level)));
        }
        id
    }

    /// `RemoveLabel(l)`: removes a user label.
    pub fn remove_label(&mut self, label: UserLabel) -> Option<(top_tree::VertexId, Level)> {
        let (v, level) = self.labels.remove(&label)?;

        let last_at_level = {
            let labels = self
                .labels_at
                .get_mut(&(v, level))
                .expect("live label must have a (vertex, level) index bucket");
            assert!(labels.remove(&label));
            labels.is_empty()
        };

        if last_at_level {
            let label = self.top_tree.first_label(v).unwrap();
            let label = self.top_tree.node_label_key(label).unwrap();

            self.top_tree
                .update_label_summary(label, |sum| sum.remove_vertex_levels(level_bit(level)));
        }
        Some((v, level))
    }

    /// `FindFirstLabel(v, w, i)`.
    ///
    /// Returns a label at level `i` whose vertex `u` satisfies
    /// `CoverLevel(u, meet(u, v, w)) >= i`, minimizing the distance from `v`
    /// to `meet(u, v, w)`. Correctness is guaranteed for levels in
    /// `-1..=LEVEL_CAP`, the range represented by the incident masks.
    pub fn find_first_label(
        &mut self,
        v: top_tree::VertexId,
        w: top_tree::VertexId,
        level: Level,
    ) -> Option<UserLabel> {
        if !self.connected(v, w) {
            return None;
        }
        let root = self.top_tree.expose_path_node(v, w)?;
        let vertex = if v == w {
            self.find_label_vertex(root, v, level)
        } else {
            self.first_path(root, v, level)
        };
        self.top_tree.deexpose(w);
        self.top_tree.deexpose(v);
        vertex.and_then(|u| self.smallest_label_at(u, level))
    }

    /// The smallest live label id at `vertex` with the given exact level.
    fn smallest_label_at(&self, vertex: top_tree::VertexId, level: Level) -> Option<UserLabel> {
        self.labels_at
            .get(&(vertex, level))
            .and_then(|labels| labels.iter().next().copied())
    }

    /// Descends a path cluster to the valid level-`i` label whose projection
    /// onto the cluster path is closest to `near` (`near` is a boundary of
    /// `node`).
    fn first_path(
        &mut self,
        node: top_tree::ClusterId,
        near: top_tree::VertexId,
        level: Level,
    ) -> Option<top_tree::VertexId> {
        self.top_tree.push_node_tag(node);

        if !incident_has(self.top_tree.node_summary(node), level) {
            return None;
        }

        if !matches!(
            self.top_tree.node_leaf_data(node),
            top_tree::NodeData::Internal
        ) {
            return None;
        }

        let (left, right) = self.top_tree.node_children(node).expect("internal node");

        let central = self.top_tree.node_central(node).expect("internal node");

        if self.top_tree.node_is_path(left) && self.top_tree.node_is_path(right) {
            let (near_child, far_child) =
                if boundary_contains(self.top_tree.node_boundary(left), near) {
                    (left, right)
                } else {
                    (right, left)
                };

            self.first_path(near_child, near, level)
                .or_else(|| self.first_path(far_child, central, level))
        } else {
            let (path_child, point_child) = if self.top_tree.node_is_path(left) {
                (left, right)
            } else {
                (right, left)
            };
            if central == near {
                self.find_label_vertex(point_child, central, level)
                    .or_else(|| self.first_path(path_child, near, level))
            } else {
                self.first_path(path_child, near, level)
                    .or_else(|| self.find_label_vertex(point_child, central, level))
            }
        }
    }

    /// Finds any valid level-`i` label reachable from `boundary_vertex`, which
    /// must be a boundary vertex of `node`.
    fn find_label_vertex(
        &mut self,
        node: top_tree::ClusterId,
        boundary_vertex: top_tree::VertexId,
        level: Level,
    ) -> Option<top_tree::VertexId> {
        let boundary = self.top_tree.node_boundary(node);
        // NOTE: this must be evaluated BEFORE pushing `node`'s tag, because
        // `push_tag` consumes the label's own `pending` state used by clean_tree.
        if !pointincident_has(
            self.top_tree.node_summary(node),
            boundary,
            boundary_vertex,
            level,
        ) {
            return None;
        }

        match self.top_tree.node_leaf_data(node) {
            top_tree::NodeData::Label(_) => {
                return self
                    .top_tree
                    .node_label_key(node)
                    .and_then(|key| self.top_tree.label_vertex(key));
            }
            top_tree::NodeData::Edge(_) => return None,
            top_tree::NodeData::Internal => {}
        }

        self.top_tree.push_node_tag(node);
        let (left, right) = self.top_tree.node_children(node).expect("internal node");
        let central = self.top_tree.node_central(node).expect("internal node");

        if boundary_vertex == central {
            return self
                .find_label_vertex(left, boundary_vertex, level)
                .or_else(|| self.find_label_vertex(right, boundary_vertex, level));
        }

        if boundary_contains(self.top_tree.node_boundary(left), boundary_vertex) {
            if let Some(found) = self.find_label_vertex(left, boundary_vertex, level) {
                return Some(found);
            }
            if self.top_tree.node_summary(left).cover >= i32::from(level) {
                return self.find_label_vertex(right, central, level);
            }
            None
        } else {
            if let Some(found) = self.find_label_vertex(right, boundary_vertex, level) {
                return Some(found);
            }
            if self.top_tree.node_summary(right).cover >= i32::from(level) {
                return self.find_label_vertex(left, central, level);
            }
            None
        }
    }
}

#[cfg(test)]
mod tests;
