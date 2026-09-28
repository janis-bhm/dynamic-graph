//! Public graph-level operations built from the tree operations in Appendix A
//! of Holm, Rotenberg, and Thorup's dynamic bridge-finding algorithm.
//!
//! [`DynamicGraph`] keeps a spanning forest in [`FindBridge`]. Tree edges are
//! represented directly in that forest, while each non-tree edge contributes
//! two labels and covers the tree path between its endpoints. The labels are
//! promoted by the paper's `RecoverPhase` procedure when tree edges are
//! deleted.

use std::collections::BTreeMap;

use crate::cover_level::{FindBridge, Level, UserLabel};

/// A stable handle for a vertex in a [`DynamicGraph`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VertexId(usize);

impl VertexId {
    /// Returns the zero-based index of this vertex in the graph.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A stable handle for an edge in a [`DynamicGraph`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeId(usize);

impl EdgeId {
    /// Returns the zero-based index of this edge in the graph.
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
enum EdgeKind {
    Tree,
    NonTree {
        label1: UserLabel,
        label2: UserLabel,
    },
}

#[derive(Clone, Copy, Debug)]
struct EdgeRecord {
    endpoints: (usize, usize),
    level: i32,
    kind: EdgeKind,
}

/// An undirected dynamic graph supporting connectivity, bridge, and
/// two-edge-connectivity queries.
///
/// The graph-level reduction from Appendix A is maintained on top of
/// [`FindBridge`]: inserted edges that join components become tree edges;
/// other edges are stored as levelled endpoint labels covering their tree
/// path. Deletions use `Swap`, `FindReplacement`, and `Recover` to maintain the
/// spanning forest and edge levels.
///
/// # Vertex-count limit
///
/// The fixed dense level cap is 32, so this structure is only guaranteed for
/// graphs with fewer than `2^31` vertices. Beyond this range, the paper's
/// `l_max = floor(log2 n)` can exceed the levels represented by the dense
/// `FindSize`/`FindFirstLabel` structures. Since non-tree edges may be promoted
/// up to `l_max`, reaching the cap can prevent further promotion, so the size
/// invariant used to guarantee a replacement edge may fail.
pub struct DynamicGraph {
    fb: FindBridge,
    /// `VertexId.0 -> FindBridge`'s internal vertex index.
    vertices: Vec<usize>,
    /// `FindBridge`'s internal vertex index -> public handle.
    internal_to_vertex: Vec<VertexId>,
    /// `EdgeId.0 -> live edge record`; deleted edge handles are never reused.
    edges: Vec<Option<EdgeRecord>>,
    /// Normalized internal endpoint pair -> the live tree-edge handle.
    tree_edge_at: BTreeMap<(usize, usize), EdgeId>,
    /// Each live non-tree label -> the graph edge that owns it.
    label_to_edge: BTreeMap<UserLabel, EdgeId>,
    /// Highest supported cover level, also used by the paper for tree edges.
    l_max: i32,
    /// Number of live graph edges (including both tree and non-tree edges).
    live_edges: usize,
}

impl Default for DynamicGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicGraph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self {
            fb: FindBridge::new(),
            vertices: Vec::new(),
            internal_to_vertex: Vec::new(),
            edges: Vec::new(),
            tree_edge_at: BTreeMap::new(),
            label_to_edge: BTreeMap::new(),
            l_max: i32::from(Level::MAX),
            live_edges: 0,
        }
    }

    /// Adds an isolated vertex and returns its stable handle.
    pub fn add_vertex(&mut self) -> VertexId {
        debug_assert!(
            self.vertices.len() < (1usize << 31) - 1,
            "adding a vertex would exceed DynamicGraph's supported range (< 2^31 vertices; level cap 32)"
        );
        let id = VertexId(self.vertices.len());
        let internal = self.fb.add_vertex(id.0);
        self.vertices.push(internal);
        self.internal_to_vertex.push(id);
        id
    }

    /// Returns the number of vertices added to this graph.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of live graph edges.
    pub fn edge_count(&self) -> usize {
        self.live_edges
    }

    /// Inserts an undirected edge, returning `None` for invalid vertices or a
    /// self-loop.
    ///
    /// As in Appendix A's `Insert`, an edge between different tree components
    /// becomes a tree edge. An edge whose endpoints are already connected
    /// becomes a level-0 non-tree edge that covers their tree path.
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId) -> Option<EdgeId> {
        let a = *self.vertices.get(u.0)?;
        let b = *self.vertices.get(v.0)?;
        if a == b {
            return None;
        }

        let edge = EdgeId(self.edges.len());
        self.edges.push(None);

        if !self.fb.connected(a, b) {
            self.fb.link(a, b);
            let _ = self.tree_edge_at.insert(norm(a, b), edge);
            self.edges[edge.0] = Some(EdgeRecord {
                endpoints: (a, b),
                level: self.l_max,
                kind: EdgeKind::Tree,
            });
        } else {
            let level = Level::new(0).expect("level zero is supported");
            let (label1, label2) = self.add_edge_labels(edge, a, b, level);
            self.fb.cover(a, b, level);
            self.edges[edge.0] = Some(EdgeRecord {
                endpoints: (a, b),
                level: 0,
                kind: EdgeKind::NonTree { label1, label2 },
            });
        }

        self.live_edges += 1;
        Some(edge)
    }

    /// Deletes a live edge, returning whether `e` existed.
    ///
    /// This is Appendix A's `Delete`: a covered tree edge is swapped with a
    /// replacement edge, then the deleted edge's labels and cover are removed
    /// and the affected levels are recovered. Edge kind is checked in addition
    /// to its level because a promoted non-tree edge can also reach `l_max`.
    pub fn delete_edge(&mut self, e: EdgeId) -> bool {
        let Some(mut record) = self.edges.get(e.0).and_then(Option::as_ref).copied() else {
            return false;
        };
        let (mut v, mut w) = record.endpoints;
        let alpha = record.level;

        // In the paper l_max identifies a tree edge. Here it is also the
        // largest representable cover level, so EdgeKind disambiguates a
        // non-tree edge promoted to that level.
        if matches!(record.kind, EdgeKind::Tree) {
            if self.fb.cover_level_between(v, w) == -1 {
                let _ = self.fb.cut(v, w);
                let _ = self.tree_edge_at.remove(&norm(v, w));
                self.edges[e.0] = None;
                self.live_edges -= 1;
                return true;
            }

            self.swap(e);
            record = *self.edges[e.0]
                .as_ref()
                .expect("Swap keeps the deleted edge live as a non-tree edge");
            (v, w) = record.endpoints;
        }

        let (label1, label2) = match record.kind {
            EdgeKind::NonTree { label1, label2 } => (label1, label2),
            EdgeKind::Tree => unreachable!("a tree edge is swapped before label removal"),
        };
        self.remove_edge_labels(label1, label2);
        let level = Level::new(alpha).expect("a non-tree edge has a supported cover level");
        self.fb.uncover(v, w, level);
        for i in (0..=alpha).rev() {
            self.recover(w, v, i);
        }

        self.edges[e.0] = None;
        self.live_edges -= 1;
        true
    }

    /// Returns the public endpoints of a live edge, or `None` for an invalid
    /// or deleted edge handle.
    pub fn edge_endpoints(&self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        let record = self.edges.get(e.0)?.as_ref()?;
        let u = *self.internal_to_vertex.get(record.endpoints.0)?;
        let v = *self.internal_to_vertex.get(record.endpoints.1)?;
        Some((u, v))
    }

    /// Returns whether two valid vertices are connected; invalid handles
    /// return `false`.
    pub fn connected(&mut self, u: VertexId, v: VertexId) -> bool {
        let (Some(a), Some(b)) = (self.internal_vertex(u), self.internal_vertex(v)) else {
            return false;
        };
        self.fb.connected(a, b)
    }

    /// Returns any bridge in `v`'s connected component, if one exists.
    pub fn find_bridge(&mut self, v: VertexId) -> Option<EdgeId> {
        let internal = self.internal_vertex(v)?;
        let edge = self.fb.find_bridge(internal)?;
        self.tree_edge_at.get(&norm(edge.0, edge.1)).copied()
    }

    /// Returns any bridge on the `u`-`v` path, if one exists.
    pub fn find_bridge_between(&mut self, u: VertexId, v: VertexId) -> Option<EdgeId> {
        let (a, b) = (self.internal_vertex(u)?, self.internal_vertex(v)?);
        let edge = self.fb.find_bridge_between(a, b)?;
        self.tree_edge_at.get(&norm(edge.0, edge.1)).copied()
    }

    /// Returns the number of vertices in `v`'s connected component, or zero
    /// for an invalid handle.
    pub fn component_size(&mut self, v: VertexId) -> u64 {
        let Some(internal) = self.internal_vertex(v) else {
            return 0;
        };
        self.fb.find_size(internal, internal, -1)
    }

    /// Returns the number of vertices in `v`'s two-edge-connected component,
    /// or zero for an invalid handle.
    pub fn two_edge_component_size(&mut self, v: VertexId) -> u64 {
        let Some(internal) = self.internal_vertex(v) else {
            return 0;
        };
        self.fb.find_size(internal, internal, 0)
    }

    /// Returns whether `u` and `v` are two-edge-connected. A valid vertex is
    /// two-edge-connected to itself; invalid or disconnected handles return
    /// `false`.
    pub fn two_edge_connected(&mut self, u: VertexId, v: VertexId) -> bool {
        let (Some(a), Some(b)) = (self.internal_vertex(u), self.internal_vertex(v)) else {
            return false;
        };
        self.fb.connected(a, b) && self.fb.cover_level_between(a, b) >= 0
    }

    fn internal_vertex(&self, vertex: VertexId) -> Option<usize> {
        self.vertices.get(vertex.0).copied()
    }

    fn add_edge_labels(
        &mut self,
        edge: EdgeId,
        u: usize,
        v: usize,
        level: Level,
    ) -> (UserLabel, UserLabel) {
        let label1 = self.fb.add_label(u, level);
        let label2 = self.fb.add_label(v, level);
        let _ = self.label_to_edge.insert(label1, edge);
        let _ = self.label_to_edge.insert(label2, edge);
        (label1, label2)
    }

    fn remove_edge_labels(&mut self, label1: UserLabel, label2: UserLabel) {
        let _ = self.fb.remove_label(label1);
        let _ = self.fb.remove_label(label2);
        let _ = self.label_to_edge.remove(&label1);
        let _ = self.label_to_edge.remove(&label2);
    }

    /// Appendix A's `Swap`: replace a covered tree edge and turn it into a
    /// non-tree edge at the path's minimum cover level.
    fn swap(&mut self, edge: EdgeId) {
        let record = *self.edges[edge.0]
            .as_ref()
            .expect("Swap is called for a live tree edge");
        let (v, w) = record.endpoints;
        let alpha = self.fb.cover_level_between(v, w);
        debug_assert!(alpha >= 0 && alpha <= self.l_max);

        let _ = self.fb.cut(v, w);
        let _ = self.tree_edge_at.remove(&norm(v, w));
        let replacement = self.find_replacement(v, w, alpha);
        let replacement_record = *self.edges[replacement.0]
            .as_ref()
            .expect("FindReplacement returns a live edge");
        let (x, y) = replacement_record.endpoints;
        let (replacement_label1, replacement_label2) = match replacement_record.kind {
            EdgeKind::NonTree { label1, label2 } => (label1, label2),
            EdgeKind::Tree => unreachable!("only labelled non-tree edges replace a tree edge"),
        };

        self.remove_edge_labels(replacement_label1, replacement_label2);
        self.fb.link(x, y);
        let _ = self.tree_edge_at.insert(norm(x, y), replacement);
        self.edges[replacement.0] = Some(EdgeRecord {
            endpoints: (x, y),
            level: self.l_max,
            kind: EdgeKind::Tree,
        });

        let level = Level::new(alpha).expect("a covered tree edge has a supported level");
        let (label1, label2) = self.add_edge_labels(edge, v, w, level);
        self.edges[edge.0] = Some(EdgeRecord {
            endpoints: (v, w),
            level: alpha,
            kind: EdgeKind::NonTree { label1, label2 },
        });
        self.fb.cover(v, w, level);
    }

    /// Appendix A's `FindReplacement` searches the smaller side after a cut.
    fn find_replacement(&mut self, v: usize, w: usize, i: i32) -> EdgeId {
        let size_v = self.fb.find_size(v, v, i);
        let size_w = self.fb.find_size(w, w, i);
        if size_v <= size_w {
            self.recover_phase(v, v, i, size_v)
        } else {
            self.recover_phase(w, w, i, size_w)
        }
        .expect(
            "a covered tree edge must have a replacement; the fixed level cap of 32 may be insufficient for graphs with at least 2^31 vertices",
        )
    }

    /// Appendix A's `Recover`: scan both orientations of a path with half its
    /// level-`i` size as the promotion threshold.
    fn recover(&mut self, v: usize, w: usize, i: i32) {
        let size = self.fb.find_size(v, w, i) / 2;
        let _ = self.recover_phase(v, w, i, size);
        let _ = self.recover_phase(w, v, i, size);
    }

    /// Appendix A's `RecoverPhase`: promote eligible labelled edges or return
    /// a non-tree edge crossing the cut currently being repaired.
    fn recover_phase(&mut self, v: usize, w: usize, i: i32, size: u64) -> Option<EdgeId> {
        let level = Level::new(i).expect("RecoverPhase uses a supported level");
        loop {
            let label = self.fb.find_first_label(v, w, level)?;
            let edge = *self
                .label_to_edge
                .get(&label)
                .expect("every live non-tree label belongs to an edge");
            let record = *self.edges[edge.0]
                .as_ref()
                .expect("a live label belongs to a live edge");
            let (q, r) = record.endpoints;

            if !self.fb.connected(q, r) {
                return Some(edge);
            }

            let next_level = i + 1;
            if next_level <= self.l_max && self.fb.find_size(q, r, next_level) <= size {
                let (label1, label2) = match record.kind {
                    EdgeKind::NonTree { label1, label2 } => (label1, label2),
                    EdgeKind::Tree => unreachable!("a label is attached to a non-tree edge"),
                };
                self.remove_edge_labels(label1, label2);
                let promoted =
                    Level::new(next_level).expect("the promotion guard keeps the level in range");
                let (new_label1, new_label2) = self.add_edge_labels(edge, q, r, promoted);
                self.edges[edge.0] = Some(EdgeRecord {
                    endpoints: (q, r),
                    level: next_level,
                    kind: EdgeKind::NonTree {
                        label1: new_label1,
                        label2: new_label2,
                    },
                });
                self.fb.cover(q, r, promoted);
            } else {
                self.fb.cover(q, r, level);
                return None;
            }
        }
    }
}

fn norm(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests;
