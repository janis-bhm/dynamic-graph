//! Public graph-level operations built from the tree operations in Appendix A
//! of Holm, Rotenberg, and Thorup's dynamic bridge-finding algorithm.
//!
//! [`DynamicGraph`] keeps a spanning forest in [`FindBridge`]. Tree edges are
//! represented directly in that forest, while each non-tree edge contributes
//! two labels and covers the tree path between its endpoints. The labels are
//! promoted by the paper's `RecoverPhase` procedure when tree edges are
//! deleted.

use std::collections::BTreeMap;

use top_tree::{ClusterId, VertexId as TopVertexId};

use crate::cover_level::{FindBridge, Level, UserLabel};

/// A stable handle for a vertex in a [`DynamicGraph`].
///
/// The index is the vertex's append-only public ID, not its current position
/// in the internal forest. Removing a vertex invalidates its handle, and later
/// additions never reuse it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VertexId(usize);

impl VertexId {
    /// Returns this vertex's append-only public ID index.
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
    endpoints: (ClusterId, ClusterId),
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
/// spanning forest and edge levels. Removing a vertex deletes each incident
/// edge through the normal deletion path before removing the forest vertex.
///
/// Public vertex handles and edge endpoints are translated through each
/// vertex's stable label-cluster ID, so endpoint IDs need no remapping, the
/// reverse table is never `swap_remove`d, and `tree_edge_at` is never rebuilt.
/// Incident edges are still deleted through `delete_edge`, whose delete/recover
/// path may promote a surviving non-tree edge and change its level and labels.
/// After those deletions, the removed handle is tombstoned and its inverse
/// cluster entry is cleared.
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
    /// `VertexId.0 -> the vertex's stable label cluster`. Public handles are
    /// append-only; a removed handle becomes a `None` tombstone and is never
    /// reused.
    vertices: Vec<Option<ClusterId>>,
    /// `ClusterId::index() -> VertexId` for live vertex label clusters.
    ///
    /// A freed vertex cluster slot may later be reused by an edge or internal
    /// cluster; those never write here, and the slot is cleared to `None` when
    /// the owning vertex is removed. Only read it for clusters known to be live
    /// vertex label clusters.
    vertex_of_cluster: Vec<Option<VertexId>>,
    /// `EdgeId.0 -> live edge record`; deleted edge handles are never reused.
    edges: Vec<Option<EdgeRecord>>,
    /// Normalized endpoint cluster pair -> the live tree-edge handle.
    tree_edge_at: BTreeMap<(ClusterId, ClusterId), EdgeId>,
    /// Each live non-tree label -> the graph edge that owns it.
    label_to_edge: BTreeMap<UserLabel, EdgeId>,
    /// Highest supported cover level, also used by the paper for tree edges.
    l_max: i32,
    /// Number of live graph edges (including both tree and non-tree edges).
    live_edges: usize,
    /// Number of live vertices.
    live_vertices: usize,
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
            vertex_of_cluster: Vec::new(),
            edges: Vec::new(),
            tree_edge_at: BTreeMap::new(),
            label_to_edge: BTreeMap::new(),
            l_max: i32::from(Level::MAX),
            live_edges: 0,
            live_vertices: 0,
        }
    }

    /// Adds an isolated vertex and returns its stable, append-only handle.
    pub fn add_vertex(&mut self) -> VertexId {
        debug_assert!(
            self.live_vertices < (1usize << 31) - 1,
            "adding a vertex would exceed DynamicGraph's supported range (< 2^31 vertices; level cap 32)"
        );
        let id = VertexId(self.vertices.len());
        let internal = self.fb.add_vertex();
        let cluster = self
            .fb
            .vertex_cluster(internal)
            .expect("every live vertex has a structural label cluster");
        self.vertices.push(Some(cluster));
        if self.vertex_of_cluster.len() <= cluster.index() {
            self.vertex_of_cluster.resize(cluster.index() + 1, None);
        }
        self.vertex_of_cluster[cluster.index()] = Some(id);
        self.live_vertices += 1;
        id
    }

    /// Returns the number of live vertices in this graph.
    pub fn vertex_count(&self) -> usize {
        self.live_vertices
    }

    /// Returns the number of live graph edges.
    pub fn edge_count(&self) -> usize {
        self.live_edges
    }

    /// Removes a vertex and all incident edges, returning whether the vertex
    /// handle was live.
    ///
    /// Every incident edge is removed through [`DynamicGraph::delete_edge`],
    /// so replacement-edge search and level recovery run as usual. Removed
    /// public vertex handles are never reused.
    pub fn remove_vertex(&mut self, vertex: VertexId) -> bool {
        let Some(cluster) = self.cluster(vertex) else {
            return false;
        };
        let internal = self.internal(cluster);

        let incident_edges = self
            .edges
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                record
                    .filter(|record| record.endpoints.0 == cluster || record.endpoints.1 == cluster)
                    .map(|_| EdgeId(index))
            })
            .collect::<Vec<_>>();
        for edge in incident_edges {
            assert!(
                self.delete_edge(edge),
                "an edge collected from a live vertex must still be live"
            );
        }

        assert!(
            self.fb.remove_vertex(internal),
            "the internal vertex remains live after deleting its edges"
        );
        self.vertices[vertex.0] = None;
        self.vertex_of_cluster[cluster.index()] = None;
        self.live_vertices -= 1;

        true
    }

    /// Inserts an undirected edge, returning `None` for invalid vertices or a
    /// self-loop.
    ///
    /// As in Appendix A's `Insert`, an edge between different tree components
    /// becomes a tree edge. An edge whose endpoints are already connected
    /// becomes a level-0 non-tree edge that covers their tree path.
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId) -> Option<EdgeId> {
        let ca = self.cluster(u)?;
        let cb = self.cluster(v)?;
        if ca == cb {
            return None;
        }
        let (a, b) = self.resolve(ca, cb);

        let edge = EdgeId(self.edges.len());
        self.edges.push(None);

        if !self.fb.connected(a, b) {
            self.fb.link(a, b);
            let _ = self.tree_edge_at.insert(norm(ca, cb), edge);
            self.edges[edge.0] = Some(EdgeRecord {
                endpoints: (ca, cb),
                level: self.l_max,
                kind: EdgeKind::Tree,
            });
        } else {
            let level = Level::new(0).expect("level zero is supported");
            let (label1, label2) = self.add_edge_labels(edge, a, b, level);
            self.fb.cover(a, b, level);
            self.edges[edge.0] = Some(EdgeRecord {
                endpoints: (ca, cb),
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
        let (cu, cw) = record.endpoints;
        let (v, w) = self.resolve(cu, cw);
        let mut alpha = record.level;

        // In the paper l_max identifies a tree edge. Here it is also the
        // largest representable cover level, so EdgeKind disambiguates a
        // non-tree edge promoted to that level.
        if matches!(record.kind, EdgeKind::Tree) {
            if self.fb.cover_level_between(v, w) == -1 {
                let _ = self.fb.cut(v, w);
                let _ = self.tree_edge_at.remove(&norm(cu, cw));
                self.edges[e.0] = None;
                self.live_edges -= 1;
                return true;
            }

            self.swap(e);
            record = *self.edges[e.0]
                .as_ref()
                .expect("Swap keeps the deleted edge live as a non-tree edge");
            // Paper Appendix A, Delete line 30: for a tree edge the original
            // level is l_max, so use the cover level that Swap turned it into.
            alpha = record.level;
        }

        let (label1, label2) = match record.kind {
            EdgeKind::NonTree { label1, label2 } => (label1, label2),
            EdgeKind::Tree => unreachable!("a tree edge is swapped before label removal"),
        };
        self.remove_edge_labels(label1, label2);
        let level = Level::new(alpha).expect("a non-tree edge has a supported cover level");
        self.fb.uncover(v, w, level);
        for i in (0..=alpha).rev() {
            self.recover(cw, cu, i);
        }

        self.edges[e.0] = None;
        self.live_edges -= 1;
        true
    }

    /// Returns the public endpoints of a live edge, or `None` for an invalid
    /// or deleted edge handle.
    pub fn edge_endpoints(&self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        let record = self.edges.get(e.0)?.as_ref()?;
        let u = self
            .vertex_of_cluster
            .get(record.endpoints.0.index())
            .copied()
            .flatten()?;
        let v = self
            .vertex_of_cluster
            .get(record.endpoints.1.index())
            .copied()
            .flatten()?;
        Some((u, v))
    }

    /// Returns whether two valid vertices are connected; invalid handles
    /// return `false`.
    pub fn connected(&mut self, u: VertexId, v: VertexId) -> bool {
        let (Some(ca), Some(cb)) = (self.cluster(u), self.cluster(v)) else {
            return false;
        };
        let (a, b) = self.resolve(ca, cb);
        self.fb.connected(a, b)
    }

    /// Returns any bridge in `v`'s connected component, if one exists.
    pub fn find_bridge(&mut self, v: VertexId) -> Option<EdgeId> {
        let cluster = self.cluster(v)?;
        let internal = self.internal(cluster);
        let edge = self.fb.find_bridge(internal)?;
        self.graph_edge_of(edge)
    }

    /// Returns any bridge on the `u`-`v` path, if one exists.
    pub fn find_bridge_between(&mut self, u: VertexId, v: VertexId) -> Option<EdgeId> {
        let (ca, cb) = (self.cluster(u)?, self.cluster(v)?);
        let (a, b) = self.resolve(ca, cb);
        let edge = self.fb.find_bridge_between(a, b)?;
        self.graph_edge_of(edge)
    }

    /// Returns the number of vertices in `v`'s connected component, or zero
    /// for an invalid handle.
    pub fn component_size(&mut self, v: VertexId) -> u64 {
        let Some(cluster) = self.cluster(v) else {
            return 0;
        };
        let internal = self.internal(cluster);
        self.fb.find_size(internal, internal, -1)
    }

    /// Returns the number of vertices in `v`'s two-edge-connected component,
    /// or zero for an invalid handle.
    pub fn two_edge_component_size(&mut self, v: VertexId) -> u64 {
        let Some(cluster) = self.cluster(v) else {
            return 0;
        };
        let internal = self.internal(cluster);
        self.fb.find_size(internal, internal, 0)
    }

    /// Returns whether `u` and `v` are two-edge-connected. A valid vertex is
    /// two-edge-connected to itself; invalid or disconnected handles return
    /// `false`.
    pub fn two_edge_connected(&mut self, u: VertexId, v: VertexId) -> bool {
        let (Some(ca), Some(cb)) = (self.cluster(u), self.cluster(v)) else {
            return false;
        };
        let (a, b) = self.resolve(ca, cb);
        self.fb.connected(a, b) && self.fb.cover_level_between(a, b) >= 0
    }

    /// The stable cluster of a live public vertex handle.
    fn cluster(&self, vertex: VertexId) -> Option<ClusterId> {
        self.vertices.get(vertex.0).copied().flatten()
    }

    /// The current internal handle of a live vertex cluster.
    ///
    /// `top_tree::VertexId`s are compacted by vertex removals, so they are
    /// resolved here at the `FindBridge` boundary and never stored in
    /// `DynamicGraph`.
    fn internal(&self, cluster: ClusterId) -> TopVertexId {
        self.fb
            .cluster_vertex(cluster)
            .expect("a live vertex owns a label cluster")
    }

    /// Both internal handles of a pair of live vertex clusters.
    fn resolve(&self, a: ClusterId, b: ClusterId) -> (TopVertexId, TopVertexId) {
        (self.internal(a), self.internal(b))
    }

    /// The live graph edge whose forest edge is `edge` (a `FindBridge` result).
    fn graph_edge_of(&self, edge: (TopVertexId, TopVertexId)) -> Option<EdgeId> {
        let a = self.fb.vertex_cluster(edge.0)?;
        let b = self.fb.vertex_cluster(edge.1)?;
        self.tree_edge_at.get(&norm(a, b)).copied()
    }

    fn add_edge_labels(
        &mut self,
        edge: EdgeId,
        u: TopVertexId,
        v: TopVertexId,
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
        let (cu, cw) = record.endpoints;
        let (v, w) = self.resolve(cu, cw);
        let alpha = self.fb.cover_level_between(v, w);
        debug_assert!(alpha >= 0 && alpha <= self.l_max);

        let _ = self.fb.cut(v, w);
        let _ = self.tree_edge_at.remove(&norm(cu, cw));
        let replacement = self.find_replacement(cu, cw, alpha);
        let replacement_record = *self.edges[replacement.0]
            .as_ref()
            .expect("FindReplacement returns a live edge");
        let (cx, cy) = replacement_record.endpoints;
        let (x, y) = self.resolve(cx, cy);
        let (replacement_label1, replacement_label2) = match replacement_record.kind {
            EdgeKind::NonTree { label1, label2 } => (label1, label2),
            EdgeKind::Tree => unreachable!("only labelled non-tree edges replace a tree edge"),
        };

        self.remove_edge_labels(replacement_label1, replacement_label2);
        self.fb.link(x, y);
        let _ = self.tree_edge_at.insert(norm(cx, cy), replacement);
        self.edges[replacement.0] = Some(EdgeRecord {
            endpoints: (cx, cy),
            level: self.l_max,
            kind: EdgeKind::Tree,
        });

        let level = Level::new(alpha).expect("a covered tree edge has a supported level");
        let (label1, label2) = self.add_edge_labels(edge, v, w, level);
        self.edges[edge.0] = Some(EdgeRecord {
            endpoints: (cu, cw),
            level: alpha,
            kind: EdgeKind::NonTree { label1, label2 },
        });
        self.fb.cover(v, w, level);
    }

    /// Appendix A's `FindReplacement` searches the smaller side after a cut.
    fn find_replacement(&mut self, v: ClusterId, w: ClusterId, i: i32) -> EdgeId {
        let (internal_v, internal_w) = self.resolve(v, w);
        let size_v = self.fb.find_size(internal_v, internal_v, i);
        let size_w = self.fb.find_size(internal_w, internal_w, i);
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
    fn recover(&mut self, v: ClusterId, w: ClusterId, i: i32) {
        let (internal_v, internal_w) = self.resolve(v, w);
        let size = self.fb.find_size(internal_v, internal_w, i) / 2;
        let _ = self.recover_phase(v, w, i, size);
        let _ = self.recover_phase(w, v, i, size);
    }

    /// Appendix A's `RecoverPhase`: promote eligible labelled edges or return
    /// a non-tree edge crossing the cut currently being repaired.
    fn recover_phase(&mut self, v: ClusterId, w: ClusterId, i: i32, size: u64) -> Option<EdgeId> {
        let level = Level::new(i).expect("RecoverPhase uses a supported level");
        let (internal_v, internal_w) = self.resolve(v, w);
        loop {
            let label = self.fb.find_first_label(internal_v, internal_w, level)?;
            let edge = *self
                .label_to_edge
                .get(&label)
                .expect("every live non-tree label belongs to an edge");
            let record = *self.edges[edge.0]
                .as_ref()
                .expect("a live label belongs to a live edge");
            let (cq, cr) = record.endpoints;
            let (q, r) = self.resolve(cq, cr);

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
                    endpoints: (cq, cr),
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

fn norm(a: ClusterId, b: ClusterId) -> (ClusterId, ClusterId) {
    if a < b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests;
