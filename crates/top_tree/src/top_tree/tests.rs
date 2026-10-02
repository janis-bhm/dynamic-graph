use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use super::*;

/// The top tree leaf that represents the forest edge `a`-`b`.
///
/// The forest stores that `ClusterId` as the edge payload, so `link` can hand
/// the leaf back to an application that wants to remember it.
fn edge_leaf<N, E, S: Summary>(
    tt: &TopTree<S, N, E>,
    a: tree::VertexId,
    b: tree::VertexId,
) -> ClusterId {
    let edge = tt.tree.edge_index_of(a, b).expect("edge must exist");
    *tt.tree.edge_weight(edge).expect("edge must exist")
}

// The summaries below take their per-leaf value from the leaf's own
// `ClusterId::index()`, the leaf's slot in the cluster arena. Two reasons:
//
// * [`Summary::tree_edge`] is handed only the leaf's [`ClusterId`] — in
//   particular not the edge weight the application passed to [`TopTree::link`],
//   which the top tree keeps in the leaf cluster's own `weight` field and never
//   shows to a summary. A slot index is unique among live leaves, so it still
//   pins down *which* edges a summary covers. The application-supplied weight is
//   checked on its own terms instead: by [`check_invariants`] (every edge leaf
//   keeps the weight `link` was given) and by [`Harness::cut`] (cutting returns
//   it).
//
// * The tests compute their expected aggregates from the leaves they recorded at
//   `link` time, reading the same slot index off the `ClusterId` directly rather
//   than through a shared helper. So a summary that misreports which edges it
//   covers, or their values, cannot agree with the expectation by construction.

/// A monoid summary over the whole cluster: number of tree edges and xor of
/// the values of their leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Agg {
    edges: u32,
    xor: u64,
}

impl Summary for Agg {
    type Tag = ();

    fn tree_edge(e: ClusterId) -> Self {
        Agg {
            edges: 1,
            xor: e.index() as u64,
        }
    }

    fn label(_v: ClusterId) -> Self {
        Agg { edges: 0, xor: 0 }
    }

    fn combine(left: &Self, right: &Self, _ctx: &MergeContext) -> Self {
        Agg {
            edges: left.edges + right.edges,
            xor: left.xor ^ right.xor,
        }
    }
}

/// A boundary aware summary over the cluster path: the number of edges on the
/// cluster path. This exercises the `is_compress`/`is_rake` distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathLen {
    len: u32,
}

impl Summary for PathLen {
    type Tag = ();

    fn tree_edge(_u: ClusterId) -> Self {
        PathLen { len: 1 }
    }

    fn label(_v: ClusterId) -> Self {
        PathLen { len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            PathLen {
                len: u32::from(ctx.left_is_path_child()) * left.len
                    + u32::from(ctx.right_is_path_child()) * right.len,
            }
        } else {
            PathLen { len: 0 }
        }
    }
}

/// An aggregate that includes every leaf, so a label summary mutation is
/// visible from an exposed path containing that label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LabelValue {
    value: u64,
}

impl Summary for LabelValue {
    type Tag = ();

    fn tree_edge(_u: ClusterId) -> Self {
        LabelValue { value: 0 }
    }

    fn label(_v: ClusterId) -> Self {
        LabelValue { value: 1 }
    }

    fn combine(left: &Self, right: &Self, _ctx: &MergeContext) -> Self {
        LabelValue {
            value: left.value + right.value,
        }
    }
}

/// The sum of the per-leaf values on the cluster path, with a lazy "add `x` to
/// every edge on the path" tag. This is the mechanism used by cover-level
/// style algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathSum {
    sum: i64,
    len: u32,
}

impl Summary for PathSum {
    type Tag = i64;

    fn tree_edge(e: ClusterId) -> Self {
        PathSum {
            sum: e.index() as i64,
            len: 1,
        }
    }

    fn label(_v: ClusterId) -> Self {
        PathSum { sum: 0, len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            let mut sum = 0;
            let mut len = 0;
            if ctx.left_is_path_child() {
                sum += left.sum;
                len += left.len;
            }
            if ctx.right_is_path_child() {
                sum += right.sum;
                len += right.len;
            }
            PathSum { sum, len }
        } else {
            PathSum { sum: 0, len: 0 }
        }
    }

    fn apply(&mut self, tag: &i64) {
        self.sum += tag * self.len as i64;
    }

    fn compose(tag: &mut i64, parent: &i64) {
        *tag += *parent;
    }
}

/// The maximum per-leaf value on the cluster path (`0` for point clusters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathMax {
    max: u64,
}

impl Summary for PathMax {
    type Tag = ();

    fn tree_edge(e: ClusterId) -> Self {
        PathMax {
            max: e.index() as u64,
        }
    }

    fn label(_v: ClusterId) -> Self {
        PathMax { max: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary.is_path() {
            let mut max = 0;
            if ctx.left_is_path_child() {
                max = max.max(left.max);
            }
            if ctx.right_is_path_child() {
                max = max.max(right.max);
            }
            PathMax { max }
        } else {
            PathMax { max: 0 }
        }
    }
}

/// The leaf identities on the cluster path, in path order. This is the
/// order-sensitive summary: reversing a cluster reverses the edge sequence, so
/// an exposed path can be checked both for *which* edges it reports and for the
/// direction it reports them in.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectedPath {
    edges: Vec<ClusterId>,
}

impl Summary for DirectedPath {
    type Tag = ();

    fn tree_edge(e: ClusterId) -> Self {
        DirectedPath { edges: vec![e] }
    }

    fn label(_v: ClusterId) -> Self {
        DirectedPath { edges: Vec::new() }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        let mut edges = Vec::new();
        if ctx.left_is_path_child() {
            edges.extend_from_slice(&left.edges);
        }
        if ctx.right_is_path_child() {
            edges.extend_from_slice(&right.edges);
        }
        DirectedPath { edges }
    }

    fn flip(&mut self) {
        self.edges.reverse();
    }
}

#[test]
fn cluster_vertex_tracks_attached_labels_across_compaction() {
    let mut tt = TopTree::<PathLen>::new();
    let removed_vertex = tt.add_vertex();
    let a = tt.add_vertex();
    let moved_vertex = tt.add_vertex();

    let removed_cluster = tt.attach(removed_vertex, ());
    let a_cluster = tt.attach(a, ());
    let moved_cluster = tt.attach(moved_vertex, ());
    assert_eq!(tt.cluster_vertex(removed_cluster), Some(removed_vertex));
    assert_eq!(tt.cluster_vertex(a_cluster), Some(a));
    assert_eq!(tt.cluster_vertex(moved_cluster), Some(moved_vertex));

    let edge_cluster = tt.link(a, moved_vertex, ());
    assert!(tt.cluster_vertex(edge_cluster).is_none());
    let internal_cluster = tt
        .expose_path_node(a, moved_vertex)
        .expect("the linked path has a cluster");
    assert!(tt.cluster_vertex(internal_cluster).is_none());

    let swap = tt
        .remove_vertex(removed_vertex)
        .expect("the isolated vertex is removed");
    let mut moved_after = moved_vertex;
    moved_after.swap(swap);
    assert_ne!(moved_after, moved_vertex);
    assert_eq!(moved_after.index(), removed_vertex.index());
    assert_eq!(tt.cluster_vertex(moved_cluster), Some(moved_after));
}

#[test]
fn cluster_vertex_rejects_freed_cluster_handles() {
    let mut tt = TopTree::<PathLen>::new();
    let v = tt.add_vertex();
    let cluster = tt.attach(v, ());
    assert_eq!(tt.cluster_vertex(cluster), Some(v));

    tt.remove_vertex(v).expect("the attached vertex is removed");
    assert!(tt.cluster_vertex(cluster).is_none());
    assert!(tt.clusters.get(cluster).is_none());
    assert!(tt.clusters.get_mut(cluster).is_none());

    let a = tt.add_vertex();
    let b = tt.add_vertex();
    let edge_cluster = tt.link(a, b, ());
    assert!(tt.cut(a, b).is_some());
    assert!(tt.cluster_vertex(edge_cluster).is_none());
    assert!(tt.clusters.get(edge_cluster).is_none());

    let a = tt.add_vertex();
    let b = tt.add_vertex();
    tt.link(a, b, ());
    tt.attach(a, ());
    let internal_cluster = tt
        .expose_path_node(a, b)
        .expect("the linked path has a cluster");
    assert_eq!(tt.node_kind(internal_cluster), NodeKind::Internal);

    tt.remove_vertex(a)
        .expect("the attached endpoint and its edge are removed");
    assert!(tt.clusters.get(internal_cluster).is_none());
    assert!(tt.cluster_vertex(internal_cluster).is_none());
}

#[test]
fn cluster_liveness_bitmap_matches_allocator() {
    let mut tt = TopTree::<PathLen>::new();
    let a = tt.add_vertex();
    let b = tt.add_vertex();
    let c = tt.add_vertex();
    let d = tt.add_vertex();
    let e = tt.add_vertex();

    let initial_label_cluster = tt.attach(a, ());
    let initial_label = tt
        .try_label_id_for_cluster(initial_label_cluster)
        .expect("attached cluster is a label leaf");
    tt.link(a, b, ());
    assert!(tt.cut(a, b).is_some());
    // Keep a live cluster above `initial_label` so that detaching it leaves a
    // free slot that is *not* the last one. `SlotVec::remove` pops the last
    // slot instead of free-listing it, so freeing the topmost cluster would
    // leave the arena empty and there would be nothing left to reuse.
    tt.attach(c, ());
    tt.detach(initial_label);

    let old_len = tt.clusters.slots.len();
    let cluster_to_remove = tt.attach(a, ());
    assert!(
        cluster_to_remove.index() < old_len,
        "the first attach must reuse a free cluster slot"
    );
    assert_eq!(
        tt.clusters.slots.len(),
        old_len,
        "reusing a free slot must not grow the arena"
    );
    tt.attach(b, ());
    tt.attach(c, ());
    let fresh_cluster = tt.attach(d, ());
    tt.attach(e, ());
    assert!(
        fresh_cluster.index() >= old_len,
        "attaching after exhausting the free list must grow the arena"
    );
    assert!(tt.clusters.slots.len() > old_len);

    tt.remove_vertex(a)
        .expect("the vertex and its attached label are removed");
    assert!(tt.clusters.get(cluster_to_remove).is_none());

    for index in 0..tt.clusters.slots.len() {
        let id =
            ClusterId(crate::NonMaxUsize::new(index).expect("cluster index must fit the ID type"));
        assert_eq!(
            tt.clusters.get(id).is_some(),
            tt.clusters.occupancy.get(index),
            "liveness mismatch at cluster index {index}"
        );
    }

    let iter_indices: Vec<_> = tt.clusters.iter().map(|(id, _)| id.index()).collect();
    let live_indices: Vec<_> = (0..tt.clusters.slots.len())
        .filter(|&index| tt.clusters.occupancy.get(index))
        .collect();
    assert_eq!(iter_indices, live_indices);
}

#[test]
fn cluster_free_list_matches_liveness_bitmap_after_reuse() {
    let mut tt = TopTree::<PathLen>::new();
    let vertices: Vec<_> = (0..6).map(|_| tt.add_vertex()).collect();
    // One label leaf per vertex, i.e. cluster slots `0..6`.
    for &vertex in vertices.iter() {
        let cluster = tt.attach(vertex, ());
        assert!(
            tt.try_label_id_for_cluster(cluster).is_some(),
            "attached cluster is a label leaf"
        );
    }

    let assert_free_list_matches_live = |tt: &TopTree<PathLen>| {
        let nodes_len = tt.clusters.slots.len();
        let mut free_slots = vec![false; nodes_len];
        let mut free_count = 0;
        let mut current = tt.clusters.first_free;

        for _ in 0..=nodes_len {
            let Some(free_index) = current else {
                break;
            };
            let index = free_index.get();
            assert!(index < nodes_len, "cluster free list link out of bounds");
            assert!(
                !free_slots[index],
                "cluster free list must not contain a cycle"
            );
            assert!(
                !tt.clusters.occupancy.get(index),
                "cluster free list must not point at a live slot"
            );
            free_slots[index] = true;
            free_count += 1;

            // SAFETY: The bounds/liveness checks above passed, and the free-list
            // invariant means this slot's first word contains its next link.
            current = unsafe { tt.clusters.slots[index].next };
        }

        assert!(
            current.is_none(),
            "cluster free list traversal exceeded arena length"
        );
        for index in 0..nodes_len {
            assert_eq!(
                free_slots[index],
                !tt.clusters.occupancy.get(index),
                "free-list/liveness mismatch at cluster index {index}"
            );
        }
        assert_eq!(
            free_count,
            (0..nodes_len)
                .filter(|&index| !tt.clusters.occupancy.get(index))
                .count(),
            "free-list length must match the number of non-live slots"
        );
    };

    // `Tree::remove_label` moves the last label into the freed slot and stamps it
    // with a fresh generation, so a label handle collected earlier can go stale.
    // Look each label up by its vertex instead.
    //
    // Detaching bottom-up matters as well: `SlotVec::remove` pops the last slot
    // rather than free-listing it, so freeing the topmost clusters in turn would
    // empty the arena and leave the free list empty for the checks below to
    // agree with the liveness bitmap only vacuously.
    let detach_label_at = |tt: &mut TopTree<PathLen>, vertex: tree::VertexId| {
        let cluster = tt.first_label(vertex).expect("vertex must have a label");
        let label = tt
            .try_label_id_for_cluster(cluster)
            .expect("incident leaf must be a label leaf");
        tt.detach(label);
    };

    for &vertex in vertices.iter().take(4) {
        detach_label_at(&mut tt, vertex);
    }
    assert!(
        tt.clusters.first_free.is_some(),
        "detaching a non-last slot must leave the arena's free list non-empty"
    );

    for &vertex in vertices.iter().take(4) {
        let cluster = tt.attach(vertex, ());
        assert!(
            tt.try_label_id_for_cluster(cluster).is_some(),
            "attached cluster is a label leaf"
        );
    }
    assert_free_list_matches_live(&tt);

    for &vertex in vertices.iter().take(3) {
        detach_label_at(&mut tt, vertex);
    }
    assert!(
        tt.clusters.first_free.is_some(),
        "detaching a non-last slot must leave the arena's free list non-empty"
    );
    assert_free_list_matches_live(&tt);
}

#[test]
fn removing_isolated_vertex_remaps_surviving_clusters() {
    let mut tt = TopTree::<DirectedPath>::new();
    let removed = tt.add_vertex();
    let left = tt.add_vertex();
    let middle = tt.add_vertex();
    let mut moved = tt.add_vertex();

    for vertex in [removed, left, middle, moved] {
        tt.attach(vertex, ());
    }
    tt.link(left, middle, ());
    tt.link(middle, moved, ());
    let left_middle = edge_leaf(&tt, left, middle);
    let middle_moved = edge_leaf(&tt, middle, moved);

    let swap = tt.remove_vertex(removed).expect("live vertex is removed");
    moved.swap(swap);
    assert_eq!(moved.index(), removed.index());
    assert!(tt.forest().node(removed).is_none());
    assert!(tt.forest().node(moved).is_some());
    assert_eq!(tt.edge_count(), 2);
    assert_eq!(tt.label_count(), 3);

    // The summary reports leaf identities, so the exposed path must consist of
    // exactly the two surviving edges, in the order `left`..`moved`.
    let summary = tt.expose_path(left, moved).expect("path remains connected");
    assert_eq!(summary.edges, vec![left_middle, middle_moved]);
    tt.deexpose(moved);
    tt.deexpose(left);

    for root in live_roots(&tt) {
        check_node_boundaries(&tt, root);
    }

    let labels_before_stale_removal = tt.label_count();
    assert_eq!(tt.remove_vertex(removed), None);
    assert_eq!(tt.label_count(), labels_before_stale_removal);
}

#[test]
fn removing_last_vertex_clears_exposure_and_reports_success() {
    let mut tt = TopTree::<PathLen>::new();
    let vertex = tt.add_vertex();
    assert_eq!(tt.expose(vertex), None);
    assert!(tt.exposed.get(vertex.index()));

    assert_eq!(tt.remove_vertex(vertex), Some(SwapResult::None));
    let added = tt.add_vertex();
    assert_eq!(added.index(), vertex.index());
    assert!(!tt.exposed.get(added.index()));
    assert_eq!(
        tt.remove_vertex(vertex),
        None,
        "stale generations are rejected"
    );
}

/// The edge weight the harness supplies for the edge between path positions `k`
/// and `k + 1`. Keeping the historical `k + 1` rule means the application
/// supplied weights along `0-1-2-...` are still `1, 2, 3, ...`.
fn harness_weight(u: usize, v: usize) -> u64 {
    (u.min(v) + 1) as u64
}

/// Unordered key for the harness edge `u`-`v`.
fn edge_key(u: usize, v: usize) -> (usize, usize) {
    (u.min(v), u.max(v))
}

/// A mirror of a small forest, used to drive the top tree and to check the
/// summaries it reports against an independently computed ground truth.
///
/// Every tree edge carries the `u64` weight [`harness_weight`]; the summaries
/// never see that weight (see the note above `Agg`) but `check_invariants` and
/// [`Harness::cut`] check that the top tree keeps it intact.
struct Harness<S: Summary> {
    tt: TopTree<S, (), u64>,
    vertices: Vec<tree::VertexId>,
    adj: HashMap<usize, HashSet<usize>>,
    /// The leaf cluster of every live edge, keyed by [`edge_key`].
    edge_leaves: HashMap<(usize, usize), ClusterId>,
}

impl<S: Summary + Clone> Harness<S> {
    fn new(n: usize) -> Self {
        let mut tt: TopTree<S, (), u64> = TopTree::new();
        let mut vertices = Vec::new();
        let mut adj = HashMap::new();
        for i in 0..n {
            let id = tt.add_vertex();
            vertices.push(id);
            adj.insert(i, Default::default());
        }
        Self {
            tt,
            vertices,
            adj,
            edge_leaves: HashMap::new(),
        }
    }

    fn v(&self, i: usize) -> tree::VertexId {
        self.vertices[i]
    }

    /// The leaf cluster representing the live edge `u`-`v`.
    fn leaf(&self, u: usize, v: usize) -> ClusterId {
        self.edge_leaves[&edge_key(u, v)]
    }

    /// The leaves of the edges on the `u`-`v` path, in path order.
    fn path_leaves(&self, u: usize, v: usize) -> Vec<ClusterId> {
        path_edges(self, u, v)
            .expect("path must exist")
            .into_iter()
            .map(|(a, b)| self.leaf(a, b))
            .collect()
    }

    /// The per-leaf values along the `u`-`v` path, in path order, as the weight
    /// sensitive summaries compute them.
    fn path_leaf_values(&self, u: usize, v: usize) -> Vec<u64> {
        self.path_leaves(u, v)
            .into_iter()
            .map(|leaf| leaf.index() as u64)
            .collect()
    }

    fn link(&mut self, u: usize, v: usize) {
        self.tt.link(self.v(u), self.v(v), harness_weight(u, v));
        let leaf = edge_leaf(&self.tt, self.v(u), self.v(v));
        let previous = self.edge_leaves.insert(edge_key(u, v), leaf);
        assert!(previous.is_none(), "edge {u}-{v} must not already exist");
        self.adj.get_mut(&u).unwrap().insert(v);
        self.adj.get_mut(&v).unwrap().insert(u);
    }

    fn cut(&mut self, u: usize, v: usize) {
        let weight = self.tt.cut(self.v(u), self.v(v)).expect("edge must exist");
        assert_eq!(
            weight,
            harness_weight(u, v),
            "cut must return the weight passed to link"
        );
        assert!(
            self.edge_leaves.remove(&edge_key(u, v)).is_some(),
            "edge {u}-{v} must have been linked"
        );
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.adj.get_mut(&v).unwrap().remove(&u);
    }

    fn expose(&mut self, v: usize) -> Option<S> {
        self.tt.expose(self.v(v))
    }

    fn expose_path(&mut self, u: usize, v: usize) -> Option<S> {
        self.tt.expose_path(self.v(u), self.v(v))
    }

    fn deexpose(&mut self, v: usize) -> Option<S> {
        self.tt.deexpose(self.v(v))
    }

    fn component_summary(&self, v: usize) -> Option<S> {
        self.tt.component_summary(self.v(v))
    }

    fn attach(&mut self, v: usize) -> tree::LabelId {
        let cluster = self.tt.attach(self.v(v), ());
        self.tt
            .try_label_id_for_cluster(cluster)
            .expect("attached cluster must represent a label")
    }

    fn detach(&mut self, label: tree::LabelId) {
        self.tt.detach(label);
    }

    fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for &start in self.adj.keys() {
            if !seen.insert(start) {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(v) = queue.pop_front() {
                comp.push(v);
                for &w in &self.adj[&v] {
                    if seen.insert(w) {
                        queue.push_back(w);
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    fn component_edges(&self, v: usize) -> Vec<(usize, usize)> {
        let comp = self
            .components()
            .into_iter()
            .find(|c| c.contains(&v))
            .unwrap();
        let mut edges: Vec<_> = comp
            .iter()
            .flat_map(|&u| {
                self.adj[&u]
                    .iter()
                    .filter(move |&&w| u < w)
                    .map(move |&w| (u, w))
            })
            .collect();
        edges.sort();
        edges
    }
}

fn live_roots<S: Summary, N, E>(tt: &TopTree<S, N, E>) -> Vec<ClusterId> {
    let clusters = (0..tt.tree.nodes.len())
        .map(|i| tt.tree.vertex_id_from_index(i))
        .filter_map(|l| tt.find_root(l))
        .collect::<HashSet<_>>();
    clusters.into_iter().collect()
}

fn collect_leaves<S: Summary, N, E>(
    tt: &TopTree<S, N, E>,
    node: ClusterId,
    leaves: &mut Vec<ClusterId>,
    nodes: &mut usize,
) {
    *nodes += 1;
    let cluster = tt.cl(node);
    if let Some(Children { left, right }) = cluster.children {
        assert_eq!(tt.cl(left).parent, Some(node), "left child parent mismatch");
        assert_eq!(
            tt.cl(right).parent,
            Some(node),
            "right child parent mismatch"
        );
        collect_leaves(tt, left, leaves, nodes);
        collect_leaves(tt, right, leaves, nodes);
    } else {
        leaves.push(node);
    }
}

/// The boundary vertices of a node in the three "slots" used by the classic
/// splay top tree: a left boundary, a (possibly shared) middle boundary, and a
/// right boundary. At most two slots are ever populated.
#[derive(Clone, Copy, Debug, Default)]
struct Boundaries {
    left: Option<tree::VertexId>,
    mid: Option<tree::VertexId>,
    right: Option<tree::VertexId>,
}

impl Boundaries {
    fn count(&self) -> usize {
        usize::from(self.left.is_some())
            + usize::from(self.mid.is_some())
            + usize::from(self.right.is_some())
    }

    fn leftmost(&self) -> Option<tree::VertexId> {
        self.left.or(self.mid)
    }

    fn rightmost(&self) -> Option<tree::VertexId> {
        self.right.or(self.mid)
    }

    fn set(&self) -> BTreeSet<tree::VertexId> {
        [self.left, self.mid, self.right]
            .into_iter()
            .flatten()
            .collect()
    }

    fn flipped(&self) -> Self {
        Self {
            left: self.right,
            mid: self.mid,
            right: self.left,
        }
    }
}

/// Counts the leaves below `node` incident to `vertex`.
fn count_incident_leaves<S: Summary, N, E>(
    tt: &TopTree<S, N, E>,
    node: ClusterId,
    vertex: tree::VertexId,
) -> usize {
    match tt.cl(node).data {
        ClusterData::Edge { tree_id: edge, .. } => {
            let (u, v) = tt.tree.edge_endpoints(edge).expect("edge must exist");
            usize::from(u == vertex || v == vertex)
        }
        ClusterData::Node { tree_id: label, .. } => {
            usize::from(tt.tree.label_vertex(label).expect("label must exist") == vertex)
        }
        ClusterData::Internal => {
            let Children { left, right } = tt.cl(node).children.expect("internal has children");
            count_incident_leaves(tt, left, vertex) + count_incident_leaves(tt, right, vertex)
        }
    }
}

/// Recomputes the boundary vertices of `node` from scratch (from the underlying
/// forest and the exposed bits) and checks them against the stored value.
///
/// The orientation invariant from the paper is a property of the *materialized*
/// tree, i.e. the frame obtained by pushing every pending flip from the root
/// down. `parity` accumulates the flip bits of the proper ancestors of `node`,
/// so the node's effective flip is `parity ^ cluster.flipped`. The returned
/// `Boundaries` are expressed in that materialized frame; the stored value is in
/// the node's local frame and must equal the returned value flipped by the
/// effective flip.
///
/// The invariant checked for every internal node is: the rightmost boundary of
/// the materialized left child and the leftmost boundary of the materialized
/// right child must both exist and equal the central vertex.
fn check_node_boundaries<S: Summary, N, E>(tt: &TopTree<S, N, E>, node: ClusterId) -> Boundaries {
    check_node_boundaries_in(tt, node, false)
}

fn check_node_boundaries_in<S: Summary, N, E>(
    tt: &TopTree<S, N, E>,
    node: ClusterId,
    parity: bool,
) -> Boundaries {
    let cluster = tt.cl(node);
    let effective_flip = parity ^ cluster.flipped;

    let materialized = match cluster.data {
        ClusterData::Edge { tree_id: edge, .. } => {
            let (left, right) = tt.tree.edge_endpoints(edge).expect("edge must exist");
            // Materialized left endpoint is `endpoints[effective_flip]`.
            let (left, right) = if effective_flip {
                (right, left)
            } else {
                (left, right)
            };
            let mut c = Boundaries::default();
            if tt.is_boundary_vertex(left) {
                c.left = Some(left);
            }
            if tt.is_boundary_vertex(right) {
                c.right = Some(right);
            }
            c
        }
        ClusterData::Node { tree_id: label, .. } => {
            let vertex = tt.tree.label_vertex(label).expect("label must exist");
            let mut c = Boundaries::default();
            if tt.is_boundary_vertex(vertex) {
                c.mid = Some(vertex);
            }
            c
        }
        ClusterData::Internal => {
            let Children { left, right } = cluster.children.expect("internal has children");
            // Materialized left/right child of the node.
            let (materialized_left, materialized_right) = if effective_flip {
                (right, left)
            } else {
                (left, right)
            };

            let bl = check_node_boundaries_in(tt, materialized_left, effective_flip);
            let br = check_node_boundaries_in(tt, materialized_right, effective_flip);

            assert_eq!(
                bl.rightmost(),
                br.leftmost(),
                "children of internal node {node:?} must share a central boundary vertex"
            );
            let central = bl.rightmost().unwrap_or_else(|| {
                panic!("left child must have a rightmost boundary at node {node:?}")
            });

            let mut c = Boundaries::default();
            let inside = count_incident_leaves(tt, node, central);
            if tt.exposed.get(central.index()) || inside < tt.tree.degree(central) {
                c.mid = Some(central);
            }
            if bl.leftmost() != bl.rightmost() {
                c.left = bl.leftmost();
            }
            if br.leftmost() != br.rightmost() {
                c.right = br.rightmost();
            }
            c
        }
    };

    // The stored value lives in the node's local frame.
    let expected = if effective_flip {
        materialized.flipped()
    } else {
        materialized
    };
    let stored = cluster.boundary_vertices;
    assert_eq!(
        expected.count() as u8,
        stored.count(),
        "boundary count mismatch at node {node:?} ({:?})",
        tt.node_kind(node)
    );

    let expected_set: BTreeSet<tree::VertexId> = expected.set();
    let stored_set: BTreeSet<tree::VertexId> = match stored {
        BoundaryVertices::None => BTreeSet::new(),
        BoundaryVertices::One(v) => BTreeSet::from([v]),
        BoundaryVertices::Two { left, right } => BTreeSet::from([left, right]),
    };
    assert_eq!(
        expected_set,
        stored_set,
        "boundary vertices mismatch at node {node} ({:?})",
        tt.node_kind(node)
    );

    let mapped = match [expected.left, expected.mid, expected.right]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => BoundaryVertices::None,
        [v] => BoundaryVertices::One(*v),
        [l, r] => BoundaryVertices::Two {
            left: *l,
            right: *r,
        },
        other => panic!("more than two boundary vertices {other:?} at node {node:?}"),
    };
    assert_eq!(
        stored,
        mapped,
        "boundary orientation mismatch at node {node} ({:?})",
        tt.node_kind(node)
    );

    if matches!(cluster.data, ClusterData::Internal) {
        assert_eq!(
            tt.has_middle_boundary(node),
            materialized.mid.is_some(),
            "has_middle_boundary mismatch at node {node}"
        );
    }

    materialized
}

fn check_invariants<S: Summary + Clone>(h: &Harness<S>) {
    let tt = &h.tt;

    for (id, cluster) in tt.clusters.iter() {
        if let Some(parent) = cluster.parent {
            let p = tt.cl(parent);
            let is_child = p.children.is_some_and(|c| c.left == id || c.right == id);
            assert!(is_child, "node {id} is not a child of its parent {parent}");
        }
    }

    // Every live node's stored boundary vertices must match the forest.
    for root in live_roots(tt) {
        check_node_boundaries(tt, root);
    }

    let mut total_nodes = 0;
    let mut covered_edges = BTreeSet::<usize>::new();
    let mut covered_labels = BTreeSet::<usize>::new();

    for root in live_roots(tt) {
        let mut leaves = Vec::new();
        let mut nodes = 0;
        collect_leaves(tt, root, &mut leaves, &mut nodes);
        assert_eq!(
            nodes,
            leaves.len() * 2 - 1,
            "a top tree with k leaves must have 2k-1 nodes"
        );
        total_nodes += nodes;

        // The boundaries of a component root are exactly its exposed vertices.
        for vertex in check_node_boundaries(tt, root).set() {
            assert!(
                tt.exposed.get(vertex.index()),
                "root boundary vertex {vertex:?} must be exposed"
            );
        }

        for leaf in leaves {
            match tt.cl(leaf).data {
                ClusterData::Edge {
                    tree_id: edge,
                    weight,
                } => {
                    assert!(
                        covered_edges.insert(edge.index()),
                        "edge leaf appears twice"
                    );
                    // The forest edge must point back at the top tree leaf that
                    // represents it, and the leaf must still carry the weight
                    // the application supplied at `link` time.
                    let (u, v) = tt.tree.edge_endpoints(edge).expect("edge must exist");
                    let key = (u.index().min(v.index()), u.index().max(v.index()));
                    let expected =
                        h.edge_leaves.get(&key).copied().unwrap_or_else(|| {
                            panic!("edge leaf {leaf} is not a linked harness edge")
                        });
                    assert_eq!(
                        tt.tree.edge_weight(edge).copied(),
                        Some(expected),
                        "forest edge {edge:?} must name its top tree leaf"
                    );
                    assert_eq!(
                        weight,
                        harness_weight(key.0, key.1),
                        "edge leaf must keep the weight passed to link"
                    );
                }
                ClusterData::Node { tree_id: label, .. } => {
                    assert!(
                        covered_labels.insert(label.index()),
                        "label leaf appears twice"
                    );
                }
                ClusterData::Internal => panic!("leaf cannot be internal"),
            }
        }
    }

    let live_nodes = tt.clusters.iter().count();
    assert_eq!(total_nodes, live_nodes, "all live nodes must be reachable");

    assert_eq!(
        covered_edges.len(),
        tt.tree.edge_count(),
        "every tree edge must occur exactly once"
    );
    assert_eq!(
        covered_labels.len(),
        tt.tree.label_count(),
        "every label must occur exactly once"
    );

    // Verify that the edges of each top tree component match the mirror graph.
    for root in live_roots(tt) {
        let mut leaves = Vec::new();
        let mut nodes = 0;
        collect_leaves(tt, root, &mut leaves, &mut nodes);

        let mut top_edges = BTreeSet::new();
        let mut sample = None;
        for leaf in leaves {
            if let ClusterData::Edge { tree_id: edge, .. } = tt.cl(leaf).data {
                let (u, v) = tt.tree.edge_endpoints(edge).unwrap();
                top_edges.insert((u.index().min(v.index()), u.index().max(v.index())));
                sample = Some(u);
            }
        }
        if let Some(sample) = sample {
            let expected: BTreeSet<_> = h.component_edges(sample.index()).into_iter().collect();
            assert_eq!(top_edges, expected, "top tree component edges mismatch");
        }
    }
}

fn path_edges(h: &Harness<impl Summary>, u: usize, v: usize) -> Option<Vec<(usize, usize)>> {
    let mut prev: BTreeMap<usize, usize> = BTreeMap::new();
    let mut seen = BTreeSet::from([u]);
    let mut queue = VecDeque::from([u]);
    while let Some(x) = queue.pop_front() {
        if x == v {
            let mut edges = Vec::new();
            let mut cur = v;
            while cur != u {
                let p = prev[&cur];
                edges.push((p.min(cur), p.max(cur)));
                cur = p;
            }
            edges.reverse();
            return Some(edges);
        }
        for &w in &h.adj[&x] {
            if seen.insert(w) {
                prev.insert(w, x);
                queue.push_back(w);
            }
        }
    }
    None
}

#[test]
fn single_edge_summary() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);

    assert_eq!(h.expose(0), Some(PathLen { len: 1 }));
    check_invariants(&h);
    assert_eq!(h.expose(1), Some(PathLen { len: 1 }));
    check_invariants(&h);

    h.cut(0, 1);
    assert_eq!(h.tt.edge_count(), 0);
    check_invariants(&h);
}

#[test]
fn path_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
        check_invariants(&h);
    }

    for (u, v, expected) in [(0, 5, 5), (0, 3, 3), (2, 5, 3), (4, 4, 0)] {
        assert_eq!(h.expose_path(u, v), Some(PathLen { len: expected }));
        check_invariants(&h);
        h.deexpose(v);
        h.deexpose(u);
        check_invariants(&h);
    }

    for i in (1..6).rev() {
        h.cut(i - 1, i);
        check_invariants(&h);
    }
}

#[test]
fn update_label_summary_recomputes_ancestors_without_relinking() {
    let mut h = Harness::<LabelValue>::new(2);
    h.link(0, 1);
    let label = h.attach(0);

    let roots_before = live_roots(&h.tt);
    let structure_before: Vec<_> =
        h.tt.clusters
            .iter()
            .map(|(id, cluster)| (id, cluster.parent, cluster.children, cluster.data))
            .collect();

    h.tt.update_label_summary(label, |summary| summary.value = 7);

    assert_eq!(live_roots(&h.tt), roots_before);
    let structure_after: Vec<_> =
        h.tt.clusters
            .iter()
            .map(|(id, cluster)| (id, cluster.parent, cluster.children, cluster.data))
            .collect();
    assert_eq!(structure_after, structure_before);
    check_invariants(&h);

    assert_eq!(
        h.expose_path(0, 1),
        Some(LabelValue { value: 7 }),
        "the exposed path aggregate must include the updated label summary"
    );
    check_invariants(&h);
}

/// The node-inspection surface a guided descent relies on, the per-vertex label
/// lookup, and the by-cluster variant of a label summary update. Both label and
/// edge leaves carry application weights here, so `leaf_weight` and
/// `leaf_weight_mut` are exercised on both `ClusterWeight` variants.
#[test]
fn node_inspection_and_label_lookup() {
    let mut tt: TopTree<LabelValue, u64, u64> = TopTree::new();
    let v: Vec<_> = (0..4).map(|_| tt.add_vertex()).collect();

    tt.link(v[0], v[1], 10);
    tt.link(v[0], v[2], 20);
    let label0 = tt.attach(v[0], 100);
    let label2 = tt.attach(v[2], 200);

    let leaf_01 = edge_leaf(&tt, v[0], v[1]);
    let leaf_02 = edge_leaf(&tt, v[0], v[2]);
    assert_ne!(leaf_01, leaf_02, "each edge is represented by its own leaf");

    // Connectivity, by vertex and by cluster.
    assert!(tt.connected(v[0], v[1]));
    assert!(tt.connected(v[1], v[2]));
    assert!(!tt.connected(v[0], v[3]));
    assert!(tt.connected_clusters(leaf_01, leaf_02));
    assert!(tt.connected_clusters(leaf_01, label2));
    let label3 = tt.attach(v[3], 1);
    assert!(!tt.connected_clusters(leaf_01, label3));

    // Leaf identification.
    assert_eq!(tt.node_kind(leaf_01), NodeKind::Edge);
    assert_eq!(tt.node_kind(label0), NodeKind::Label);
    assert_eq!(tt.edge_endpoints(leaf_01), Some((v[0], v[1])));
    assert_eq!(
        tt.edge_endpoints(leaf_02),
        Some((v[0], v[2])),
        "endpoints follow the forest, not the cluster orientation"
    );
    assert_eq!(
        tt.edge_endpoints(label0),
        None,
        "a label leaf is not a tree edge"
    );
    assert_eq!(
        tt.label_vertex(tt.try_label_id_for_cluster(label2).expect("label leaf")),
        Some(v[2])
    );
    assert_eq!(
        tt.try_label_id_for_cluster(leaf_01),
        None,
        "an edge leaf is not a label"
    );

    // Application weights on both leaf kinds, readable and writable.
    assert_eq!(tt.leaf_weight(leaf_02).into_edge(), Some(&20));
    assert_eq!(tt.leaf_weight(label0).into_label(), Some(&100));
    let ClusterWeight::Edge(weight) = tt.leaf_weight_mut(leaf_01) else {
        panic!("an edge leaf must report an edge weight");
    };
    *weight = 11;
    assert_eq!(tt.leaf_weight(leaf_01).into_edge(), Some(&11));
    let ClusterWeight::Label(weight) = tt.leaf_weight_mut(label0) else {
        panic!("a label leaf must report a label weight");
    };
    *weight = 101;
    assert_eq!(tt.leaf_weight(label0).into_label(), Some(&101));
    assert_eq!(
        tt.leaf_weight(leaf_01).into_label(),
        None,
        "an edge leaf carries no label weight"
    );

    // Label lookup by vertex.
    assert_eq!(tt.first_label(v[0]), Some(label0));
    assert_eq!(tt.first_label(v[2]), Some(label2));
    assert_eq!(tt.first_label(v[1]), None, "a bare vertex has no label");
    assert_eq!(
        tt.label_clusters(v[0]).collect::<BTreeSet<_>>(),
        BTreeSet::from([label0])
    );
    assert_eq!(tt.label_clusters(v[1]).count(), 0);

    // Inspection of the exposed path, using only logical (flip-aware) accessors.
    let root = tt
        .expose_path_node(v[1], v[2])
        .expect("path 1..2 is exposable");
    assert!(tt.node_is_path(root));
    assert_eq!(
        tt.node_boundary(root),
        Boundary::Two {
            left: v[1],
            right: v[2]
        }
    );
    assert_eq!(
        *tt.node_summary(root),
        LabelValue { value: 2 },
        "LabelValue counts every leaf, so only the two labels contribute"
    );

    // The exposed root rakes the label at `v[2]` off the path: labels are never
    // path children, so it becomes a point cluster joined at the right
    // boundary, and the path cluster on the left is what spans `v[1]`..`v[2]`.
    let (left, right) = tt
        .node_children(root)
        .expect("an exposed path root is internal");
    assert_eq!(tt.node_kind(right), NodeKind::Label);
    assert_eq!(tt.node_boundary(right), Boundary::One(v[2]));
    assert_eq!(
        tt.label_vertex(tt.try_label_id_for_cluster(right).expect("label leaf")),
        Some(v[2])
    );
    assert_eq!(tt.node_central(right), None, "a leaf has no central vertex");

    assert_eq!(tt.node_kind(left), NodeKind::Internal);
    assert!(tt.node_is_path(left));
    assert_eq!(
        tt.node_boundary(left),
        Boundary::Two {
            left: v[1],
            right: v[2]
        }
    );
    assert_eq!(
        tt.node_central(left),
        Some(v[0]),
        "the two edges of the path meet at v[0]"
    );

    let mut leaves = Vec::new();
    let mut nodes = 0;
    collect_leaves(&tt, root, &mut leaves, &mut nodes);
    assert_eq!(nodes, leaves.len() * 2 - 1);
    assert_eq!(
        leaves.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([leaf_01, leaf_02, label0, label2]),
        "the exposed path root covers both edges and both labels"
    );

    // A label summary update by cluster leaves the structure untouched and every
    // ancestor, including the exposed path root, sees the new value.
    let structure_before: Vec<_> = tt
        .clusters
        .iter()
        .map(|(id, cluster)| (id, cluster.parent, cluster.children))
        .collect();
    tt.update_label_summary_by_cluster(label0, |summary| summary.value = 5);
    assert_eq!(
        tt.clusters
            .iter()
            .map(|(id, cluster)| (id, cluster.parent, cluster.children))
            .collect::<Vec<_>>(),
        structure_before
    );
    assert_eq!(
        *tt.node_summary(root),
        LabelValue { value: 6 },
        "the updated label contributes 5 and the other label 1"
    );
    assert_eq!(tt.component_summary(v[1]), Some(LabelValue { value: 6 }));
    assert_eq!(
        tt.expose_path(v[2], v[1]),
        Some(LabelValue { value: 6 }),
        "the reversed orientation covers the same leaves"
    );
    tt.deexpose(v[1]);
    tt.deexpose(v[2]);

    // The other entry point identifies the leaf by the forest's `LabelId`.
    let key0 = tt.try_label_id_for_cluster(label0).expect("label leaf");
    tt.update_label_summary(key0, |summary| summary.value = 9);
    assert_eq!(
        tt.component_summary(v[3]),
        Some(LabelValue { value: 1 }),
        "the isolated vertex only has its own label"
    );
    assert_eq!(tt.component_summary(v[1]), Some(LabelValue { value: 10 }));
}

#[test]
fn orientation_sensitive_summary_flips_with_cluster() {
    let mut tt: TopTree<DirectedPath> = TopTree::new();
    let u = tt.add_vertex();
    let v = tt.add_vertex();
    let w = tt.add_vertex();
    tt.link(u, v, ());
    tt.link(v, w, ());

    // The payload of a forest edge is the `ClusterId` of its top tree leaf.
    let uv = edge_leaf(&tt, u, v);
    let vw = edge_leaf(&tt, v, w);

    // A two-edge path, so that reversing the cluster is visible in the
    // sequence of reported edges: a one-edge leaf holds a single element and
    // cannot visibly reverse.
    let path = tt
        .expose_path_node(u, w)
        .expect("the exposed path has a cluster");
    assert_eq!(tt.cl(path).sum.edges, vec![uv, vw]);

    tt.toggle_flipped(path);
    assert_eq!(tt.cl(path).sum.edges, vec![vw, uv]);

    // Materializing the lazy flip must not reverse the already-flipped sum again.
    tt.push_flip(path);
    assert_eq!(tt.cl(path).sum.edges, vec![vw, uv]);

    tt.toggle_flipped(path);
    assert_eq!(tt.cl(path).sum.edges, vec![uv, vw]);
}

#[test]
fn star_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(0, i);
        check_invariants(&h);
    }

    for i in 1..6 {
        assert_eq!(h.expose_path(0, i), Some(PathLen { len: 1 }));
        h.deexpose(i);
        h.deexpose(0);
        assert_eq!(h.expose_path(i, 0), Some(PathLen { len: 1 }));
        h.deexpose(0);
        h.deexpose(i);
        check_invariants(&h);
    }
}

#[test]
fn component_aggregate_is_whole_component() {
    let mut h = Harness::<Agg>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
    }

    let expected_xor = (1..6).fold(0, |acc, i| acc ^ h.leaf(i - 1, i).index() as u64);
    let summary = h.expose_path(0, 5).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.deexpose(5);
    h.deexpose(0);

    // Even exposing a sub-path returns the whole component's summary, since
    // the root cluster still contains the whole tree.
    let summary = h.expose_path(1, 3).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.deexpose(3);
    h.deexpose(1);

    assert_eq!(
        h.component_summary(4).unwrap(),
        Agg {
            edges: 5,
            xor: expected_xor
        }
    );
}

#[test]
fn path_max_summary() {
    let mut h = Harness::<PathMax>::new(6);
    for i in 1..6 {
        h.link(i - 1, i);
    }
    for i in 1..6 {
        for j in i + 1..6 {
            let max = h
                .path_leaf_values(i, j)
                .into_iter()
                .max()
                .expect("the path must contain at least one edge");
            assert_eq!(h.expose_path(i, j), Some(PathMax { max }));
            h.deexpose(j);
            h.deexpose(i);
        }
    }
}

#[test]
fn lazy_path_tag_propagates() {
    let mut tt: TopTree<PathSum> = TopTree::new();
    let vertices: Vec<_> = (0..5).map(|_| tt.add_vertex()).collect();
    // `leaves[k]` is the leaf of edge `k` (between positions `k` and `k + 1`).
    let mut leaves = Vec::new();
    for i in 1..5 {
        tt.link(vertices[i - 1], vertices[i], ());
        leaves.push(edge_leaf(&tt, vertices[i - 1], vertices[i]));
    }
    let weight = |edge: usize| leaves[edge].index() as i64;

    // Path 1..3 has edges 1 and 2.
    assert_eq!(
        tt.expose_path(vertices[1], vertices[3]),
        Some(PathSum {
            sum: weight(1) + weight(2),
            len: 2
        })
    );
    tt.deexpose(vertices[3]);
    tt.deexpose(vertices[1]);

    // Add 10 to every edge of path 1..3.
    let tagged = tt.expose_path_tagged(vertices[1], vertices[3], 10);
    assert_eq!(
        tagged,
        Some(PathSum {
            sum: weight(1) + weight(2) + 20,
            len: 2
        })
    );
    tt.deexpose(vertices[3]);
    tt.deexpose(vertices[1]);

    // The tag must be visible from a different path covering the same edges.
    assert_eq!(
        tt.expose_path(vertices[0], vertices[4]),
        Some(PathSum {
            sum: weight(0) + weight(1) + 10 + weight(2) + 10 + weight(3),
            len: 4
        })
    );

    // A guided descent reaches the same conclusion per leaf: pushing the
    // pending tags down must leave the two tagged edges at their own weight
    // plus 10 and the untagged ones alone.
    let root = tt
        .expose_path_node(vertices[0], vertices[4])
        .expect("path 0..4 must be exposed");
    let mut stack = vec![root];
    let mut leaf_sums = Vec::new();
    while let Some(node) = stack.pop() {
        tt.push_node_tag(node);
        match tt.node_children(node) {
            Some((left, right)) => {
                stack.push(right);
                stack.push(left);
            }
            None => leaf_sums.push(*tt.node_summary(node)),
        }
    }
    leaf_sums.sort_by_key(|sum| sum.sum);
    let mut expected_leaf_sums = vec![
        PathSum {
            sum: weight(0),
            len: 1,
        },
        PathSum {
            sum: weight(1) + 10,
            len: 1,
        },
        PathSum {
            sum: weight(2) + 10,
            len: 1,
        },
        PathSum {
            sum: weight(3),
            len: 1,
        },
    ];
    expected_leaf_sums.sort_by_key(|sum| sum.sum);
    assert_eq!(leaf_sums, expected_leaf_sums);

    tt.deexpose(vertices[4]);
    tt.deexpose(vertices[0]);

    assert_eq!(
        tt.expose_path(vertices[1], vertices[2]),
        Some(PathSum {
            sum: weight(1) + 10,
            len: 1
        })
    );
    tt.deexpose(vertices[2]);
    tt.deexpose(vertices[1]);
}

#[test]
fn labels_count_towards_component() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);
    let label = h.attach(0);
    check_invariants(&h);

    assert_eq!(h.tt.label_count(), 1);
    // With no exposed boundary vertices the root is a point cluster, so the
    // path length is zero; exposing the two endpoints gives the path.
    assert_eq!(h.component_summary(0), Some(PathLen { len: 0 }));
    assert_eq!(h.expose_path(0, 1), Some(PathLen { len: 1 }));
    h.deexpose(1);
    h.deexpose(0);

    h.detach(label);
    assert_eq!(h.tt.label_count(), 0);
    check_invariants(&h);
}

#[test]
fn label_on_isolated_vertex() {
    let mut h = Harness::<PathLen>::new(2);
    let label = h.attach(0);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 1);

    h.detach(label);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 0);
}

#[test]
fn link_cut_and_expose_paths() {
    let mut h = Harness::<PathLen>::new(10);
    for i in 1..10 {
        h.link(i - 1, i);
    }
    for u in 0..10 {
        for v in u + 1..10 {
            let expected = (v - u) as u32;
            assert_eq!(
                h.expose_path(u, v),
                Some(PathLen { len: expected }),
                "path length mismatch for {u}..{v}"
            );
            check_invariants(&h);
            h.deexpose(v);
            h.deexpose(u);
            check_invariants(&h);
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

#[test]
fn randomized_link_cut() {
    for seed in 0..25 {
        let mut rng = Rng(seed);
        let mut h = Harness::<PathLen>::new(9);

        for _ in 0..120 {
            let comps = h.components();
            let edges: Vec<(usize, usize)> = h
                .adj
                .iter()
                .flat_map(|(&u, nbrs)| nbrs.iter().map(move |&v| (u, v)))
                .filter(|(u, v)| u < v)
                .collect();

            match rng.next() % 3 {
                0 if comps.len() >= 2 => {
                    let a = (rng.next() as usize) % comps.len();
                    let mut b = (rng.next() as usize) % (comps.len() - 1);
                    if b >= a {
                        b += 1;
                    }
                    let u = comps[a][(rng.next() as usize) % comps[a].len()];
                    let v = comps[b][(rng.next() as usize) % comps[b].len()];
                    h.link(u, v);
                }
                1 if !edges.is_empty() => {
                    let (u, v) = edges[(rng.next() as usize) % edges.len()];
                    h.cut(u, v);
                }
                _ => {
                    let a = (rng.next() as usize) % 9;
                    let b = (rng.next() as usize) % 9;
                    if a != b
                        && let Some(path) = path_edges(&h, a, b)
                    {
                        assert_eq!(
                            h.expose_path(a, b),
                            Some(PathLen {
                                len: path.len() as u32
                            })
                        );
                        h.deexpose(b);
                        h.deexpose(a);
                    }
                }
            }

            check_invariants(&h);
        }
    }
}

#[test]
fn bit_vec_packs_bits() {
    let mut bits = BitVec::new();
    assert!(!bits.get(0));

    bits.grow_to(200);
    bits.set(0, true);
    bits.set(64, true);
    bits.set(65, true);
    bits.set(199, true);

    assert!(bits.get(0));
    assert!(!bits.get(1));
    assert!(bits.get(64));
    assert!(bits.get(65));
    assert!(bits.get(199));

    bits.set(64, false);
    assert!(!bits.get(64));
    assert!(bits.get(65));

    assert_eq!(bits.blocks.len(), 4);
}

/// Regression test: removing a label must undo the boundary vertices that
/// attaching it added, even though the boundaries are maintained relative to
/// each cluster's parent context.
#[test]
fn detach_restores_boundary_count() {
    // A vertex whose degree drops from 2 to 1 stops being a boundary.
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1);
    let l = h.attach(0);
    h.detach(l);
    check_invariants(&h);
    for root in live_roots(&h.tt) {
        assert_eq!(
            h.tt.cl(root).boundary_vertices,
            BoundaryVertices::None,
            "stale boundary after detach"
        );
    }

    // A vertex that remains a boundary must not be over-decremented.
    let mut h = Harness::<PathLen>::new(3);
    h.link(0, 1);
    h.link(0, 2);
    let l = h.attach(0);
    h.detach(l);
    check_invariants(&h);
    for root in live_roots(&h.tt) {
        assert_eq!(
            h.tt.cl(root).boundary_vertices,
            BoundaryVertices::None,
            "vertex 0 is internal to the merged edge cluster"
        );
    }
}
