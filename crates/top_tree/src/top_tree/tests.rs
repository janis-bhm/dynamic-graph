use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::*;

/// A monoid summary over the whole cluster: number of tree edges and xor of
/// their weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Agg {
    edges: u32,
    xor: u64,
}

impl Summary<u64> for Agg {
    type Tag = ();

    fn tree_edge(w: &u64, _u: usize, _v: usize) -> Self {
        Agg { edges: 1, xor: *w }
    }

    fn label(_w: &u64, _v: usize) -> Self {
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

impl Summary<u64> for PathLen {
    type Tag = ();

    fn tree_edge(_w: &u64, _u: usize, _v: usize) -> Self {
        PathLen { len: 1 }
    }

    fn label(_w: &u64, _v: usize) -> Self {
        PathLen { len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary == 2 {
            PathLen {
                len: u32::from(ctx.left_is_path_child()) * left.len
                    + u32::from(ctx.right_is_path_child()) * right.len,
            }
        } else {
            PathLen { len: 0 }
        }
    }
}

/// The sum of the edge weights on the cluster path, with a lazy "add `x` to
/// every edge on the path" tag. This is the mechanism used by cover-level
/// style algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathSum {
    sum: i64,
    len: u32,
}

impl Summary<i64> for PathSum {
    type Tag = i64;

    fn tree_edge(w: &i64, _u: usize, _v: usize) -> Self {
        PathSum { sum: *w, len: 1 }
    }

    fn label(_w: &i64, _v: usize) -> Self {
        PathSum { sum: 0, len: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary == 2 {
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

/// The maximum edge weight on the cluster path (`0` for point clusters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathMax {
    max: u64,
}

impl Summary<u64> for PathMax {
    type Tag = ();

    fn tree_edge(w: &u64, _u: usize, _v: usize) -> Self {
        PathMax { max: *w }
    }

    fn label(_w: &u64, _v: usize) -> Self {
        PathMax { max: 0 }
    }

    fn combine(left: &Self, right: &Self, ctx: &MergeContext) -> Self {
        if ctx.boundary == 2 {
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectedPath {
    edges: Vec<(usize, usize)>,
}

impl Summary<u64> for DirectedPath {
    type Tag = ();

    fn tree_edge(_w: &u64, u: usize, v: usize) -> Self {
        DirectedPath {
            edges: vec![(u, v)],
        }
    }

    fn label(_w: &u64, _v: usize) -> Self {
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
        self.edges = self.edges.drain(..).rev().map(|(u, v)| (v, u)).collect();
    }
}

struct Harness<S: Summary<u64>> {
    tt: TopTree<u32, S, u64, ()>,
    adj: BTreeMap<usize, BTreeMap<usize, u64>>,
}

impl<S: Summary<u64>> Harness<S> {
    fn new(n: usize) -> Self {
        let mut tt = TopTree::new();
        let mut adj = BTreeMap::new();
        for i in 0..n {
            tt.add_vertex(i as u32, ());
            adj.insert(i, BTreeMap::new());
        }
        Self { tt, adj }
    }

    fn link(&mut self, u: usize, v: usize, w: u64) {
        self.tt.link(u, v, w);
        self.adj.get_mut(&u).unwrap().insert(v, w);
        self.adj.get_mut(&v).unwrap().insert(u, w);
    }

    fn cut(&mut self, u: usize, v: usize) {
        self.tt.cut(u, v);
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.adj.get_mut(&v).unwrap().remove(&u);
    }

    fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for &start in self.adj.keys() {
            if !seen.insert(start) {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(v) = queue.pop_front() {
                comp.push(v);
                for &w in self.adj[&v].keys() {
                    if seen.insert(w) {
                        queue.push_back(w);
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    fn component_edges(&self, v: usize) -> Vec<(usize, usize, u64)> {
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
                    .map(|(&w, &weight)| (w, weight))
                    .filter(move |&(w, _)| u < w)
                    .map(move |(w, weight)| (u, w, weight))
            })
            .collect();
        edges.sort();
        edges
    }
}

fn live_roots<S: Summary<u64>>(tt: &TopTree<u32, S, u64, ()>) -> Vec<usize> {
    tt.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            node.as_ref()
                .filter(|c| c.parent.get().is_none())
                .map(|_| i)
        })
        .collect()
}

fn collect_leaves<S: Summary<u64>>(
    tt: &TopTree<u32, S, u64, ()>,
    node: usize,
    leaves: &mut Vec<usize>,
    nodes: &mut usize,
) {
    *nodes += 1;
    let cluster = tt.cl(node);
    if let (Some(left), Some(right)) = (cluster.left.get(), cluster.right.get()) {
        assert_eq!(
            tt.cl(left).parent.get(),
            Some(node),
            "left child parent mismatch"
        );
        assert_eq!(
            tt.cl(right).parent.get(),
            Some(node),
            "right child parent mismatch"
        );
        collect_leaves(tt, left, leaves, nodes);
        collect_leaves(tt, right, leaves, nodes);
    } else {
        assert!(cluster.left.get().is_none() && cluster.right.get().is_none());
        leaves.push(node);
    }
}

fn check_invariants<S: Summary<u64>>(h: &Harness<S>) {
    let tt = &h.tt;

    for (i, node) in tt.nodes.iter().enumerate() {
        if let Some(cluster) = node
            && let Some(parent) = cluster.parent.get()
        {
            let p = tt.nodes[parent].as_ref().expect("parent must be live");
            assert!(
                p.left.get() == Some(i) || p.right.get() == Some(i),
                "node {i} is not a child of its parent {parent}"
            );
        }
    }

    let mut total_nodes = 0;
    let mut covered_edges = BTreeSet::new();
    let mut covered_labels = BTreeSet::new();

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

        for leaf in leaves {
            match tt.cl(leaf).data {
                ClusterData::Edge(edge) => {
                    assert!(covered_edges.insert(edge), "edge leaf appears twice");
                }
                ClusterData::Label(label) => {
                    assert!(covered_labels.insert(label), "label leaf appears twice");
                }
                ClusterData::Internal => panic!("leaf cannot be internal"),
            }
        }
    }

    let live_nodes = tt.nodes.iter().filter(|n| n.is_some()).count();
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
            if let ClusterData::Edge(edge) = tt.cl(leaf).data {
                let (u, v) = tt.tree.edge_endpoints(edge).unwrap();
                top_edges.insert((u.min(v), u.max(v)));
                sample = Some(u);
            }
        }
        if let Some(sample) = sample {
            let expected: BTreeSet<_> = h
                .component_edges(sample)
                .into_iter()
                .map(|(u, v, _)| (u, v))
                .collect();
            assert_eq!(top_edges, expected, "top tree component edges mismatch");
        }
    }
}

fn path_edges(h: &Harness<impl Summary<u64>>, u: usize, v: usize) -> Option<Vec<u64>> {
    let mut prev: BTreeMap<usize, (usize, u64)> = BTreeMap::new();
    let mut seen = BTreeSet::from([u]);
    let mut queue = VecDeque::from([u]);
    while let Some(x) = queue.pop_front() {
        if x == v {
            let mut weights = Vec::new();
            let mut cur = v;
            while cur != u {
                let (p, w) = prev[&cur];
                weights.push(w);
                cur = p;
            }
            weights.reverse();
            return Some(weights);
        }
        for (&w, &weight) in &h.adj[&x] {
            if seen.insert(w) {
                prev.insert(w, (x, weight));
                queue.push_back(w);
            }
        }
    }
    None
}

#[test]
fn single_edge_summary() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1, 42);

    assert_eq!(h.tt.expose(0), Some(PathLen { len: 1 }));
    check_invariants(&h);
    assert_eq!(h.tt.expose(1), Some(PathLen { len: 1 }));
    check_invariants(&h);

    h.tt.cut(0, 1);
    assert_eq!(h.tt.edge_count(), 0);
    check_invariants(&h);
}

#[test]
fn path_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(i - 1, i, i as u64);
        check_invariants(&h);
    }

    for (u, v, expected) in [(0, 5, 5), (0, 3, 3), (2, 5, 3), (4, 4, 0)] {
        assert_eq!(h.tt.expose_path(u, v), Some(PathLen { len: expected }));
        check_invariants(&h);
        h.tt.deexpose(v);
        h.tt.deexpose(u);
        check_invariants(&h);
    }

    for i in (1..6).rev() {
        h.cut(i - 1, i);
        check_invariants(&h);
    }
}

#[test]
fn orientation_sensitive_summary_flips_with_cluster() {
    let mut tt: TopTree<u32, DirectedPath, u64, ()> = TopTree::new();
    let u = tt.add_vertex(0, ());
    let v = tt.add_vertex(1, ());
    tt.link(u, v, 1);

    let leaf = tt.edge_leaf[0].get().unwrap();
    tt.toggle_flipped(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(1, 0)]);

    // Materializing the lazy flip must not reverse the already-flipped sum again.
    tt.push_flip(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(1, 0)]);

    tt.toggle_flipped(leaf);
    assert_eq!(tt.cl(leaf).sum.edges, vec![(0, 1)]);
}

#[test]
fn star_summary() {
    let mut h = Harness::<PathLen>::new(6);
    for i in 1..6 {
        h.link(0, i, i as u64);
        check_invariants(&h);
    }

    for i in 1..6 {
        assert_eq!(h.tt.expose_path(0, i), Some(PathLen { len: 1 }));
        h.tt.deexpose(i);
        h.tt.deexpose(0);
        assert_eq!(h.tt.expose_path(i, 0), Some(PathLen { len: 1 }));
        h.tt.deexpose(0);
        h.tt.deexpose(i);
        check_invariants(&h);
    }
}

#[test]
fn component_aggregate_is_whole_component() {
    let mut h = Harness::<Agg>::new(6);
    for i in 1..6 {
        h.link(i - 1, i, i as u64);
    }

    let expected_xor = (1..6u64).fold(0, |a, b| a ^ b);
    let summary = h.tt.expose_path(0, 5).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.tt.deexpose(5);
    h.tt.deexpose(0);

    // Even exposing a sub-path returns the whole component's summary, since
    // the root cluster still contains the whole tree.
    let summary = h.tt.expose_path(1, 3).unwrap();
    assert_eq!(summary.edges, 5);
    assert_eq!(summary.xor, expected_xor);
    h.tt.deexpose(3);
    h.tt.deexpose(1);

    assert_eq!(
        h.tt.component_summary(4).unwrap(),
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
        h.link(i - 1, i, (i as u64) * 10);
    }
    for i in 1..6 {
        for j in i + 1..6 {
            let max = (i..j).map(|k| (k as u64 + 1) * 10).max().unwrap();
            assert_eq!(h.tt.expose_path(i, j), Some(PathMax { max }));
            h.tt.deexpose(j);
            h.tt.deexpose(i);
        }
    }
}

#[test]
fn lazy_path_tag_propagates() {
    let mut tt: TopTree<u32, PathSum, i64, ()> = TopTree::new();
    for i in 0..5 {
        tt.add_vertex(i as u32, ());
    }
    for i in 1..5 {
        tt.link(i - 1, i, i as i64);
    }

    // Path 1..3 has edges 2 and 3.
    assert_eq!(tt.expose_path(1, 3), Some(PathSum { sum: 5, len: 2 }));
    tt.deexpose(3);
    tt.deexpose(1);

    // Add 10 to every edge of path 1..3.
    let tagged = tt.expose_path_tagged(1, 3, 10);
    assert_eq!(tagged, Some(PathSum { sum: 25, len: 2 }));
    tt.deexpose(3);
    tt.deexpose(1);

    // The tag must be visible from a different path covering the same edges.
    assert_eq!(
        tt.expose_path(0, 4),
        Some(PathSum {
            sum: 1 + 12 + 13 + 4,
            len: 4
        })
    );
    tt.deexpose(4);
    tt.deexpose(0);

    assert_eq!(tt.expose_path(1, 2), Some(PathSum { sum: 12, len: 1 }));
    tt.deexpose(2);
    tt.deexpose(1);
}

#[test]
fn labels_count_towards_component() {
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1, 5);
    let label = h.tt.attach(0, 9);
    check_invariants(&h);

    assert_eq!(h.tt.label_count(), 1);
    // With no exposed boundary vertices the root is a point cluster, so the
    // path length is zero; exposing the two endpoints gives the path.
    assert_eq!(h.tt.component_summary(0), Some(PathLen { len: 0 }));
    assert_eq!(h.tt.expose_path(0, 1), Some(PathLen { len: 1 }));
    h.tt.deexpose(1);
    h.tt.deexpose(0);

    h.tt.detach(label);
    assert_eq!(h.tt.label_count(), 0);
    check_invariants(&h);
}

#[test]
fn label_on_isolated_vertex() {
    let mut h = Harness::<PathLen>::new(2);
    let label = h.tt.attach(0, 5);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 1);

    h.tt.detach(label).expect("label exists");
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 0);
}

#[test]
fn stable_label_handles() {
    let mut h = Harness::<PathLen>::new(3);
    h.link(0, 1, 7);

    let a = h.tt.attach(0, 1);
    let b = h.tt.attach(0, 2);
    let c = h.tt.attach(1, 3);
    check_invariants(&h);
    assert_eq!(h.tt.label_count(), 3);

    // Removing `b` must not invalidate the other handles even though the
    // underlying label storage swap-removes.
    h.tt.detach(b).expect("b exists");
    assert_eq!(h.tt.label_count(), 2);
    check_invariants(&h);

    h.tt.detach(a).expect("a still exists");
    check_invariants(&h);
    h.tt.detach(c).expect("c still exists");
    assert_eq!(h.tt.label_count(), 0);
    check_invariants(&h);
}

#[test]
fn link_cut_and_expose_paths() {
    let mut h = Harness::<PathLen>::new(10);
    for i in 1..10 {
        h.link(i - 1, i, (i as u64) * 7 + 1);
    }
    for u in 0..10 {
        for v in u + 1..10 {
            let expected = (v - u) as u32;
            assert_eq!(
                h.tt.expose_path(u, v),
                Some(PathLen { len: expected }),
                "path length mismatch for {u}..{v}"
            );
            check_invariants(&h);
            h.tt.deexpose(v);
            h.tt.deexpose(u);
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
                .flat_map(|(&u, nbrs)| nbrs.keys().map(move |&v| (u, v)))
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
                    h.link(u, v, rng.next());
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
                            h.tt.expose_path(a, b),
                            Some(PathLen {
                                len: path.len() as u32
                            })
                        );
                        h.tt.deexpose(b);
                        h.tt.deexpose(a);
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

/// Regression test: removing a label must undo the boundary count that
/// attaching it added, even though `num_boundary` is maintained relative to
/// each cluster's parent context.
#[test]
fn detach_restores_boundary_count() {
    // A vertex whose degree drops from 2 to 1 stops being a boundary.
    let mut h = Harness::<PathLen>::new(2);
    h.link(0, 1, 1);
    let label = h.tt.attach(0, 9);
    h.tt.detach(label);

    let boundary_counts: Vec<u8> =
        h.tt.nodes
            .iter()
            .filter_map(|node| node.as_ref().map(|c| c.num_boundary))
            .collect();
    assert_eq!(boundary_counts, vec![0], "stale boundary after detach");
    check_invariants(&h);

    // A vertex that remains a boundary must not be over-decremented.
    let mut h = Harness::<PathLen>::new(3);
    h.link(0, 1, 1);
    h.link(0, 2, 1);
    let label = h.tt.attach(0, 9);
    h.tt.detach(label);

    let root = live_roots(&h.tt).pop().expect("a root must exist");
    assert_eq!(
        h.tt.cl(root).num_boundary,
        0,
        "vertex 0 is internal to the merged edge cluster"
    );
    check_invariants(&h);
}
