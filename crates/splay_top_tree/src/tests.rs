#![cfg(test)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::tree::{self, EdgeKey};
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Xor(u64);

impl Reduce for Xor {
    fn reduce(&self, other: &Self) -> Self {
        Xor(self.0 ^ other.0)
    }
}

type NN = std::ptr::NonNull<Node<Xor>>;

struct Harness {
    tree: tree::Tree<Xor>,
    adj: BTreeMap<Index, BTreeMap<Index, u64>>,
    exposed: BTreeSet<Index>,
}

impl Harness {
    fn new(num_vertices: usize) -> Self {
        let mut tree = tree::Tree::new();
        let mut adj = BTreeMap::new();
        for _ in 0..num_vertices {
            let v = tree.add_vertex();
            adj.insert(v, BTreeMap::new());
        }
        Self {
            tree,
            adj,
            exposed: BTreeSet::new(),
        }
    }

    fn vertices(&self) -> impl Iterator<Item = Index> + '_ {
        self.adj.keys().copied()
    }

    fn link(&mut self, u: Index, v: Index, w: u64) {
        debug_assert_ne!(u, v);
        link(u, v, Xor(w), &mut self.tree);
        self.adj.get_mut(&u).unwrap().insert(v, w);
        self.adj.get_mut(&v).unwrap().insert(u, w);
    }

    fn cut(&mut self, u: Index, v: Index) {
        cut(u, v, &mut self.tree);
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.adj.get_mut(&v).unwrap().remove(&u);
    }

    fn expose(&mut self, v: Index) {
        let returned = expose(v, &mut self.tree);
        if let Some(root) = returned {
            let root_of_component = self.edge_leaves_of(v).next().map(climb).unwrap();
            assert_eq!(root.node, root_of_component, "expose must return the root");
        }
        self.exposed.insert(v);
    }

    fn deexpose(&mut self, v: Index) {
        deexpose(v, &mut self.tree);
        self.exposed.remove(&v);
    }

    fn deexpose_all(&mut self) {
        let exposed: Vec<_> = self.exposed.iter().copied().collect();
        for v in exposed {
            self.deexpose(v);
        }
    }

    fn incident(&self, v: Index) -> Vec<Index> {
        self.adj.get(&v).unwrap().keys().copied().collect()
    }

    fn edge_leaves_of<'a>(&'a self, v: Index) -> impl Iterator<Item = NN> + 'a {
        self.tree
            .incident_edges(v)
            .map(|e| e.edge_node.cast::<Node<Xor>>())
    }

    fn components(&self) -> Vec<Vec<Index>> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for start in self.vertices() {
            if seen.contains(&start) {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([start]);
            seen.insert(start);
            while let Some(v) = queue.pop_front() {
                comp.push(v);
                for w in self.incident(v) {
                    if seen.insert(w) {
                        queue.push_back(w);
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    fn comp_of(&self, v: Index) -> Vec<Index> {
        self.components()
            .into_iter()
            .find(|c| c.contains(&v))
            .unwrap()
    }

    fn has_exposed(&self, comp: &[Index]) -> bool {
        comp.iter().any(|v| self.exposed.contains(v))
    }

    fn exposed_count(&self, comp: &[Index]) -> usize {
        comp.iter().filter(|v| self.exposed.contains(v)).count()
    }
}

fn climb(mut node: NN) -> NN {
    loop {
        let parent = unsafe { &*node.as_ptr() }.parent_node();
        let Some(p) = parent else { return node };
        node = p;
    }
}

fn degree(h: &tree::Tree<Xor>, v: Index) -> usize {
    h.degree(v).unwrap_or(0)
}

fn is_boundary_vertex(h: &tree::Tree<Xor>, v: Index) -> bool {
    let exposed = h.vertices.get(&v).is_some_and(|v| v.exposed);
    exposed || degree(h, v) >= 2
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Cii {
    left: Option<Index>,
    mid: Option<Index>,
    right: Option<Index>,
}

impl Cii {
    fn count(&self) -> usize {
        usize::from(self.left.is_some())
            + usize::from(self.mid.is_some())
            + usize::from(self.right.is_some())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterKey {
    Edge(EdgeKey),
    Label(Index),
}

fn cluster_keys(h: &tree::Tree<Xor>, node: NN) -> Vec<ClusterKey> {
    match Node::force_ptr(node) {
        LeafOrInternal::Edge(leaf) => unsafe {
            vec![ClusterKey::Edge(leaf.as_ref().edge.downcast())]
        },
        LeafOrInternal::Label(label) => vec![ClusterKey::Label(unsafe { label.as_ref().vertex })],
        LeafOrInternal::Internal(internal) => {
            let internal = unsafe { internal.as_ref() };
            let mut out = cluster_keys(h, internal.children[0]);
            out.extend(cluster_keys(h, internal.children[1]));
            out
        }
    }
}

fn count_leaves_with(h: &tree::Tree<Xor>, node: NN, v: Index) -> usize {
    match Node::force_ptr(node) {
        LeafOrInternal::Edge(leaf) => {
            let leaf = unsafe { leaf.as_ref() };
            let endpoints = h.endpoints(&leaf.edge);
            usize::from(endpoints.left == v || endpoints.right == v)
        }
        LeafOrInternal::Label(label) => usize::from(unsafe { label.as_ref().vertex } == v),
        LeafOrInternal::Internal(internal) => {
            let internal = unsafe { internal.as_ref() };
            count_leaves_with(h, internal.children[0], v)
                + count_leaves_with(h, internal.children[1], v)
        }
    }
}

fn check_node(h: &tree::Tree<Xor>, node: NN) -> Cii {
    let node_ref = unsafe { &*node.as_ptr() };

    let cii = match Node::force_ptr(node) {
        LeafOrInternal::Edge(leaf) => {
            let node = unsafe { leaf.as_ref() };
            let endpoints = h.endpoints(&node.edge);
            let flip = node_ref.is_flipped();
            let (ep_left, ep_right) = if flip {
                (endpoints.right, endpoints.left)
            } else {
                (endpoints.left, endpoints.right)
            };
            let mut c = Cii::default();
            if is_boundary_vertex(h, ep_left) {
                c.left = Some(ep_left);
            }
            if is_boundary_vertex(h, ep_right) {
                c.right = Some(ep_right);
            }
            c
        }
        LeafOrInternal::Label(label) => {
            let node = unsafe { label.as_ref() };

            let mut c = Cii::default();
            if is_boundary_vertex(h, node.vertex) {
                c.mid = Some(node.vertex);
            }

            c
        }
        LeafOrInternal::Internal(internal) => {
            let internal = unsafe { internal.as_ref() };
            let bl = check_node(h, internal.children[0]);
            let br = check_node(h, internal.children[1]);

            let bl_rightmost = bl.right.or(bl.mid);
            let bl_leftmost = bl.left.or(bl.mid);
            let br_rightmost = br.right.or(br.mid);
            let br_leftmost = br.left.or(br.mid);

            assert!(
                bl_rightmost.is_some() && br_leftmost.is_some(),
                "children of an internal node must have a shared boundary vertex; \
             node cluster = {:?}",
                cluster_keys(h, node)
            );
            assert_eq!(
                bl_rightmost,
                br_leftmost,
                "orientation invariant: rightmost boundary of left child must equal \
             leftmost boundary of right child (the central vertex); node cluster = {:?}",
                cluster_keys(h, node)
            );
            let central = bl_rightmost.unwrap();

            let mut c = Cii::default();
            let inside = count_leaves_with(h, node, central);
            if h.vertices.get(&central).is_some_and(|v| v.exposed) || inside < degree(h, central) {
                c.mid = Some(central);
            }
            if bl_leftmost != bl_rightmost {
                c.left = bl_leftmost;
            }
            if br_leftmost != br_rightmost {
                c.right = br_rightmost;
            }
            if node_ref.is_flipped() {
                std::mem::swap(&mut c.left, &mut c.right);
            }
            c
        }
    };

    assert_eq!(
        cii.count(),
        node_ref.num_boundary(),
        "num_boundary mismatch at node (leaf={}, cluster={:?})",
        node_ref.is_edge(),
        cluster_keys(h, node)
    );
    cii
}

fn fold_weight(node: NN) -> u64 {
    match Node::force_ptr(node) {
        LeafOrInternal::Edge(leaf) => unsafe { leaf.as_ref().weight.0 },
        LeafOrInternal::Label(label) => unsafe { label.as_ref().weight.0 },
        LeafOrInternal::Internal(internal) => {
            let internal = unsafe { internal.as_ref() };
            fold_weight(internal.children[0]) ^ fold_weight(internal.children[1])
        }
    }
}

fn check_node_weights_and_edges(
    h: &tree::Tree<Xor>,
    node: NN,
    leaves: &mut Vec<NN>,
    nodes: &mut Vec<NN>,
) {
    let node_ref = unsafe { &*node.as_ptr() };
    nodes.push(node);

    match Node::force_ptr(node) {
        LeafOrInternal::Label(_) | LeafOrInternal::Edge(_) => {
            leaves.push(node);
        }
        LeafOrInternal::Internal(internal) => {
            let internal = unsafe { internal.as_ref() };
            for child in internal.children {
                let child_ref = unsafe { &*child.as_ptr() };
                assert_eq!(
                    child_ref.parent(),
                    Some(NonNull::from(internal)),
                    "child->parent back pointer is inconsistent"
                );
                check_node_weights_and_edges(h, child, leaves, nodes);
            }
            assert_eq!(
                node_ref.weight.0,
                fold_weight(node),
                "stored weight does not match reduce over leaves in subtree; cluster = {:?}",
                cluster_keys(h, node)
            );
        }
    }
}

fn all_roots(h: &Harness) -> Vec<NN> {
    let mut roots: Vec<NN> = h
        .tree
        .edges
        .values()
        .map(|e| climb(e.edge_node.cast()))
        .collect();
    roots.sort_by_key(|r| r.as_ptr() as usize);
    roots.dedup();
    roots
}

fn assert_invariants(h: &Harness) {
    let tree = &h.tree;
    let mut total_leaves = 0usize;

    for (key, edge) in tree.edges.iter() {
        let leaf_nn: NN = edge.edge_node.cast();
        let node = unsafe { &*leaf_nn.cast::<LeafNode<Xor>>().as_ptr() };
        assert!(node.is_edge(), "edge must map to a leaf node");
        assert_eq!(node.edge, *key, "leaf edge key must match its map entry");
        let stored = node.weight.0;
        let expected = h.adj[&edge.endpoints[0]].get(&edge.endpoints[1]).copied();
        match expected {
            Some(w) => assert_eq!(stored, w, "leaf weight must match user supplied weight"),
            None => panic!("underlying tree and mirror disagree about edge existence"),
        }
        total_leaves += 1;
    }

    let mut covered: BTreeSet<EdgeKey> = BTreeSet::new();
    for root in all_roots(h) {
        assert!(unsafe { &*root.as_ptr() }.parent().is_none());

        let mut leaves = Vec::new();
        let mut nodes = Vec::new();
        check_node_weights_and_edges(tree, root, &mut leaves, &mut nodes);
        assert_eq!(
            leaves.len() * 2 - 1,
            nodes.len(),
            "a top tree with k leaves must have 2k-1 nodes"
        );

        let cii = check_node(tree, root);
        for (slot, label) in [
            (&cii.left, "left"),
            (&cii.mid, "middle"),
            (&cii.right, "right"),
        ] {
            if let Some(v) = slot {
                let exposed = tree.vertices.get(v).is_some_and(|v| v.exposed);
                assert!(exposed, "root boundary vertex ({label}) must be exposed");
            }
        }
        assert_eq!(cii.count(), unsafe { &*root.as_ptr() }.num_boundary());

        let mut comp_leaves = BTreeSet::new();
        for leaf in leaves {
            let key = unsafe { &*leaf.cast::<LeafNode<Xor>>().as_ptr() }
                .edge
                .downcast();
            assert!(
                covered.insert(key),
                "leaf appears in more than one top tree component"
            );
            comp_leaves.insert(key);
        }

        let sample = comp_leaves
            .iter()
            .next()
            .expect("component must contain an edge");
        let comp: BTreeSet<Index> = h
            .components()
            .into_iter()
            .find(|c| {
                c.iter()
                    .any(|&v| h.adj[&v].keys().any(|&w| EdgeKey::new(v, w) == *sample))
            })
            .unwrap()
            .into_iter()
            .collect();

        let expected: BTreeSet<EdgeKey> = tree
            .edges
            .keys()
            .filter(|k| {
                let [a, b] = k.0;
                comp.contains(&a) || comp.contains(&b)
            })
            .map(|k| k.downcast())
            .collect();
        assert_eq!(
            comp_leaves, expected,
            "top tree leaves must exactly match the underlying component's edges"
        );
    }

    assert_eq!(
        covered.len(),
        total_leaves,
        "all edges covered exactly once"
    );
}

#[test]
fn single_edge_tree() {
    let mut h = Harness::new(2);
    let verts: Vec<_> = h.vertices().collect();
    h.link(verts[0], verts[1], 42);
    assert_invariants(&h);

    h.expose(verts[0]);
    assert_invariants(&h);
    h.expose(verts[1]);
    assert_invariants(&h);

    h.deexpose(verts[0]);
    assert_invariants(&h);
    h.deexpose(verts[1]);
    assert_invariants(&h);

    h.cut(verts[0], verts[1]);
    assert_invariants(&h);
}

#[test]
fn expose_isolated_vertex() {
    let mut h = Harness::new(3);
    let verts: Vec<_> = h.vertices().collect();
    h.link(verts[0], verts[1], 1);
    assert_invariants(&h);
    h.expose(verts[2]);
    assert_invariants(&h);
    h.deexpose(verts[2]);
    assert_invariants(&h);
    h.cut(verts[0], verts[1]);
    assert_invariants(&h);
}

fn build_path(h: &mut Harness) -> Vec<Index> {
    let verts: Vec<_> = h.vertices().collect();
    for i in 1..verts.len() {
        h.link(verts[i - 1], verts[i], i as u64 * 31 + 7);
        assert_invariants(h);
    }
    verts
}

#[test]
fn path_link_and_cut_ops() {
    let mut h = Harness::new(8);
    let verts = build_path(&mut h);

    for (i, &v) in verts.iter().enumerate() {
        h.expose(v);
        assert_invariants(&h);
        if i + 1 < verts.len() {
            h.expose(verts[i + 1]);
            assert_invariants(&h);
            h.deexpose(verts[i + 1]);
            assert_invariants(&h);
        }
        h.deexpose(v);
        assert_invariants(&h);
    }

    for i in (1..verts.len()).rev() {
        h.cut(verts[i - 1], verts[i]);
        assert_invariants(&h);
    }
}

#[test]
fn star_link_and_cut() {
    let mut h = Harness::new(7);
    let verts: Vec<_> = h.vertices().collect();
    let center = verts[0];
    for &leaf in &verts[1..] {
        h.link(center, leaf, 3);
        assert_invariants(&h);
    }
    for (i, &leaf) in verts[1..].iter().enumerate() {
        h.cut(center, leaf);
        assert_invariants(&h);
        if i == 0 {
            h.link(center, leaf, 9);
            assert_invariants(&h);
        }
    }
    for &leaf in &verts[1..] {
        if !h.adj[&center].contains_key(&leaf) {
            h.link(center, leaf, 5);
            assert_invariants(&h);
        }
    }
    h.deexpose_all();
}

#[test]
fn link_cut_then_expose_both_ends_of_a_path() {
    let mut h = Harness::new(12);
    let verts = build_path(&mut h);
    let a = verts[0];
    let b = verts[11];
    h.expose(a);
    assert_invariants(&h);
    h.expose(b);
    assert_invariants(&h);

    let root = climb(h.edge_leaves_of(a).next().unwrap());
    let root_ref = unsafe { &*root.as_ptr() };
    assert!(
        root_ref.is_path(),
        "root should be a path cluster with two boundaries"
    );
    assert_eq!(root_ref.num_boundary(), 2);
    let expected: u64 = (0..11)
        .map(|i| (i + 1) as u64 * 31 + 7)
        .fold(0, |a, b| a ^ b);
    assert_eq!(
        root_ref.weight.0, expected,
        "root weight must xor all edges of the path"
    );

    assert_invariants(&h);
    h.deexpose(b);
    assert_invariants(&h);
    h.deexpose(a);
    assert_invariants(&h);
    for i in (1..verts.len()).rev() {
        h.cut(verts[i - 1], verts[i]);
        assert_invariants(&h);
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
fn grow_a_small_tree_with_exposes() {
    let mut rng = Rng(12345);
    let mut h = Harness::new(5);
    let n = 40;
    for _ in 0..n {
        let r = rng.next() % 3;
        let comps = h.components();
        let clean: Vec<Vec<Index>> = comps
            .iter()
            .filter(|c| !h.has_exposed(c))
            .cloned()
            .collect();
        let edges: Vec<(Index, Index)> = clean
            .iter()
            .flat_map(|c| {
                c.iter().flat_map(|v| {
                    h.adj[v]
                        .keys()
                        .copied()
                        .filter(|w| *w > *v)
                        .map(move |w| (*v, w))
                })
            })
            .collect();
        let mut acted = false;
        if r == 0 && clean.len() >= 2 {
            let ia = (rng.next() as usize) % clean.len();
            let ib = (rng.next() as usize) % (clean.len() - 1);
            let ib = if ib >= ia { ib + 1 } else { ib };
            let u = clean[ia][(rng.next() as usize) % clean[ia].len()];
            let v = clean[ib][(rng.next() as usize) % clean[ib].len()];
            h.link(u, v, rng.next() | 1);
            acted = true;
        } else if r == 1 && !edges.is_empty() {
            let (u, v) = edges[(rng.next() as usize) % edges.len()];
            h.cut(u, v);
            acted = true;
        } else if h.exposed.len() < 2 {
            let verts: Vec<_> = h.vertices().collect();
            for _ in 0..verts.len() {
                let v = verts[(rng.next() as usize) % verts.len()];
                if !h.exposed.contains(&v) && h.exposed_count(&h.comp_of(v)) <= 1 {
                    h.expose(v);
                    acted = true;
                    break;
                }
            }
        }
        if !acted && !h.exposed.is_empty() {
            let v = *h.exposed.iter().next().unwrap();
            h.deexpose(v);
            acted = true;
        }
        if !acted {
            break;
        }
        assert_invariants(&h);
    }
    h.deexpose_all();
    assert_invariants(&h);
}

fn run_random_ops(seed: u64, num_vertices: usize, num_ops: usize) {
    let mut rng = Rng(seed);
    let mut h = Harness::new(num_vertices);

    for _ in 0..num_ops {
        let verts: Vec<_> = h.vertices().collect();
        let comps = h.components();
        let clean: Vec<Vec<Index>> = comps
            .iter()
            .filter(|c| !h.has_exposed(c))
            .cloned()
            .collect();
        let clean_edges: Vec<(Index, Index)> = clean
            .iter()
            .flat_map(|c| {
                c.iter().flat_map(|v| {
                    h.adj[v]
                        .keys()
                        .copied()
                        .filter(|w| *w > *v)
                        .map(move |w| (*v, w))
                })
            })
            .collect();
        let can_deexpose = !h.exposed.is_empty();
        let can_link = clean.len() >= 2;
        let can_expose = verts
            .iter()
            .any(|&v| !h.exposed.contains(&v) && h.exposed_count(&h.comp_of(v)) <= 1);

        let r = rng.next() % 4;
        let mut acted = false;
        if r == 0 && can_deexpose {
            let v = *h.exposed.iter().next().unwrap();
            h.deexpose(v);
            acted = true;
        } else if r == 1 && can_link {
            let ia = (rng.next() as usize) % clean.len();
            let ib = (rng.next() as usize) % (clean.len() - 1);
            let ib = if ib >= ia { ib + 1 } else { ib };
            let a = &clean[ia];
            let b = &clean[ib];
            let u = a[(rng.next() as usize) % a.len()];
            let v = b[(rng.next() as usize) % b.len()];
            h.link(u, v, rng.next() | 1);
            acted = true;
        } else if r == 2 && !clean_edges.is_empty() {
            let (u, v) = clean_edges[(rng.next() as usize) % clean_edges.len()];
            h.cut(u, v);
            acted = true;
        } else if can_expose {
            let v = verts[(rng.next() as usize) % verts.len()];
            if !h.exposed.contains(&v) && h.exposed_count(&h.comp_of(v)) <= 1 {
                h.expose(v);
                acted = true;
            }
        }

        if !acted {
            h.deexpose_all();
            continue;
        }

        assert_invariants(&h);
    }
    h.deexpose_all();
    assert_invariants(&h);
}

#[test]
fn randomized_small_forest() {
    for seed in 0..40 {
        run_random_ops(seed, 7, 220);
    }
}

#[test]
fn randomized_medium_forest() {
    for seed in 0..10 {
        run_random_ops(seed, 14, 500);
    }
}
