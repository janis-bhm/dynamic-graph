use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::*;

/// A straightforward (slow) reference implementation of the same operations.
struct Naive {
    adj: BTreeMap<usize, BTreeSet<usize>>,
    cover: BTreeMap<(usize, usize), i32>,
}

fn key(u: usize, v: usize) -> (usize, usize) {
    if u < v { (u, v) } else { (v, u) }
}

impl Naive {
    fn new(n: usize) -> Self {
        let mut adj = BTreeMap::new();
        for i in 0..n {
            adj.insert(i, BTreeSet::new());
        }
        Naive {
            adj,
            cover: BTreeMap::new(),
        }
    }

    fn link(&mut self, u: usize, v: usize) {
        self.adj.get_mut(&u).unwrap().insert(v);
        self.adj.get_mut(&v).unwrap().insert(u);
        self.cover.insert(key(u, v), -1);
    }

    fn cut(&mut self, u: usize, v: usize) {
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.adj.get_mut(&v).unwrap().remove(&u);
        self.cover.remove(&key(u, v));
    }

    fn path_edges(&self, u: usize, v: usize) -> Option<Vec<(usize, usize)>> {
        if u == v {
            return Some(Vec::new());
        }
        let mut prev: BTreeMap<usize, usize> = BTreeMap::new();
        let mut seen = BTreeSet::from([u]);
        let mut queue = VecDeque::from([u]);
        while let Some(x) = queue.pop_front() {
            if x == v {
                let mut edges = Vec::new();
                let mut cur = v;
                while cur != u {
                    let p = prev[&cur];
                    edges.push(key(p, cur));
                    cur = p;
                }
                return Some(edges);
            }
            for &w in &self.adj[&x] {
                if seen.insert(w) {
                    prev.insert(w, x);
                    queue.push_back(w);
                }
            }
        }
        None
    }

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
            for &w in &self.adj[&x] {
                if seen.insert(w) {
                    prev.insert(w, x);
                    queue.push_back(w);
                }
            }
        }
        None
    }

    fn projections(&self, path: &[usize]) -> BTreeMap<usize, usize> {
        let mut projection: BTreeMap<usize, usize> = BTreeMap::new();
        let mut queue = VecDeque::new();
        for &m in path {
            projection.insert(m, m);
            queue.push_back(m);
        }
        while let Some(x) = queue.pop_front() {
            let px = projection[&x];
            for &y in &self.adj[&x] {
                if !projection.contains_key(&y) {
                    projection.insert(y, px);
                    queue.push_back(y);
                }
            }
        }
        projection
    }

    fn cover_between(&self, u: usize, v: usize) -> i32 {
        if u == v {
            return NO_COVER;
        }
        match self.path_vertices(u, v) {
            None => NO_COVER,
            Some(path) => path
                .windows(2)
                .map(|pair| self.cover[&key(pair[0], pair[1])])
                .min()
                .unwrap(),
        }
    }

    fn component(&self, v: usize) -> Vec<usize> {
        let mut comp = Vec::new();
        let mut seen = BTreeSet::from([v]);
        let mut queue = VecDeque::from([v]);
        while let Some(x) = queue.pop_front() {
            comp.push(x);
            for &w in &self.adj[&x] {
                if seen.insert(w) {
                    queue.push_back(w);
                }
            }
        }
        comp
    }

    fn component_edges(&self, v: usize) -> Vec<(usize, usize)> {
        let comp: BTreeSet<_> = self.component(v).into_iter().collect();
        self.cover
            .keys()
            .copied()
            .filter(|(a, _)| comp.contains(a))
            .collect()
    }

    fn cover_path(&mut self, u: usize, v: usize, level: i32) {
        for e in self.path_edges(u, v).unwrap() {
            let c = self.cover.get_mut(&e).unwrap();
            *c = (*c).max(level);
        }
    }

    fn uncover_path(&mut self, u: usize, v: usize, level: i32) {
        for e in self.path_edges(u, v).unwrap() {
            let c = self.cover.get_mut(&e).unwrap();
            if *c <= level {
                *c = -1;
            }
        }
    }

    fn cover_level(&self, v: usize) -> i32 {
        self.component_edges(v)
            .into_iter()
            .map(|e| self.cover[&e])
            .min()
            .unwrap_or(NO_COVER)
    }

    fn cover_level_between(&self, u: usize, v: usize) -> i32 {
        if u == v {
            return NO_COVER;
        }
        self.path_edges(u, v)
            .map(|edges| edges.into_iter().map(|e| self.cover[&e]).min().unwrap())
            .unwrap_or(NO_COVER)
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

fn check_against_naive(fb: &mut FindBridge, naive: &Naive, n: usize) {
    for v in 0..n {
        assert_eq!(
            fb.cover_level(v),
            naive.cover_level(v),
            "cover_level({v}) mismatch"
        );
        let expected = naive.cover_level(v);
        match fb.min_covered_edge(v) {
            Some(e) => {
                assert!(naive.cover.contains_key(&key(e.0, e.1)));
                assert_eq!(naive.cover[&key(e.0, e.1)], expected);
            }
            None => assert_eq!(expected, NO_COVER),
        }
        let found = fb.find_bridge(v);
        if expected == -1 {
            let e = found.expect("a bridge must be reported");
            assert_eq!(naive.cover[&key(e.0, e.1)], -1);
            assert!(naive.component(v).contains(&e.0));
        } else {
            assert!(found.is_none());
        }
    }

    for u in 0..n {
        assert_eq!(fb.cover_level_between(u, u), NO_COVER);
        assert!(fb.min_covered_edge_between(u, u).is_none());
        for v in u + 1..n {
            let expected = naive.cover_level_between(u, v);
            assert_eq!(
                fb.cover_level_between(u, v),
                expected,
                "cover_level_between({u}, {v}) mismatch"
            );
            assert_eq!(expected, naive.cover_level_between(v, u));
            match fb.min_covered_edge_between(u, v) {
                Some(e) => {
                    assert_eq!(naive.cover[&key(e.0, e.1)], expected);
                    let path: BTreeSet<_> = naive.path_edges(u, v).unwrap().into_iter().collect();
                    assert!(path.contains(&key(e.0, e.1)));
                }
                None => assert_eq!(expected, NO_COVER),
            }
            let found = fb.find_bridge_between(u, v);
            if expected == -1 {
                let e = found.expect("a bridge must be reported");
                assert_eq!(naive.cover[&key(e.0, e.1)], -1);
            } else {
                assert!(found.is_none());
            }
        }
    }
}

#[test]
fn single_edge_is_a_bridge() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex(0);
    let b = fb.add_vertex(1);
    fb.link(a, b);

    assert!(fb.connected(a, b));
    assert_eq!(fb.cover_level(a), -1);
    assert_eq!(fb.cover_level_between(a, b), -1);
    assert!(fb.find_bridge(a).is_some());
    assert!(fb.find_bridge_between(a, b).is_some());

    fb.cover(a, b, 2);
    assert_eq!(fb.cover_level(a), 2);
    assert_eq!(fb.cover_level_between(a, b), 2);
    assert!(fb.find_bridge(a).is_none());

    fb.uncover(a, b, 2);
    assert_eq!(fb.cover_level(a), -1);
    assert!(fb.find_bridge_between(a, b).is_some());

    // Uncovering with a too small level does nothing.
    fb.cover(a, b, 4);
    fb.uncover(a, b, 3);
    assert_eq!(fb.cover_level_between(a, b), 4);
}

#[test]
fn disconnected_vertices() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex(0);
    let b = fb.add_vertex(1);
    let c = fb.add_vertex(2);
    fb.link(a, b);
    assert!(!fb.connected(a, c));
    assert_eq!(fb.cover_level_between(a, c), NO_COVER);
    assert!(fb.find_bridge_between(a, c).is_none());
    // `a` is still connected to `b` after the negative connectivity test.
    assert!(fb.connected(a, b));
}

#[test]
fn trivial_path_is_not_a_bridge() {
    let mut fb = FindBridge::new();
    let a = fb.add_vertex(0);
    let b = fb.add_vertex(1);
    fb.link(a, b);
    assert!(fb.find_bridge_between(a, a).is_none());
}

#[test]
fn path_cover_and_uncover() {
    let n = 7;
    let mut fb = FindBridge::new();
    for i in 0..n {
        fb.add_vertex(i);
    }
    for i in 1..n {
        fb.link(i - 1, i);
    }
    let mut naive = Naive::new(n);
    for i in 1..n {
        naive.link(i - 1, i);
    }

    fb.cover(1, 5, 3);
    naive.cover_path(1, 5, 3);
    assert_eq!(fb.cover_level_between(0, 6), -1);
    assert_eq!(fb.cover_level_between(2, 4), 3);
    assert_eq!(fb.cover_level_between(1, 6), -1);
    assert_eq!(fb.cover_level_between(0, 5), -1);
    check_against_naive(&mut fb, &naive, n);

    fb.cover(0, 6, 5);
    naive.cover_path(0, 6, 5);
    check_against_naive(&mut fb, &naive, n);

    fb.uncover(0, 6, 4);
    naive.uncover_path(0, 6, 4);
    check_against_naive(&mut fb, &naive, n);
}

#[test]
fn randomized_cover_uncover() {
    for seed in 0..20 {
        let n = 9;
        let levels = 6;
        let mut rng = Rng(seed);
        let mut fb = FindBridge::new();
        let mut naive = Naive::new(n);
        for i in 0..n {
            fb.add_vertex(i);
        }

        // Build a growing tree.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for v in 1..n {
            let u = (rng.next() as usize) % v;
            fb.link(u, v);
            naive.link(u, v);
            edges.push((u, v));
        }

        for _ in 0..400 {
            let op = rng.next() % 4;
            let u = (rng.next() as usize) % n;
            let v = (rng.next() as usize) % n;
            let reachable = u != v && naive.path_edges(u, v).is_some();
            match op {
                0 if reachable => {
                    let level = (rng.next() % levels) as i32;
                    fb.cover(u, v, level);
                    naive.cover_path(u, v, level);
                }
                1 if reachable => {
                    let level = (rng.next() % levels) as i32;
                    fb.uncover(u, v, level);
                    naive.uncover_path(u, v, level);
                }
                2 if !edges.is_empty() => {
                    let (a, b) = edges.swap_remove((rng.next() as usize) % edges.len());
                    fb.cut(a, b);
                    naive.cut(a, b);
                }
                3 => {
                    // Link two vertices that are in different components, if
                    // we can find such a pair.
                    for _ in 0..8 {
                        let a = (rng.next() as usize) % n;
                        let b = (rng.next() as usize) % n;
                        if a != b && naive.path_edges(a, b).is_none() {
                            fb.link(a, b);
                            naive.link(a, b);
                            edges.push((a, b));
                            break;
                        }
                    }
                }
                _ => {}
            }

            check_against_naive(&mut fb, &naive, n);
        }
    }
}

fn naive_find_size(naive: &Naive, n: usize, v: usize, w: usize, i: i32) -> u64 {
    if naive.path_vertices(v, w).is_none() {
        return 0;
    }
    let path = naive.path_vertices(v, w).unwrap();
    let projection = naive.projections(&path);
    let mut count = 0;
    for u in 0..n {
        if let Some(&m) = projection.get(&u)
            && naive.cover_between(u, m) >= i
        {
            count += 1;
        }
    }
    count
}

fn naive_find_first_label(
    naive: &Naive,
    labels: &BTreeMap<LabelId, (usize, i32)>,
    v: usize,
    w: usize,
    i: i32,
) -> Option<LabelId> {
    let path = naive.path_vertices(v, w)?;
    let projection = naive.projections(&path);
    for &m in &path {
        for (&id, &(u, level)) in labels {
            if level == i && projection.get(&u) == Some(&m) && naive.cover_between(u, m) >= i {
                return Some(id);
            }
        }
    }
    None
}

#[test]
fn component_size_and_labels() {
    let n = 6;
    let mut fb = FindBridge::new();
    for i in 0..n {
        fb.add_vertex(i);
    }
    for i in 1..n {
        fb.link(i - 1, i);
    }

    // The whole path is one component of size 6.
    assert_eq!(fb.find_size(0, 0, -1), 6);
    assert_eq!(fb.find_size(3, 3, -1), 6);

    // Cover 1..4 at level 2.
    fb.cover(1, 4, 2);
    // With threshold 3, only the path edges are all "vertex-level", so every
    // vertex counts (CoverLevel of a vertex to itself is the sentinel).
    // With threshold 2, vertices whose projection path uses an uncovered edge
    // are excluded.
    // All path vertices count at any level; off-path vertices here do not
    // exist except the path itself, so the counts are just the component size
    // for i <= 2.
    assert_eq!(fb.find_size(0, 5, 2), 6);

    let a = fb.add_label(2, 1);
    let b = fb.add_label(4, 2);
    let c = fb.add_label(0, 1);
    assert_eq!(fb.find_first_label(0, 5, 1), Some(c));
    assert_eq!(fb.find_first_label(0, 5, 2), Some(b));
    assert_eq!(fb.find_first_label(3, 5, 1), Some(a));
    assert_eq!(fb.find_first_label(0, 1, 2), Some(b));

    fb.remove_label(b);
    assert_eq!(fb.find_first_label(0, 5, 2), None);
    let _ = a;
}

#[test]
fn randomized_find_size_and_labels() {
    for seed in 0..8 {
        let n = 8;
        let levels = 5;
        let mut rng = Rng(seed);
        let mut fb = FindBridge::new();
        let mut naive = Naive::new(n);
        for i in 0..n {
            fb.add_vertex(i);
        }

        let mut edges: Vec<(usize, usize)> = Vec::new();
        for v in 1..n {
            let u = (rng.next() as usize) % v;
            fb.link(u, v);
            naive.link(u, v);
            edges.push((u, v));
        }

        let mut labels: BTreeMap<LabelId, (usize, i32)> = BTreeMap::new();

        for _ in 0..120 {
            match rng.next() % 6 {
                0 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.cover(u, v, level);
                        naive.cover_path(u, v, level);
                    }
                }
                1 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.uncover(u, v, level);
                        naive.uncover_path(u, v, level);
                    }
                }
                2 if !edges.is_empty() => {
                    let (a, b) = edges.swap_remove((rng.next() as usize) % edges.len());
                    fb.cut(a, b);
                    naive.cut(a, b);
                }
                3 => {
                    for _ in 0..8 {
                        let a = (rng.next() as usize) % n;
                        let b = (rng.next() as usize) % n;
                        if a != b && naive.path_vertices(a, b).is_none() {
                            fb.link(a, b);
                            naive.link(a, b);
                            edges.push((a, b));
                            break;
                        }
                    }
                }
                4 => {
                    let v = (rng.next() as usize) % n;
                    let level = (rng.next() % levels) as i32;
                    let id = fb.add_label(v, level);
                    labels.insert(id, (v, level));
                }
                _ => {
                    if labels.len() > 1 {
                        let ids: Vec<_> = labels.keys().copied().collect();
                        let id = ids[(rng.next() as usize) % ids.len()];
                        fb.remove_label(id);
                        labels.remove(&id);
                    }
                }
            }

            // Check FindSize and FindFirstLabel against the brute force.
            for _ in 0..4 {
                let v = (rng.next() as usize) % n;
                let w = (rng.next() as usize) % n;
                let i = (rng.next() % (levels as u64 + 1)) as i32 - 1;
                assert_eq!(
                    fb.find_size(v, w, i),
                    naive_find_size(&naive, n, v, w, i),
                    "find_size({v}, {w}, {i}) mismatch (seed {seed})"
                );
                let expected = naive_find_first_label(&naive, &labels, v, w, i);
                assert_eq!(
                    fb.find_first_label(v, w, i),
                    expected,
                    "find_first_label({v}, {w}, {i}) mismatch (seed {seed})"
                );
            }
        }
    }
}
