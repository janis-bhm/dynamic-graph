use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::*;

fn lvl(i: i32) -> Level {
    Level::new(i).unwrap()
}

#[test]
fn level_bounds_and_conversions() {
    assert_eq!(Level::new(-1), Some(Level::MIN));
    assert_eq!(Level::new(LEVEL_CAP), Some(Level::MAX));
    assert_eq!(Level::new(-2), None);
    assert_eq!(Level::new(LEVEL_CAP + 1), None);
    assert_eq!(Level::new(NO_COVER), None);

    assert_eq!(Level::try_from(-1), Ok(Level::MIN));
    assert_eq!(Level::try_from(LEVEL_CAP), Ok(Level::MAX));
    assert_eq!(Level::try_from(-2), Err(()));
    assert_eq!(Level::try_from(LEVEL_CAP + 1), Err(()));
    assert_eq!(Level::try_from(NO_COVER), Err(()));

    for (value, level) in [(-1, Level::MIN), (0, lvl(0)), (LEVEL_CAP, Level::MAX)] {
        assert_eq!(i32::from(level), value);
        assert_eq!(Level::try_from(value), Ok(level));
    }
}

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

#[test]
fn size_vector_trims_trailing_zeros() {
    let empty = SizeVector::empty();
    assert_eq!(empty.0.len(), 0);
    assert_eq!(empty.get(0), 0);

    let vector = SizeVector::from_vec(vec![0, 0, 3, 0, 0]);
    assert_eq!(vector.0.len(), 3);
    assert_eq!(vector.get(0), 0);
    assert_eq!(vector.get(1), 0);
    assert_eq!(vector.get(2), 3);
    assert_eq!(vector.get(3), 0);

    // With direct slot indexing, this is the input whose first three slots
    // are 0, 3, 0.
    let vector = SizeVector::from_vec(vec![0, 3, 0, 0, 0]);
    assert_eq!(vector.get(0), 0);
    assert_eq!(vector.get(1), 3);
    assert_eq!(vector.get(2), 0);
}

#[test]
fn size_vector_add_and_mask() {
    let left = SizeVector::from_vec(vec![1, 0, 0]);
    let right = SizeVector::from_vec(vec![0, 2, 0, 4]);
    let sum = left.add(&right);
    assert_eq!(
        (0..4).map(|slot| sum.get(slot)).collect::<Vec<_>>(),
        vec![1, 2, 0, 4]
    );

    let masked = sum.masked(-1);
    assert_eq!(
        (0..3).map(|slot| masked.get(slot)).collect::<Vec<_>>(),
        vec![1, 0, 0]
    );

    let masked = sum.masked(0);
    assert_eq!(
        (0..3).map(|slot| masked.get(slot)).collect::<Vec<_>>(),
        vec![1, 2, 0]
    );
}

fn assert_size_vector_matches_dense(
    vector: &SizeVector,
    dense: &[u64; SLOTS],
    case: usize,
    operation: &str,
) {
    for (slot, &expected) in dense.iter().enumerate() {
        assert_eq!(
            vector.get(slot),
            expected,
            "{operation} mismatch in case {case}, slot {slot}"
        );
    }
    assert_eq!(
        vector.get(SLOTS),
        0,
        "{operation} should read zero beyond the logical slots in case {case}"
    );
}

fn random_size_vector_data(rng: &mut Rng) -> (Vec<u64>, [u64; SLOTS]) {
    let len = (rng.next() as usize) % (SLOTS + 1);
    let trailing_zeros = if len == 0 {
        0
    } else {
        (rng.next() as usize) % (len + 1)
    };
    let nonzero_prefix = len - trailing_zeros;
    let data: Vec<_> = (0..len)
        .map(|slot| {
            if slot < nonzero_prefix {
                rng.next() % 8
            } else {
                0
            }
        })
        .collect();
    let mut dense = [0; SLOTS];
    for (slot, &value) in data.iter().enumerate() {
        dense[slot] = value;
    }
    (data, dense)
}

#[test]
fn size_vector_matches_dense_randomized_reference() {
    let mut rng = Rng(0x51_2e_5eed);
    for case in 0..200 {
        let (left_data, left_dense) = random_size_vector_data(&mut rng);
        let (right_data, right_dense) = random_size_vector_data(&mut rng);
        let left = SizeVector::from_vec(left_data);
        let right = SizeVector::from_vec(right_data);
        assert_size_vector_matches_dense(&left, &left_dense, case, "from_vec(left)");
        assert_size_vector_matches_dense(&right, &right_dense, case, "from_vec(right)");

        let sum = left.add(&right);
        let mut sum_dense = [0; SLOTS];
        for ((sum, &left), &right) in sum_dense.iter_mut().zip(&left_dense).zip(&right_dense) {
            *sum = left + right;
        }
        assert_size_vector_matches_dense(&sum, &sum_dense, case, "add");

        let key = match rng.next() % 8 {
            0 => i32::MIN,
            1 => -2,
            2 => -1,
            3 => LEVEL_CAP + 1,
            4 => NO_COVER,
            5 => i32::MAX,
            _ => (rng.next() % (SLOTS as u64 + 8)) as i32 - 4,
        };
        let masked_key = if key == NO_COVER {
            LEVEL_CAP
        } else {
            key.clamp(-1, LEVEL_CAP)
        };
        let max_slots = (masked_key + 2).clamp(0, SLOTS as i32) as usize;
        let mut masked_dense = sum_dense;
        for slot in &mut masked_dense[max_slots..] {
            *slot = 0;
        }
        let masked = sum.masked(key);
        assert_size_vector_matches_dense(&masked, &masked_dense, case, "masked");
    }
}

#[test]
fn part_tree_diagonal_aggregates_are_recomputed() {
    let mut tree = PartTree::new();
    add_at(
        &mut tree,
        -4,
        &SizeVector::from_vec(vec![2, 3, 5, 7]),
        (1u64 << 0) | (1u64 << 2),
    );
    add_at(
        &mut tree,
        -1,
        &SizeVector::from_vec(vec![11, 13, 17]),
        (1u64 << 0) | (1u64 << 3),
    );
    add_at(
        &mut tree,
        0,
        &SizeVector::from_vec(vec![1, 4, 9, 16]),
        (1u64 << 1) | (1u64 << 3),
    );
    add_at(
        &mut tree,
        4,
        &SizeVector::from_vec(vec![6, 7, 8, 9, 10, 11, 12]),
        (1u64 << 0) | (1u64 << 5) | (1u64 << 6),
    );
    add_at(
        &mut tree,
        8,
        &SizeVector::from_vec(vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31]),
        (1u64 << 4) | (1u64 << 9) | (1u64 << 10),
    );
    add_at(
        &mut tree,
        LEVEL_CAP,
        &SizeVector::from_vec((0..SLOTS).map(|slot| (slot as u64 + 1) * 3).collect()),
        (1u64 << 32) | (1u64 << 33),
    );
    add_at(
        &mut tree,
        NO_COVER,
        &SizeVector::from_vec(vec![23, 29, 31, 37, 41, 43]),
        (1u64 << 1) | (1u64 << 30),
    );

    // The negative keys and the cap/sentinel keys each coalesce into one entry.
    assert_eq!(tree.iter().count(), 5);

    let ranges = [(-1, -1), (0, 4), (5, LEVEL_CAP - 1), (-1, LEVEL_CAP)];
    for (klo, khi) in ranges {
        let mut dense = [0; SLOTS];
        let mut expected_inc = 0;
        for (key, entry) in tree.iter() {
            if *key >= klo && *key <= khi {
                let masked = entry.raw.masked(*key);
                for (slot, sum) in dense.iter_mut().enumerate() {
                    *sum += masked.get(slot);
                }
                expected_inc |= inc_m_apply(*key, entry.inc);
            }
        }

        let diagonal = diagonal_sum(&tree, klo, khi);
        for (slot, expected) in dense.iter().enumerate() {
            assert_eq!(
                diagonal.get(slot),
                *expected,
                "diagonal sum mismatch in range [{klo}, {khi}], slot {slot}"
            );
        }
        assert_eq!(diagonal.get(SLOTS), 0);
        assert_eq!(
            range_inc_diag(&tree, klo, khi),
            expected_inc,
            "incident diagonal mismatch in range [{klo}, {khi}]"
        );
    }

    let expected_total_inc = tree
        .iter()
        .fold(0, |acc, (key, entry)| acc | inc_m_apply(*key, entry.inc));
    assert_eq!(total_inc_diag(&tree), expected_total_inc);
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

    fb.cover(a, b, lvl(2));
    assert_eq!(fb.cover_level(a), 2);
    assert_eq!(fb.cover_level_between(a, b), 2);
    assert!(fb.find_bridge(a).is_none());

    fb.uncover(a, b, lvl(2));
    assert_eq!(fb.cover_level(a), -1);
    assert!(fb.find_bridge_between(a, b).is_some());

    // Uncovering with a too small level does nothing.
    fb.cover(a, b, lvl(4));
    fb.uncover(a, b, lvl(3));
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

    fb.cover(1, 5, lvl(3));
    naive.cover_path(1, 5, 3);
    assert_eq!(fb.cover_level_between(0, 6), -1);
    assert_eq!(fb.cover_level_between(2, 4), 3);
    assert_eq!(fb.cover_level_between(1, 6), -1);
    assert_eq!(fb.cover_level_between(0, 5), -1);
    check_against_naive(&mut fb, &naive, n);

    fb.cover(0, 6, lvl(5));
    naive.cover_path(0, 6, 5);
    check_against_naive(&mut fb, &naive, n);

    fb.uncover(0, 6, lvl(4));
    naive.uncover_path(0, 6, 4);
    check_against_naive(&mut fb, &naive, n);
}

#[test]
fn randomized_cover_uncover_no_naive() {
    for seed in 0..8 {
        let n = 9;
        let levels = 6;
        let mut rng = Rng(seed);
        let mut fb = FindBridge::new();
        for i in 0..n {
            fb.add_vertex(i);
        }

        // Build a growing tree.
        for v in 1..n {
            let u = (rng.next() as usize) % v;
            fb.link(u, v);
        }

        for _ in 0..60 {
            let op = rng.next() % 2;
            let u = (rng.next() as usize) % n;
            let v = (rng.next() as usize) % n;
            if u != v && fb.connected(u, v) {
                match op {
                    0 => {
                        let level = (rng.next() % levels) as i32;
                        fb.cover(u, v, lvl(level));
                    }
                    1 => {
                        let level = (rng.next() % levels) as i32;
                        fb.uncover(u, v, lvl(level));
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn randomized_cover_uncover() {
    for seed in 0..8 {
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

        for _ in 0..60 {
            let op = rng.next() % 4;
            let u = (rng.next() as usize) % n;
            let v = (rng.next() as usize) % n;
            let reachable = u != v && naive.path_edges(u, v).is_some();
            match op {
                0 if reachable => {
                    let level = (rng.next() % levels) as i32;
                    fb.cover(u, v, lvl(level));
                    naive.cover_path(u, v, level);
                }
                1 if reachable => {
                    let level = (rng.next() % levels) as i32;
                    fb.uncover(u, v, lvl(level));
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

fn naive_has_incident(
    naive: &Naive,
    labels: &BTreeMap<LabelId, (usize, i32)>,
    v: usize,
    w: usize,
    level: i32,
) -> bool {
    let Some(path) = naive.path_vertices(v, w) else {
        return false;
    };
    let projection = naive.projections(&path);
    labels.values().any(|&(u, label_level)| {
        if label_level != level {
            return false;
        }
        let Some(&m) = projection.get(&u) else {
            return false;
        };
        naive.cover_between(u, m) >= level
    })
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
    fb.cover(1, 4, lvl(2));
    // With threshold 3, only the path edges are all "vertex-level", so every
    // vertex counts (CoverLevel of a vertex to itself is the sentinel).
    // With threshold 2, vertices whose projection path uses an uncovered edge
    // are excluded.
    // All path vertices count at any level; off-path vertices here do not
    // exist except the path itself, so the counts are just the component size
    // for i <= 2.
    assert_eq!(fb.find_size(0, 5, 2), 6);

    let a = fb.add_label(2, lvl(1));
    let b = fb.add_label(4, lvl(2));
    let c = fb.add_label(0, lvl(1));
    assert_eq!(fb.find_first_label(0, 5, lvl(1)), Some(c));
    assert_eq!(fb.find_first_label(0, 5, lvl(2)), Some(b));
    assert_eq!(fb.find_first_label(3, 5, lvl(1)), Some(a));
    assert_eq!(fb.find_first_label(0, 1, lvl(2)), Some(b));

    assert_eq!(fb.remove_label(b), Some((4, lvl(2))));
    assert_eq!(fb.find_first_label(0, 5, lvl(2)), None);
    let _ = a;
}

#[test]
fn find_first_label_multiple_labels_at_same_vertex_and_level() {
    let mut fb = FindBridge::new();
    for vertex in 0..=2 {
        fb.add_vertex(vertex);
    }
    fb.link(0, 1);
    fb.link(1, 2);

    let smaller = fb.add_label(1, lvl(1));
    let other = fb.add_label(1, lvl(1));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(smaller));

    assert_eq!(fb.remove_label(smaller), Some((1, lvl(1))));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(other));

    assert_eq!(fb.remove_label(other), Some((1, lvl(1))));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), None);
}

#[test]
fn find_first_label_removing_one_level_preserves_another() {
    let mut fb = FindBridge::new();
    for vertex in 0..=2 {
        fb.add_vertex(vertex);
    }
    fb.link(0, 1);
    fb.link(1, 2);

    let level_zero = fb.add_label(1, lvl(0));
    let level_one = fb.add_label(1, lvl(1));
    assert_eq!(fb.find_first_label(0, 2, lvl(0)), Some(level_zero));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(level_one));

    assert_eq!(fb.remove_label(level_zero), Some((1, lvl(0))));
    assert_eq!(fb.find_first_label(0, 2, lvl(0)), None);
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(level_one));
}

#[test]
fn find_first_label_off_path_branch() {
    let mut fb = FindBridge::new();
    for vertex in 0..=6 {
        fb.add_vertex(vertex);
    }
    for (u, v) in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6)] {
        fb.link(u, v);
    }

    fb.cover(1, 6, lvl(2));
    let near = fb.add_label(6, lvl(2));
    fb.add_label(3, lvl(2));

    assert_eq!(fb.find_first_label(0, 4, lvl(2)), Some(near));
}

#[test]
fn find_first_label_star_cover() {
    let mut fb = FindBridge::new();
    for vertex in 0..=4 {
        fb.add_vertex(vertex);
    }
    for leaf in 1..=4 {
        fb.link(0, leaf);
    }

    fb.cover(0, 2, lvl(1));
    let good = fb.add_label(2, lvl(1));
    fb.add_label(3, lvl(1));
    assert_eq!(fb.find_first_label(0, 1, lvl(1)), Some(good));

    fb.remove_label(good);
    assert_eq!(fb.find_first_label(0, 1, lvl(1)), None);

    fb.cover(0, 4, lvl(1));
    let good2 = fb.add_label(4, lvl(1));
    assert_eq!(fb.find_first_label(0, 1, lvl(1)), Some(good2));
}

#[test]
fn find_first_label_v_equals_w() {
    let mut fb = FindBridge::new();
    for vertex in 0..=2 {
        fb.add_vertex(vertex);
    }
    fb.link(0, 1);
    let l = fb.add_label(0, lvl(1));
    assert_eq!(fb.find_first_label(0, 0, lvl(1)), Some(l));

    let l2 = fb.add_label(2, lvl(1));
    assert_eq!(fb.find_first_label(2, 2, lvl(1)), Some(l2));
}

#[test]
fn find_first_label_disconnected() {
    let mut fb = FindBridge::new();
    for vertex in 0..4 {
        fb.add_vertex(vertex);
    }
    fb.link(0, 1);
    fb.link(2, 3);
    fb.add_label(0, lvl(1));
    fb.add_label(2, lvl(1));

    assert_eq!(fb.find_first_label(0, 2, lvl(1)), None);
}

#[test]
fn find_first_label_prefers_nearest() {
    let mut fb = FindBridge::new();
    for vertex in 0..=5 {
        fb.add_vertex(vertex);
    }
    for vertex in 1..=5 {
        fb.link(vertex - 1, vertex);
    }
    let id_at_1 = fb.add_label(1, lvl(0));
    fb.add_label(4, lvl(0));

    assert_eq!(fb.find_first_label(0, 5, lvl(0)), Some(id_at_1));
}

#[test]
fn find_first_label_after_uncover() {
    let mut fb = FindBridge::new();
    for vertex in 0..=3 {
        fb.add_vertex(vertex);
    }
    for (u, v) in [(0, 1), (1, 2), (1, 3)] {
        fb.link(u, v);
    }
    let path_label = fb.add_label(1, lvl(1));
    let branch_label = fb.add_label(3, lvl(1));

    fb.cover(0, 2, lvl(1));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(path_label));
    fb.uncover(0, 2, lvl(1));
    // The label remains valid because its projection is its own vertex, whose
    // CoverLevel is the no-cover sentinel.
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(path_label));

    fb.remove_label(path_label);
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), None);
    fb.cover(1, 3, lvl(1));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), Some(branch_label));
    fb.uncover(1, 3, lvl(1));
    assert_eq!(fb.find_first_label(0, 2, lvl(1)), None);
}

#[test]
#[ignore] // This test is slow, so we ignore it by default.
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

        for _ in 0..60 {
            match rng.next() % 6 {
                0 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.cover(u, v, lvl(level));
                        naive.cover_path(u, v, level);
                    }
                }
                1 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.uncover(u, v, lvl(level));
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
                    let id = fb.add_label(v, lvl(level));
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
                let got = fb.find_first_label(v, w, lvl(i));
                match naive_find_first_label(&naive, &labels, v, w, i) {
                    None => assert!(
                        got.is_none(),
                        "expected no label ({v},{w},{i}, seed {seed})"
                    ),
                    Some(_) => {
                        let id = got.expect("a label must be found");
                        let &(u, l) = labels.get(&id).expect("returned label must be live");
                        assert_eq!(l, i, "wrong level ({v},{w},{i}, seed {seed})");
                        let path = naive.path_vertices(v, w).unwrap();
                        let projection = naive.projections(&path);
                        let m = *projection.get(&u).expect("label must be in the component");
                        assert!(
                            naive.cover_between(u, m) >= i,
                            "invalid label ({v},{w},{i}, seed {seed})"
                        );
                        let dist = path.iter().position(|&x| x == m).unwrap();
                        let best = labels
                            .values()
                            .filter_map(|&(u2, l2)| {
                                if l2 != i {
                                    return None;
                                }
                                let m2 = *projection.get(&u2)?;
                                if naive.cover_between(u2, m2) >= i {
                                    Some(path.iter().position(|&x| x == m2).unwrap())
                                } else {
                                    None
                                }
                            })
                            .min()
                            .unwrap();
                        assert_eq!(
                            dist, best,
                            "not a minimal-distance label ({v},{w},{i}, seed {seed})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn incident_mask_matches_naive() {
    for seed in 0..5 {
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
        for _ in 0..30 {
            match rng.next() % 6 {
                0 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.cover(u, v, lvl(level));
                        naive.cover_path(u, v, level);
                    }
                }
                1 => {
                    let u = (rng.next() as usize) % n;
                    let v = (rng.next() as usize) % n;
                    if u != v && naive.path_vertices(u, v).is_some() {
                        let level = (rng.next() % levels) as i32;
                        fb.uncover(u, v, lvl(level));
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
                    let level = (rng.next() % (levels as u64 + 1)) as i32 - 1;
                    let id = fb.add_label(v, lvl(level));
                    labels.insert(id, (v, level));
                }
                _ => {
                    if !labels.is_empty() {
                        let ids: Vec<_> = labels.keys().copied().collect();
                        let id = ids[(rng.next() as usize) % ids.len()];
                        fb.remove_label(id);
                        labels.remove(&id);
                    }
                }
            }

            for _ in 0..4 {
                let v = (rng.next() as usize) % n;
                let w = (rng.next() as usize) % n;
                let level = (rng.next() % (levels as u64 + 2)) as i32 - 1;
                let connected = naive.path_vertices(v, w).is_some();
                assert_eq!(
                    fb.connected(v, w),
                    connected,
                    "connected({v}, {w}) mismatch (seed {seed})"
                );
                if connected {
                    let expected = naive_has_incident(&naive, &labels, v, w, level);
                    let mask = fb.debug_root_incident(v, w);
                    let actual = mask & super::level_bit(lvl(level)) != 0;
                    assert_eq!(
                        actual, expected,
                        "incident({v}, {w}, {level}) mismatch (seed {seed})"
                    );

                    // Exercise the scalar query helper against the same root
                    // summary whose mask was returned by the debug method.
                    let summary_has = {
                        let summary = fb.top_tree.expose_path(v, w);
                        summary.map_or(false, |s| super::incident_has(&s, lvl(level)))
                    };
                    fb.top_tree.deexpose(w);
                    fb.top_tree.deexpose(v);
                    assert_eq!(summary_has, actual, "incident_has mismatch");
                }
            }
        }
    }
}

#[test]
fn find_size_above_level_cap_counts_path_vertices() {
    let mut fb = FindBridge::new();
    for vertex in 0..4 {
        fb.add_vertex(vertex);
    }
    for vertex in 1..4 {
        fb.link(vertex - 1, vertex);
    }

    assert_eq!(fb.find_size(0, 3, LEVEL_CAP + 1), 4);
}

#[test]
fn find_size_above_level_cap_on_point_cluster() {
    let mut fb = FindBridge::new();
    for vertex in 0..6 {
        fb.add_vertex(vertex);
    }
    for vertex in 1..6 {
        fb.link(vertex - 1, vertex);
    }

    for vertex in 0..6 {
        assert_eq!(fb.find_size(vertex, vertex, LEVEL_CAP + 1), 1);
        assert_eq!(fb.find_size(vertex, vertex, LEVEL_CAP + 100), 1);
    }
}

#[test]
fn find_size_above_level_cap_is_path_length() {
    let mut fb = FindBridge::new();
    for vertex in 0..6 {
        fb.add_vertex(vertex);
    }
    for vertex in 1..6 {
        fb.link(vertex - 1, vertex);
    }

    for (u, v) in [(1, 4), (0, 5), (2, 3)] {
        let expected = (v - u + 1) as u64;
        assert_eq!(
            fb.find_size(u, v, LEVEL_CAP + 1),
            expected,
            "forward pair {u}->{v}"
        );
        assert_eq!(
            fb.find_size(v, u, LEVEL_CAP + 1),
            expected,
            "reverse pair {v}->{u}"
        );
    }
}

#[test]
fn find_size_is_symmetric() {
    for seed in 0..4 {
        let n = 12;
        let component_size = 4;
        let mut rng = Rng(seed);
        let mut fb = FindBridge::new();
        for vertex in 0..n {
            fb.add_vertex(vertex);
        }

        // Generate three small random trees, hence a random forest.
        for component_start in (0..n).step_by(component_size) {
            for vertex in component_start + 1..component_start + component_size {
                let parent = component_start + (rng.next() as usize % (vertex - component_start));
                fb.link(parent, vertex);
            }

            // Apply at least one cover to every component.
            let level = (rng.next() % (LEVEL_CAP as u64 + 1)) as i32;
            fb.cover(
                component_start,
                component_start + component_size - 1,
                lvl(level),
            );
        }

        // Add more randomly oriented covers within the generated components.
        for _ in 0..24 {
            let component_start = (rng.next() as usize % (n / component_size)) * component_size;
            let u = component_start + (rng.next() as usize % component_size);
            let v = component_start + (rng.next() as usize % component_size);
            let level = (rng.next() % (LEVEL_CAP as u64 + 1)) as i32;
            fb.cover(u, v, lvl(level));
        }

        // Check every represented level for random endpoint pairs, including
        // pairs from different components.
        for level in -1..=LEVEL_CAP {
            for _ in 0..8 {
                let u = (rng.next() as usize) % n;
                let v = (rng.next() as usize) % n;
                let forward = fb.find_size(u, v, level);
                let reverse = fb.find_size(v, u, level);
                assert_eq!(
                    forward, reverse,
                    "find_size({u}, {v}, {level}) != find_size({v}, {u}, {level}) (seed {seed})"
                );
            }
        }

        // Sample levels randomly as well, over the inclusive -1..=LEVEL_CAP
        // domain.
        for _ in 0..64 {
            let u = (rng.next() as usize) % n;
            let v = (rng.next() as usize) % n;
            let level = (rng.next() % (LEVEL_CAP as u64 + 2)) as i32 - 1;
            let forward = fb.find_size(u, v, level);
            let reverse = fb.find_size(v, u, level);
            assert_eq!(
                forward, reverse,
                "find_size({u}, {v}, {level}) != find_size({v}, {u}, {level}) (seed {seed})"
            );
        }
    }
}

#[test]
fn find_size_at_minus_one_is_component_size() {
    let n = 7;
    let mut fb = FindBridge::new();
    let mut naive = Naive::new(n);
    for vertex in 0..n {
        fb.add_vertex(vertex);
    }
    for &(u, v) in &[(0, 1), (1, 2), (1, 3), (4, 5), (5, 6)] {
        fb.link(u, v);
        naive.link(u, v);
    }

    for vertex in 0..n {
        let expected = naive.component(vertex).len() as u64;
        assert_eq!(
            fb.find_size(vertex, vertex, -1),
            expected,
            "component size mismatch at vertex {vertex}"
        );
    }
}

#[test]
fn find_size_above_level_cap_on_single_edge() {
    let mut fb = FindBridge::new();
    fb.add_vertex(0);
    fb.add_vertex(1);
    fb.link(0, 1);

    assert_eq!(fb.find_size(0, 0, LEVEL_CAP + 1), 1);
    assert_eq!(fb.find_size(1, 1, LEVEL_CAP + 1), 1);
    assert_eq!(fb.find_size(0, 1, LEVEL_CAP + 1), 2);
    assert_eq!(fb.find_size(1, 0, LEVEL_CAP + 1), 2);
}

#[test]
fn find_size_above_level_cap_on_isolated_vertex() {
    let mut fb = FindBridge::new();
    let vertex = fb.add_vertex(0);

    assert_eq!(fb.find_size(vertex, vertex, LEVEL_CAP + 1), 1);
    assert_eq!(fb.find_size(vertex, vertex, LEVEL_CAP + 100), 1);
}

#[test]
fn find_size_above_level_cap_on_star() {
    let mut fb = FindBridge::new();
    for vertex in 0..=5 {
        fb.add_vertex(vertex);
    }
    for leaf in 1..=5 {
        fb.link(0, leaf);
    }

    assert_eq!(fb.find_size(0, 0, LEVEL_CAP + 1), 1);
    for leaf in 1..=5 {
        assert_eq!(fb.find_size(0, leaf, LEVEL_CAP + 1), 2);
    }
    assert_eq!(fb.find_size(1, 2, LEVEL_CAP + 1), 3);
}
