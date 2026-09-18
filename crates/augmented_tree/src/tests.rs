use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sum(i64);

impl Aggregate for Sum {
    fn reduce(&self, other: &Self) -> Self {
        Sum(self.0 + other.0)
    }
    fn identity() -> Self {
        Sum(0)
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
fn empty_tree() {
    let tree: BTree<i32, Sum> = BTree::new();
    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
    assert_eq!(tree.get(&1), None);
    assert_eq!(tree.first_key_value(), None);
    assert_eq!(tree.last_key_value(), None);
    assert_eq!(tree.aggregate(), &Sum(0));
    assert_eq!(tree.iter().count(), 0);
}

#[test]
fn simple_insert_get_remove() {
    let mut tree = BTree::new();
    assert_eq!(tree.insert(1, Sum(10)), None);
    assert_eq!(tree.insert(2, Sum(20)), None);
    assert_eq!(tree.insert(3, Sum(30)), None);
    assert_eq!(tree.len(), 3);
    assert_eq!(tree.get(&2), Some(&Sum(20)));
    assert_eq!(tree.aggregate(), &Sum(60));

    assert_eq!(tree.insert(2, Sum(200)), Some(Sum(20)));
    assert_eq!(tree.len(), 3);
    assert_eq!(tree.aggregate(), &Sum(240));
    assert_eq!(tree.get(&2), Some(&Sum(200)));

    assert_eq!(tree.remove(&2), Some(Sum(200)));
    assert_eq!(tree.remove(&2), None);
    assert_eq!(tree.len(), 2);
    assert_eq!(tree.aggregate(), &Sum(40));

    assert_eq!(tree.remove(&3), Some(Sum(30)));
    assert_eq!(tree.remove(&1), Some(Sum(10)));
    assert!(tree.is_empty());
    assert_eq!(tree.aggregate(), &Sum(0));
}

#[test]
fn first_last_pop() {
    let mut tree = BTree::new();
    for i in 0..20 {
        tree.insert(i, Sum(i as i64));
    }
    assert_eq!(tree.first_key_value(), Some((&0, &Sum(0))));
    assert_eq!(tree.last_key_value(), Some((&19, &Sum(19))));
    assert_eq!(tree.pop_first(), Some((0, Sum(0))));
    assert_eq!(tree.pop_last(), Some((19, Sum(19))));
    assert_eq!(tree.len(), 18);
    assert_eq!(tree.aggregate().0, (1..19).sum::<i64>());
}

#[test]
fn matches_std_btreemap() {
    let mut rng = Rng(0xdead_beef);
    let mut tree: BTree<i64, Sum> = BTree::new();
    let mut reference: BTreeMap<i64, i64> = BTreeMap::new();

    let steps = if cfg!(miri) { 400 } else { 4000 };
    for step in 0..steps {
        let key = (rng.next() % 200) as i64;
        match rng.next() % 3 {
            0 | 1 => {
                let value = (rng.next() % 1000) as i64;
                let got = tree.insert(key, Sum(value));
                let want = reference.insert(key, value);
                assert_eq!(got.map(|s| s.0), want, "insert at step {step}");
            }
            _ => {
                let got = tree.remove(&key);
                let want = reference.remove(&key);
                assert_eq!(got.map(|s| s.0), want, "remove at step {step}");
            }
        }
        assert_eq!(tree.len(), reference.len());
        unsafe { check_node(tree.root.node, tree.root.height) };
    }

    let expected: Vec<(i64, i64)> = reference.iter().map(|(&k, &v)| (k, v)).collect();
    let actual: Vec<(i64, i64)> = tree.iter().map(|(&k, v)| (k, v.0)).collect();
    assert_eq!(actual, expected);

    let total: i64 = reference.values().sum();
    assert_eq!(tree.aggregate().0, total);
}

unsafe fn check_node<V: Aggregate + std::fmt::Debug + PartialEq>(
    node: NonNull<LeafNode<i64, V>>,
    height: usize,
) -> V {
    unsafe {
        let leaf = &*node.as_ptr();
        let len = usize::from(leaf.len);
        if height == 0 {
            let mut agg = V::identity();
            for i in 0..len {
                agg = agg.reduce(leaf.values[i].assume_init_ref());
            }
            assert_eq!(agg, leaf.aggregate, "leaf aggregate mismatch (len={len})");
            agg
        } else {
            let internal = &*node.as_ptr().cast::<InternalNode<i64, V>>();
            let mut agg = V::identity();
            for i in 0..len {
                let child = internal.edges[i].assume_init();
                let ca = check_node(child.cast::<LeafNode<i64, V>>(), height - 1);
                assert_eq!(
                    ca,
                    (*child.cast::<LeafNode<i64, V>>().as_ptr()).aggregate,
                    "child aggregate mismatch at child {i}"
                );
                agg = agg.reduce(&ca);
                agg = agg.reduce(internal.leaf.values[i].assume_init_ref());
            }
            let child = internal.edges[len].assume_init();
            let ca = check_node(child.cast::<LeafNode<i64, V>>(), height - 1);
            agg = agg.reduce(&ca);
            assert_eq!(agg, internal.leaf.aggregate, "internal aggregate mismatch");
            agg
        }
    }
}

#[test]
fn grows_tall_and_shrinks() {
    let mut tree = BTree::new();
    let n = if cfg!(miri) { 150 } else { 1000 } as i64;
    for i in 0..n {
        tree.insert(i, Sum(1));
    }
    assert_eq!(tree.len(), n as usize);
    assert_eq!(tree.aggregate(), &Sum(n));
    for i in 0..n {
        assert_eq!(tree.get(&i), Some(&Sum(1)));
    }
    // remove in a scrambled order
    let mut order: Vec<i64> = (0..n).collect();
    let mut rng = Rng(42);
    for i in (1..order.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        order.swap(i, j);
    }
    for (removed, key) in order.into_iter().enumerate() {
        assert_eq!(tree.remove(&key), Some(Sum(1)));
        unsafe { check_node(tree.root.node, tree.root.height) };
        assert_eq!(
            tree.aggregate(),
            &Sum(n - removed as i64 - 1),
            "removed {removed}"
        );
    }
    assert!(tree.is_empty());
    assert_eq!(tree.aggregate(), &Sum(0));
}

#[test]
fn get_mut_updates_aggregate() {
    let mut tree = BTree::new();
    for i in 0..50 {
        tree.insert(i, Sum(i as i64));
    }
    let old_total: i64 = (0..50).sum();
    assert_eq!(tree.aggregate().0, old_total);

    tree.update(&10, |v| v.0 += 100).unwrap();
    assert_eq!(tree.aggregate().0, old_total + 100);
    assert_eq!(tree.get(&10), Some(&Sum(110)));

    {
        let mut guard = tree.get_mut(&20).unwrap();
        guard.get_mut().0 = -1000;
    }
    assert_eq!(tree.get(&20), Some(&Sum(-1000)));
    assert_eq!(tree.aggregate().0, old_total + 100 - 20 - 1000);
}

#[test]
fn entry_or_insert_and_modify() {
    let mut tree: BTree<i64, Sum> = BTree::new();

    {
        let guard = tree.entry(1).or_insert(Sum(10));
        assert_eq!(guard.get(), &Sum(10));
    }
    assert_eq!(tree.get(&1), Some(&Sum(10)));
    assert_eq!(tree.aggregate(), &Sum(10));

    // Occupied: the existing value is kept, and mutating the guard updates
    // the aggregate when the guard drops.
    {
        let mut guard = tree.entry(1).or_insert(Sum(999));
        assert_eq!(guard.get(), &Sum(10));
        guard.get_mut().0 += 5;
    }
    assert_eq!(tree.get(&1), Some(&Sum(15)));
    assert_eq!(tree.aggregate(), &Sum(15));

    tree.entry(1).and_modify(|v| v.0 += 100).or_insert(Sum(0));
    assert_eq!(tree.get(&1), Some(&Sum(115)));
    assert_eq!(tree.aggregate(), &Sum(115));

    tree.entry(2).and_modify(|v| v.0 += 100).or_insert(Sum(7));
    assert_eq!(tree.get(&2), Some(&Sum(7)));
    assert_eq!(tree.aggregate(), &Sum(122));

    tree.entry(3).or_insert_with(|| Sum(1));
    tree.entry(4).or_insert_with_key(|&k| Sum(k));
    tree.entry(5).or_default();
    assert_eq!(tree.get(&5), Some(&Sum(0)));
    assert_eq!(tree.aggregate(), &Sum(127));
}

#[test]
fn entry_vacant_and_occupied_details() {
    let mut tree: BTree<i64, Sum> = BTree::new();

    match tree.entry(5) {
        Entry::Vacant(v) => {
            assert_eq!(v.key(), &5);
            let guard = v.insert(Sum(50));
            assert_eq!(guard.get(), &Sum(50));
        }
        Entry::Occupied(_) => panic!("expected vacant"),
    }
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.aggregate(), &Sum(50));

    match tree.entry(5) {
        Entry::Occupied(mut o) => {
            assert_eq!(o.key(), &5);
            assert_eq!(o.get(), &Sum(50));
            assert_eq!(o.insert(Sum(60)), Sum(50));
            assert_eq!(o.get(), &Sum(60));
        }
        Entry::Vacant(_) => panic!("expected occupied"),
    }
    assert_eq!(tree.aggregate(), &Sum(60));

    match tree.entry(5) {
        Entry::Occupied(o) => assert_eq!(o.remove_entry(), (5, Sum(60))),
        Entry::Vacant(_) => panic!("expected occupied"),
    }
    assert!(tree.is_empty());
    assert_eq!(tree.aggregate(), &Sum(0));

    match tree.entry(9) {
        Entry::Vacant(v) => assert_eq!(v.into_key(), 9),
        Entry::Occupied(_) => panic!("expected vacant"),
    }
    assert!(tree.is_empty());
}

#[test]
fn entry_updates_aggregate_across_splits() {
    let mut tree: BTree<i64, Sum> = BTree::new();
    for i in 0..200i64 {
        let _ = tree.entry(i).or_insert(Sum(1));
    }
    assert_eq!(tree.len(), 200);
    assert_eq!(tree.aggregate(), &Sum(200));
    unsafe { check_node(tree.root.node, tree.root.height) };

    // Modify every 7th entry, each contributing +10.
    let mut expected = 200i64;
    for i in (0..200i64).step_by(7) {
        tree.entry(i).and_modify(|v| v.0 += 10).or_insert(Sum(0));
        expected += 10;
    }
    assert_eq!(tree.aggregate(), &Sum(expected));
    unsafe { check_node(tree.root.node, tree.root.height) };

    // Removing through the entry keeps the aggregate correct at every step.
    for i in (0..200i64).step_by(5) {
        let removed = match tree.entry(i) {
            Entry::Occupied(o) => o.remove(),
            Entry::Vacant(_) => panic!("missing key {i}"),
        };
        expected -= removed.0;
        assert_eq!(tree.aggregate(), &Sum(expected), "after removing {i}");
        unsafe { check_node(tree.root.node, tree.root.height) };
    }
}

#[test]
fn clear_reuses_tree() {
    let mut tree = BTree::new();
    for i in 0..100 {
        tree.insert(i, Sum(2));
    }
    tree.clear();
    assert!(tree.is_empty());
    assert_eq!(tree.aggregate(), &Sum(0));
    assert_eq!(tree.iter().count(), 0);
    for i in 0..10 {
        tree.insert(i, Sum(3));
    }
    assert_eq!(tree.len(), 10);
    assert_eq!(tree.aggregate(), &Sum(30));
}

#[test]
fn keys_and_values_iterators() {
    let mut tree = BTree::new();
    for i in (0..10).rev() {
        tree.insert(i, Sum(i as i64 * 2));
    }
    let keys: Vec<i32> = tree.keys().copied().collect();
    assert_eq!(keys, (0..10).collect::<Vec<_>>());
    let values: Vec<i64> = tree.values().map(|v| v.0).collect();
    assert_eq!(values, (0..10).map(|i| i as i64 * 2).collect::<Vec<_>>());
    assert_eq!(tree.iter().len(), 10);
    assert_eq!(tree.iter().size_hint(), (10, Some(10)));
}

#[test]
fn should_not_leak_on_drop() {
    // Miri catches leaks/UB here.
    let n = if cfg!(miri) { 80 } else { 500 };
    let mut tree = BTree::new();
    for i in 0..n {
        tree.insert(i, Sum(1));
    }
    for i in 0..n / 2 {
        tree.remove(&i);
    }
    drop(tree);
}

#[test]
fn double_ended_iteration() {
    let mut tree = BTree::new();
    let n = if cfg!(miri) { 40 } else { 200 };
    for i in 0..n {
        tree.insert(i, Sum(i as i64));
    }

    let forward: Vec<i32> = tree.iter().map(|(k, _)| *k).collect();
    let backward: Vec<i32> = tree.iter().rev().map(|(k, _)| *k).collect();
    assert_eq!(forward, (0..n).collect::<Vec<_>>());
    assert_eq!(backward, (0..n).rev().collect::<Vec<_>>());

    // Interleave front and back consumption.
    let mut iter = tree.iter();
    let mut seen = Vec::new();
    while let Some((k, _)) = iter.next() {
        seen.push(*k);
        if let Some((k, _)) = iter.next_back() {
            seen.push(*k);
        }
    }
    assert_eq!(seen.len(), n as usize);
    assert_eq!(iter.len(), 0);
}

/// A key that counts how many keys have been created and dropped.
mod key_drop {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    #[derive(Default)]
    pub struct Counts {
        pub created: AtomicUsize,
        pub dropped: AtomicUsize,
    }

    pub struct Key(pub i32, pub Arc<Counts>);

    impl Key {
        pub fn new(value: i32, counts: &Arc<Counts>) -> Self {
            counts.created.fetch_add(1, AtomicOrdering::SeqCst);
            Key(value, counts.clone())
        }
    }

    impl Drop for Key {
        fn drop(&mut self) {
            self.1.dropped.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    impl std::borrow::Borrow<i32> for Key {
        fn borrow(&self) -> &i32 {
            &self.0
        }
    }
}

#[test]
fn drops_keys_exactly_once() {
    use key_drop::{Counts, Key};
    use std::sync::Arc;

    let counts = Arc::new(Counts::default());
    let n = if cfg!(miri) { 60 } else { 300 };
    {
        let mut tree: BTree<Key, Sum> = BTree::new();
        for i in 0..n {
            tree.insert(Key::new(i, &counts), Sum(1));
        }
        // Remove half by a borrowed `i32`, which does not create keys.
        for i in 0..n / 2 {
            assert!(tree.remove(&i).is_some());
        }
    }
    assert_eq!(
        counts.created.load(std::sync::atomic::Ordering::SeqCst),
        n as usize
    );
    assert_eq!(
        counts.dropped.load(std::sync::atomic::Ordering::SeqCst),
        n as usize,
        "every key must be dropped exactly once"
    );
}

/// A value that tracks how many live instances exist.
mod val_drop {
    use std::sync::atomic::{AtomicIsize, Ordering as AtomicOrdering};

    pub static LIVE: AtomicIsize = AtomicIsize::new(0);

    #[derive(Debug)]
    pub struct Val(pub i64);

    impl Val {
        pub fn new(value: i64) -> Self {
            LIVE.fetch_add(1, AtomicOrdering::SeqCst);
            Val(value)
        }
    }

    impl Drop for Val {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, AtomicOrdering::SeqCst);
        }
    }

    impl crate::Aggregate for Val {
        fn reduce(&self, other: &Self) -> Self {
            Val::new(self.0 + other.0)
        }
        fn identity() -> Self {
            Val::new(0)
        }
    }
}

#[test]
fn drops_values_exactly_once() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    use val_drop::{LIVE, Val};

    assert_eq!(LIVE.load(AtomicOrdering::SeqCst), 0);
    let n = if cfg!(miri) { 50 } else { 300 };
    {
        let mut tree: BTree<i32, Val> = BTree::new();
        for i in 0..n {
            tree.insert(i, Val::new(1));
        }
        for i in 0..n / 2 {
            assert!(tree.remove(&i).is_some());
        }
        assert_eq!(tree.aggregate().0, i64::from(n) / 2);
    }
    assert_eq!(
        LIVE.load(AtomicOrdering::SeqCst),
        0,
        "leaked or double-dropped values"
    );
}
