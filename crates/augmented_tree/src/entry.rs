use super::{Aggregate, BTree, DormantMutRef, Handle, LeafNode, NodeRef, SearchResult, marker};
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

impl<K: Ord, V: Aggregate> BTree<K, V> {
    /// Returns a view into the entry for `key`, which may be vacant or
    /// occupied.
    ///
    /// Unlike [`std::collections::BTreeMap::entry`], the mutable accessors of an
    /// [`OccupiedEntry`] hand out an [`OccupiedValue`] guard rather than a plain
    /// `&mut V`. Mutating through the guard keeps the cached aggregates
    /// consistent, because they are recomputed when the guard is dropped.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let (map, dormant_map) = DormantMutRef::new(self);
        match map.root.borrow_mut().search_tree(&key) {
            SearchResult::Found(handle) => Entry::Occupied(OccupiedEntry {
                handle,
                dormant_map,
                _marker: PhantomData,
            }),
            SearchResult::GoDown(handle) => Entry::Vacant(VacantEntry {
                key,
                handle,
                dormant_map,
                _marker: PhantomData,
            }),
        }
    }
}

/// A view into a single entry of a [`BTree`], which may be vacant or occupied.
///
/// Constructed by [`BTree::entry`].
pub enum Entry<'a, K, V> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
}

/// A view into a vacant entry of a [`BTree`].
///
/// Part of the [`Entry`] enum.
pub struct VacantEntry<'a, K, V> {
    key: K,
    handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    dormant_map: DormantMutRef<'a, BTree<K, V>>,
    _marker: PhantomData<&'a mut (K, V)>,
}

/// A view into an occupied entry of a [`BTree`].
///
/// Part of the [`Entry`] enum.
pub struct OccupiedEntry<'a, K, V> {
    handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Either>, marker::KV>,
    dormant_map: DormantMutRef<'a, BTree<K, V>>,
    _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K, V> Entry<'a, K, V> {
    /// The key this entry refers to.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }
}

impl<'a, K: Ord, V: Aggregate> Entry<'a, K, V> {
    /// Ensures a value is present, inserting `default` when vacant, and returns
    /// a guard over the value.
    pub fn or_insert(self, default: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is present, inserting the result of `default` when
    /// vacant, and returns a guard over the value.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is present, inserting the result of `default` when
    /// vacant, and returns a guard over the value.
    ///
    /// The function receives the entry's key, so key-derived values can be
    /// produced without cloning the key.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Applies `f` to the value if the entry is occupied, recomputing the
    /// cached aggregates immediately, then returns the entry unchanged.
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut entry) => {
                {
                    let mut guard = entry.get_mut();
                    f(guard.get_mut());
                }
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }

    /// Calls `update` on the value if the entry is occupied, or inserts the result of `default` if vacant, and returns the occupied entry.
    pub fn update_or_insert_with<F: FnOnce(&mut V), D: FnOnce() -> V>(
        self,
        update: F,
        default: D,
    ) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                {
                    let mut guard = entry.get_mut();
                    update(guard.get_mut());
                }
                entry
            }
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }
}

impl<'a, K: Ord, V: Aggregate + Default> Entry<'a, K, V> {
    /// Ensures a value is present, inserting [`Default::default`] when vacant,
    /// and returns a guard over the value.
    pub fn or_default(self) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// The key that would be inserted.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes ownership of the key without inserting anything.
    pub fn into_key(self) -> K {
        let VacantEntry { key, .. } = self;
        key
    }
}

impl<'a, K: Ord, V: Aggregate> VacantEntry<'a, K, V> {
    /// Inserts `value` at this entry's key and returns a guard over it.
    pub fn insert(self, value: V) -> OccupiedEntry<'a, K, V> {
        let VacantEntry {
            key,
            handle,
            mut dormant_map,
            ..
        } = self;

        let handle = handle.insert_recursing(key, value, |ins| {
            // SAFETY: the split recursed through the root, and the original
            // reborrow is not used while this callback runs.
            let map = unsafe { dormant_map.reborrow() };
            map.root
                .push_internal_level()
                .push(ins.kv.0, ins.kv.1, ins.right);
        });

        unsafe {
            core::ptr::read(&handle.node)
                .forget_type()
                .recompute_and_ascend()
        };

        // SAFETY: modifying the length doesn't invalidate handles to any nodes.
        unsafe {
            dormant_map.reborrow().length += 1;
        }

        OccupiedEntry {
            handle: handle.forget_node_type(),
            dormant_map,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// The key of this entry.
    pub fn key(&self) -> &K {
        self.handle.reborrow().into_kv().0
    }

    /// The value of this entry.
    pub fn get(&self) -> &V {
        self.handle.reborrow().into_kv().1
    }
}

impl<'a, K: Ord, V: Aggregate> OccupiedEntry<'a, K, V> {
    /// Returns a guard over the value, recomputing the cached aggregates when
    /// it is dropped.
    pub fn get_mut(&mut self) -> OccupiedValue<'_, K, V> {
        OccupiedValue {
            node: self.handle.node.node,
            height: self.handle.node.height,
            index: self.handle.index,
            _marker: PhantomData,
        }
    }

    /// Converts the entry into a guard over its value.
    pub fn into_mut(self) -> OccupiedValue<'a, K, V> {
        OccupiedValue {
            node: self.handle.node.node,
            height: self.handle.node.height,
            index: self.handle.index,
            _marker: PhantomData,
        }
    }

    /// Replaces the value, returning the old one.
    pub fn insert(&mut self, value: V) -> V {
        let old = {
            let (_, slot) = self.handle.kv_mut();
            mem::replace(slot, value)
        };
        self.recompute();
        old
    }

    /// Removes the entry, returning its value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry, returning its key and value.
    pub fn remove_entry(self) -> (K, V) {
        let OccupiedEntry {
            handle,
            dormant_map,
            ..
        } = self;

        let mut emptied_internal_root = false;
        let (old_kv, pos) = handle.remove_kv_tracking(|| emptied_internal_root = true);
        pos.into_node().forget_type().recompute_and_ascend();

        // SAFETY: the removal handle is no longer used.
        let map = unsafe { dormant_map.awaken() };
        map.length -= 1;
        if emptied_internal_root {
            map.root.pop_internal_level();
        }
        old_kv
    }

    /// Recomputes the aggregate of this entry's node and all of its ancestors.
    fn recompute(&mut self) {
        // SAFETY: `self.handle` is not used while the reborrow is alive.
        let handle = unsafe { self.handle.reborrow_mut() };
        handle.into_node().recompute_and_ascend();
    }
}

/// A guard over a single value, produced by [`BTree::get_mut`].
///
/// On drop the cached aggregates along the path to the root are recomputed.
pub struct OccupiedValue<'a, K, V: Aggregate> {
    pub(crate) node: NonNull<LeafNode<K, V>>,
    pub(crate) height: usize,
    pub(crate) index: usize,
    pub(crate) _marker: PhantomData<(&'a mut BTree<K, V>, &'a K, &'a mut V)>,
}

impl<'a, K, V: Aggregate> OccupiedValue<'a, K, V> {
    /// The key this guard refers to.
    pub fn key(&self) -> &K {
        // SAFETY: `index < len`.
        unsafe { (*self.node.as_ptr()).keys[self.index].assume_init_ref() }
    }

    /// The value this guard refers to.
    pub fn get(&self) -> &V {
        // SAFETY: `index < len`.
        unsafe { (*self.node.as_ptr()).values[self.index].assume_init_ref() }
    }

    /// A mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: `index < len` and we have exclusive access.
        unsafe { (*self.node.as_ptr()).values[self.index].assume_init_mut() }
    }
}

impl<K, V: Aggregate> Drop for OccupiedValue<'_, K, V> {
    fn drop(&mut self) {
        let node = NodeRef::<marker::Mut<'_>, K, V, marker::Either> {
            node: self.node,
            height: self.height,
            _marker: PhantomData,
        };
        node.recompute_and_ascend();
    }
}
