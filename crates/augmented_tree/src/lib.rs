use std::{mem::MaybeUninit, ptr::NonNull};

mod node {
    trait NodePtr<K, V> {
        const CAPACITY: usize;
        const MIN_LEN: usize = Self::CAPACITY.div_ceil(2) - 1;
    }
}
