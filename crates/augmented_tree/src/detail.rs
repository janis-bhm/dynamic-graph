//! ---------------------------------------------------------------------------
//! Low-level slice helpers
//! ---------------------------------------------------------------------------

use std::mem::MaybeUninit;
use std::ptr;

/// Inserts a value into a slice of initialized elements followed by one
/// uninitialized element.
///
/// # Safety
/// `idx < slice.len()`.
pub(super) unsafe fn slice_insert<T>(slice: &mut [MaybeUninit<T>], idx: usize, val: T) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let len = slice.len();
        debug_assert!(len > idx);
        let slice_ptr = slice.as_mut_ptr();
        if len > idx + 1 {
            ptr::copy(slice_ptr.add(idx), slice_ptr.add(idx + 1), len - idx - 1);
        }
        (*slice_ptr.add(idx)).write(val);
    }
}

/// Removes and returns a value from a slice of all initialized elements,
/// leaving a trailing uninitialized element.
///
/// # Safety
/// `idx < slice.len()`.
pub(super) unsafe fn slice_remove<T>(slice: &mut [MaybeUninit<T>], idx: usize) -> T {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let len = slice.len();
        debug_assert!(idx < len);
        let slice_ptr = slice.as_mut_ptr();
        let ret = (*slice_ptr.add(idx)).assume_init_read();
        ptr::copy(slice_ptr.add(idx + 1), slice_ptr.add(idx), len - idx - 1);
        ret
    }
}

/// Shifts the elements in a slice `distance` positions to the left.
///
/// # Safety
/// `slice.len() >= distance`.
pub(super) unsafe fn slice_shl<T>(slice: &mut [MaybeUninit<T>], distance: usize) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let slice_ptr = slice.as_mut_ptr();
        ptr::copy(slice_ptr.add(distance), slice_ptr, slice.len() - distance);
    }
}

/// Shifts the elements in a slice `distance` positions to the right.
///
/// # Safety
/// `slice.len() >= distance`.
pub(super) unsafe fn slice_shr<T>(slice: &mut [MaybeUninit<T>], distance: usize) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let slice_ptr = slice.as_mut_ptr();
        ptr::copy(slice_ptr, slice_ptr.add(distance), slice.len() - distance);
    }
}

/// Moves initialized elements into an uninitialized slice.
pub(super) fn move_to_slice<T>(src: &mut [MaybeUninit<T>], dst: &mut [MaybeUninit<T>]) {
    assert_eq!(src.len(), dst.len());
    // SAFETY: the slices do not overlap and `src` is fully initialized.
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
    }
}
