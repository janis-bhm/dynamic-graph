use std::{mem::ManuallyDrop, ptr::NonNull};

pub struct WithDrop<F: FnOnce(&mut T), T> {
    t: T,
    f: ManuallyDrop<F>,
}

impl<F: FnOnce(&mut T), T> Iterator for WithDrop<F, T>
where
    T: Iterator,
{
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next()
    }
}

impl<F: FnOnce(&mut T), T> std::ops::Deref for WithDrop<F, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl<F: FnOnce(&mut T), T> std::ops::DerefMut for WithDrop<F, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

impl<F: FnOnce(&mut T), T> WithDrop<F, T> {
    fn new(t: T, f: F) -> Self {
        Self {
            t,
            f: ManuallyDrop::new(f),
        }
    }
}

pub trait WithDropExt<T> {
    fn with_drop<F: FnOnce(&mut T)>(self, f: F) -> WithDrop<F, T>;
}

impl<T> WithDropExt<T> for T {
    fn with_drop<F: FnOnce(&mut T)>(self, f: F) -> WithDrop<F, T> {
        WithDrop::new(self, f)
    }
}

impl<F: FnOnce(&mut T), T> Drop for WithDrop<F, T> {
    fn drop(&mut self) {
        let f = unsafe { ManuallyDrop::take(&mut self.f) };
        f(&mut self.t);
    }
}

/// # Safety
/// `BITS` must not be less than the number of bits required to store the tag.
/// That is to say, 2^`BITS` must be greater than any value produced by `into_usize`.
pub unsafe trait Tag: Copy {
    const BITS: u32;
    fn into_usize(self) -> usize;
    unsafe fn from_usize(tag: usize) -> Self;
}

const fn align_of<T: ?Sized + Aligned>() -> usize {
    T::ALIGN
}

/// # Safety
/// `ALIGN` must be the alignment of `Self`.
pub unsafe trait Aligned {
    /// Alignment of `Self`.
    const ALIGN: usize;
}

unsafe impl<T> Aligned for T {
    const ALIGN: usize = core::mem::align_of::<Self>();
}

unsafe impl<T> Aligned for [T] {
    const ALIGN: usize = core::mem::align_of::<T>();
}

const fn bits_for<T: ?Sized + Aligned>() -> u32 {
    let align = align_of::<T>();
    align.trailing_zeros()
}

const fn bits_for_tags(mut tags: &[usize]) -> u32 {
    let mut bits = 0;
    while let &[tag, ref rest @ ..] = tags {
        tags = rest;
        let b = usize::BITS - tag.leading_zeros();
        if b > bits {
            bits = b;
        }
    }

    bits
}

#[derive(Debug, Clone, Copy)]
pub struct TaggedPtr<P: Aligned + ?Sized, T> {
    packed: *mut P,
    _marker: core::marker::PhantomData<T>,
}

impl<P, T> TaggedPtr<P, T>
where
    T: Tag,
    P: Aligned + ?Sized,
{
    pub fn new(ptr: *mut P, tag: T) -> Self {
        Self {
            packed: Self::pack(ptr, tag),
            _marker: core::marker::PhantomData,
        }
    }

    const ASSERTION: () = { assert!(T::BITS <= bits_for::<P>(), "Not enough bits to store tag") };
    const TAG_BIT_SHIFT: u32 = usize::BITS - T::BITS;

    pub fn pack(ptr: *mut P, tag: T) -> *mut P {
        let () = Self::ASSERTION;

        let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

        ptr.map_addr(|addr| (addr >> T::BITS) | packed_tag)
    }

    pub fn tag(&self) -> T {
        let packed_addr = self.packed.addr();
        let tag_bits = packed_addr >> Self::TAG_BIT_SHIFT;
        unsafe { T::from_usize(tag_bits) }
    }

    pub fn set_tag(&mut self, tag: T) {
        let ptr = self.as_ptr();
        self.packed = Self::pack(ptr, tag);
    }

    pub fn update_tag<F: FnOnce(&mut T)>(&mut self, f: F) {
        let mut tag = self.tag();
        f(&mut tag);
        self.set_tag(tag);
    }

    pub fn as_ptr(&self) -> *mut P {
        self.packed.map_addr(|addr| addr << T::BITS)
    }

    pub fn set_ptr(&mut self, ptr: *mut P) {
        let tag = self.tag();
        self.packed = Self::pack(ptr, tag);
    }

    pub fn as_non_null(&self) -> Option<NonNull<P>> {
        NonNull::new(self.as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tagged_ptr() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum MyTag {
            A,
            B,
            C,
        }

        unsafe impl Tag for MyTag {
            const BITS: u32 = 2;

            fn into_usize(self) -> usize {
                match self {
                    MyTag::A => 0,
                    MyTag::B => 1,
                    MyTag::C => 2,
                }
            }

            unsafe fn from_usize(tag: usize) -> Self {
                match tag {
                    0 => MyTag::A,
                    1 => MyTag::B,
                    2 => MyTag::C,
                    _ => panic!("Invalid tag value"),
                }
            }
        }

        let mut x = 42;
        let mut tagged_ptr = TaggedPtr::<i32, MyTag>::new(&mut x as *mut i32, MyTag::A);

        assert_eq!(tagged_ptr.tag(), MyTag::A);
        assert_eq!(unsafe { *tagged_ptr.as_ptr() }, 42);

        tagged_ptr.set_tag(MyTag::B);
        assert_eq!(tagged_ptr.tag(), MyTag::B);

        tagged_ptr.update_tag(|tag| {
            if let MyTag::B = tag {
                *tag = MyTag::C;
            }
        });
        assert_eq!(tagged_ptr.tag(), MyTag::C);
    }
}
