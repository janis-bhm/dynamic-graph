use std::ptr::NonNull;

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
    align.trailing_zeros() + (64 - 56)
}

#[expect(dead_code)]
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

#[expect(dead_code)]
pub trait AssertNumeric {
    fn assert_eq(self, other: Self) -> Self;
    fn assert_ne(self, other: Self) -> Self;
    fn assert_lt(self, other: Self) -> Self;
    fn assert_le(self, other: Self) -> Self;
    fn assert_gt(self, other: Self) -> Self;
    fn assert_ge(self, other: Self) -> Self;
    fn assert_in(self, range: impl std::ops::RangeBounds<Self>) -> Self;
}

macro_rules! impl_assert_num {
    ($($t:ty),*) => {
        $(
            impl AssertNumeric for $t {
                fn assert_eq(self, other: Self) -> Self {
                    assert_eq!(self, other);
                    self
                }
                fn assert_ne(self, other: Self) -> Self {
                    assert_ne!(self, other);
                    self
                }
                fn assert_lt(self, other: Self) -> Self {
                    assert!(self < other);
                    self
                }
                fn assert_le(self, other: Self) -> Self {
                    assert!(self <= other);
                    self
                }
                fn assert_gt(self, other: Self) -> Self {
                    assert!(self > other);
                    self
                }
                fn assert_ge(self, other: Self) -> Self {
                    assert!(self >= other);
                    self
                }
                fn assert_in(self, range: impl std::ops::RangeBounds<Self>) -> Self {
                    assert!(range.contains(&self));
                    self
                }
            }
        )*
    };
}

impl_assert_num!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

// debug
#[allow(dead_code)]
pub(crate) struct DisplayOpt<T>(Option<T>);

impl<T> core::fmt::Display for DisplayOpt<T>
where
    T: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.0 {
            Some(value) => value.fmt(f),
            None => write!(f, "None"),
        }
    }
}

#[allow(dead_code)]
pub(crate) trait DisplayOptExt<T> {
    fn display(&self) -> DisplayOpt<&T>;
}

impl<T: core::fmt::Display> DisplayOptExt<T> for Option<T> {
    fn display(&self) -> DisplayOpt<&T> {
        DisplayOpt(self.as_ref())
    }
}
