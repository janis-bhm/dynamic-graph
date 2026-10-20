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
