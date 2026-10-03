macro_rules! impl_nonmax_type {
    ($($vis:vis struct $name:ident($int:ident is $pat:pat)),* $(,)?) => {
        $(impl_nonmax_type!(@impl $vis struct $name($int is $pat));)*
    };
    (@impl $vis:vis struct $name:ident($int:ident is $pat:pat)) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        $vis struct $name(pattern_type!($int is $pat));

        const _: () = {
            assert!(core::mem::size_of::<$name>() == core::mem::size_of::<$int>());
            assert!(core::mem::size_of::<Option<$name>>() == core::mem::size_of::<$int>());
        };

        impl $name {
            pub const fn new(value: $int) -> Option<Self> {
                if let $pat = value {
                    // SAFETY: The value is guaranteed to be in the valid range for this type.
                    Some($name(unsafe { core::mem::transmute::<$int, pattern_type!($int is $pat)>(value) }))
                } else {
                    None
                }
            }

            /// # Safety
            /// The caller must ensure that `value` is in the valid range for this type.
            pub const unsafe fn new_unchecked(value: $int) -> Self {
                    // SAFETY: The value is guaranteed to be in the valid range
                    // for this type by the caller
                    $name(unsafe { core::mem::transmute::<$int, pattern_type!($int is $pat)>(value) })
            }

            pub const fn get(self) -> $int {
                unsafe { core::mem::transmute(self) }
            }

            pub const fn as_inner(self) -> $int {
                self.get()
            }
        }

        impl core::marker::StructuralPartialEq for $name {}
        impl Eq for $name {}
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.get() == other.get()
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                Ord::cmp(&self.get(), &other.get())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(Ord::cmp(self, other))
            }
        }

        impl core::hash::Hash for $name {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                core::hash::Hash::hash(&self.get(), state)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$int as core::fmt::Debug>::fmt(&self.get(), f)
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$int as core::fmt::Display>::fmt(&self.get(), f)
            }
        }
    };
}

impl_nonmax_type!(
    pub struct NonMaxUsize(usize is 0..=0xFFFFFFFFFFFFFFFE),
    pub struct NonMaxIsize(isize is 0..=0x7FFFFFFFFFFFFFFE),
    pub struct NonMaxU32(u32 is 0..=0xFFFFFFFE),
    pub struct NonMaxI32(i32 is 0..=0x7FFFFFFE)
);
