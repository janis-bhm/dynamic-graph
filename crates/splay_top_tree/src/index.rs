use slotvec::Indexing;

#[repr(transparent)]
#[derive(Clone, Copy, Default, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(u32);

impl Generation {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn increment(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }

    pub fn increment_and_get(&mut self) -> Self {
        let current = *self;
        self.increment();
        current
    }

    pub fn current(&self) -> Self {
        *self
    }
}

#[macro_export]
macro_rules! impl_id {
    ($($vis:vis struct $name:ident),* $(,)?) => {
        $(impl_id!(@impl $vis struct $name);)*
    };
    (@impl $vis:vis struct $name:ident) => {
        #[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
        $vis struct $name {
            index: $crate::non_max::NonMaxUsize,
            #[cfg(debug_assertions)]
            generation: $crate::index::Generation,
        }

        impl $name {
            fn new(index: $crate::non_max::NonMaxUsize, #[cfg(debug_assertions)] generation: $crate::index::Generation) -> Self {
                Self {
                    index,
                    #[cfg(debug_assertions)]
                    generation,
                }
            }

            fn new_from_usize(index: usize, generation: $crate::index::Generation) -> Self {
                Self {
                    index: $crate::non_max::NonMaxUsize::new(index).expect("Index must be non-zero"),
                    #[cfg(debug_assertions)]
                    generation,
                }
            }

            pub fn index(&self) -> usize {
                self.index.get()
            }

            #[cfg(debug_assertions)]
            fn generation(&self) -> $crate::index::Generation {
                self.generation
            }
        }
    };
}

impl_id! {
    pub struct VertexId,
    pub struct EdgeId,
    pub struct LabelId,
}

impl Indexing for VertexId {
    type Optional = Option<VertexId>;

    fn get(&self) -> usize {
        self.index.get()
    }

    fn new(value: usize) -> Self {
        Self::new_from_usize(value, Generation(0))
    }
}

impl Indexing for EdgeId {
    type Optional = Option<EdgeId>;

    fn get(&self) -> usize {
        self.index.get()
    }

    fn new(value: usize) -> Self {
        Self::new_from_usize(value, Generation(0))
    }
}

impl Indexing for LabelId {
    type Optional = Option<LabelId>;

    fn get(&self) -> usize {
        self.index.get()
    }

    fn new(value: usize) -> Self {
        Self::new_from_usize(value, Generation(0))
    }
}
