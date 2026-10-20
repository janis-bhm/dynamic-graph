#![expect(internal_features)]
#![feature(pattern_type_macro, pattern_types, step_trait)]

use core::ops::{Add, AddAssign, BitAnd, Sub, SubAssign};

use augmented_tree::Aggregate;
use splay_top_tree::{Boundary, EdgeId, LabelId, Summary, VertexId};

#[expect(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct u5(pattern_type!(u8 is 0..=32));

impl u5 {
    pub const fn as_inner(self) -> u8 {
        unsafe { core::mem::transmute::<u5, u8>(self) }
    }

    pub const fn new(value: u8) -> Self {
        if let 0..=32 = value {
            unsafe { core::mem::transmute::<u8, u5>(value) }
        } else {
            panic!("Value out of range for u5");
        }
    }
}

impl PartialEq for u5 {
    fn eq(&self, other: &Self) -> bool {
        self.as_inner() == other.as_inner()
    }
}

impl Eq for u5 {}

impl PartialOrd for u5 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for u5 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_inner().cmp(&other.as_inner())
    }
}

impl core::hash::Hash for u5 {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_inner().hash(state);
    }
}

impl core::fmt::Debug for u5 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "u5({})", self.as_inner())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Uncovered,
    Covered(u5),
    NoCover,
}

impl Level {
    const MAX_LEVEL: u8 = 32;
    pub const MAX: Self = Level::new(Self::MAX_LEVEL);
    pub const MIN: Self = Level::Uncovered;

    pub const fn new(level: u8) -> Self {
        assert!(level <= 32);
        Self::Covered(unsafe { core::mem::transmute::<u8, u5>(level) })
    }

    pub const fn increment(self) -> Self {
        match self {
            Level::NoCover => Level::NoCover,
            Level::Uncovered => Level::new(0),
            Level::Covered(l) => {
                let l: u8 = l.as_inner();
                Level::new(l + 1)
            }
        }
    }

    pub const fn decrement(self) -> Self {
        match self {
            Level::NoCover => Level::new(Self::MAX_LEVEL - 1),
            Level::Uncovered => Level::Uncovered,
            Level::Covered(l) => {
                let l: u8 = l.as_inner();
                if l == 0 {
                    Level::Uncovered
                } else {
                    Level::new(l - 1)
                }
            }
        }
    }

    pub const fn as_bitmask(self) -> u64 {
        match self {
            Level::Uncovered => 1,
            Level::Covered(l) => 1 << (l.as_inner() + 1),
            Level::NoCover => Self::MAX.as_bitmask(),
        }
    }
}

impl core::iter::Step for Level {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        let distance = (end.as_bitmask() - start.as_bitmask()) as usize;

        (distance, Some(distance))
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        if count == 0 {
            return Some(start);
        }

        match start {
            Self::Uncovered => Some(Self::new(count as u8 - 1u8)),
            Self::Covered(_) if start == Self::MAX => None,
            Self::Covered(l) => {
                let new_level = l.as_inner() + count as u8;
                if new_level > Self::MAX_LEVEL {
                    None
                } else {
                    Some(Self::new(new_level))
                }
            }
            Self::NoCover => None,
        }
    }

    fn forward_overflowing(start: Self, count: usize) -> (Self, bool) {
        <Self as core::iter::Step>::forward_checked(start, count)
            .map_or((Self::MAX, true), |new_level| (new_level, false))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        if count == 0 {
            return Some(start);
        }

        match start {
            Self::Uncovered => None,
            Self::Covered(l) => {
                let new_level = l.as_inner() as isize - count as isize;
                if new_level < 0 {
                    None
                } else {
                    Some(Self::new(new_level as u8))
                }
            }
            Self::NoCover => {
                let new_level = Self::MAX_LEVEL as isize - count as isize;
                if new_level < 0 {
                    None
                } else {
                    Some(Self::new(new_level as u8))
                }
            }
        }
    }

    fn backward_overflowing(start: Self, count: usize) -> (Self, bool) {
        <Self as core::iter::Step>::backward_checked(start, count)
            .map_or((Self::Uncovered, true), |new_level| (new_level, false))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Incident {
    bits: u64,
}

impl Incident {
    pub fn empty() -> Self {
        Self { bits: 0 }
    }
    pub fn contains(&self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }
    pub fn from_level(level: Level) -> Self {
        Self {
            bits: level.as_bitmask(),
        }
    }
}

impl Add for Incident {
    type Output = Self;

    #[expect(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl AddAssign for Incident {
    #[expect(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl Sub for Incident {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & !rhs.bits,
        }
    }
}

impl SubAssign for Incident {
    fn sub_assign(&mut self, rhs: Self) {
        self.bits &= !rhs.bits;
    }
}

impl BitAnd<Level> for Incident {
    type Output = Incident;

    fn bitand(self, rhs: Level) -> Self::Output {
        let mask = (1u64 << rhs.as_bitmask()) - 1;
        Incident {
            bits: self.bits & mask,
        }
    }
}

impl BitAnd<Level> for &Incident {
    type Output = Incident;

    fn bitand(self, rhs: Level) -> Self::Output {
        let mask = (1u64 << rhs.as_bitmask()) - 1;
        Incident {
            bits: self.bits & mask,
        }
    }
}

/// A size vector over slots `0..SLOTS` (slot = level + 1). Trailing zero slots
/// are implicit: the backing slice ends at the last nonzero slot and `get`
/// returns 0 beyond it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Size(Box<[u32]>);

impl core::ops::Index<Level> for Size {
    type Output = u32;

    fn index(&self, index: Level) -> &Self::Output {
        let slot = index.as_bitmask() as usize;
        if slot < self.0.len() {
            &self.0[slot]
        } else {
            &0
        }
    }
}

impl core::ops::IndexMut<Level> for Size {
    fn index_mut(&mut self, index: Level) -> &mut Self::Output {
        let slot = index.as_bitmask() as usize;
        if slot >= self.0.len() {
            self.grow_to_fit(index);
        }
        &mut self.0[slot]
    }
}

impl Size {
    /// Canonical form: drop trailing zeros.
    fn from_vec(mut data: Vec<u32>) -> Self {
        while data.last() == Some(&0) {
            data.pop();
        }
        Size(data.into_boxed_slice())
    }

    fn zeroed(size: usize) -> Self {
        Size(unsafe { Box::new_zeroed_slice(size).assume_init() })
    }

    fn grow_to_fit(&mut self, level: Level) {
        let slot = level.as_bitmask() as usize;
        if slot >= self.0.len() {
            let mut new = Self::zeroed(slot);
            new.0[..self.0.len()].copy_from_slice(&self.0);
            *self = new;
        }
    }

    fn empty() -> Self {
        Self(Box::new([]))
    }

    fn get(&self, level: Level) -> u32 {
        self.0
            .get(level.as_bitmask() as usize)
            .copied()
            .unwrap_or(0)
    }

    fn add_impl(&self, other: &Self) -> Self {
        let new_len = self.0.len().max(other.0.len());
        let mut out = Self::zeroed(new_len);
        for slot in Level::MIN
            ..=Level::new(u8::try_from(new_len).expect("Size vector length exceeds u8::MAX"))
        {
            out[slot] = self.get(slot) + other.get(slot);
        }
        out
    }

    fn add_assign_impl(&mut self, other: &Self) {
        if self.0.len() < other.0.len() {
            *self = self.add_impl(other);
        } else {
            for slot in 0..other.0.len() {
                self.0[slot] += other.0[slot];
            }
        }
    }
}

impl Add for &Size {
    type Output = Size;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl Add<Size> for &Size {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl Add for Size {
    type Output = Size;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign_impl(&rhs);
        self
    }
}

impl Add<&Size> for Size {
    type Output = Size;

    fn add(mut self, rhs: &Size) -> Self::Output {
        self.add_assign_impl(rhs);
        self
    }
}

impl AddAssign<&Size> for Size {
    fn add_assign(&mut self, rhs: &Size) {
        self.add_assign_impl(rhs)
    }
}

impl AddAssign<Size> for Size {
    fn add_assign(&mut self, rhs: Size) {
        self.add_assign_impl(&rhs)
    }
}

impl BitAnd<Level> for &Size {
    type Output = Size;

    fn bitand(self, rhs: Level) -> Self::Output {
        let slots = rhs.as_bitmask() as usize;
        let mut out = Size::zeroed(slots);
        out.0.copy_from_slice(&self.0[..slots.min(self.0.len())]);

        out
    }
}

impl BitAnd<Level> for Size {
    type Output = Size;

    fn bitand(self, rhs: Level) -> Self::Output {
        let slots = rhs.as_bitmask() as usize;
        let mut out = Size::zeroed(slots);
        out.0.copy_from_slice(&self.0[..slots.min(self.0.len())]);

        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PartEntry {
    part: Size,
    diag: Size,
    incident: Incident,
}

impl Default for PartEntry {
    fn default() -> Self {
        Self {
            part: Size::empty(),
            diag: Size::empty(),
            incident: Incident::empty(),
        }
    }
}

impl Aggregate for PartEntry {
    fn reduce(&self, other: &Self) -> Self {
        Self {
            part: &self.part + &other.part,
            diag: &self.diag + &other.diag,
            incident: self.incident + other.incident,
        }
    }

    fn identity() -> Self {
        Self {
            part: Size::empty(),
            diag: Size::empty(),
            incident: Incident::empty(),
        }
    }
}

type PartTree = augmented_tree::BTree<Level, PartEntry>;

pub struct ClusterData {
    pub cover: Level,
    pub global_cover: Level,
    pending_cover: Level,
    pending_uncover: Level,
    pub min_path_edge: Option<EdgeId>,
    pub min_global_edge: Option<EdgeId>,
    pub incident: Incident,
    size: Size,
    num_vertices: u32,
    part: [PartTree; 2],
}

impl Summary for ClusterData {
    fn create_edge(e: EdgeId) -> Self {
        Self {
            cover: Level::Uncovered,
            global_cover: Level::Uncovered,
            pending_cover: Level::Uncovered,
            pending_uncover: Level::Uncovered,
            min_path_edge: Some(e),
            min_global_edge: Some(e),
            incident: Incident::empty(),
            size: Size::zeroed(0),
            num_vertices: 2,
            part: [PartTree::new(), PartTree::new()],
        }
    }

    fn create_label(_l: LabelId) -> Self {
        let mut size = Size::zeroed(1);
        size[Level::Uncovered] = 1;

        Self {
            cover: Level::NoCover,
            global_cover: Level::NoCover,
            pending_cover: Level::Uncovered,
            pending_uncover: Level::Uncovered,
            min_path_edge: None,
            min_global_edge: None,
            incident: Incident::empty(),
            size,
            num_vertices: 1,
            part: [PartTree::new(), PartTree::new()],
        }
    }

    fn flip(&mut self) {
        self.part.swap(0, 1);
    }

    fn split(&self, left: &mut Self, right: &mut Self, ctx: &splay_top_tree::MergeContext) {
        // CL.Split
        for (child, _) in [(left, ctx.left_boundary), (right, ctx.right_boundary)]
            .into_iter()
            .filter(|(_, boundary)| boundary.is_path())
        {
            if child.cover.max(child.pending_uncover) <= self.pending_uncover {
                child.pending_uncover = self.pending_uncover;
            }
            if child.cover <= self.pending_cover.max(self.pending_uncover) {
                child.cover = self.pending_cover;
                child.pending_cover = self.pending_cover;
            }
        }
    }

    fn merge(left: &Self, right: &Self, ctx: &splay_top_tree::MergeContext) -> Self {
        let parent_is_path = ctx.parent_boundary.is_path();
        let left_is_path = ctx.left_boundary.is_path();
        let right_is_path = ctx.right_boundary.is_path();

        // CL.Merge
        let min_cover_path_child = parent_is_path.then(|| {
            [(left, left_is_path), (right, right_is_path)]
                .into_iter()
                .filter(|(_, b)| *b)
                .min_by_key(|(c, _)| c.cover)
                .expect("at least one child must be a path child")
        });

        let cover = min_cover_path_child
            .map(|(c, _)| c.cover)
            .unwrap_or(Level::MAX);

        let min_path_edge = min_cover_path_child.and_then(|(c, _)| c.min_path_edge);

        // global cover and min global edge are the minimal of globalcover' and minglobaledge' of the children, where:
        //
        // globalcover' = if path_child or globalcover<=cover {globalcover} else {cover}
        //
        // minglobaledge' = if path_child or globalcover<=cover {minglobaledge} else {minpathedge}

        let (global_cover, min_global_edge) = [(left, left_is_path), (right, right_is_path)]
            .into_iter()
            .map(|(c, b)| {
                if (parent_is_path && b) || c.global_cover <= cover {
                    (c.global_cover, c.min_global_edge)
                } else {
                    (cover, c.min_path_edge)
                }
            })
            .reduce(|(rc, re), (lc, le)| (rc.min(lc), re.min(le)))
            .expect("two children exist, reduction returns `Some`");

        // FS.Merge

        struct ChildData<'a> {
            v: VertexId,
            b: Boundary,
            cl: &'a ClusterData,
            part: [PartTree; 2],
            undo: [Option<PartTree>; 2],
        }

        match ((left, ctx.left_boundary), (right, ctx.right_boundary)) {
            (
                (path, path_boundary @ Boundary::Two { .. }),
                (point, point_boundary @ Boundary::One(_)),
            )
            | (
                (point, point_boundary @ Boundary::One(_)),
                (path, path_boundary @ Boundary::Two { .. }),
            ) if !ctx.parent_boundary.is_path() => {
                // heterogeneous point cluster
                _ = (path, path_boundary, point, point_boundary);
            }
            ((l, lb), (r, rb)) => {
                let size = &l.size + &r.size;

                let c = lb.shared(&rb).expect("children must share a vertex");

                let (vl, vr) = match ctx.parent_boundary {
                    Boundary::Two { left, right } => {
                        assert!(lb.contains(left) && rb.contains(right));
                        (left, right)
                    }
                    _ => (c, c),
                };

                let mut left = ChildData {
                    v: vl,
                    b: lb,
                    cl: l,
                    part: [PartTree::new(), PartTree::new()],
                    undo: [None, None],
                };
                let mut right = ChildData {
                    v: vr,
                    b: rb,
                    cl: r,
                    part: [PartTree::new(), PartTree::new()],
                    undo: [None, None],
                };

                let mut parts = [PartTree::new(), PartTree::new()];

                for child @ &mut ChildData { v, b, cl, .. } in [&mut left, &mut right] {
                    let l = cl.pending_cover.max(cl.pending_uncover);

                    if v != c {
                        for v in [v, c] {
                            let tree = &cl.part[b.index_of(v).unwrap()];
                            let sum = &tree.aggregate().part;
                            let diag = sum & cl.pending_cover;

                            let leq_tree: PartTree =
                                tree.range(..=l).map(|(k, v)| (*k, v.clone())).collect();

                            let mut gt_tree: PartTree =
                                tree.range(l..).map(|(k, v)| (*k, v.clone())).collect();

                            gt_tree.entry(cl.pending_cover).update_or_insert_with(
                                |entry| {
                                    entry.part += sum;
                                    entry.diag += &diag;
                                },
                                || PartEntry {
                                    part: sum.clone(),
                                    diag: diag.clone(),
                                    incident: Incident::empty(),
                                },
                            );

                            child.undo[b.index_of(v).unwrap()] = Some(leq_tree);
                            child.part[b.index_of(v).unwrap()] = gt_tree;
                        }
                    } else {
                        let tree = cl.part[b.index_of(v).unwrap()].clone();
                        child.part[b.index_of(c).unwrap()] = tree;
                        child.undo = [None, None];
                    };
                }

                let mut asdf = |x: &mut ChildData, y: &mut ChildData| {
                    let x_tree = &x.part[x.b.index_of(x.v).unwrap()];
                    let y_tree = &y.part[y.b.index_of(c).unwrap()];
                    //asd
                    let suffix = y_tree.range_aggregate(x.cl.cover..);

                    let x_at_cover = x_tree.get(&x.cl.cover).cloned().unwrap_or_default();

                    let part = &x_at_cover.part + &suffix.part;
                    let diag = &x_at_cover.diag + &(&suffix.part & x.cl.cover);

                    let (xtree2, xundo2) = if c == x.v {
                        (
                            PartTree::from_iter([(
                                Level::MAX,
                                PartEntry {
                                    part: x.cl.size.clone(),
                                    diag: x.cl.size.clone(),
                                    incident: Incident::empty(),
                                },
                            )]),
                            None,
                        )
                    } else {
                        (
                            x_tree
                                .range(x.cl.cover.increment()..)
                                .map(|(k, v)| (*k, v.clone()))
                                .collect::<PartTree>(),
                            Some(
                                x_tree
                                    .range(..=x.cl.cover)
                                    .map(|(k, v)| (*k, v.clone()))
                                    .collect::<PartTree>(),
                            ),
                        )
                    };

                    let (ytree3, yundo3) = if c == y.v {
                        (
                            None,
                            PartTree::from_iter([(
                                Level::MAX,
                                PartEntry {
                                    part: y.cl.size.clone(),
                                    diag: y.cl.size.clone(),
                                    incident: Incident::empty(),
                                },
                            )]),
                        )
                    } else {
                        (
                            Some(
                                y_tree
                                    .range(..x.cl.cover)
                                    .map(|(k, v)| (*k, v.clone()))
                                    .collect::<PartTree>(),
                            ),
                            y_tree
                                .range(x.cl.cover..)
                                .map(|(k, v)| (*k, v.clone()))
                                .collect::<PartTree>(),
                        )
                    };

                    let mut tree = xtree2.clone();
                    tree.insert(
                        x.cl.cover,
                        PartEntry {
                            part,
                            diag,
                            incident: Incident::empty(),
                        },
                    );
                    tree.extend(&ytree3.unwrap_or_default());

                    parts[ctx.parent_boundary.index_of(x.v).unwrap()] = tree;

                    let _ = (xtree2, xundo2, ytree3, yundo3, tree);
                };

                asdf(&mut left, &mut right);
                asdf(&mut right, &mut left);

                _ = (l, r, c);
            }
        }

        if parent_is_path {
            let size = &left.size + &right.size;
        } else if let Some((path, point)) = match ((left, left_is_path), (right, right_is_path)) {
            ((l, true), (r, false)) => Some((l, r)),
            ((l, false), (r, true)) => Some((r, l)),
            _ => None,
        } {
            // heterogeneous point cluster
        } else {
            // both children are point clusters
        }

        Self {
            cover,
            global_cover,
            pending_cover: Level::Uncovered,
            pending_uncover: Level::Uncovered,
            min_path_edge,
            min_global_edge,
            ..todo!()
        }
    }
}
