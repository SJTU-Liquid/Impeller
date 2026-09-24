use std::fmt::Debug;

use crate::transit::type2::memory_planning::prelude::*;

#[derive(Debug, Clone)]
struct FreeBlock<Cf> {
    free_since: usize,
    criterion: Cf,
}

impl<Cf> mm_heap::Priority<Cf> for FreeBlock<Cf>
where
    Cf: Clone,
{
    fn priority(&self) -> Cf {
        self.criterion.clone()
    }
}

#[derive(Clone)]
struct SplittedBlock<C, P> {
    live_since: usize,
    criterion: C,
    pointer: P,
}

impl<C, P> mm_heap::Priority<C> for SplittedBlock<C, P>
where
    C: Clone,
{
    fn priority(&self) -> C {
        self.criterion.clone()
    }
}

#[derive(Clone)]
struct OccupiedBlock<C, P> {
    live_since: usize,
    criterion: C,
    pointer: P,
}

impl<C, P> mm_heap::Priority<C> for OccupiedBlock<C, P>
where
    C: Clone,
{
    fn priority(&self) -> C {
        self.criterion.clone()
    }
}

enum Block<'b, C, Cf, P> {
    Free(&'b FreeBlock<Cf>),
    Occupied(&'b OccupiedBlock<C, P>),
    Splitted(&'b SplittedBlock<C, P>),
}

impl<'b, C, Cf, P> Block<'b, C, Cf, P> {
    fn live_since(&self) -> Option<usize> {
        use Block::*;
        match self {
            Occupied(OccupiedBlock { live_since, .. })
            | Splitted(SplittedBlock { live_since, .. }) => Some(*live_since),
            _ => None,
        }
    }

    fn victim_criterion(&self) -> Option<C>
    where
        C: Clone,
    {
        use Block::*;
        match self {
            Occupied(OccupiedBlock { criterion, .. })
            | Splitted(SplittedBlock { criterion, .. }) => Some(criterion.clone()),
            _ => None,
        }
    }

    fn _pointer(&self) -> Option<P>
    where
        P: Clone,
    {
        use Block::*;
        match self {
            Occupied(OccupiedBlock { pointer, .. }) | Splitted(SplittedBlock { pointer, .. }) => {
                Some(pointer.clone())
            }
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Map<T, C>(BTreeMap<IntegralSize, MmHeap<Addr, C, T>>);

impl<T, C> Map<T, C> {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn of_size(&self, size: IntegralSize) -> Option<&MmHeap<Addr, C, T>> {
        self.0.get(&size)
    }
}

impl<T, C> Map<T, C>
where
    C: Clone + Ord,
{
    fn at(&self, size: IntegralSize, addr: Addr) -> Option<&T> {
        self.0.get(&size)?.get(&addr)
    }
}

impl<T, C> Map<T, C>
where
    C: Clone + Ord,
    T: mm_heap::Priority<C>,
{
    fn update_at(&mut self, size: IntegralSize, addr: Addr, t: T) -> Option<T> {
        let blocks = self.0.entry(size).or_default();
        blocks.insert(addr, t)
    }

    fn update_at_with(
        &mut self,
        size: IntegralSize,
        addr: Addr,
        f: impl FnOnce(T) -> T,
    ) -> Option<()> {
        let blocks = self.0.entry(size).or_default();
        blocks.update_with(addr, f)
    }

    fn remove(&mut self, size: IntegralSize, addr: Addr) -> Option<T> {
        self.0.get_mut(&size)?.remove(&addr)
    }

    fn best_of_size(&self, size: IntegralSize) -> Option<(Addr, &T)>
    where
        C: Criterion,
    {
        let blocks = self.of_size(size)?;
        let (addr, block) = blocks.max()?;

        if block.priority() == C::bottom() {
            None
        } else {
            Some((*addr, block))
        }
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (IntegralSize, Addr)> + 'a {
        self.0
            .iter()
            .map(|(size, blocks)| {
                let size = size.clone();
                blocks.iter().map(move |(addr, _)| (size, addr.clone()))
            })
            .flatten()
    }
}

#[derive(Clone)]
struct Blocks<C, Cf, P> {
    free: Map<FreeBlock<Cf>, Cf>,
    splitted: Map<SplittedBlock<C, P>, C>,
    occupied: Map<OccupiedBlock<C, P>, C>,
    max_bs: IntegralSize,
}

impl<C, Cf, P> Blocks<C, Cf, P> {
    fn new(max_bs: IntegralSize) -> Self {
        Self {
            free: Map::new(),
            splitted: Map::new(),
            occupied: Map::new(),
            max_bs,
        }
    }

    fn parent(&self, size: IntegralSize, addr: Addr) -> Option<(IntegralSize, Addr)> {
        if size >= self.max_bs {
            None
        } else {
            Some(parent(size, addr))
        }
    }
}

/// Make sure `size` is not already `max_bs` of [`Blocks`]
fn parent(size: IntegralSize, addr: Addr) -> (IntegralSize, Addr) {
    let is = size.double();
    let parent_addr = Addr(addr.0 & (!0u64 << is.0));
    (is, parent_addr)
}

/// Make sure `size` is not already 1
fn children(size: IntegralSize, addr: Addr) -> (IntegralSize, [Addr; 2]) {
    (size.half(), [addr, Addr(addr.0 + 2u64.pow(size.0 - 1))])
}

fn apply_optional<T>(a: Option<T>, b: Option<T>, f: impl FnOnce(T, T) -> T) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn min_optional<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    apply_optional(a, b, |a, b| a.min(b))
}

impl<C, Cf, P> Blocks<C, Cf, P>
where
    Cf: Clone + Ord,
    C: Clone + Ord,
{
    fn at<'a>(&'a self, size: IntegralSize, addr: Addr) -> Option<Block<'a, C, Cf, P>> {
        use Block::*;
        if let Some(fb) = self.free.at(size, addr) {
            Some(Free(fb))
        } else if let Some(ob) = self.occupied.at(size, addr) {
            Some(Occupied(ob))
        } else if let Some(sb) = self.splitted.at(size, addr) {
            Some(Splitted(sb))
        } else {
            None
        }
    }
}

pub type Pointers<P> = Heap<P, (IntegralSize, Addr)>;

struct NonfreeMaker<'p, C, P> {
    now: usize,
    c: C,
    p: Option<P>,
    pointers: &'p mut Pointers<P>,
}

impl<'p, C, P> NonfreeMaker<'p, C, P>
where
    C: Clone,
    P: UsizeId,
{
    fn new(now: usize, c: C, pointers: &'p mut Pointers<P>) -> Self {
        Self {
            now,
            c,
            p: None,
            pointers,
        }
    }

    fn occupied(&self) -> OccupiedBlock<C, P> {
        OccupiedBlock {
            live_since: self.now,
            criterion: self.c.clone(),
            pointer: self.p.clone().unwrap(),
        }
    }

    fn splitted(&self) -> SplittedBlock<C, P> {
        SplittedBlock {
            live_since: self.now,
            criterion: self.c.clone(),
            pointer: self.p.clone().unwrap(),
        }
    }

    fn pointer(&self) -> P {
        self.p.unwrap()
    }

    fn with_addr(&mut self, size: IntegralSize, addr: Addr) -> &mut Self {
        let p = self.pointers.push((size, addr));
        self.p = Some(p);
        self
    }
}

struct InvariantChecker<C, Cf, P> {
    visited: BTreeSet<(IntegralSize, Addr)>,
    _phantom: PhantomData<(C, Cf, P)>,
}

impl<C, Cf, P> InvariantChecker<C, Cf, P>
where
    Cf: Criterion + Debug,
    C: Criterion + Debug,
    P: UsizeId,
{
    fn new() -> Self {
        Self {
            visited: BTreeSet::new(),
            _phantom: PhantomData,
        }
    }

    fn visit(&mut self, size: IntegralSize, addr: Addr) {
        self.visited.insert((size, addr));
    }

    fn visited(&self, size: IntegralSize, addr: Addr) -> bool {
        self.visited.contains(&(size, addr))
    }

    fn check_recurse(&mut self, blocks: &Blocks<C, Cf, P>, size: IntegralSize, addr: Addr) {
        self.visit(size, addr);

        match blocks.at(size, addr) {
            None => panic!("expect to find some block at ({}, {})", size.0, addr.0),
            Some(Block::Free(..)) => {}
            Some(Block::Occupied(..)) => {}
            Some(Block::Splitted(block)) => {
                if size.0 == 0 {
                    panic!("block at {} of size 0 cannot be splitted", addr.0);
                }

                let (children_size, children_addrs) = children(size, addr);

                let [left_block, right_block] = children_addrs.map(|c| {
                    blocks.at(children_size, c).unwrap_or_else(|| {
                        panic!("expect children block at ({}, {}) to exist", size.0, addr.0)
                    })
                });

                let criterion = apply_optional(
                    left_block.victim_criterion(),
                    right_block.victim_criterion(),
                    Criterion::join,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "expect at least one children of ({}, {}) to be non-free",
                        size, addr.0
                    )
                });

                let live_since = min_optional(left_block.live_since(), right_block.live_since())
                    .unwrap_or_else(|| {
                        panic!(
                            "expect at least one children of ({}, {}) to be non-free",
                            size, addr.0
                        )
                    });

                if block.criterion != criterion {
                    panic!("criterion {:?} of splitted block at ({}, {}) is not the join of {:?} and {:?}", &block.criterion, size, addr.0, left_block.victim_criterion(), right_block.victim_criterion());
                }

                if block.live_since != live_since {
                    panic!("live_since {:?} of splitted block at ({}, {}) is not the min of {:?} and {:?}", &block.live_since, size, addr.0, left_block.live_since(), right_block.live_since());
                }

                children_addrs
                    .into_iter()
                    .for_each(|c| self.check_recurse(blocks, children_size, c));
            }
        }
    }

    fn finish(self, blocks: &Blocks<C, Cf, P>) {
        for ((size, addr), kind) in blocks
            .free
            .iter()
            .zip(std::iter::repeat("free"))
            .chain(blocks.splitted.iter().zip(std::iter::repeat("splitted")))
            .chain(blocks.occupied.iter().zip(std::iter::repeat("occupied")))
        {
            if !self.visited(size, addr) {
                panic!(
                    "{} block ({}, {}) is not found in tree but not deleted from blocks",
                    kind, size, addr.0
                )
            }
        }
    }
}

impl<C, Cf, P> Blocks<C, Cf, P>
where
    Cf: Criterion,
    C: Criterion,
    P: UsizeId,
{
    fn propagate_up(&mut self, size: IntegralSize, addr: Addr) {
        if let Some((parent_size, parent_addr)) = self.parent(size, addr) {
            let (children_size, children_addrs) = children(parent_size, parent_addr);
            let [left_block, right_block] =
                children_addrs.map(|c| self.at(size, c).expect("children block to exist"));

            let mut changed = false;

            if let Some(live_since) =
                min_optional(left_block.live_since(), right_block.live_since())
            {
                let criterion = apply_optional(
                    left_block.victim_criterion(),
                    right_block.victim_criterion(),
                    Criterion::join,
                )
                .expect("at least one children non-free");

                self.splitted
                    .update_at_with(parent_size, parent_addr, |old_block| {
                        changed = (live_since != old_block.live_since)
                            || (criterion != old_block.criterion);
                        SplittedBlock {
                            live_since,
                            criterion,
                            pointer: old_block.pointer,
                        }
                    })
                    .expect("a splitted block to exist here");
            } else {
                changed = true;

                let [left_block, right_block] = children_addrs.map(|c| {
                    self.free
                        .remove(children_size, c)
                        .expect("both children free")
                });

                let criterion = Cf::join(left_block.criterion, right_block.criterion);
                let free_since = Ord::max(left_block.free_since, right_block.free_since);

                let _old_splitted = self
                    .splitted
                    .remove(parent_size, parent_addr)
                    .expect("parent to be splitted");
                let _ = self
                    .free
                    .update_at(
                        parent_size,
                        parent_addr,
                        FreeBlock {
                            free_since,
                            criterion,
                        },
                    )
                    .is_some_and(|_| panic!("target address is already free"));
            }

            if changed {
                self.propagate_up(parent_size, parent_addr);
            }
        }
    }

    fn mark_free_to_occupied(
        &mut self,
        size: IntegralSize,
        addr: Addr,
        mut block: NonfreeMaker<C, P>,
    ) -> P {
        let _old = self
            .free
            .remove(size, addr)
            .expect("free block here exists");

        let pointer = block.with_addr(size, addr).pointer();

        let _ = self
            .occupied
            .update_at(size, addr, block.occupied())
            .is_some_and(|_| panic!("target address is already occupied"));

        self.propagate_up(size, addr);

        pointer
    }

    fn split_free_until(
        &mut self,
        top_size: IntegralSize,
        top_addr: Addr,
        size: IntegralSize,
        mut block: NonfreeMaker<C, P>,
    ) -> P {
        let FreeBlock {
            free_since,
            criterion,
        } = self
            .free
            .remove(top_size, top_addr)
            .expect("to have top block as a free block");

        for size in (size.double().iter_to(top_size)).rev() {
            let _ = self
                .splitted
                .update_at(size, top_addr, block.with_addr(size, top_addr).splitted())
                .is_some_and(|_| panic!("target address is already splitted"));

            let (children_size, [_, right_addr]) = children(size, top_addr);

            let _ = self
                .free
                .update_at(
                    children_size,
                    right_addr,
                    FreeBlock {
                        free_since,
                        criterion: criterion.clone(),
                    },
                )
                .is_some_and(|_| panic!("target address is already free"));
        }

        let pointer = block.with_addr(size, top_addr).pointer();

        let _ = self
            .occupied
            .update_at(size, top_addr, block.occupied())
            .is_some_and(|_| panic!("target address is already occupied"));

        self.propagate_up(top_size, top_addr);

        pointer
    }

    fn new_top_occupied_block(&mut self, addr: Addr, mut block: NonfreeMaker<C, P>) -> P {
        let pointer = block.with_addr(self.max_bs, addr).pointer();

        let _ = self
            .occupied
            .update_at(self.max_bs, addr, block.occupied())
            .is_some_and(|_| panic!("target address is already occupied"));

        pointer
    }

    fn new_top_free_block(&mut self, addr: Addr, block: FreeBlock<Cf>) {
        let _ = self
            .free
            .update_at(self.max_bs, addr, block)
            .is_some_and(|_| panic!("target address is already free"));
    }

    fn collect_children_recurse(&self, size: IntegralSize, addr: Addr, r: &mut Vec<P>) {
        match self.at(size, addr).expect("some block to be here") {
            Block::Occupied(block) => r.push(block.pointer),
            Block::Splitted(..) => {
                let (children_size, children_addrs) = children(size, addr);
                children_addrs
                    .into_iter()
                    .for_each(|addr| self.collect_children_recurse(children_size, addr, r));
            }
            _ => {}
        }
    }

    fn collect_children(&self, size: IntegralSize, addr: Addr) -> Vec<P> {
        let mut r = Vec::new();
        self.collect_children_recurse(size, addr, &mut r);
        r
    }

    /// Recursively removes all blocks within block (`size`, `addr`), including the block it self.
    ///
    /// CAUTION Dangerous Method: inconsistent state since the criterions are not propagated up,
    /// and parent blocks are not removed.
    fn remove_block(&mut self, size: IntegralSize, addr: Addr) {
        match self.at(size, addr).expect("some block to be here") {
            Block::Occupied(..) => {
                self.occupied.remove(size, addr).unwrap();
            }
            Block::Free(..) => {
                self.free.remove(size, addr).unwrap();
            }
            Block::Splitted(..) => {
                let (children_size, children_addrs) = children(size, addr);
                children_addrs
                    .into_iter()
                    .for_each(|addr| self.remove_block(children_size, addr));
                self.splitted.remove(size, addr).unwrap();
            }
        }
    }

    fn free_then_occupy_grandchild(
        &mut self,
        size: IntegralSize,
        addr: Addr,
        grandchild_size: IntegralSize,
        free_block: FreeBlock<Cf>,
        occupied_block: NonfreeMaker<C, P>,
    ) -> P {
        self.remove_block(size, addr);
        self.free.update_at(size, addr, free_block);
        self.split_free_until(size, addr, grandchild_size, occupied_block)
    }

    fn mark_occupied_free(&mut self, size: IntegralSize, addr: Addr, block: FreeBlock<Cf>) {
        let _occupied = self
            .occupied
            .remove(size, addr)
            .unwrap_or_else(|| panic!("freeing unoccupied block at ({}, {})", size.0, addr.0));
        self.free.update_at(size, addr, block);
        self.propagate_up(size, addr);
    }

    fn update_occupied_criterion(
        &mut self,
        size: IntegralSize,
        addr: Addr,
        c: C,
        force_update: bool,
    ) {
        self.occupied
            .update_at_with(size, addr, |mut b| {
                let original = &mut b.criterion;
                if *original != C::bottom() || force_update {
                    *original = c;
                }
                b
            })
            .expect("an occupied block to exist here");
        self.propagate_up(size, addr);
    }
}

#[derive(Clone)]
pub struct Pool<C, Cf, P> {
    capacity: u64,
    blocks: Blocks<C, Cf, P>,
    next_new_top_block_addr: Addr,
}

struct Indent(usize);

impl std::fmt::Display for Indent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.0 {
            write!(f, " ")?;
        }
        Ok(())
    }
}

fn format_blocks_recurse<C, Cf, P>(
    blocks: &Blocks<C, Cf, P>,
    size: IntegralSize,
    addr: Addr,
    indentation: usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    C: Ord + Clone + std::fmt::Debug,
    Cf: Ord + Clone + std::fmt::Debug,
    P: std::fmt::Debug,
{
    match blocks.at(size, addr) {
        None => write!(f, "{}<ERROR>", Indent(indentation))?,
        Some(Block::Free(block)) => {
            writeln!(
                f,
                "{}<Free bs={} addr={} since={} criterion={:?}>",
                Indent(indentation),
                size,
                addr,
                block.free_since,
                block.criterion
            )?;
        }
        Some(Block::Occupied(block)) => {
            writeln!(
                f,
                "{}<Occupied bs={} addr={} since={} criterion={:?} pointer={:?}>",
                Indent(indentation),
                size,
                addr,
                block.live_since,
                block.criterion,
                block.pointer
            )?;
        }
        Some(Block::Splitted(block)) => {
            writeln!(
                f,
                "{}<Splitted bs={} addr={} since={} criterion={:?} pointer={:?}>",
                Indent(indentation),
                size,
                addr,
                block.live_since,
                block.criterion,
                block.pointer
            )?;
            let (child_size, child_addrs) = children(size, addr);

            for child_addr in child_addrs {
                format_blocks_recurse(blocks, child_size, child_addr, indentation + 2, f)?;
            }

            writeln!(f, "{}</Splitted>", Indent(indentation))?;
        }
    }
    Ok(())
}

impl<C, Cf, P> std::fmt::Debug for Pool<C, Cf, P>
where
    C: Ord + Clone + std::fmt::Debug,
    Cf: Ord + Clone + std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "<Buddy capacity={} max_bs={} next_new_top_block_addr={}>",
            self.capacity, self.blocks.max_bs, self.next_new_top_block_addr.0
        )?;

        for (size, addr) in self.top_blocks() {
            format_blocks_recurse(&self.blocks, size, addr, 2, f)?;
        }

        writeln!(f, "</Buddy>")?;

        Ok(())
    }
}

impl<C, Cf, P> Pool<C, Cf, P> {
    pub fn new(capacity: u64, max_bs: IntegralSize) -> Self {
        Self {
            capacity,
            blocks: Blocks::new(max_bs),
            next_new_top_block_addr: Addr(0),
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity
    }

    pub fn occupied_size(&self) -> u64 {
        self.next_new_top_block_addr.0
    }

    fn top_blocks(&self) -> impl Iterator<Item = (IntegralSize, Addr)> {
        let lbs = self.blocks.max_bs.0;
        (0..self.next_new_top_block_addr.0)
            .step_by(2usize.pow(lbs))
            .map(move |a| (IntegralSize(lbs), Addr(a)))
    }
}

impl<C, Cf, P> Pool<C, Cf, P>
where
    C: Criterion,
    Cf: Criterion,
    P: UsizeId,
{
    pub fn check_invariants(&self)
    where
        C: std::fmt::Debug,
        Cf: std::fmt::Debug,
    {
        if self.next_new_top_block_addr.0 > self.capacity_bytes() {
            panic!(
                "allocated more bytes {} then capacity {}",
                self.next_new_top_block_addr.0,
                self.capacity_bytes()
            );
        }

        let mut checker = InvariantChecker::new();
        for (size, addr) in self.top_blocks() {
            checker.check_recurse(&self.blocks, size, addr);
        }
        checker.finish(&self.blocks);
    }

    fn increment_top_block_addr(&mut self) -> Option<Addr> {
        let new_top = self.next_new_top_block_addr.0 + 2u64.pow(self.blocks.max_bs.0);

        if new_top > self.capacity {
            return None;
        }
        let addr = self.next_new_top_block_addr;
        self.next_new_top_block_addr.0 = new_top;
        Some(addr)
    }

    fn allocate_exact_size(&mut self, size: IntegralSize, block: NonfreeMaker<C, P>) -> Option<P> {
        let (addr, _) = self.blocks.free.best_of_size(size)?;
        let p = self.blocks.mark_free_to_occupied(size, addr, block);
        Some(p)
    }

    fn allocate_maximum_bs(&mut self, block: NonfreeMaker<C, P>) -> Option<P> {
        let addr = self.increment_top_block_addr()?;

        let p = self.blocks.new_top_occupied_block(addr, block);

        Some(p)
    }

    fn allocate_by_split(&mut self, size: IntegralSize, block: NonfreeMaker<C, P>) -> Option<P> {
        let (top_size, top_addr) = if let Some((top_size, top_addr)) =
            (size.double().iter_to(self.blocks.max_bs))
                .filter_map(|size| {
                    self.blocks
                        .free
                        .best_of_size(size)
                        .map(|(addr, _)| (size, addr))
                })
                .next()
        {
            (top_size, top_addr)
        } else {
            let addr = self.increment_top_block_addr()?;
            self.blocks.new_top_free_block(
                addr,
                FreeBlock {
                    free_since: 0,
                    criterion: Cf::top(),
                },
            );
            (self.blocks.max_bs, addr)
        };

        let p = self
            .blocks
            .split_free_until(top_size, top_addr, size, block);

        Some(p)
    }

    pub fn allocate(
        &mut self,
        size: IntegralSize,
        c: C,
        now: usize,
        pointers: &mut Pointers<P>,
    ) -> Option<P>
    where
        C: std::fmt::Debug,
        Cf: std::fmt::Debug,
    {
        let r = self.allocate_(size, c, now, pointers);
        r
    }

    fn allocate_(
        &mut self,
        size: IntegralSize,
        c: C,
        now: usize,
        pointers: &mut Pointers<P>,
    ) -> Option<P> {
        if size > self.blocks.max_bs {
            panic!(
                "invalid block size {:?} bigger than configured maximum {:?}",
                size, self.blocks.max_bs
            );
        }

        if let Some(p) = self.allocate_exact_size(size, NonfreeMaker::new(now, c.clone(), pointers))
        {
            return Some(p);
        }

        if size == self.blocks.max_bs {
            if let Some(p) = self.allocate_maximum_bs(NonfreeMaker::new(now, c.clone(), pointers)) {
                return Some(p);
            } else {
                return None;
            }
        }

        self.allocate_by_split(size, NonfreeMaker::new(now, c, pointers))
    }

    fn victims_exact_size(
        &self,
        size: IntegralSize,
        requested_size: IntegralSize,
    ) -> Option<Victims<P>> {
        let occupied = self.blocks.occupied.best_of_size(size);
        let splitted = self.blocks.splitted.best_of_size(size);

        match (occupied, splitted) {
            (Some((_, ob)), Some((_, sb))) if &ob.criterion >= &sb.criterion => Some(Victims {
                victims: vec![ob.pointer],
                freed: ob.pointer,
                allocated: requested_size,
            }),
            (Some((_, ob)), None) => Some(Victims {
                victims: vec![ob.pointer],
                freed: ob.pointer,
                allocated: requested_size,
            }),
            (None, Some((addr, sb))) => Some(Victims {
                victims: self.blocks.collect_children(size, addr),
                freed: sb.pointer,
                allocated: requested_size,
            }),
            (Some(..), Some((addr, sb))) => Some(Victims {
                victims: self.blocks.collect_children(size, addr),
                freed: sb.pointer,
                allocated: requested_size,
            }),
            (None, None) => None,
        }
    }

    pub fn victims(&self, size: IntegralSize) -> Option<Victims<P>> {
        for try_size in size.iter_to(self.blocks.max_bs) {
            if let Some(victims) = self.victims_exact_size(try_size, size) {
                return Some(victims);
            }
        }
        None
    }

    pub fn commit_victims(
        &mut self,
        victims: &Victims<P>,
        c: C,
        criterion_free: Cf,
        now: usize,
        pointers: &mut Pointers<P>,
    ) -> P
    where
        C: Debug,
        Cf: Debug,
    {
        let (victim_size, victim_addr) = pointers[victims.freed];
        let r = self.blocks.free_then_occupy_grandchild(
            victim_size,
            victim_addr,
            victims.allocated,
            FreeBlock {
                free_since: now,
                criterion: criterion_free,
            },
            NonfreeMaker::new(now, c, pointers),
        );

        r
    }

    pub fn deallocate(&mut self, p: P, now: usize, criterion_free: Cf, pointers: &Pointers<P>)
    where
        C: Debug,
        Cf: Debug,
    {
        let (size, addr) = pointers[p];

        self.blocks.mark_occupied_free(
            size,
            addr,
            FreeBlock {
                free_since: now + 1,
                criterion: criterion_free,
            },
        );
    }

    pub fn update_criterion(&mut self, size: IntegralSize, addr: Addr, c: C, force_udpate: bool) {
        self.blocks
            .update_occupied_criterion(size, addr, c, force_udpate);
    }
}

#[derive(Debug)]
pub struct Victims<P> {
    pub victims: Vec<P>,
    pub freed: P,
    pub allocated: IntegralSize,
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
    struct C(i32);

    impl Criterion for C {
        fn join(self, rhs: Self) -> Self {
            self.max(rhs)
        }

        fn top() -> Self {
            Self(i32::MAX)
        }

        fn bottom() -> Self {
            Self(i32::MIN)
        }
    }

    define_usize_id!(PId);

    fn pool() -> Pool<C, C, PId> {
        Pool::new(2u64.pow(5), IntegralSize(3))
    }

    #[test]
    fn test_allocate() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        for (now, (size, c)) in [(2, 4), (2, 4), (1, 4), (3, 4), (3, 4)]
            .into_iter()
            .enumerate()
        {
            pool.allocate(IntegralSize(size), C(c), now, &mut pointers)
                .unwrap();
            pool.check_invariants();
        }
    }

    #[test]
    fn test_parent_children_relationship() {
        // Test the parent-child relationship logic
        let size = IntegralSize(2); // Size 4 blocks (2^2)
        let addr = Addr(0b1000); // 8 in decimal

        // Test parent calculation - should be size 3 (8 bytes)
        let (parent_size, parent_addr) = parent(size, addr);
        assert_eq!(parent_size, IntegralSize(3));
        assert_eq!(parent_addr, Addr(0b1000)); // Should be 8 (same as child since it's aligned)

        // Test children calculation of the parent
        let (child_size, children_addrs) = children(parent_size, parent_addr);
        assert_eq!(child_size, size);
        assert_eq!(children_addrs[0], Addr(0b1000)); // First child at 8
        assert_eq!(children_addrs[1], Addr(0b1100)); // Second child at 12 (8 + 4)
    }

    #[test]
    fn test_allocate_different_sizes() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Allocate blocks of different sizes
        let p1 = pool
            .allocate(IntegralSize(1), C(10), 1, &mut pointers)
            .unwrap(); // Size 2
        pool.check_invariants();
        let (_size1, addr1) = pointers[p1];
        assert_eq!(addr1, Addr(0));

        let p2 = pool
            .allocate(IntegralSize(2), C(20), 2, &mut pointers)
            .unwrap(); // Size 4
        pool.check_invariants();
        let (_size2, addr2) = pointers[p2];
        assert_eq!(addr2, Addr(4)); // Should be at 8 since first block took 0-2

        let p3 = pool
            .allocate(IntegralSize(3), C(30), 3, &mut pointers)
            .unwrap(); // Size 8
        pool.check_invariants();
        let (_size3, addr3) = pointers[p3];
        assert_eq!(addr3, Addr(8)); // Should be at 16
    }

    #[test]
    fn test_victims_function() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Allocate several blocks
        let _p1 = pool
            .allocate(IntegralSize(2), C(10), 1, &mut pointers)
            .unwrap(); // Low priority
        let p2 = pool
            .allocate(IntegralSize(2), C(30), 2, &mut pointers)
            .unwrap(); // High priority
        let _p3 = pool
            .allocate(IntegralSize(2), C(20), 3, &mut pointers)
            .unwrap(); // Medium priority

        pool.check_invariants();

        // Request victims for a block of size 2
        let victims = pool.victims(IntegralSize(2)).unwrap();

        // Should select the highest priority block
        assert_eq!(victims.victims, vec![p2]);
        assert_eq!(victims.freed, p2);
    }

    #[test]
    fn test_deallocate_function() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Allocate a block
        let p = pool
            .allocate(IntegralSize(2), C(10), 1, &mut pointers)
            .unwrap();
        pool.check_invariants();

        // Deallocate the block
        pool.deallocate(p, 5, C(0), &pointers);

        pool.check_invariants();

        // Try to allocate again - should get the same address
        let new_p = pool
            .allocate(IntegralSize(2), C(20), 6, &mut pointers)
            .unwrap();
        assert_eq!(pointers[p], pointers[new_p]);
    }

    #[test]
    fn test_commit_victims() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Allocate several blocks
        let _p1 = pool
            .allocate(IntegralSize(2), C(10), 1, &mut pointers)
            .unwrap(); // Will be victim
        let _p2 = pool
            .allocate(IntegralSize(2), C(30), 2, &mut pointers)
            .unwrap(); // High priority

        pool.check_invariants();

        // Get victims
        let victims = pool.victims(IntegralSize(2)).unwrap();

        // Commit victims to allocate a new block
        pool.commit_victims(&victims, C(25), C(5), 10, &mut pointers);
        pool.check_invariants();

        // The victim address should now be reused
    }

    #[test]
    fn test_commit_victims_when_full() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        let _ = pool
            .allocate(IntegralSize(2), C(50), 1, &mut pointers)
            .unwrap();
        let _ = pool
            .allocate(IntegralSize(3), C(10), 1, &mut pointers)
            .unwrap();
        let _ = pool
            .allocate(IntegralSize(2), C(50), 1, &mut pointers)
            .unwrap();
        let _ = pool
            .allocate(IntegralSize(2), C(10), 1, &mut pointers)
            .unwrap();
        let _ = pool
            .allocate(IntegralSize(3), C(10), 1, &mut pointers)
            .unwrap();

        let r = pool.allocate(IntegralSize(3), C(10), 1, &mut pointers);
        assert!(r.is_none());
        pool.check_invariants();

        let victims = pool.victims(IntegralSize(3)).unwrap();

        pool.commit_victims(&victims, C(25), C(5), 10, &mut pointers);
        pool.check_invariants();

        let _ = pool
            .allocate(IntegralSize(2), C(10), 1, &mut pointers)
            .unwrap();
        pool.check_invariants();
    }

    #[test]
    fn test_edge_cases() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Test allocating maximum block size
        let p = pool
            .allocate(pool.blocks.max_bs, C(10), 1, &mut pointers)
            .unwrap();
        pool.check_invariants();
        let (_, addr) = pointers[p];
        assert_eq!(addr, Addr(0));

        // Test allocating when pool is full
        let mut full_pool = Pool::<C, C, PId>::new(2u64.pow(3), IntegralSize(3)); // Small pool
        let mut full_pointers = Pointers::new();
        let _p1 = full_pool
            .allocate(IntegralSize(1), C(10), 1, &mut full_pointers)
            .unwrap();
        let _p2 = full_pool
            .allocate(IntegralSize(1), C(20), 2, &mut full_pointers)
            .unwrap();
        let _p3 = full_pool
            .allocate(IntegralSize(1), C(30), 3, &mut full_pointers)
            .unwrap();
        let _p4 = full_pool
            .allocate(IntegralSize(1), C(40), 4, &mut full_pointers)
            .unwrap();
        // Next allocation should fail
        let result = full_pool.allocate(IntegralSize(3), C(50), 5, &mut full_pointers);
        assert!(result.is_none());
    }

    #[test]
    fn test_split_and_merge() {
        let mut pool = pool();
        let mut pointers = Pointers::new();

        // Allocate a large block
        let large_p = pool
            .allocate(IntegralSize(3), C(10), 1, &mut pointers)
            .unwrap(); // Size 8
        pool.check_invariants();

        // Deallocate it
        pool.deallocate(large_p, 2, C(0), &pointers);
        pool.check_invariants();

        // Now allocate smaller blocks that should be split from the large one
        let small_p1 = pool
            .allocate(IntegralSize(1), C(20), 3, &mut pointers)
            .unwrap(); // Size 2
        let small_p2 = pool
            .allocate(IntegralSize(1), C(30), 4, &mut pointers)
            .unwrap(); // Size 2
        pool.check_invariants();

        // Check that addresses are within the original large block range
        let (_, small_addr1) = pointers[small_p1];
        let (_, small_addr2) = pointers[small_p2];
        assert!(small_addr1.0 < 8);
        assert!(small_addr2.0 < 8);
    }
}
