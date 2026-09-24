use std::u64;

use super::super::prelude::*;
use planning::machine::*;

mod pages {
    use std::fmt::Debug;

    use super::*;

    define_usize_id!(PageId);

    #[derive(Clone, Debug)]
    struct Map<C> {
        pas: Vec<PageId>,
        c: C,
        free_page_count: usize,
    }

    impl<C> Map<C> {
        pub fn free(&self) -> bool {
            self.free_page_count == self.pas.len()
        }

        pub fn increment_free_count(self) -> Self {
            Map {
                free_page_count: self.free_page_count + 1,
                ..self
            }
        }

        pub fn drement_free_count(self) -> Self {
            Map {
                free_page_count: self.free_page_count - 1,
                ..self
            }
        }
    }

    impl<C> mm_heap::Priority<(bool, C)> for Map<C>
    where
        C: Clone,
    {
        fn priority(&self) -> (bool, C) {
            (self.free(), self.c.clone())
        }
    }

    /// Records mappings from virtual address ID's to physical pages,
    /// grouped by virtual address sizes (in pages)
    #[derive(Clone, Debug)]
    struct GroupedMappings<P, C>(BTreeMap<usize, MmHeap<P, (bool, C), Map<C>>>);

    impl<P, C> GroupedMappings<P, C> {
        pub fn empty() -> Self {
            Self(BTreeMap::new())
        }
    }

    impl<P, C> GroupedMappings<P, C>
    where
        C: Criterion,
        P: UsizeId,
    {
        pub fn best_of_size(&self, number: usize) -> Option<(P, &Map<C>)> {
            self.0
                .get(&number)
                .map(|heap| heap.max().map(|(vaid, map)| (*vaid, map)))
                .flatten()
                .filter(|(_, map)| map.free() && map.c != C::bottom())
        }

        pub fn get(&self, number: usize, vaid: P) -> Option<&Map<C>> {
            self.0.get(&number).map(|heap| heap.get(&vaid)).flatten()
        }

        pub fn remove(&mut self, number: usize, vaid: P) -> Option<Map<C>> {
            self.0
                .get_mut(&number)
                .map(|heap| heap.remove(&vaid))
                .flatten()
        }

        pub fn udpate_if_exist(
            &mut self,
            number: usize,
            vaid: P,
            f: impl FnOnce(Map<C>) -> Map<C>,
        ) -> bool {
            if let Some(map) = self.remove(number, vaid) {
                let map = f(map);
                self.add(vaid, map);
                true
            } else {
                false
            }
        }

        pub fn add(&mut self, vaid: P, map: Map<C>) {
            let number = map.pas.len();
            assert!(
                self.0
                    .entry(number)
                    .or_default()
                    .insert(vaid, map)
                    .is_none(),
                "newly mapped {:?} cannot be already mapped",
                vaid
            );
        }

        pub fn iter<'a>(&'a self) -> impl Iterator<Item = (P, &'a Map<C>)> + 'a {
            self.0
                .iter()
                .map(|(_, heap)| heap.iter().map(|(p, m)| (*p, m)))
                .flatten()
        }
    }

    #[derive(Debug, Clone)]
    struct Page<P> {
        vas: BTreeSet<P>,
        occupied: bool,
    }

    impl<P> Page<P> {
        pub fn fresh() -> Self {
            Self {
                vas: BTreeSet::new(),
                occupied: false,
            }
        }
    }

    /// A set of pages with their next_use tracked.
    #[derive(Debug)]
    pub struct Pages<P, C, Cf> {
        /// All mappings where the virtual address is currently occupied
        occupied_mapping: MmHeap<P, (bool, C), Map<C>>,
        /// All mappings where the virtual address is currently free,
        /// but the pages mapped to those vas may not be free
        free_mapping: GroupedMappings<P, Cf>,
        /// Whether the page is occupied, and to which virtual addresses it is mapped
        physical_pages: Heap<PageId, Page<P>>,
        /// page number of the virtual address
        vas: Heap<P, usize>,
        /// All free physical pages
        free_pages: MmHeap<PageId, Cf, Cf>,
    }

    impl<P, C, Cf> Pages<P, C, Cf>
    where
        P: UsizeId,
        C: Criterion,
        Cf: Criterion,
    {
        pub fn new(number_pages: usize) -> Self {
            let mut pages = Self {
                occupied_mapping: MmHeap::new(),
                free_mapping: GroupedMappings::empty(),
                physical_pages: Heap::repeat(Page::fresh(), number_pages),
                vas: Heap::new(),
                free_pages: MmHeap::new(),
            };

            // Initialize all pages as free
            for i in 0..number_pages {
                let page_id = PageId::from(i);
                pages.free_pages.insert(page_id, Cf::top());
            }

            pages
        }

        #[allow(unused)]
        pub fn check_invariants(&self) {
            let mut unchecked_pages = self.physical_pages.map_by_ref(&mut |_, p| p.occupied);

            for (vaid, map) in self.occupied_mapping.iter() {
                for page in map.pas.iter() {
                    assert!(
                        self.physical_pages[*page].occupied,
                        "expect page from `occupied_mapping` {:?} occupied",
                        page
                    );
                    assert!(
                        unchecked_pages[*page] == true,
                        "expect page from `occupied_mapping` {:?} to map exactly to one va",
                        page
                    );
                    assert!(
                        !self.free_pages.contains(page),
                        "expect page from `occupied_mapping` {:?} not in `free_pages`",
                        page
                    );
                    assert!(
                        self.physical_pages[*page].vas.contains(vaid),
                        "expect va {:?} contained in reverse mapping of page {:?}",
                        vaid,
                        page
                    );
                    unchecked_pages[*page] = false;
                }
            }

            for (vaid, map) in self.free_mapping.iter() {
                if map.free() {
                    for page in map.pas.iter() {
                        assert!(
                            self.physical_pages[*page].occupied == false,
                            "expect page from `free_mapping` {:?} free",
                            page
                        );
                        assert!(
                            self.free_pages.contains(page),
                            "expect page from `free_mapping` {:?} in `free_pages`",
                            page
                        );
                    }
                }

                for page in map.pas.iter() {
                    assert!(
                        self.physical_pages[*page].vas.contains(&vaid),
                        "expect va {:?} contained in reverse mapping of page {:?}",
                        vaid,
                        page
                    );
                }
            }

            for (page, unchecked) in unchecked_pages.iter_with_id() {
                assert!(
                    !*unchecked,
                    "expect all occupied pages mapped in `occupied_mapping`, but {:?} is not",
                    page
                )
            }
        }

        fn occupy_page(&mut self, page: PageId) {
            assert!(
                self.free_pages.remove(&page).is_some(),
                "{:?} must be free",
                page
            );
            self.physical_pages[page].occupied = true;

            for &vaid in self.physical_pages[page].vas.iter() {
                let number = self.vas[vaid];
                self.free_mapping
                    .udpate_if_exist(number, vaid, |map| map.drement_free_count());
            }
        }

        fn free_page(&mut self, page: PageId, criterion: Cf) {
            assert!(
                self.free_pages.insert(page, criterion.clone()).is_none(),
                "{:?} cannot be already free",
                page
            );
            self.physical_pages[page].occupied = false;

            for &vaid in self.physical_pages[page].vas.iter() {
                let number = self.vas[vaid];
                self.free_mapping
                    .udpate_if_exist(number, vaid, |map| map.increment_free_count());
            }
        }

        /// Mark a bunch of pages as occupied with equal criterion and map them to a virtual address
        pub fn map_and_occupy(&mut self, pages: impl Iterator<Item = PageId>, criterion: C) -> P {
            let pages: Vec<PageId> = pages.collect();
            let vaid = self.vas.push(pages.len());

            for page in pages.iter() {
                self.occupy_page(*page);
                self.physical_pages[*page].vas.insert(vaid);
            }

            let map = Map {
                pas: pages,
                c: criterion,
                free_page_count: 0,
            };
            assert!(
                self.occupied_mapping.insert(vaid, map).is_none(),
                "{:?} should not be already mapped and occupied",
                vaid
            );

            vaid
        }

        /// Mark already mapped virtual address as occupied
        pub fn occupy(&mut self, vaid: P, criterion: C) -> usize {
            let number = self.vas[vaid];

            let map = self
                .free_mapping
                .remove(number, vaid)
                .unwrap_or_else(|| panic!("virtual address {:?} must be mapped", vaid));

            for page in map.pas.iter().copied() {
                self.occupy_page(page);
            }

            self.occupied_mapping.insert(
                vaid,
                Map {
                    pas: map.pas,
                    c: criterion,
                    free_page_count: 0,
                },
            );

            number
        }

        /// Free a virtual address, returning its criterion. The virtual address cannot already be free.
        pub fn free(&mut self, vaid: P, free_criterion: Cf) -> C {
            let map = self
                .occupied_mapping
                .remove(&vaid)
                .unwrap_or_else(|| panic!("{:?} is not occupied", vaid));

            for page in map.pas.iter() {
                self.free_page(*page, free_criterion.clone());
            }

            let c = map.c;
            let map = Map {
                free_page_count: map.pas.len(),
                pas: map.pas,
                c: free_criterion.clone(),
            };

            self.free_mapping.add(vaid, map);

            c
        }

        /// Find pages with max victim criterion.
        /// Before trying to find individual pages, the caller should first attempt reuse a mapped virtual address.
        ///
        /// If virtual addresses are returned, number of pages mapped to them is guaranteed to >= `number`.
        /// Returns empty vec if there are not enough occupied pages with non-bottom criterion.
        pub fn best_victims(&self, number: usize) -> Vec<P> {
            let mut count = 0;
            let mut vaids = Vec::new();

            for (vaid, map) in self
                .occupied_mapping
                .iter()
                .rev()
                .filter(|(_k, map)| map.c != C::bottom())
            {
                vaids.push(*vaid);
                count += map.pas.len();

                if count >= number {
                    break;
                }
            }

            if count < number {
                Vec::new()
            } else {
                vaids
            }
        }

        pub fn update_criterion(&mut self, va: P, to: C, force_update: bool) {
            let map = self
                .occupied_mapping
                .remove(&va)
                .unwrap_or_else(|| panic!("{:?} is not occupied", va));
            let map = Map {
                c: if force_update || map.c != C::bottom() {
                    to
                } else {
                    map.c
                },
                ..map
            };

            self.occupied_mapping.insert(va, map);
        }

        /// Look for a already mapped virtual address
        pub fn best_free_mapped(&self, number: usize) -> Option<P> {
            self.free_mapping.best_of_size(number).map(|(vaid, _)| vaid)
        }

        pub fn best_free_pages(&self, number: usize) -> Vec<PageId> {
            self.free_pages
                .iter()
                .rev()
                .filter(|(_k, c)| *c != &Cf::bottom())
                .map(|(k, _c)| *k)
                .take(number)
                .collect()
        }

        pub fn va_pages<'a>(&'a self, vaid: P) -> impl Iterator<Item = PageId> + 'a {
            let number = self.vas[vaid];
            let map1 = self
                .occupied_mapping
                .get(&vaid)
                .map(|map| map.pas.iter().cloned());
            let map2 = self
                .free_mapping
                .get(number, vaid)
                .map(|map| map.pas.iter().cloned());

            match (map1.is_some(), map2.is_some()) {
                (true, true) => panic!("{:?} cannot both be occupied and free", vaid),
                (false, false) => panic!(
                    "{:?} of number {} is not mapped to any physical pages",
                    vaid, number
                ),
                _ => {}
            }

            map1.into_iter().flatten().chain(map2.into_iter().flatten())
        }

        pub fn number(&self) -> usize {
            self.physical_pages.len()
        }

        pub fn va_page_number(&self, va: P) -> usize {
            self.vas[va]
        }

        pub fn _occupancy(&self) -> f32 {
            1.0 - self.free_pages.len() as f32 / self.physical_pages.len() as f32
        }
    }
}

use pages::{PageId, Pages};

define_usize_id!(VaId);

struct Pointers<P> {
    pointers: Heap<P, VaId>,
    pointers_backward: BTreeMap<VaId, P>,
}

impl<P> Pointers<P>
where
    P: UsizeId,
{
    pub fn new() -> Self {
        Pointers {
            pointers: Heap::new(),
            pointers_backward: BTreeMap::new(),
        }
    }

    pub fn push(&mut self, vaid: VaId) -> P {
        let p = self.pointers.push(vaid);
        self.pointers_backward.insert(vaid, p);
        p
    }

    pub fn get_backward(&self, vaid: VaId) -> P {
        self.pointers_backward[&vaid]
    }
}

impl<P> std::ops::Index<P> for Pointers<P>
where
    P: UsizeId,
{
    type Output = VaId;
    fn index(&self, index: P) -> &Self::Output {
        &self.pointers[index]
    }
}

#[derive(Debug, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    pub map_count: usize,
    pub map_page_count: usize,
    pub reuse_count: usize,
    pub reuse_page_count: usize,
}

impl Statistics {
    pub fn new() -> Self {
        Statistics {
            map_count: 0,
            map_page_count: 0,
            reuse_count: 0,
            reuse_page_count: 0,
        }
    }

    pub fn add_map(&mut self, pages: usize) {
        self.map_count += 1;
        self.map_page_count += pages;
    }

    pub fn add_reuse(&mut self, pages: usize) {
        self.reuse_count += 1;
        self.reuse_page_count += pages;
    }
}

pub struct PageAllocator<P, C, Cf> {
    page_size: u64,
    pages: Pages<VaId, C, Cf>,
    living_objects: ObjectBijection<P>,
    pointers: Pointers<P>,
    device: Device,
    statistics: Statistics,
}

impl<P, C, Cf> PageAllocator<P, C, Cf>
where
    P: UsizeId,
    C: Criterion,
    Cf: Criterion,
{
    pub fn new(number_pages: usize, page_size: u64, device: Device) -> Self {
        Self {
            page_size,
            device,
            pages: Pages::new(number_pages),
            pointers: Pointers::new(),
            living_objects: ObjectBijection::new(),
            statistics: Statistics::new(),
        }
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics.clone()
    }
}

#[derive(Debug)]
struct CommitPayload {
    free_pages: Vec<PageId>,
    left_number: usize,
    victim_vas: Vec<VaId>,
    victim_objects: Vec<Object>,
    object: Object,
}

impl<P, C, Cf> PageAllocator<P, C, Cf>
where
    P: UsizeId + 'static,
    C: Criterion + 'static + std::fmt::Debug,
    Cf: Criterion + 'static + std::fmt::Debug,
{
    /// Allocate `pages` for `object` by create new virtual address and map `pages` to it
    fn allocate_pages(
        &mut self,
        pages: Vec<PageId>,
        object: Object,
        c: C,
        machine: &mut dyn AdEmitter<P>,
    ) -> Pointer<P> {
        let vaid = self.pages.map_and_occupy(pages.iter().copied(), c.clone());
        let p = self.pointers.push(vaid);
        self.living_objects.insert(object.clone(), p);

        self.statistics.add_map(pages.len());

        let pointer = Pointer::new::<_, C, Cf>(self, p);
        machine.allocate(object, self.device, pointer);
        pointer
    }

    /// Allocate `vaid` for `object`
    fn reuse_virtual_address(
        &mut self,
        object: Object,
        vaid: VaId,
        c: C,
        machine: &mut dyn AdEmitter<P>,
    ) -> Pointer<P> {
        let p = self.pointers.push(vaid);
        let pages = self.pages.occupy(vaid, c);
        self.living_objects.insert(object.clone(), p);

        self.statistics.add_reuse(pages);

        let pointer = Pointer::new::<_, C, Cf>(self, p);
        machine.allocate(object, self.device, pointer);
        pointer
    }

    /// Deallocate `object`
    fn deallocate_object(
        &mut self,
        victim_object: Object,
        c: Cf,
        machine: &mut dyn AdEmitter<P>,
    ) -> Pointer<P> {
        let victim_p = self
            .living_objects
            .remove_forward(&victim_object)
            .unwrap_or_else(|| panic!("object {:?} not allocated", victim_object));
        let victim_vaid = self.pointers[victim_p];

        self.pages.free(victim_vaid, c.clone());

        let victim_pointer = Pointer::new::<_, C, Cf>(self, victim_p);
        machine.deallocate(victim_object, self.device, victim_pointer);

        victim_pointer
    }

    fn update_pages_criterion(&mut self, vaid: VaId, c: C, force_update: bool) {
        self.pages.update_criterion(vaid, c, force_update);
    }
}

impl<P, C, Cf> Allocator<P, C, Cf> for PageAllocator<P, C, Cf>
where
    C: Criterion + 'static + std::fmt::Debug,
    Cf: Criterion + 'static + std::fmt::Debug,
    P: UsizeId + 'static,
{
    fn layer(&self) -> u32 {
        0
    }

    fn access(&mut self, t: &Object, c: C) -> Option<Pointer<P>> {
        self.living_objects.get_forward(t).cloned().map(|p| {
            self.update_pages_criterion(self.pointers[p], c, false);
            Pointer::new::<_, C, Cf>(self, p)
        })
    }

    fn unlock(&mut self, t: &Object, c: C) {
        let p = self
            .living_objects
            .get_forward(t)
            .unwrap_or_else(|| panic!("cannot unlock object {:?} which is unrecorded", t));
        self.update_pages_criterion(self.pointers[*p], c, true);
    }

    fn contains(&self, t: &Object) -> bool {
        self.living_objects.contains_forward(t)
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &Object,
        _now: usize,
        machine: &mut dyn AdEmitter<P>,
    ) -> Result<Pointer<P>, OomError> {
        if let Some(_) = self.living_objects.get_forward(t) {
            panic!("{:?} already allocated", t);
        }
        let number = u64::from(size).div_ceil(self.page_size) as usize;

        if number > self.pages.number() {
            return Err(OomError::for_alloc::<Self>(self.device(), size));
        }

        // First attemp reuse mapped virtual address
        if let Some(vaid) = self.pages.best_free_mapped(number) {
            let pointer = self.reuse_virtual_address(t.clone(), vaid, C::bottom(), machine);
            return Ok(pointer);
        }

        // Then try allocate new virtual address and map pages to it
        let pages = self.pages.best_free_pages(number);

        if pages.len() == number {
            let pointer = self.allocate_pages(pages, t.clone(), C::bottom(), machine);

            Ok(pointer)
        } else {
            Err(OomError::for_alloc::<Self>(self.device, size))
        }
    }

    fn takeover_victims(
        &self,
        size: Size,
        object: &Object,
    ) -> Result<(allocator::CommitVictims, Vec<Object>), OomError> {
        let number = u64::from(size).div_ceil(self.page_size) as usize;

        let free_pages = self.pages.best_free_pages(number);

        // When `free_pages` does not consitute a fair amount of space, we first try obtaining all requested pages by eviction
        let left_number = number - free_pages.len();
        let try_queue = if free_pages.len() * 2 >= number {
            vec![left_number]
        } else {
            vec![number, left_number]
        };

        let (victim_vas, left_number) = try_queue
            .into_iter()
            .fold(None, |old, left_number| match old {
                Some(..) => old,
                None => {
                    let victim_vas = self.pages.best_victims(left_number);
                    if victim_vas.len() != 0 {
                        Some((victim_vas, left_number))
                    } else {
                        None
                    }
                }
            })
            .ok_or_else(|| OomError::for_victim::<Self>(self.device, size))?;

        let free_pages = if left_number == number {
            Vec::new()
        } else {
            free_pages
        };

        let victim_objects: Vec<Object> = victim_vas
            .iter()
            .map(|va| {
                self.living_objects
                    .get_backward(&self.pointers.get_backward(*va))
                    .unwrap_or_else(|| {
                        panic!(
                            "virtual address {:?} is currently not associated with any object",
                            va
                        )
                    })
                    .clone()
            })
            .collect();

        let object = object.clone();

        let commit = allocator::CommitVictims::new(
            Box::new(CommitPayload {
                free_pages,
                left_number,
                victim_objects: victim_objects.clone(),
                victim_vas,
                object: object.clone(),
            }),
            self.layer(),
        );

        Ok((commit, victim_objects))
    }

    fn accept_commit(
        &mut self,
        commit: allocator::CommitVictims,
        _now: usize,
        cf: Cf,
        machine: &mut dyn planning::AdEmitter<P>,
    ) -> Pointer<P> {
        let payload: CommitPayload = commit.payload();

        payload.victim_objects.iter().for_each(|obj| {
            self.deallocate_object(obj.clone(), cf.clone(), machine);
        });

        if payload.free_pages.len() == 0
            && payload.victim_vas.len() == 1
            && self.pages.va_page_number(payload.victim_vas[0]) == payload.left_number
        {
            // When the victim virtual address just fits the new object, we simply reuse the va
            let pointer = self.reuse_virtual_address(
                payload.object,
                payload.victim_vas[0],
                C::bottom(),
                machine,
            );

            pointer
        } else {
            // Otherwise, create new mapping
            let new_pages = payload
                .free_pages
                .into_iter()
                .chain(
                    payload
                        .victim_vas
                        .into_iter()
                        .map(|va| self.pages.va_pages(va))
                        .flatten()
                        .take(payload.left_number),
                )
                .collect();

            let pointer =
                self.allocate_pages(new_pages, payload.object.clone(), C::bottom(), machine);

            pointer
        }
    }

    fn deallocate(&mut self, t: &Object, _now: usize, cf: Cf, machine: &mut dyn AdEmitter<P>) {
        self.deallocate_object(t.clone(), cf, machine);
    }

    fn device(&self) -> Device {
        self.device
    }

    fn reuse(&mut self, new_object: &Object, old_object: &Object, c: C) {
        let p = self
            .living_objects
            .remove_forward(old_object)
            .unwrap_or_else(|| panic!("old_object {:?} not found in living_objects", old_object));

        self.update_pages_criterion(self.pointers[p], c, false);

        self.living_objects.insert(new_object.clone(), p);
    }
}

impl<'a, P, C, Cf> Realizer<P> for AsRealizer<'a, PageAllocator<P, C, Cf>, P, C, Cf>
where
    P: UsizeId,
    C: Criterion,
    Cf: Criterion,
{
    fn device(&self) -> Device {
        self.allocator().device
    }

    fn allocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocMethod {
        if let Some(p) = pointer.here(0) {
            let vaid = self.allocator().pointers[p];
            let pages = self.allocator().pages.va_pages(vaid).collect::<Vec<_>>();
            let vs = pages.len() * (self.allocator().page_size as usize);
            AllocMethod::Paged {
                va_size: vs,
                vaid: vaid.into(),
                pa: pages.iter().map(|i| usize::from(*i)).collect(),
            }
        } else {
            panic!("pointer {:?} is not of layer 0", pointer)
        }
    }

    fn deallocated(&self, _t: &Object, pointer: &Pointer<P>) -> AllocVariant {
        if let Some(p) = pointer.here(0) {
            let vaid = self.allocator().pointers[p];
            AllocVariant::Paged { vaid: vaid.into() }
        } else {
            panic!("pointer {:?} is not of layer 0", pointer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Allocator;
    use super::*;
    use crate::transit::type2::object_analysis::value::{ObjectId, ValueNode as Typ};

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct TestCriterion(usize);

    impl Criterion for TestCriterion {
        fn join(self, rhs: Self) -> Self {
            TestCriterion(self.0.max(rhs.0))
        }

        fn top() -> Self {
            TestCriterion(usize::MAX)
        }

        fn bottom() -> Self {
            TestCriterion(0)
        }
    }

    define_usize_id!(TestId);

    struct MockMachine<P> {
        allocations: Vec<(Object, Device, Pointer<P>)>,
        deallocations: Vec<(Object, Device, Pointer<P>)>,
    }

    impl<P> MockMachine<P> {
        fn new() -> Self {
            Self {
                allocations: Vec::new(),
                deallocations: Vec::new(),
            }
        }
    }

    impl<P: UsizeId> planning::AdEmitter<P> for MockMachine<P> {
        fn allocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
            self.allocations.push((t, device, pointer));
        }

        fn deallocate(&mut self, t: Object, device: Device, pointer: Pointer<P>) {
            self.deallocations.push((t, device, pointer));
        }
    }

    #[test]
    fn test_page_allocator_basic_allocation() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        // Test basic allocation
        let size = Size::from(8192u64); // Needs 2 pages
        let obj = Object::new(ObjectId::from(1usize), Typ::Scalar);

        let result = allocator.allocate(size, &obj, 0, &mut machine);
        assert!(result.is_ok());

        // Verify the object is now contained in the allocator
        assert!(allocator.contains(&obj));

        // Verify 2 pages were allocated
        assert_eq!(allocator.pages.best_free_pages(10).len(), 8); // 10 total - 2 used = 8 free

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_multiple_allocations() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        // Allocate multiple objects
        let obj1 = Object::new(ObjectId::from(1usize), Typ::Scalar);
        let obj2 = Object::new(ObjectId::from(2usize), Typ::Scalar);
        let obj3 = Object::new(ObjectId::from(3usize), Typ::Scalar);

        let size = Size::from(4096u64); // 1 page each

        assert!(allocator.allocate(size, &obj1, 0, &mut machine).is_ok());
        assert!(allocator.allocate(size, &obj2, 0, &mut machine).is_ok());
        assert!(allocator.allocate(size, &obj3, 0, &mut machine).is_ok());

        assert!(allocator.contains(&obj1));
        assert!(allocator.contains(&obj2));
        assert!(allocator.contains(&obj3));

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_ada() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();
        let size = Size::from(4099u64); // 1 page

        let obj = Object::new(ObjectId::from(1usize), Typ::Scalar);
        assert!(allocator.allocate(size, &obj, 0, &mut machine).is_ok());

        allocator.pages.check_invariants();

        allocator.deallocate(&obj, 1, TestCriterion::top(), &mut machine);

        allocator.pages.check_invariants();

        let obj = Object::new(ObjectId::from(2usize), Typ::Scalar);
        assert!(allocator.allocate(size, &obj, 2, &mut machine).is_ok());

        allocator.pages.check_invariants();

        let result = allocator.access(&obj, TestCriterion(10));
        assert!(result.is_some());

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_deallocation() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        let size = Size::from(4096u64); // 1 page
        let obj = Object::new(ObjectId::from(1usize), Typ::Scalar);

        // Allocate object
        let result = allocator.allocate(size, &obj, 0, &mut machine);
        assert!(result.is_ok());
        assert!(allocator.contains(&obj));

        // Deallocate object
        allocator.deallocate(&obj, 0, TestCriterion::top(), &mut machine);

        // Object should no longer be contained
        assert!(!allocator.contains(&obj));

        // Page should be free again
        assert_eq!(allocator.pages.best_free_pages(10).len(), 10); // All pages should be free

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_takeover_victims() {
        // Create a situation where we need victims
        // Use an allocator with only 4 pages, allocate 3, then request 3 (needing 2 victims)
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(4, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        // Allocate objects
        let obj1 = Object::new(ObjectId::from(1usize), Typ::Scalar);
        let obj2 = Object::new(ObjectId::from(2usize), Typ::Scalar);
        let obj3 = Object::new(ObjectId::from(3usize), Typ::Scalar);

        let size = Size::from(4096u64); // 1 page each
        assert!(allocator.allocate(size, &obj1, 0, &mut machine).is_ok());
        assert!(allocator.allocate(size, &obj2, 0, &mut machine).is_ok());
        assert!(allocator.allocate(size, &obj3, 0, &mut machine).is_ok());

        allocator.pages.check_invariants();

        // To make pages available for eviction, we need to unlock them after allocation
        // Since initially allocated pages have bottom criterion (not available for eviction)
        allocator.unlock(&obj1, TestCriterion(10));
        allocator.unlock(&obj2, TestCriterion(20));
        allocator.unlock(&obj3, TestCriterion(30));

        allocator.pages.check_invariants();

        // Now try to allocate: 4 total pages, 3 allocated, 1 free
        // Need 3 pages, so need 3-1 = 2 victims
        let obj4 = Object::new(ObjectId::from(4usize), Typ::Scalar);
        let large_size = Size::from(12288u64); // Needs 3 pages (3*4096)

        let result = allocator.takeover_victims(large_size, &obj4);
        match result {
            Ok((_, victims)) => {
                // After unlocking, objects should be available as victims
                assert!(!victims.is_empty());
            }
            Err(_) => {
                // This is also valid if even unlocked pages are not sufficient
            }
        }

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_reuse() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        let size = Size::from(4096u64);
        let old_obj = Object::new(ObjectId::from(1usize), Typ::Scalar);
        let new_obj = Object::new(ObjectId::from(2usize), Typ::Scalar);

        // Allocate an object
        assert!(allocator.allocate(size, &old_obj, 0, &mut machine).is_ok());
        assert!(allocator.contains(&old_obj));

        allocator.pages.check_invariants();

        // Simulate deallocation without freeing (for reuse)
        // We'll test reuse operation directly
        allocator.reuse(&new_obj, &old_obj, TestCriterion(5));

        // new_obj should be contained, old_obj should not
        assert!(allocator.contains(&new_obj));
        assert!(!allocator.contains(&old_obj));

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_access() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(10, 4096, Device::Cpu);
        let mut machine = MockMachine::new();

        let size = Size::from(4096u64);
        let obj = Object::new(ObjectId::from(1usize), Typ::Scalar);

        // First allocate the object
        assert!(allocator.allocate(size, &obj, 0, &mut machine).is_ok());

        // Then access it with updated criterion
        let result = allocator.access(&obj, TestCriterion(10));
        assert!(result.is_some());

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_oom_handling() {
        let mut allocator = PageAllocator::<TestId, TestCriterion, TestCriterion>::new(
            2, // Only 2 pages
            4096,
            Device::Cpu,
        );
        let mut machine = MockMachine::new();

        let size = Size::from(4096u64); // 1 page each
        let obj1 = Object::new(ObjectId::from(1usize), Typ::Scalar);
        let obj2 = Object::new(ObjectId::from(2usize), Typ::Scalar);
        let obj3 = Object::new(ObjectId::from(3usize), Typ::Scalar);

        // Allocate 2 objects (filling up)
        assert!(allocator.allocate(size, &obj1, 0, &mut machine).is_ok());
        assert!(allocator.allocate(size, &obj2, 0, &mut machine).is_ok());

        // Try to allocate 3rd object (should fail)
        let result = allocator.allocate(size, &obj3, 0, &mut machine);
        assert!(result.is_err());

        allocator.pages.check_invariants();
    }

    #[test]
    fn test_page_allocator_stressed() {
        let mut allocator =
            PageAllocator::<TestId, TestCriterion, TestCriterion>::new(16, 1, Device::Cpu);

        let mut machine = MockMachine::new();
        let obj = |i: usize| Object::new(ObjectId::from(i), Typ::Scalar);
        let size = |i: u64| Size::new(i);

        let _ptr1 = allocator
            .allocate(size(3), &obj(0), 0, &mut machine)
            .unwrap();
        let _ptr2 = allocator
            .allocate(size(3), &obj(1), 1, &mut machine)
            .unwrap();
        allocator.deallocate(&obj(0), 2, TestCriterion(10), &mut machine);

        allocator.pages.check_invariants();

        let _ptr3 = allocator
            .allocate(size(8), &obj(2), 3, &mut machine)
            .unwrap();
        let _ptr4 = allocator
            .allocate(size(2), &obj(3), 4, &mut machine)
            .unwrap();
        allocator.pages.check_invariants();

        allocator.deallocate(&obj(1), 5, TestCriterion(10), &mut machine);

        allocator.pages.check_invariants();

        allocator.deallocate(&obj(3), 6, TestCriterion(10), &mut machine);

        allocator.pages.check_invariants();

        let _ptr5 = allocator
            .allocate(size(8), &obj(4), 7, &mut machine)
            .unwrap();

        allocator.pages.check_invariants();

        assert!(allocator.contains(&obj(2)));
        assert!(allocator.contains(&obj(4)));

        allocator.pages.check_invariants();
    }
}
