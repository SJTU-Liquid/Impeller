use std::collections::VecDeque;

use super::*;

#[derive(Debug)]
struct HalfMemory {
    assigned_tasks: HashSet<TaskId>,
    used: usize,
    offset: usize,
    total: usize,
    estimated_free_at: Instant,
}

/// A first-fit allocator for task-sized views into the single CPU static allocator.
///
/// FIFO scheduling uses this instead of the heuristic policy's two fixed halves, so a task may
/// occupy any currently free contiguous range. The scheduler owns all allocation and release of
/// these ranges; individual `CpuStaticAllocator` slices deliberately do not share bookkeeping.
#[derive(Debug)]
struct FifoMemory {
    free_ranges: BTreeMap<usize, usize>,
    total: usize,
    split_boundary: Option<usize>,
}

impl FifoMemory {
    fn new(total: usize, split_in_two: bool) -> Self {
        let mut free_ranges = BTreeMap::new();
        let split_boundary = if split_in_two {
            assert!(
                total % 2 == 0,
                "FIFO split CPU memory requires an even byte capacity"
            );
            Some(total / 2)
        } else {
            None
        };

        match split_boundary {
            Some(boundary) if boundary != 0 => {
                free_ranges.insert(0, boundary);
                free_ranges.insert(boundary, total);
            }
            Some(_) => {}
            None if total != 0 => {
                free_ranges.insert(0, total);
            }
            None => {}
        }

        Self {
            free_ranges,
            total,
            split_boundary,
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        let (start, end) = self
            .free_ranges
            .iter()
            .find(|(start, end)| **end - **start >= size)
            .map(|(start, end)| (*start, *end))?;

        self.free_ranges.remove(&start);
        let allocated_end = start.checked_add(size).expect("CPU allocation overflow");
        if allocated_end != end {
            self.free_ranges.insert(allocated_end, end);
        }
        Some(start)
    }

    /// Allocate from one physical pinned-memory region.  Region zero is the
    /// first backing allocation and region one the second.  Without a split
    /// there is only region zero.
    fn allocate_in_region(&mut self, size: usize, region: usize) -> Option<usize> {
        let (region_begin, region_end) = match self.split_boundary {
            Some(boundary) => match region {
                0 => (0, boundary),
                1 => (boundary, self.total),
                _ => return None,
            },
            None if region == 0 => (0, self.total),
            None => return None,
        };

        let (start, end) = self
            .free_ranges
            .range(region_begin..region_end)
            .find(|(start, end)| **start >= region_begin && **end <= region_end && **end - **start >= size)
            .map(|(start, end)| (*start, *end))?;

        self.free_ranges.remove(&start);
        let allocated_end = start.checked_add(size).expect("CPU allocation overflow");
        if allocated_end != end {
            self.free_ranges.insert(allocated_end, end);
        }
        Some(start)
    }

    fn deallocate(&mut self, offset: usize, size: usize) {
        let mut start = offset;
        let mut end = offset.checked_add(size).expect("CPU deallocation overflow");
        assert!(end <= self.total, "CPU deallocation is out of bounds");

        if let Some((previous_start, previous_end)) = self
            .free_ranges
            .range(..=offset)
            .next_back()
            .map(|(start, end)| (*start, *end))
        {
            assert!(
                previous_end <= offset,
                "CPU deallocation overlaps an existing free range"
            );
            if previous_end == offset && self.split_boundary != Some(offset) {
                self.free_ranges.remove(&previous_start);
                start = previous_start;
            }
        }

        if let Some((next_start, next_end)) = self
            .free_ranges
            .range(offset..)
            .next()
            .map(|(start, end)| (*start, *end))
        {
            assert!(
                next_start >= end,
                "CPU deallocation overlaps an existing free range"
            );
            if next_start == end && self.split_boundary != Some(end) {
                self.free_ranges.remove(&next_start);
                end = next_end;
            }
        }

        self.free_ranges.insert(start, end);
    }
}

impl HalfMemory {
    fn empty(offset: usize, size: usize) -> Self {
        Self {
            assigned_tasks: HashSet::new(),
            used: 0,
            offset,
            total: size,
            estimated_free_at: Instant::now(),
        }
    }

    fn free(&self) -> bool {
        self.assigned_tasks.is_empty()
    }

    fn add_task(&mut self, task: &AcceptedTask, estimated_free_at: Instant) -> usize {
        let offset = self.offset + self.used;

        let task_size = task.version.as_ref().unwrap().memory_limit() as usize;
        self.assigned_tasks.insert(task.id);
        self.used += task_size;
        self.estimated_free_at = self.estimated_free_at.max(estimated_free_at);

        offset
    }

    fn space_left(&mut self) -> usize {
        if self.assigned_tasks.is_empty() {
            self.used = 0;
            self.estimated_free_at = Instant::now();
        }
        self.total - self.used
    }
}

#[derive(Debug)]
enum AvailableMemory {
    Intact(Option<TaskId>),
    Halved([HalfMemory; 2]),
}

impl AvailableMemory {
    fn try_switch_to_intact(&mut self) {
        match &self {
            AvailableMemory::Halved([h1, h2]) if h1.free() && h2.free() => {
                *self = AvailableMemory::Intact(None)
            }
            _ => {}
        }
    }

    fn try_switch_to_halved(&mut self, total_memory: usize) {
        if let AvailableMemory::Intact(None) = &self {
            let half_size = total_memory / 2;
            *self = AvailableMemory::Halved([
                HalfMemory::empty(0, half_size),
                HalfMemory::empty(half_size, total_memory - half_size),
            ]);
        }
    }
}

#[derive(Debug)]
pub struct GlobalResources {
    available_cards: Vec<i32>,
    available_memory: AvailableMemory,
    fifo_memory: FifoMemory,
    /// When set, logical CUDA device IDs are constrained to the matching
    /// physical pinned-memory region under FIFO scheduling.
    fifo_card_regions: Option<Vec<usize>>,
    /// Total bytes of CPU memory
    total_memory: usize,
}

impl GlobalResources {
    pub fn new(
        cards: Vec<i32>,
        available_memory: usize,
        fifo_split_cpu_memory: bool,
        fifo_card_regions: Option<Vec<usize>>,
    ) -> Self {
        if let Some(regions) = &fifo_card_regions {
            assert_eq!(regions.len(), cards.len(), "FIFO GPU NUMA map does not match cards");
            assert!(fifo_split_cpu_memory, "FIFO GPU NUMA map requires split CPU memory");
        }
        Self {
            available_memory: AvailableMemory::Intact(None),
            fifo_memory: FifoMemory::new(available_memory, fifo_split_cpu_memory),
            fifo_card_regions,
            total_memory: available_memory,
            available_cards: cards,
        }
    }
}

fn try_take_cards(cards: &mut Vec<i32>, n: usize) -> Option<Vec<i32>> {
    if cards.len() >= n {
        Some((0..n).map(|_| cards.pop().unwrap()).collect())
    } else {
        None
    }
}

impl GlobalResources {
    fn try_take_cards<T>(
        &mut self,
        n: usize,
        f: impl FnOnce(Vec<i32>, &mut Self) -> Option<T>,
    ) -> Option<T> {
        let cards = try_take_cards(&mut self.available_cards, n)?;
        if let Some(r) = f(cards.clone(), self) {
            Some(r)
        } else {
            self.available_cards.extend_from_slice(&cards);
            None
        }
    }

    fn try_take_fifo_card_and_memory(&mut self, size: usize) -> Option<(Vec<i32>, usize)> {
        if self.fifo_card_regions.is_none() {
            let card = self.available_cards.pop()?;
            return match self.fifo_memory.allocate(size) {
                Some(offset) => Some((vec![card], offset)),
                None => {
                    self.available_cards.push(card);
                    None
                }
            };
        }

        for position in (0..self.available_cards.len()).rev() {
            let card = self.available_cards[position];
            let region = self
                .fifo_card_regions
                .as_ref()
                .map(|regions| regions[card as usize])
                .unwrap_or(0);
            if let Some(offset) = self.fifo_memory.allocate_in_region(size, region) {
                self.available_cards.remove(position);
                return Some((vec![card], offset));
            }
        }
        None
    }
}
pub struct Core {
    pub config: SchedulerConfig,
    /// Newly submitted tasks go here
    pending_tasks: VecDeque<AcceptedTask>,
    /// We schedule tasks window by window. by algorithms described in
    ///   Approximate Algorithms for Scheduling Parallelizable Tasks
    /// Takss from `pending_tasks` are dumped to this queue periodically
    scheduling_window: Vec<AcceptedTask>,
    scheduling_window_sorted: bool,
    scheduling_window_allotted: bool,
    resources: GlobalResources,
    programs: Heap<ProgramId, erased::BoxedArtifect>,
    task_id_allocator: IdAllocator<TaskId>,
    knowledge: Knowlede,
}

impl Core {
    pub fn new(
        config: SchedulerConfig,
        resources: GlobalResources,
        programs: Heap<ProgramId, erased::BoxedArtifect>,
    ) -> Self {
        Self {
            config: config,
            pending_tasks: VecDeque::new(),
            scheduling_window: Vec::new(),
            scheduling_window_sorted: false,
            scheduling_window_allotted: false,
            resources: resources,
            knowledge: Knowlede {
                time_experience: programs.map_by_ref(&mut |_, artifect| {
                    artifect
                        .versions()
                        .map(|ver| (ver.clone(), Duration::from_secs(1)))
                        .collect()
                }),
            },
            programs: programs,
            task_id_allocator: IdAllocator::new(),
        }
    }

    pub fn add_artifect(&mut self, art: erased::BoxedArtifect) -> ProgramId {
        self.knowledge.time_experience.push(
            art.versions()
                .map(|ver| (ver.clone(), Duration::from_secs(1)))
                .collect(),
        );
        self.programs.push(art)
    }

    pub fn artifect(&self, program: ProgramId) -> &erased::BoxedArtifect {
        &self.programs[program]
    }

    pub fn add_task(&mut self, submit: submitter::SubmittedTask) -> TaskId {
        let id = self.task_id_allocator.alloc();
        let accepted = AcceptedTask {
            id,
            submitted: submit,
            cards: Vec::new(),
            version: None,
            memory_offset: None,
        };
        self.pending_tasks.push_back(accepted);
        id
    }

    fn dump_tasks_to_scheduling_window(&mut self) {
        let n = self
            .config
            .schedule_window_size
            .min(self.pending_tasks.len());
        self.scheduling_window.extend(self.pending_tasks.drain(..n));
        self.scheduling_window_sorted = false;
        self.scheduling_window_allotted = false;
    }

    /// Determine CPU memory assigned to each task in the scheduling window.
    /// Window must be non-empty>
    fn allot_memory(&mut self) {
        // Initially set
        //   beta_j^1 = arg min_i t_j (i) dot i
        // where beta_j^k is the CPU memory allocated to task j at iteration k,
        // and t_j (i) is estimated execution of task j with i bytes assigned
        self.scheduling_window.iter_mut().for_each(|task| {
            let version = self.programs[task.submitted.program]
                .versions()
                .min_by_key(|m| {
                    self.knowledge
                        .estimate_time(task.submitted.program, *m)
                        .mul_f64(m.memory_limit_gigabytes())
                })
                .expect("artifect does not provide any version");
            task.version = Some(version.clone());
        });

        loop {
            // In each iteration k, we find task j_0 such that
            //   j0 = arg max t_j (beta_j^(k - 1))
            // and allot more memory to it
            //   beta_j0^k = arg min_(i > beta_j0^(k - 1)) t_j (i) dot i
            // while maintaining
            //   beta_j^k = beta_j^(k - 1)
            // for all j != j0
            let bottoleneck = self
                .scheduling_window
                .iter_mut()
                .max_by_key(|task| {
                    self.knowledge
                        .estimate_time(task.submitted.program, task.version.as_ref().unwrap())
                })
                .expect("empty schedule window");
            let version = self.programs[bottoleneck.submitted.program]
                .versions()
                .filter(|m| m.memory_limit() > bottoleneck.version.as_ref().unwrap().memory_limit())
                .min_by_key(|m| {
                    self.knowledge
                        .estimate_time(bottoleneck.submitted.program, *m)
                        .mul_f64(m.memory_limit_gigabytes())
                });
            if let Some(version) = version {
                bottoleneck.version = Some(version.clone());
            } else {
                self.scheduling_window_allotted = true;
                return;
            }
        }
    }

    /// Choose from schedule window a task to run.
    /// The returned [`AcceptedTask`] guarantees that the `version`, `cards` and `memory_offset` fields are set.
    pub fn schedule(&mut self) -> Option<AcceptedTask> {
        match self.config.scheduling_policy {
            SchedulingPolicy::Heuristic => self.schedule_heuristic(),
            SchedulingPolicy::Fifo => self.schedule_fifo(),
        }
    }

    /// Existing heuristic scheduler. It is kept separate from FIFO so the default behavior is
    /// unchanged.
    fn schedule_heuristic(&mut self) -> Option<AcceptedTask> {
        let cards_per_request = 1;

        if self.scheduling_window.is_empty() {
            self.dump_tasks_to_scheduling_window();
        }

        if self.scheduling_window.is_empty() {
            return None;
        }

        println!(
            "Scheduling queue: {:?}",
            self.scheduling_window
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
        );

        if !self.scheduling_window_allotted {
            self.allot_memory();
        }

        // If there are tasks consuming more than half of total, try schedule them first
        if let Some(j) = self.scheduling_window.iter().position(|task| {
            task.version.as_ref().unwrap().memory_limit() * 2 > self.resources.total_memory as u64
        }) {
            self.resources.available_memory.try_switch_to_intact();

            return self
                .resources
                .try_take_cards(cards_per_request, |cards, resources| {
                    if let AvailableMemory::Intact(None) = resources.available_memory {
                        let mut task = self.scheduling_window.remove(j);
                        task.cards = cards;
                        task.memory_offset = Some(0);
                        resources.available_memory = AvailableMemory::Intact(Some(task.id));

                        println!("Scheduling task {:?} with an intact memory block", &task);

                        Some(task)
                    } else {
                        None
                    }
                });
        }

        self.resources
            .available_memory
            .try_switch_to_halved(self.resources.total_memory);

        // Otherwise, sort the schedule window by execution time,
        // then try schedule first one to the half that is estimated to become free earlier
        if !self.scheduling_window_sorted {
            self.scheduling_window.sort_by(|a, b| {
                self.knowledge
                    .estimate_time(a.submitted.program, a.version.as_ref().unwrap())
                    .cmp(
                        &self
                            .knowledge
                            .estimate_time(b.submitted.program, b.version.as_ref().unwrap()),
                    )
                    .reverse()
            });
            self.scheduling_window_sorted = true;
        }

        if let Some(task) = self.scheduling_window.last() {
            let version = task.version.clone();

            return self
                .resources
                .try_take_cards(cards_per_request, |cards, resources| {
                    if let AvailableMemory::Halved(halves) = &mut resources.available_memory {
                        halves.sort_by_key(|halve| halve.estimated_free_at);

                        for half in halves {
                            if half.space_left()
                                >= version.as_ref().unwrap().memory_limit() as usize
                            {
                                let mut task = self.scheduling_window.pop().unwrap();
                                let estimated_free_at = Instant::now()
                                    + self.knowledge.estimate_time(
                                        task.submitted.program,
                                        task.version.as_ref().unwrap(),
                                    );
                                task.cards = cards;
                                task.memory_offset = Some(half.add_task(&task, estimated_free_at));

                                println!("Scheduling task {:?} with half a memory block", &task);

                                return Some(task);
                            }
                        }
                    }
                    None
                });
        }

        None
    }

    /// FIFO admission with first-fit CPU-range allocation.
    ///
    /// FIFO deliberately does no duration-based reordering or look-ahead. It selects the largest
    /// available artifact version, matching the effective final version selected by the legacy
    /// `allot_memory` loop, then admits only the oldest pending task when both a card and a
    /// contiguous CPU range are available.
    fn schedule_fifo(&mut self) -> Option<AcceptedTask> {
        const CARDS_PER_REQUEST: usize = 1;

        if self.scheduling_window.is_empty() {
            self.dump_tasks_to_scheduling_window();
        }

        if self.scheduling_window.is_empty() {
            return None;
        }

        if !self.scheduling_window_allotted {
            for task in &mut self.scheduling_window {
                let version = self.programs[task.submitted.program]
                    .versions()
                    .max_by_key(|version| version.memory_limit())
                    .expect("artifect does not provide any version");
                task.version = Some(version.clone());
            }
            self.scheduling_window_allotted = true;
        }

        println!(
            "FIFO Scheduling queue: {:?}",
            self.scheduling_window
                .iter()
                .map(|task| task.id)
                .collect::<Vec<_>>()
        );

        let version = self.scheduling_window.first()?.version.clone().unwrap();
        assert_eq!(CARDS_PER_REQUEST, 1, "NUMA FIFO currently supports one GPU per task");
        let (cards, offset) = self
            .resources
            .try_take_fifo_card_and_memory(version.memory_limit() as usize)?;
        let mut task = self.scheduling_window.remove(0);
        task.cards = cards;
        task.memory_offset = Some(offset);

        println!("FIFO scheduling task {:?}", &task);
        Some(task)
    }

    pub fn finish(
        &mut self,
        id: TaskId,
        cards: &[i32],
        program: ProgramId,
        version: &MemoryInfo,
        memory_offset: usize,
        duration: Duration,
    ) {
        self.knowledge.update(program, version, duration);

        match self.config.scheduling_policy {
            SchedulingPolicy::Heuristic => match &mut self.resources.available_memory {
                AvailableMemory::Intact(t) => {
                    if t.is_some_and(|x| x == id) {
                        self.resources.available_memory = AvailableMemory::Intact(None)
                    } else {
                        panic!(
                            "finishing a task {:?} but availabel memory is Intact({:?})",
                            id, t
                        )
                    }
                }
                AvailableMemory::Halved([h1, h2]) => {
                    if !h1.assigned_tasks.remove(&id) && !h2.assigned_tasks.remove(&id) {
                        panic!(
                            "finishing a task {:?} but no memory is assigned to it in neither half",
                            id
                        );
                    }
                }
            },
            SchedulingPolicy::Fifo => self
                .resources
                .fifo_memory
                .deallocate(memory_offset, version.memory_limit() as usize),
        }

        self.resources.available_cards.extend_from_slice(cards);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct DummyArtifect {
        version: MemoryInfo,
    }

    struct DummyInputs;

    fn submit_dummy_task(core: &mut Core, program: ProgramId) -> TaskId {
        core.add_task(submitter::SubmittedTask {
            program,
            inputs: Box::new(DummyInputs),
        })
    }

    impl erased::Artifects for DummyArtifect {
        fn versions<'a>(&'a self) -> Box<dyn Iterator<Item = &'a MemoryInfo> + 'a> {
            Box::new(std::iter::once(&self.version))
        }

        fn prepare_dispatcher(
            &self,
            _version: &MemoryInfo,
            _pools: Pools,
            _rng: AsyncRng,
            _gpu_mapping: GpuMapping,
        ) -> erased::BoxedRuntime {
            panic!("DummyArtifect cannot be run")
        }
    }

    impl erased::VariableTables for DummyInputs {}

    struct Iter<'c>(&'c mut Core);

    impl<'c> Iterator for Iter<'c> {
        type Item = AcceptedTask;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.schedule()
        }
    }
    impl Core {
        fn schedule_iter(&mut self) -> Iter<'_> {
            Iter(self)
        }
        fn finish_accepted_task(&mut self, t: &AcceptedTask, time: Duration) {
            self.finish(
                t.id,
                &t.cards,
                t.submitted.program,
                t.version.as_ref().unwrap(),
                t.memory_offset.unwrap(),
                time,
            );
        }
    }

    struct RunningTasks {
        running_time: HashMap<TaskId, usize>,
        running_tasks: Vec<(AcceptedTask, usize)>,
        now: usize,
    }

    impl RunningTasks {
        fn new() -> Self {
            Self {
                running_time: HashMap::new(),
                running_tasks: Vec::new(),
                now: 0,
            }
        }

        fn schedule(&mut self, core: &mut Core) {
            self.running_tasks.extend(core.schedule_iter().map(|t| {
                println!("[+{}] Started running {:?}", self.now, t.id);
                (t, self.now)
            }));
        }

        fn tick(&mut self, delta: usize, core: &mut Core) {
            for _ in 0..delta {
                self.tick1(core);
            }
        }

        fn tick1(&mut self, core: &mut Core) {
            self.now += 1;

            while let Some(idx) = self
                .running_tasks
                .iter()
                .position(|(rt, begin)| self.now - *begin > self.running_time[&rt.id])
            {
                let (t, begin) = self.running_tasks.remove(idx);
                let dur = self.now - begin;
                println!("[+{}] Finished {:?} in {}", self.now, t.id, dur);
                core.finish_accepted_task(&t, Duration::from_millis(dur as u64));
            }

            self.schedule(core);
        }

        fn submit(&mut self, task: submitter::SubmittedTask, running_time: usize, core: &mut Core) {
            let id = core.add_task(task);
            self.running_time.insert(id, running_time);

            self.schedule(core);
        }
    }

    #[test]
    fn test_core_4card() {
        let mut core = Core::new(
            SchedulerConfig::default(),
            GlobalResources::new(vec![0, 1, 2, 3], 2usize.pow(30), false, None),
            Heap::new(),
        );

        let prog_id = core.add_artifect(Box::new(DummyArtifect {
            version: MemoryInfo::new(2u64.pow(28)),
        }));

        let req = || submitter::SubmittedTask {
            program: prog_id,
            inputs: Box::new(DummyInputs),
        };

        let mut rts = RunningTasks::new();

        let core = &mut core;

        rts.submit(req(), 10, core);
        rts.submit(req(), 11, core);
        rts.submit(req(), 12, core);
        rts.submit(req(), 13, core);

        rts.tick(8, core);
        rts.submit(req(), 10, core);
        rts.submit(req(), 11, core);

        rts.tick(7, core);
        rts.submit(req(), 11, core);
        rts.submit(req(), 12, core);

        rts.tick(10, core);
        rts.submit(req(), 10, core);
        rts.submit(req(), 14, core);

        rts.tick(100, core);

        assert!(rts.running_tasks.is_empty());
        assert!(core.pending_tasks.is_empty());
        assert!(core.scheduling_window.is_empty());
    }

    #[test]
    fn test_fifo_admits_in_submission_order_with_global_cpu_ranges() {
        let mut core = Core::new(
            SchedulerConfig::default().with_scheduling_policy(SchedulingPolicy::Fifo),
            GlobalResources::new(vec![0, 1, 2, 3], 500, false, None),
            Heap::new(),
        );
        let program = core.add_artifect(Box::new(DummyArtifect {
            version: MemoryInfo::new(160),
        }));
        let ids = [
            submit_dummy_task(&mut core, program),
            submit_dummy_task(&mut core, program),
            submit_dummy_task(&mut core, program),
            submit_dummy_task(&mut core, program),
        ];
        let first = core.schedule().unwrap();
        let second = core.schedule().unwrap();
        let third = core.schedule().unwrap();

        assert_eq!([first.id, second.id, third.id], ids[..3]);
        assert_eq!(first.memory_offset, Some(0));
        assert_eq!(second.memory_offset, Some(160));
        assert_eq!(third.memory_offset, Some(320));
        assert!(core.schedule().is_none(), "only 20 bytes remain");

        core.finish_accepted_task(&second, Duration::from_secs(1));
        let fourth = core.schedule().unwrap();
        assert_eq!(fourth.id, ids[3]);
        assert_eq!(fourth.memory_offset, Some(160));

        core.finish_accepted_task(&first, Duration::from_secs(1));
        core.finish_accepted_task(&third, Duration::from_secs(1));
        core.finish_accepted_task(&fourth, Duration::from_secs(1));

        let next_id = submit_dummy_task(&mut core, program);
        let next = core.schedule().unwrap();
        assert_eq!(next.id, next_id);
        assert_eq!(next.memory_offset, Some(0));
    }

    #[test]
    fn test_fifo_split_cpu_memory_never_crosses_pinned_region_boundary() {
        let mut core = Core::new(
            SchedulerConfig::default().with_scheduling_policy(SchedulingPolicy::Fifo),
            GlobalResources::new(vec![0, 1, 2, 3], 700, true, None),
            Heap::new(),
        );
        let program = core.add_artifect(Box::new(DummyArtifect {
            version: MemoryInfo::new(160),
        }));
        for _ in 0..4 {
            submit_dummy_task(&mut core, program);
        }

        let tasks = (0..4)
            .map(|_| {
                core.schedule()
                    .expect("FIFO should admit four 160-byte tasks")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            tasks
                .iter()
                .map(|task| task.memory_offset)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(160), Some(350), Some(510)]
        );

        for task in &tasks {
            core.finish_accepted_task(task, Duration::from_secs(1));
        }
        let oversized_program = core.add_artifect(Box::new(DummyArtifect {
            version: MemoryInfo::new(400),
        }));
        submit_dummy_task(&mut core, oversized_program);
        assert!(
            core.schedule().is_none(),
            "a FIFO task must not span the two pinned-memory regions"
        );
    }

    #[test]
    fn test_fifo_numa_regions_keep_cards_with_local_memory() {
        let mut core = Core::new(
            SchedulerConfig::default().with_scheduling_policy(SchedulingPolicy::Fifo),
            // GPUs 0/1 use region 1; GPUs 2/3 use region 0.
            GlobalResources::new(vec![0, 1, 2, 3], 700, true, Some(vec![1, 1, 0, 0])),
            Heap::new(),
        );
        let program = core.add_artifect(Box::new(DummyArtifect {
            version: MemoryInfo::new(160),
        }));
        for _ in 0..4 {
            submit_dummy_task(&mut core, program);
        }

        let tasks = (0..4)
            .map(|_| core.schedule().expect("NUMA FIFO should admit four tasks"))
            .collect::<Vec<_>>();
        assert_eq!(
            tasks.iter().map(|task| task.cards.clone()).collect::<Vec<_>>(),
            vec![vec![3], vec![2], vec![1], vec![0]]
        );
        assert_eq!(
            tasks.iter().map(|task| task.memory_offset).collect::<Vec<_>>(),
            vec![Some(0), Some(160), Some(350), Some(510)]
        );
    }
}
