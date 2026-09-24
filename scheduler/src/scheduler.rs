mod prelude {
    pub use std::collections::{BTreeMap, HashMap, HashSet};
    pub use std::sync::{mpsc, Arc, Mutex};
    pub use std::thread::{self};
    pub use std::time::{Duration, Instant};
    pub use zkpoly_common::define_usize_id;
    pub use zkpoly_common::heap::{Heap, IdAllocator, UsizeId};
    pub use zkpoly_compiler::driver::artifect::{GpuMapping, Pools};
    pub use zkpoly_compiler::driver::{Artifect, HardwareInfo, MemoryInfo, MemoryPlanningConfig};
    pub use zkpoly_memory_pool::buddy_disk_pool::DiskMemoryPool;
    pub use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
    pub use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
    pub use zkpoly_runtime::async_rng::AsyncRng;
    pub use zkpoly_runtime::debug::Log;
    pub use zkpoly_runtime::runtime::{Runtime, RuntimeDebug};
}

use prelude::*;

define_usize_id!(TaskId);

mod erased;
mod submitter;

pub use submitter::exposed::*;
use submitter::ProgramId;
pub use submitter::Submitter;

struct WorkerFinishRespond {
    r: submitter::RunReturn,
    id: TaskId,
    program: ProgramId,
    version: MemoryInfo,
    cards: Vec<i32>,
    memory_offset: usize,
}

struct LaunchedTask {
    /// The `version`, `cards` and `memory_offset` fields in `accepted`
    /// are expected to be set here.
    accepted: AcceptedTask,
    runtime: erased::BoxedRuntime,
    runtime_debug: RuntimeDebug,
}

struct AcceptedTask {
    id: TaskId,
    submitted: submitter::SubmittedTask,
    version: Option<MemoryInfo>,
    cards: Vec<i32>,
    memory_offset: Option<usize>,
}

impl std::fmt::Debug for AcceptedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptedTask")
            .field("id", &self.id)
            .field("version", &self.version)
            .field("cards", &self.cards)
            .field("program", &self.submitted.program)
            .field("memory_offset", &self.memory_offset)
            .finish()
    }
}

struct ExecutorHandle {
    thread: thread::JoinHandle<()>,
}

/// Handle to the scheduler thread.
#[derive(Debug)]
pub struct SchedulerHandle {
    shutdown_sender: mpsc::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl SchedulerHandle {
    /// Shutdown the scheduler thread and executor threads.
    pub fn shutdown(self) {
        self.shutdown_sender
            .send(())
            .expect("send shutdown command failed");
        self.thread.join().expect("join scheduler failed");
    }
}

/// Configuration for the scheduler.
pub struct SchedulerConfig {
    num_executors: usize,
    memory_check: bool,
    runtime_debug: RuntimeDebug,
    schedule_window_size: usize,
    scheduling_policy: SchedulingPolicy,
    fifo_split_cpu_memory: bool,
}

/// Selects how the scheduler admits pending tasks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// The existing heuristic that estimates task durations and partitions CPU memory in halves.
    Heuristic,
    /// Admit tasks in submission order using a first-fit allocation from the whole CPU memory pool.
    Fifo,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            num_executors: 4,
            memory_check: true,
            runtime_debug: RuntimeDebug::none(),
            schedule_window_size: 32,
            scheduling_policy: SchedulingPolicy::Heuristic,
            fifo_split_cpu_memory: false,
        }
    }
}

impl SchedulerConfig {
    /// Controls number of executors launched by the scheduler.
    pub fn with_num_executors(self, x: usize) -> Self {
        Self {
            num_executors: x,
            ..self
        }
    }

    /// Controls whether memory sanity checks are enabled, for debugging.
    pub fn with_memory_check(self, x: bool) -> Self {
        Self {
            memory_check: x,
            ..self
        }
    }

    /// Controls runtime debug options.
    pub fn with_runtime_debug(self, x: RuntimeDebug) -> Self {
        Self {
            runtime_debug: x,
            ..self
        }
    }

    /// Configures the schedule window size.
    pub fn with_schedule_window_size(self, x: usize) -> Self {
        Self {
            schedule_window_size: x,
            ..self
        }
    }

    /// Selects the task admission policy. Defaults to [`SchedulingPolicy::Heuristic`].
    pub fn with_scheduling_policy(self, x: SchedulingPolicy) -> Self {
        Self {
            scheduling_policy: x,
            ..self
        }
    }

    /// For FIFO scheduling, allocate the pinned CPU pool as two equal independent regions.
    ///
    /// This is useful on hosts where one large CUDA pinned-memory allocation fails despite two
    /// smaller allocations fitting. FIFO allocations never span the boundary between the regions.
    /// The option has no effect for [`SchedulingPolicy::Heuristic`].
    pub fn with_fifo_split_cpu_memory(self, x: bool) -> Self {
        Self {
            fifo_split_cpu_memory: x,
            ..self
        }
    }
}

/// The pinned CPU backing memory for all dispatched tasks.
///
/// The split variant intentionally consists of two independent CUDA allocations. A logical FIFO
/// allocation is translated to an offset inside one backing allocation when a dispatcher is made.
enum SchedulerCpuMemory {
    Single(CpuStaticAllocator),
    Split {
        first: CpuStaticAllocator,
        second: CpuStaticAllocator,
        half: usize,
    },
}

/// Host-specific opt-in for the two-region FIFO allocator.  Each logical CUDA
/// device is assigned to one pinned-memory region, while the two backing
/// allocations are made by threads bound to the corresponding NUMA CPUs.
///
/// This intentionally lives behind environment variables: it is a diagnostic
/// execution policy and must not silently alter the portable default scheduler.
struct FifoNumaConfig {
    cpu_sets: [Vec<usize>; 2],
    gpu_regions: Vec<usize>,
}

impl FifoNumaConfig {
    fn parse_cpu_set(value: &str) -> Vec<usize> {
        value
            .split(',')
            .filter(|s| !s.is_empty())
            .flat_map(|part| match part.split_once('-') {
                Some((begin, end)) => {
                    let begin = begin.parse::<usize>().expect("invalid NUMA CPU range start");
                    let end = end.parse::<usize>().expect("invalid NUMA CPU range end");
                    assert!(begin <= end, "invalid descending NUMA CPU range");
                    (begin..=end).collect::<Vec<_>>()
                }
                None => vec![part.parse::<usize>().expect("invalid NUMA CPU id")],
            })
            .collect()
    }

    fn from_env(card_count: usize, fifo_split_cpu_memory: bool) -> Option<Self> {
        if std::env::var("ZKPOLY_FIFO_NUMA_LOCAL").ok().as_deref() != Some("1") {
            return None;
        }
        assert!(
            fifo_split_cpu_memory,
            "ZKPOLY_FIFO_NUMA_LOCAL requires FIFO split CPU memory"
        );

        let cpu_sets = std::env::var("ZKPOLY_FIFO_NUMA_CPUSETS")
            .expect("ZKPOLY_FIFO_NUMA_CPUSETS must contain two semicolon-separated CPU sets");
        let mut cpu_sets = cpu_sets.split(';').map(Self::parse_cpu_set);
        let first = cpu_sets.next().expect("missing first NUMA CPU set");
        let second = cpu_sets.next().expect("missing second NUMA CPU set");
        assert!(cpu_sets.next().is_none(), "expected exactly two NUMA CPU sets");
        assert!(!first.is_empty() && !second.is_empty(), "NUMA CPU set cannot be empty");

        let gpu_regions = std::env::var("ZKPOLY_FIFO_NUMA_GPU_REGIONS")
            .expect("ZKPOLY_FIFO_NUMA_GPU_REGIONS must map every logical GPU to region 0 or 1")
            .split(',')
            .map(|part| part.parse::<usize>().expect("invalid GPU NUMA region"))
            .collect::<Vec<_>>();
        assert_eq!(
            gpu_regions.len(),
            card_count,
            "NUMA GPU-region mapping must have one entry per visible logical GPU"
        );
        assert!(
            gpu_regions.iter().all(|region| *region < 2),
            "NUMA GPU-region values must be 0 or 1"
        );

        println!(
            "enabling NUMA-local FIFO pinned memory: regions {:?}, CPU sets {:?} / {:?}",
            gpu_regions, first, second
        );
        Some(Self {
            cpu_sets: [first, second],
            gpu_regions,
        })
    }
}

impl SchedulerCpuMemory {
    fn new(
        capacity: usize,
        memory_check: bool,
        split_in_two: bool,
        numa: Option<&FifoNumaConfig>,
    ) -> Self {
        if split_in_two {
            assert!(
                capacity % 2 == 0,
                "FIFO split CPU memory requires an even byte capacity"
            );
            let half = capacity / 2;
            Self::Split {
                first: numa.map_or_else(
                    || CpuStaticAllocator::new(half, memory_check),
                    |numa| CpuStaticAllocator::new_on_cpu_set(half, memory_check, numa.cpu_sets[0].clone()),
                ),
                second: numa.map_or_else(
                    || CpuStaticAllocator::new(half, memory_check),
                    |numa| CpuStaticAllocator::new_on_cpu_set(half, memory_check, numa.cpu_sets[1].clone()),
                ),
                half,
            }
        } else {
            Self::Single(CpuStaticAllocator::new(capacity, memory_check))
        }
    }

    fn slice(&self, offset: usize, size: usize) -> CpuStaticAllocator {
        match self {
            Self::Single(memory) => memory.slice(offset, size),
            Self::Split {
                first,
                second,
                half,
            } => {
                let end = offset.checked_add(size).expect("CPU slice offset overflow");
                if offset < *half {
                    assert!(
                        end <= *half,
                        "FIFO CPU allocation cannot cross a pinned-memory split boundary"
                    );
                    first.slice(offset, size)
                } else {
                    assert!(
                        end <= half.checked_mul(2).expect("CPU slice size overflow"),
                        "FIFO CPU allocation is out of bounds"
                    );
                    second.slice(offset - *half, size)
                }
            }
        }
    }
}

struct Knowlede {
    /// Duration for all versions are expected to be found here.
    /// There fore, they must be initialized to some estimations before hand.
    time_experience: Heap<ProgramId, Vec<(MemoryInfo, Duration)>>,
}

impl Knowlede {
    fn estimate_time(&self, program: ProgramId, version: &MemoryInfo) -> Duration {
        let assoc = &self.time_experience[program];
        assoc
            .iter()
            .find(|(v1, _)| v1 == version)
            .map(|(_, t)| *t)
            .unwrap()
    }

    fn update(&mut self, program: ProgramId, version: &MemoryInfo, duration: Duration) {
        const SMOOTH_FACTOR: f32 = 0.4;

        let assoc = &mut self.time_experience[program];
        let (_, old) = assoc.iter_mut().find(|(v1, _)| v1 == version).unwrap();

        *old = old.mul_f32(SMOOTH_FACTOR) + duration.mul_f32(1.0 - SMOOTH_FACTOR);
    }
}

struct Control {
    reception: mpsc::Receiver<submitter::Message>,
    executors: Vec<ExecutorHandle>,
    shutdown_receiver: mpsc::Receiver<()>,
    task_launcher: crossbeam_channel::Sender<LaunchedTask>,
    result_receiver: crossbeam_channel::Receiver<WorkerFinishRespond>,
    result_relayer: HashMap<TaskId, crossbeam_channel::Sender<submitter::RunReturn>>,
}

mod core;

use core::Core;
use std::marker::PhantomData;

/// The scheduler.
/// It runs in a dedicated thread, and communicates with executors and users with channels.
pub struct Scheduler {
    hd_info: HardwareInfo,
    memory_config: MemoryPlanningConfig,
    control: Control,
    core: Core,
    cpu_memory: SchedulerCpuMemory,
    disk_memory: Arc<Mutex<DiskMemoryPool>>,
    rng: AsyncRng,
    shutdown_sender: mpsc::Sender<()>,
}

impl Scheduler {
    fn add_task(&mut self, submit: submitter::Submit) -> TaskId {
        let id = self.core.add_task(submit.task);
        self.control.result_relayer.insert(id, submit.result_sender);
        id
    }

    fn finish_task(&mut self, resp: WorkerFinishRespond) {
        self.core.finish(
            resp.id,
            &resp.cards,
            resp.program,
            &resp.version,
            resp.memory_offset,
            resp.r.time,
        );

        let _ = self
            .control
            .result_relayer
            .remove(&resp.id)
            .unwrap()
            .try_send(resp.r);
    }

    fn schedule_task(&mut self) {
        while let Some(accepted) = self.core.schedule() {
            let pools = Pools {
                cpu: self.cpu_memory.slice(
                    accepted.memory_offset.unwrap(),
                    accepted.version.as_ref().unwrap().memory_limit() as usize,
                ),
                gpu: self.hd_info.gpu_allocators_for(
                    self.core.config.memory_check,
                    accepted.cards.iter().copied(),
                    &self.memory_config,
                ),
                disk: self.disk_memory.clone(),
            };

            let runtime = self
                .core
                .artifect(accepted.submitted.program)
                .prepare_dispatcher(
                    accepted.version.as_ref().unwrap(),
                    pools,
                    self.rng.clone(),
                    Arc::new({
                        let mapping = accepted.cards.clone();
                        move |i| mapping[i as usize]
                    }),
                );

            let launched = LaunchedTask {
                accepted,
                runtime,
                runtime_debug: self.core.config.runtime_debug.clone(),
            };
            self.control
                .task_launcher
                .send(launched)
                .expect("all task spawn closed");
        }
    }

    /// Launch the scheduler thread.
    pub fn launch(mut self) -> SchedulerHandle {
        let shutdown_sender = self.shutdown_sender.clone();

        let jh = thread::spawn(move || {
            println!(
                "Scheduler thread started with {} executors",
                self.core.config.num_executors
            );

            loop {
                // Check for newly submitted user tasks.
                while let Ok(msg) = self.control.reception.try_recv() {
                    match msg {
                        submitter::Message::Submit(submitted) => {
                            let id = self.add_task(submitted);
                            self.schedule_task();
                            println!("Scheduler received a new task {:?}", id);
                        }
                        submitter::Message::Add(art, sender) => {
                            let id = self.core.add_artifect(art);
                            let _ = sender.send(id);
                        }
                    }
                }

                // Executor completed a task.
                while let Ok(resp) = self.control.result_receiver.try_recv() {
                    let id = resp.id;
                    self.finish_task(resp);
                    println!("Task {:?} completed", id);
                    self.schedule_task();
                }

                if let Ok(..) = self.control.shutdown_receiver.try_recv() {
                    break;
                }

                // Sleep briefly to avoid busy waiting.
                thread::sleep(Duration::from_millis(10));
            }

            drop(self.control.task_launcher);

            println!("Shutting down the scheduler; waiting for worker threads to finish.");
            for exe in self.control.executors {
                exe.thread.join().unwrap();
            }

            println!("Scheduler stopped");
        });

        SchedulerHandle {
            shutdown_sender,
            thread: jh,
        }
    }
}

/// Make a new scheduler and launch executor threads.
pub fn make_scheduler<Rt: RuntimeType>(
    hd_info: HardwareInfo,
    memory_config: MemoryPlanningConfig,
    config: SchedulerConfig,
    rng: AsyncRng,
    disk_pool: DiskMemoryPool,
    programs: Programs<Rt>,
) -> (Scheduler, submitter::Submitter<Rt>) {
    let cards_per_request = 1; // For now this is a constant.

    assert!(
        hd_info.gpus().count() % cards_per_request == 0,
        "Total cards must be a multiple of cards per request"
    );

    // Create the task submission channel (user -> scheduler).
    let (task_submitter, task_reception) = mpsc::channel::<submitter::Message>();

    // Create the task execution channel (scheduler -> executor).
    let (task_launcher, task_spawn) = crossbeam_channel::unbounded::<LaunchedTask>();

    // Create the executor status channel (executor -> scheduler).
    let (result_sender, result_receiver) = crossbeam_channel::unbounded::<WorkerFinishRespond>();

    let (shutdown_sender, shutdown_receiver) = mpsc::channel::<()>();

    let executors = (0..config.num_executors)
        .map(|i| {
            let task_spawn = task_spawn.clone();
            let result_sender = result_sender.clone();

            let executor_thread = thread::spawn(move || {
                println!("Executor thread {} started", i);

                while let Ok(task) = task_spawn.recv() {
                    println!("Executor {} started task {:?}", i, task.accepted.id,);

                    // Execute the task.
                    let mut runtime = task.runtime;
                    let (r, log) =
                        runtime.run(task.accepted.submitted.inputs.as_ref(), task.runtime_debug);
                    let elapsed = log.total_time();

                    drop(runtime);

                    println!(
                        "Executor {} completed task {:?} at {:?}",
                        i, task.accepted.id, elapsed,
                    );
                    if !log.is_empty() {
                        let statistics = log.statistics();
                        println!("Statistics for task {:?}: {:#?}", task.accepted.id, statistics);
                    }

                    // Send the result.
                    if let Err(e) = result_sender.send(WorkerFinishRespond {
                        r: submitter::RunReturn {
                            ret_value: r,
                            log,
                            time: elapsed,
                        },
                        id: task.accepted.id,
                        program: task.accepted.submitted.program,
                        version: task.accepted.version.unwrap(),
                        cards: task.accepted.cards,
                        memory_offset: task.accepted.memory_offset.unwrap(),
                    }) {
                        eprintln!(
                            "Executor {} failed to send the result for task {:?}: {:?}",
                            i, task.accepted.id, e
                        );
                    }
                }
            });

            ExecutorHandle {
                thread: executor_thread,
            }
        })
        .collect::<Vec<_>>();

    let memory_check = config.memory_check;
    let fifo_split_cpu_memory =
        config.scheduling_policy == SchedulingPolicy::Fifo && config.fifo_split_cpu_memory;
    let fifo_numa = FifoNumaConfig::from_env(hd_info.gpus().count(), fifo_split_cpu_memory);
    let cpu_memory = SchedulerCpuMemory::new(
        hd_info.cpu().memory_limit() as usize,
        memory_check,
        fifo_split_cpu_memory,
        fifo_numa.as_ref(),
    );
    let core = Core::new(
        config,
        core::GlobalResources::new(
            (0..hd_info.gpus().count()).map(|x| x as i32).collect(),
            hd_info.cpu().memory_limit() as usize,
            fifo_split_cpu_memory,
            fifo_numa.map(|numa| numa.gpu_regions),
        ),
        programs.erase(),
    );

    let control = Control {
        reception: task_reception,
        executors,
        shutdown_receiver,
        task_launcher,
        result_receiver,
        result_relayer: HashMap::new(),
    };

    let sch = Scheduler {
        cpu_memory,
        memory_config,
        hd_info,
        control,
        core,
        disk_memory: Arc::new(Mutex::new(disk_pool)),
        rng,
        shutdown_sender,
    };

    let smt = submitter::Submitter {
        sender: task_submitter,
        _phantom: PhantomData,
    };

    (sch, smt)
}
