use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock, RwLockReadGuard},
    time::{Duration, Instant},
};
use zkpoly_common::bijection::Bijection;

use zkpoly_cuda_api::stream::{CudaEventRaw, CudaStream};

use crate::{
    devices::ThreadId,
    functions::FuncMeta,
    instructions::{self, Instruction},
};

type StreamTable = BTreeMap<instructions::Stream, (CudaStream, CudaEventRaw, Instant)>;

#[derive(Clone)]
pub(crate) struct Writer {
    sender: crossbeam_channel::Sender<Message>,
    thread: ThreadId,
    streams: Arc<RwLock<StreamTable>>,
    run_begin_time: Instant,
}

enum RuntimeUptime {
    Stream(CudaEventRaw),
    Sync(Duration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimedItem {
    Inst(Instruction),
    SubgraphLoad(instructions::Track, Option<instructions::Stream>),
    SubgraphStore(instructions::Track, Option<instructions::Stream>),
}

impl From<Instruction> for TimedItem {
    fn from(value: Instruction) -> Self {
        Self::Inst(value)
    }
}

impl TimedItem {
    pub fn need_timing_stream(&self) -> Option<instructions::Stream> {
        use TimedItem::*;
        match self {
            Inst(inst) => inst.need_timing_stream(),
            SubgraphLoad(_, s) | SubgraphStore(_, s) => s.clone(),
        }
    }

    pub fn track(&self) -> instructions::Track {
        use TimedItem::*;
        match self {
            Inst(inst) => inst.track(),
            SubgraphLoad(t, _) | SubgraphStore(t, _) => t.clone(),
        }
    }
}

pub struct ItemBeginGuard {
    item: TimedItem,
    begin_uptime: RuntimeUptime,
    n_iter: Option<usize>,
}

impl Writer {
    pub fn with_thread_id(self, thread: ThreadId) -> Self {
        Self { thread, ..self }
    }

    pub fn debug_print(&self, t: &impl std::fmt::Debug) {
        println!("[{:?}] {:?}", self.run_begin_time.elapsed(), t)
    }

    fn stream_uptime(
        &self,
        guard: &RwLockReadGuard<StreamTable>,
        stream_number: instructions::Stream,
    ) -> CudaEventRaw {
        let (stream, _stream_begin, _) = guard.get(&stream_number).unwrap();
        let event = CudaEventRaw::new(stream.get_device());
        event.record(stream);
        event
    }

    fn sync_uptime(&self) -> Duration {
        let dur = Instant::now().duration_since(self.run_begin_time);
        dur
    }

    pub fn begin_item<'a>(&'a self, item: TimedItem, n_iter: Option<usize>) -> ItemBeginGuard {
        if let Some(stream_number) = item.need_timing_stream() {
            let guard = self.streams.read().unwrap();
            let begin_uptime = self.stream_uptime(&guard, stream_number);

            ItemBeginGuard {
                item,
                begin_uptime: RuntimeUptime::Stream(begin_uptime),
                n_iter,
            }
        } else {
            let begin_uptime = self.sync_uptime();

            ItemBeginGuard {
                item,
                begin_uptime: RuntimeUptime::Sync(begin_uptime),
                n_iter,
            }
        }
    }

    pub fn end_item(&self, ibg: ItemBeginGuard, function_meta: Option<FuncMeta>) {
        match ibg.begin_uptime {
            RuntimeUptime::Stream(..) => {
                let guard = self.streams.read().unwrap();
                let end_uptime = self.stream_uptime(&guard, ibg.item.need_timing_stream().unwrap());

                self.sender
                    .send(Message {
                        thread: self.thread,
                        node: MessageNode::InstructionExecuted(ItemExecution {
                            item: ibg.item,
                            start: ibg.begin_uptime,
                            end: RuntimeUptime::Stream(end_uptime),
                            function_meta,
                            thread: self.thread,
                            plot_it: ibg.n_iter.is_none_or(|n| n < 5),
                        }),
                    })
                    .expect("channel closed");
            }
            RuntimeUptime::Sync(..) => {
                let end_uptime = self.sync_uptime();

                self.sender
                    .send(Message {
                        thread: self.thread,
                        node: MessageNode::InstructionExecuted(ItemExecution {
                            item: ibg.item,
                            start: ibg.begin_uptime,
                            end: RuntimeUptime::Sync(end_uptime),
                            function_meta,
                            thread: self.thread,
                            plot_it: ibg.n_iter.is_none_or(|n| n < 5),
                        }),
                    })
                    .expect("channel closed");
            }
        }
    }

    pub fn new_stream(
        &self,
        stream_number: instructions::Stream,
        stream: CudaStream,
        begin_event: CudaEventRaw,
    ) {
        self.streams
            .write()
            .unwrap()
            .insert(stream_number, (stream, begin_event, Instant::now()));
    }
}

pub(crate) struct Logger {
    executed_instructions: Vec<ItemExecution<RuntimeUptime>>,
}

impl Logger {
    pub(crate) fn new() -> Self {
        Self {
            executed_instructions: Vec::new(),
        }
    }

    pub(crate) fn spawn(self) -> LoggerHandle {
        let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
        let handle = std::thread::spawn(move || {
            let mut logger = self;
            loop {
                use MessageNode::*;
                match receiver.recv().expect("channel closed").node {
                    Terminate => break,
                    InstructionExecuted(ie) => logger.executed_instructions.push(ie),
                }
            }
            logger
        });
        LoggerHandle {
            writer: Writer {
                sender: sender,
                thread: ThreadId::from(0),
                streams: Arc::new(RwLock::new(BTreeMap::new())),
                run_begin_time: Instant::now(),
            },
            join_handle: handle,
        }
    }
}

pub(crate) struct LoggerHandle {
    join_handle: std::thread::JoinHandle<Logger>,
    writer: Writer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// The execution time recorded for each instruction.
pub struct Log {
    executed_instructions: Vec<ItemExecution<Uptime>>,
    total_time: std::time::Duration,
}

impl LoggerHandle {
    pub(crate) fn writer(&self) -> Writer {
        self.writer.clone()
    }

    pub fn join(self, begin: std::time::Instant) -> Log {
        let total_time = begin.elapsed();

        self.writer
            .sender
            .send(Message {
                thread: ThreadId::from(0),
                node: MessageNode::Terminate,
            })
            .expect("channel closed");

        let logger = self.join_handle.join().expect("thread panicked");
        Log {
            executed_instructions: logger
                .executed_instructions
                .into_iter()
                .map(|ie| {
                    let stream_number = ie.item.need_timing_stream();
                    ItemExecution {
                        item: ie.item,
                        start: ie.start.realize(&self.writer, stream_number),
                        end: ie.end.realize(&self.writer, stream_number),
                        function_meta: ie.function_meta,
                        thread: ie.thread,
                        plot_it: ie.plot_it,
                    }
                })
                .collect(),
            total_time,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uptime {
    nanos: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemExecution<U> {
    item: TimedItem,
    start: U,
    end: U,
    function_meta: Option<FuncMeta>,
    thread: ThreadId,
    plot_it: bool,
}

impl RuntimeUptime {
    pub fn realize(self, writer: &Writer, stream_number: Option<instructions::Stream>) -> Uptime {
        use RuntimeUptime::*;
        match self {
            Stream(event) => {
                let guard = writer.streams.read().unwrap();
                let (_, stream_begin, stream_begin_instant) =
                    &guard.get(&stream_number.unwrap()).unwrap();
                let t = stream_begin.elapsed(&event);
                Uptime {
                    nanos: (t * 1e6) as u128
                        + stream_begin_instant
                            .duration_since(writer.run_begin_time)
                            .as_nanos(),
                }
            }
            Sync(dur) => Uptime {
                nanos: dur.as_nanos(),
            },
        }
    }
}

enum MessageNode {
    InstructionExecuted(ItemExecution<RuntimeUptime>),
    Terminate,
}

#[allow(dead_code)]
struct Message {
    thread: ThreadId,
    node: MessageNode,
}

impl ItemExecution<Uptime> {
    fn duration_nanos(&self) -> u128 {
        self.end.nanos - self.start.nanos
    }
}

impl Log {
    /// Build waterfall graph from this log.
    pub fn waterfall(&self) -> waterfall::Builder {
        if self.is_empty() {
            panic!("no log recorded");
        }

        let mut builder = waterfall::Builder::new("Runtime Statistics".to_string());

        let mut track_id_acc: usize = 0;
        let track_ids = self
            .executed_instructions
            .iter()
            .fold(Bijection::new(), |mut acc, ie| {
                let track = ie.item.track();
                if acc.get_forward(&track).is_none() {
                    acc.insert(track, track_id_acc);
                    track_id_acc += 1;
                }
                acc
            });

        let mut track_ids_list = track_ids.iter().collect::<Vec<_>>();
        track_ids_list.sort_by_key(|(_, x)| **x);

        track_ids_list
            .into_iter()
            .zip(waterfall::color_loop())
            .for_each(|((track, _), color)| {
                builder.add_category(waterfall::Category::new(
                    format!("{:?}", track),
                    color.to_string(),
                ));
            });

        self.executed_instructions
            .iter()
            .enumerate()
            .filter(|(_, ie)| ie.plot_it)
            .for_each(|(idx, ie)| {
                let tid = track_ids.get_forward(&ie.item.track()).unwrap();

                use instructions::InstructionNode::*;

                let label = match &ie.item {
                    TimedItem::Inst(inst) => match inst.node() {
                        Wait { .. } | Record { .. } | Fork { .. } => None,
                        _ => Some(instructions::instruction_label_by_meta(
                            inst.node(),
                            ie.function_meta.as_ref(),
                        )),
                    },
                    TimedItem::SubgraphLoad(..) => Some("Load".to_string()),
                    TimedItem::SubgraphStore(..) => Some("Store".to_string()),
                };

                if let Some(label) = label {
                    builder.add_entry(
                        *tid,
                        waterfall::Entry {
                            id: format!("{}", idx),
                            label,
                            start: ie.start.nanos,
                            end: ie.end.nanos,
                            success: true,
                            worker: format!("Thread {}", usize::from(ie.thread)),
                            content: format!("{:?}", &ie.item),
                        },
                    );
                }
            });

        builder
    }

    pub fn total_time(&self) -> std::time::Duration {
        self.total_time
    }

    pub fn is_empty(&self) -> bool {
        self.executed_instructions.is_empty()
    }
}

pub mod statistics;
