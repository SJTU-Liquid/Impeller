use crate::{
    args::{RuntimeType, VariableId},
    debug::TimedItem,
    instructions::{self, Lcs},
    runtime::{transfer::Transfer, RuntimeInfo, SubGraphInfo},
};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::{
    sync::{Arc, Mutex},
    thread,
};
use zkpoly_common::devices::DeviceType;

struct ProducerConsumerPipeline {
    num_producers: usize,
    num_consumers: usize,
    stage_states: Vec<Mutex<StageState>>,
    producer_queues: ParticipantQueues,
    consumer_queues: ParticipantQueues,
}

struct ParticipantQueues {
    tx: Vec<Sender<usize>>,
    rx: Vec<Receiver<usize>>,
}

impl ParticipantQueues {
    fn new(participants: usize, capacity: usize) -> Self {
        assert!(participants > 0, "pipeline needs participants");
        let mut tx = Vec::with_capacity(participants);
        let mut rx = Vec::with_capacity(participants);
        for _ in 0..participants {
            let (sender, receiver) = bounded(capacity);
            tx.push(sender);
            rx.push(receiver);
        }
        Self { tx, rx }
    }

    fn broadcast(&self, stage: usize) {
        for sender in &self.tx {
            sender
                .send(stage)
                .expect("pipeline queue unexpectedly closed");
        }
    }

    fn recv(&self, lane: usize) -> usize {
        self.rx
            .get(lane)
            .unwrap_or_else(|| panic!("invalid pipeline lane {}", lane))
            .recv()
            .expect("pipeline queue unexpectedly closed")
    }
}

struct StageState {
    producers_remaining: usize,
    consumers_remaining: usize,
}

impl ProducerConsumerPipeline {
    fn stage_state(&self, stage: usize) -> &Mutex<StageState> {
        self.stage_states
            .get(stage)
            .unwrap_or_else(|| panic!("invalid stage {}", stage))
    }

    pub fn new(num_producers: usize, num_consumers: usize, num_stages: usize) -> Self {
        assert!(
            num_producers > 0,
            "pipeline needs at least one producer lane"
        );
        assert!(
            num_consumers > 0,
            "pipeline needs at least one consumer lane"
        );
        assert!(num_stages > 0, "pipeline needs at least one stage");

        let stage_states = (0..num_stages)
            .map(|_| {
                Mutex::new(StageState {
                    producers_remaining: num_producers,
                    consumers_remaining: 0,
                })
            })
            .collect::<Vec<_>>();

        let producer_queues = ParticipantQueues::new(num_producers, num_stages);
        let consumer_queues = ParticipantQueues::new(num_consumers, num_stages);

        for stage in 0..num_stages {
            producer_queues.broadcast(stage);
        }

        Self {
            num_producers,
            num_consumers,
            stage_states,
            producer_queues,
            consumer_queues,
        }
    }

    pub fn producer_acquire(&self, lane: usize) -> usize {
        self.producer_queues.recv(lane)
    }

    pub fn producer_release(&self, stage: usize) {
        let mut state = self
            .stage_state(stage)
            .lock()
            .expect("pipeline stage poisoned");
        assert!(
            state.producers_remaining > 0,
            "producer release overflow on stage {}",
            stage
        );
        state.producers_remaining -= 1;
        if state.producers_remaining == 0 {
            state.consumers_remaining = self.num_consumers;
            drop(state);
            self.consumer_queues.broadcast(stage);
        }
    }

    pub fn consumer_acquire(&self, lane: usize) -> usize {
        self.consumer_queues.recv(lane)
    }

    pub fn consumer_release(&self, stage: usize) {
        let mut state = self
            .stage_state(stage)
            .lock()
            .expect("pipeline stage poisoned");
        assert!(
            state.consumers_remaining > 0,
            "consumer release overflow on stage {}",
            stage
        );
        state.consumers_remaining -= 1;
        if state.consumers_remaining == 0 {
            state.producers_remaining = self.num_producers;
            drop(state);
            self.producer_queues.broadcast(stage);
        }
    }

    pub fn total_stages(&self) -> usize {
        self.stage_states.len()
    }
}

impl<T: RuntimeType> RuntimeInfo<T> {
    pub fn sub_graph_for(
        &self,
        prologue: Lcs,
        body: Vec<Lcs>,
        total_iters: usize,
        to_gpu_stream: VariableId,
        from_gpu_stream: VariableId,
        device_types: Vec<DeviceType>,
        compute_stream_id: VariableId,
        _thread_id: usize,
    ) {
        let load_pipeline = Arc::new(ProducerConsumerPipeline::new(
            device_types.len(), // cpu, gpu, disk
            1,                  // executor
            body.len(),         // stages
        ));

        let store_pipeline = Arc::new(ProducerConsumerPipeline::new(
            1,                  // executor
            device_types.len(), // cpu, gpu, disk
            body.len(),         // stages
        ));

        let compute_stream = unsafe {
            (&*self.variable)[compute_stream_id]
                .as_ref()
                .unwrap()
                .unwrap_stream()
        };

        let prologue_arc = Arc::new(prologue);
        let body_arc = body.into_iter().map(Arc::new).collect::<Vec<_>>();

        let mut thread_handles = vec![];
        // spawn threads for all device types to load/store data
        for (device_id, dev) in device_types.iter().enumerate() {
            // spawn load thread
            let prologue = prologue_arc.clone();
            let body = body_arc.clone();
            let load_pipeline = load_pipeline.clone();

            let mut sub_info = self.clone();
            sub_info.main_thread = false;

            let src_device = dev.clone();

            thread_handles.push(thread::spawn(move || {
                sub_info.sub_graph_load(
                    prologue,
                    body,
                    to_gpu_stream,
                    total_iters,
                    load_pipeline,
                    src_device,
                    device_id,
                );
            }));

            // spawn store thread
            let prologue = prologue_arc.clone();
            let body = body_arc.clone();
            let store_pipeline = store_pipeline.clone();

            let mut sub_info = self.clone();
            sub_info.main_thread = false;
            // TODO: add sub_graph info for sub_info here

            let dst_device = dev.clone();

            thread_handles.push(thread::spawn(move || {
                sub_info.sub_graph_store(
                    prologue,
                    body,
                    from_gpu_stream,
                    total_iters,
                    store_pipeline,
                    dst_device,
                    device_id,
                );
            }));
        }

        // execute the body mainloop
        for iter in 0..total_iters {
            let stage = iter % body_arc.len();

            // which lcs to execute
            let (cur_lcs, steady_round) = if iter == 0 {
                (prologue_arc.clone(), 0)
            } else {
                let cur_iter_steady = iter - 1;
                (
                    body_arc[cur_iter_steady % body_arc.len()].clone(),
                    (cur_iter_steady / body_arc.len()),
                )
            };

            // load pipeline: consumer acquire
            assert_eq!(stage, load_pipeline.consumer_acquire(0));

            // store pipeline: producer acquire
            assert_eq!(stage, store_pipeline.producer_acquire(0));

            let (compute_insts, _) = &cur_lcs.compute;

            let mut sub_info = self.clone();
            sub_info.sub_graph_info = Some(SubGraphInfo {
                steady_round,
                num_stages: body_arc.len(),
                lcs: cur_lcs.clone(),
            });

            // execute the compute and pre-store
            unsafe {
                sub_info.run(compute_insts, None, None, None, _thread_id);
            }

            compute_stream.sync();

            // load pipeline: consumer release
            load_pipeline.consumer_release(stage);

            // store pipeline: producer release
            store_pipeline.producer_release(stage);
        }

        // wait for all load/store threads to complete
        for handle in thread_handles {
            handle.join().expect("sub graph thread panicked");
        }
    }

    #[allow(dangerous_implicit_autorefs)]
    fn sub_graph_load(
        &self,
        prologue: Arc<Lcs>,
        body: Vec<Arc<Lcs>>,
        to_gpu_stream: VariableId,
        total_iters: usize,
        load_pipeline: Arc<ProducerConsumerPipeline>,
        src_device: DeviceType,
        producer_id: usize,
    ) {
        let stream_guard = unsafe { (*self.variable)[to_gpu_stream].clone().unwrap() };

        let stream = stream_guard.unwrap_stream();

        let total_stages = load_pipeline.total_stages();
        assert_eq!(total_stages, body.len());
        for iter in 0..total_iters {
            let stage = iter % total_stages;
            // acquire load
            assert_eq!(load_pipeline.producer_acquire(producer_id), stage);

            // which lcs to execute
            let (cur_lcs, steady_round) = if iter == 0 {
                (prologue.clone(), 0)
            } else {
                let cur_iter_steady = iter - 1;
                (
                    body[cur_iter_steady % body.len()].clone(),
                    (cur_iter_steady / body.len()) as u64,
                )
            };

            // perform load
            unsafe {
                for load in &cur_lcs.loads {
                    let mut ibg = None;

                    // load data into buffer
                    let src = (*self.variable)[load.src_reg].as_ref().unwrap();
                    match src {
                        crate::args::Variable::ScalarArray(scalar_array) => {
                            if scalar_array.device == src_device {
                                let start_slice = load.begin_range.unwrap();
                                let delta = start_slice.len() * body.len() as u64 * steady_round;
                                let cur_slice =
                                    start_slice.rotated(delta as i64, scalar_array.len() as u64);

                                let mut allocated = self.allocate_offset(load.alloc_dst.clone());

                                let dst = allocated.unwrap_scalar_array_mut();
                                match src_device {
                                    DeviceType::CPU => {
                                        let src_slice = scalar_array
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphLoad(
                                                        instructions::Track::ToGpu,
                                                        Some(instructions::Stream::ToGpu),
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src_slice.cpu2gpu(dst, &stream);
                                    }
                                    DeviceType::GPU { .. } => {
                                        let src_slice = scalar_array
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphLoad(
                                                        instructions::Track::GpuMemory(0),
                                                        Some(instructions::Stream::GpuMemory),
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src_slice.gpu2gpu(dst, &stream);
                                    }
                                    DeviceType::Disk => {
                                        let src_slice = scalar_array
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphLoad(
                                                        instructions::Track::FromDisk,
                                                        None,
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src_slice.disk2gpu(dst);
                                    }
                                }
                            }
                        }
                        crate::args::Variable::Scalar(src_scalar) => {
                            if src_scalar.device == src_device {
                                let mut allocated = self.allocate_offset(load.alloc_dst.clone());

                                let dst = allocated.unwrap_scalar_mut();
                                match src_device {
                                    DeviceType::CPU => src_scalar.cpu2gpu(dst, &stream),
                                    DeviceType::GPU { .. } => src_scalar.gpu2gpu(dst, &stream),
                                    DeviceType::Disk => unreachable!("scalar cannot be on disk"),
                                }
                            }
                        }
                        _ => unreachable!("unsupported variable type for sub graph load"),
                    }

                    if let Some(ibg) = ibg {
                        self.debug_writer.end_item(ibg, None);
                    }
                }
            }

            stream.sync(); // ensure load complete

            // notify load complete
            load_pipeline.producer_release(stage);
        }
    }

    #[allow(dangerous_implicit_autorefs)]
    fn sub_graph_store(
        &self,
        prologue: Arc<Lcs>,
        body: Vec<Arc<Lcs>>,
        from_gpu_stream: VariableId,
        total_iters: usize,
        store_pipeline: Arc<ProducerConsumerPipeline>,
        dst_device: DeviceType,
        consumer_id: usize,
    ) {
        let stream_guard = unsafe { (*self.variable)[from_gpu_stream].clone().unwrap() };

        let stream = stream_guard.unwrap_stream();

        let total_stages = store_pipeline.total_stages();
        assert_eq!(total_stages, body.len());
        for iter in 0..total_iters {
            let stage = iter % total_stages;

            // consumer acquire
            assert_eq!(stage, store_pipeline.consumer_acquire(consumer_id));

            // which lcs to execute
            let (cur_lcs, steady_round) = if iter == 0 {
                (prologue.clone(), 0)
            } else {
                let cur_iter_steady = iter - 1;
                (
                    body[cur_iter_steady % body.len()].clone(),
                    (cur_iter_steady / body.len()) as u64,
                )
            };

            // perform store
            unsafe {
                for store in &cur_lcs.stores {
                    let mut ibg = None;

                    // load data into buffer
                    let dst = (*self.variable)[store.dst_reg].as_mut().unwrap_or_else(|| {
                        panic!("dst {:?} is not defined", store.dst_reg);
                    });

                    let src = (*self.variable)[store.src_reg].as_ref().unwrap();
                    match dst {
                        crate::args::Variable::ScalarArray(dst_poly) => {
                            if dst_poly.device == dst_device {
                                let start_slice = store.begin_range.unwrap();
                                let delta = start_slice.len() * body.len() as u64 * steady_round;
                                let cur_slice =
                                    start_slice.rotated(delta as i64, dst_poly.len() as u64);
                                let src = src.unwrap_scalar_array();
                                match dst_device {
                                    DeviceType::CPU => {
                                        let mut dst_slice = dst_poly
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphStore(
                                                        instructions::Track::FromGpu,
                                                        Some(instructions::Stream::FromGpu),
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src.gpu2cpu(&mut dst_slice, &stream);
                                    }
                                    DeviceType::GPU { .. } => {
                                        let mut dst_slice = dst_poly
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphStore(
                                                        instructions::Track::GpuMemory(0),
                                                        Some(instructions::Stream::GpuMemory),
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src.gpu2gpu(&mut dst_slice, &stream);
                                    }
                                    DeviceType::Disk => {
                                        let mut dst_slice = dst_poly
                                            .set_slice_raw(
                                                cur_slice.begin() as usize,
                                                cur_slice.len() as usize,
                                            )
                                            .unwrap();

                                        if self.debug_option.record_time {
                                            ibg = Some(
                                                self.debug_writer.begin_item(
                                                    TimedItem::SubgraphStore(
                                                        instructions::Track::ToDisk,
                                                        None,
                                                    ),
                                                    self.sub_graph_info
                                                        .as_ref()
                                                        .map(|sgi| sgi.steady_round),
                                                ),
                                            );
                                        }
                                        src.gpu2disk(&mut dst_slice);
                                    }
                                }
                            }
                        }
                        crate::args::Variable::Scalar(dst) => {
                            if dst.device == dst_device {
                                let src = src.unwrap_scalar();
                                match dst.device {
                                    DeviceType::CPU => src.gpu2cpu(dst, &stream),
                                    DeviceType::GPU { .. } => src.gpu2gpu(dst, &stream),
                                    DeviceType::Disk => unreachable!("scalar cannot be on disk"),
                                }
                            }
                        }
                        _ => unreachable!("unsupported variable type for sub graph load"),
                    }

                    if let Some(ibg) = ibg {
                        self.debug_writer.end_item(ibg, None);
                    }
                }
            }

            stream.sync(); // ensure store complete

            // notify complete
            store_pipeline.consumer_release(stage);
        }
    }
}
