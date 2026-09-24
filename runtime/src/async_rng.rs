use crossbeam_channel::{bounded, Receiver};
use rand_core::RngCore;
use std::thread;

/// An asynchronous random number generator.
/// Generates random numbers on a dedicated thread and lets multiple consumers receive them through `subscribe`.
#[derive(Clone)]
pub struct AsyncRng {
    rx: Receiver<u32>,
}

impl AsyncRng {
    /// Creates a new asynchronous random number generator.
    ///
    /// # Returns
    /// * `AsyncRng` - The asynchronous random number generator.
    pub fn new(cap: usize, mut rng: impl RngCore + Send + 'static) -> Self {
        let (tx, rx) = bounded(cap);

        // Start the producer thread.
        thread::spawn(move || loop {
            let random_u32 = rng.next_u32();
            if let Err(_) = tx.send(random_u32) {
                break;
            };
        });

        Self { rx }
    }

    /// Tries to receive a random number without blocking.
    ///
    /// # Returns
    /// * `Option<u32>` - `Some` if a random number is available, or `None` otherwise.
    pub fn try_recv(&self) -> Option<u32> {
        self.rx.try_recv().ok()
    }

    /// Blocks until a random number is received.
    ///
    /// # Returns
    /// * `Option<u32>` - `Some` if a random number is received, or `None` if the channel is closed.
    pub fn recv(&self) -> Option<u32> {
        self.rx.recv().ok()
    }
}

impl rand_core::RngCore for AsyncRng {
    fn next_u32(&mut self) -> u32 {
        self.recv().unwrap()
    }

    fn next_u64(&mut self) -> u64 {
        (self.recv().unwrap() as u64) | ((self.recv().unwrap() as u64) << 32)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand_core::OsRng;

    use super::*;
    use std::time::Duration;

    #[test]
    fn test_async_rng() {
        let rng1 = AsyncRng::new(10, OsRng::default());
        let rng2 = rng1.clone();

        // Test that both consumers can receive random numbers.
        let v1 = rng1.recv().unwrap();
        let v2 = rng2.recv().unwrap();

        assert_ne!(v1, 0);
        assert_ne!(v2, 0);

        // Test cloned handles across threads.
        let rng3 = rng1.clone();

        thread::sleep(Duration::from_millis(100));
        let v3 = rng3.recv().unwrap();
        assert_ne!(v3, 0);
    }

    #[test]
    fn test_non_blocking() {
        let rng = AsyncRng::new(10, OsRng::default());

        // Test non-blocking receive.
        match rng.try_recv() {
            Some(v) => assert_ne!(v, 0),
            _ => (), // This is expected if no random number has been generated yet.
        }
    }

    #[test]
    fn test_multi_thread() {
        let rng = AsyncRng::new(10, OsRng::default());
        let rng2 = rng.clone();

        let handle = thread::spawn(move || {
            let val = rng2.recv().unwrap();
            assert_ne!(val, 0);
        });

        let val = rng.recv().unwrap();
        assert_ne!(val, 0);

        handle.join().unwrap();
    }
}
