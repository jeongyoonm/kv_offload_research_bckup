use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

/// A slot belongs to one model call and never spans an agent's tool wait.
pub(crate) struct RequestGate {
    semaphore: Option<Arc<Semaphore>>,
    pub(crate) waiting: AtomicUsize,
    pub(crate) in_flight: AtomicUsize,
    pub(crate) peak: AtomicUsize,
}

pub(crate) struct RequestPermit {
    gate: Arc<RequestGate>,
    _permit: Option<OwnedSemaphorePermit>,
}

impl RequestGate {
    pub(crate) fn new(limit: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            semaphore: limit.map(|n| Arc::new(Semaphore::new(n))),
            waiting: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }

    pub(crate) async fn acquire(self: &Arc<Self>, end: Option<Instant>) -> Option<RequestPermit> {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let permit = tokio::select! {
            biased;
            _ = stop_at(end) => {
                self.waiting.fetch_sub(1, Ordering::Relaxed);
                return None;
            }
            permit = async {
                match &self.semaphore {
                    Some(semaphore) => semaphore.clone().acquire_owned().await.ok(),
                    None => None,
                }
            } => permit,
        };
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(active, Ordering::Relaxed);
        Some(RequestPermit {
            gate: self.clone(),
            _permit: permit,
        })
    }
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.gate.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(crate) async fn stop_at(end: Option<Instant>) {
    match end {
        Some(end) => tokio::time::sleep_until(end).await,
        None => pending::<()>().await,
    }
}

pub(crate) async fn wait_or_stop(delay: Duration, end: Option<Instant>) -> bool {
    tokio::select! {
        biased;
        _ = stop_at(end) => false,
        _ = tokio::time::sleep(delay) => true,
    }
}
