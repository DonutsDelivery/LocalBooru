use std::convert::Infallible;
use std::sync::Arc;

use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::broadcast;

/// Channel capacity per broadcaster.
const CHANNEL_CAPACITY: usize = 256;

// ─── Event type constants ────────────────────────────────────────────────────

pub mod event_type {
    pub const IMAGE_ADDED: &str = "image_added";
    pub const IMAGE_UPDATED: &str = "image_updated";
    pub const IMAGE_DELETED: &str = "image_deleted";
    pub const TASK_COMPLETED: &str = "task_completed";
}

pub mod migration_event_type {
    pub const STARTED: &str = "migration_started";
    pub const PROGRESS: &str = "migration_progress";
    pub const COMPLETED: &str = "migration_completed";
    pub const ERROR: &str = "migration_error";
}

// ─── EventBroadcaster ────────────────────────────────────────────────────────

/// Simple pub/sub event broadcaster backed by `tokio::sync::broadcast`.
///
/// Subscribers receive formatted SSE messages (`data: {...}\n\n`).
/// Slow subscribers that fall behind the capacity will skip missed events.
#[derive(Clone)]
pub struct EventBroadcaster {
    tx: broadcast::Sender<String>,
}

impl EventBroadcaster {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self { tx }
    }

    /// Broadcast an event to all current subscribers.
    pub fn broadcast(&self, event_type: &str, data: Value) {
        let event = json!({
            "type": event_type,
            "data": data,
            "timestamp": Utc::now().to_rfc3339()
        });
        let message = format!("data: {}\n\n", event);
        // Ignore error if no subscribers
        let _ = self.tx.send(message);
    }

    /// Subscribe and get a receiver that yields SSE-formatted strings.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.tx.subscribe()
    }

    /// Create an SSE-compatible stream for axum.
    pub fn sse_stream(
        &self,
    ) -> impl futures_core::Stream<Item = Result<axum::response::sse::Event, Infallible>> {
        let mut rx = self.subscribe();

        async_stream::stream! {
            // Initial connection message
            let connected = json!({"type": "connected", "timestamp": Utc::now().to_rfc3339()});
            yield Ok(axum::response::sse::Event::default().data(connected.to_string()));

            loop {
                match rx.recv().await {
                    Ok(msg) => {
                        // The message is already formatted as "data: {...}\n\n"
                        // but axum SSE wraps it, so extract the JSON part
                        let json_part = msg
                            .strip_prefix("data: ")
                            .and_then(|s| s.strip_suffix("\n\n"))
                            .unwrap_or(&msg);
                        yield Ok(axum::response::sse::Event::default().data(json_part.to_string()));
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("SSE subscriber lagged by {} events", n);
                        // Continue receiving — subscriber just missed some events
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
        }
    }

    /// Number of active subscribers (approximate — includes lagged receivers).
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for EventBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Global instances ────────────────────────────────────────────────────────

/// Holds all global event broadcaster instances.
pub struct Events {
    pub library: EventBroadcaster,
    pub migration: EventBroadcaster,
}

impl Events {
    pub fn new() -> Self {
        Self {
            library: EventBroadcaster::new(),
            migration: EventBroadcaster::new(),
        }
    }
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared reference to global events.
pub type SharedEvents = Arc<Events>;

/// Create a new shared events instance.
pub fn create_events() -> SharedEvents {
    Arc::new(Events::new())
}
