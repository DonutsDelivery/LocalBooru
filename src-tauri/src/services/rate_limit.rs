use std::time::Instant;

use dashmap::DashMap;

use crate::server::error::AppError;

/// In-memory rate limiter using a sliding window per key (typically IP address).
///
/// Expired entries are cleaned up lazily on each `check_rate_limit` call for the
/// given key, plus a periodic full sweep when the map grows large.
#[derive(Debug)]
pub struct RateLimiter {
    /// Maps a key (e.g. IP address) to a list of attempt timestamps.
    attempts: DashMap<String, Vec<Instant>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            attempts: DashMap::new(),
        }
    }

    /// Check whether `key` has exceeded `max_attempts` within the last
    /// `window_secs` seconds.
    ///
    /// Returns `Ok(())` if the request is allowed, or a 429 `AppError` if the
    /// rate limit has been exceeded. Each call to this method records a new
    /// attempt for the key.
    pub fn check_rate_limit(
        &self,
        key: &str,
        max_attempts: u32,
        window_secs: u64,
    ) -> Result<(), AppError> {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(window_secs);

        let mut entry = self.attempts.entry(key.to_string()).or_default();
        let timestamps = entry.value_mut();

        // Evict expired attempts outside the window
        timestamps.retain(|t| now.duration_since(*t) < window);

        if timestamps.len() >= max_attempts as usize {
            let oldest = timestamps.first().copied();
            let retry_after = oldest
                .map(|t| {
                    let elapsed = now.duration_since(t);
                    if elapsed < window {
                        (window - elapsed).as_secs() + 1
                    } else {
                        0
                    }
                })
                .unwrap_or(window_secs);

            return Err(AppError::TooManyRequests(format!(
                "Rate limit exceeded. Try again in {} seconds",
                retry_after
            )));
        }

        // Record this attempt
        timestamps.push(now);

        // Periodic cleanup: if the global map has grown large, spawn a
        // lightweight sweep (non-blocking for the caller).
        if self.attempts.len() > 1000 {
            self.sweep_expired(window_secs);
        }

        Ok(())
    }

    /// Record a failed attempt without checking the limit.
    /// Useful when you want to only count failures (e.g. wrong password)
    /// rather than every login attempt.
    pub fn record_attempt(&self, key: &str) {
        let now = Instant::now();
        self.attempts
            .entry(key.to_string())
            .or_default()
            .push(now);
    }

    /// Remove all expired entries from the map to free memory.
    fn sweep_expired(&self, window_secs: u64) {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(window_secs);

        // Collect keys to remove (can't remove while iterating DashMap)
        let stale_keys: Vec<String> = self
            .attempts
            .iter()
            .filter_map(|entry| {
                let dominated = entry.value().iter().all(|t| now.duration_since(*t) >= window);
                if dominated {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();

        for key in stale_keys {
            self.attempts.remove(&key);
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn allows_under_limit() {
        let limiter = RateLimiter::new();
        for _ in 0..5 {
            assert!(limiter.check_rate_limit("test-ip", 5, 60).is_ok());
        }
    }

    #[test]
    fn blocks_over_limit() {
        let limiter = RateLimiter::new();
        for _ in 0..5 {
            limiter.check_rate_limit("test-ip", 5, 60).unwrap();
        }
        assert!(limiter.check_rate_limit("test-ip", 5, 60).is_err());
    }

    #[test]
    fn separate_keys_independent() {
        let limiter = RateLimiter::new();
        for _ in 0..5 {
            limiter.check_rate_limit("ip-a", 5, 60).unwrap();
        }
        // Different key should still be allowed
        assert!(limiter.check_rate_limit("ip-b", 5, 60).is_ok());
    }

    #[test]
    fn expires_after_window() {
        let limiter = RateLimiter::new();
        // Use a 1-second window
        for _ in 0..3 {
            limiter.check_rate_limit("test-ip", 3, 1).unwrap();
        }
        assert!(limiter.check_rate_limit("test-ip", 3, 1).is_err());
        // Wait for the window to expire
        sleep(Duration::from_secs(2));
        assert!(limiter.check_rate_limit("test-ip", 3, 1).is_ok());
    }
}
