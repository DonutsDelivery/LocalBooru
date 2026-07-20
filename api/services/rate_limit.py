"""
Rate limiting service for protecting endpoints from brute-force attacks.

Uses in-memory storage with automatic cleanup of expired entries.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List
import threading


@dataclass
class RateLimitEntry:
    """Tracks failed attempts for a single IP address."""
    timestamps: List[datetime] = field(default_factory=list)


class RateLimiter:
    """
    Simple in-memory rate limiter for tracking failed login attempts by IP.

    Thread-safe implementation using a lock for concurrent access.
    """

    def __init__(self, max_attempts: int = 5, window_minutes: int = 15):
        """
        Initialize the rate limiter.

        Args:
            max_attempts: Maximum failed attempts allowed within the window
            window_minutes: Time window in minutes for tracking attempts
        """
        self.max_attempts = max_attempts
        self.window_minutes = window_minutes
        self._attempts: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()

    def _cleanup_expired(self, cutoff: datetime) -> None:
        """Remove entries older than the cutoff time."""
        expired_ips = []
        for ip, entry in self._attempts.items():
            # Filter out old timestamps
            entry.timestamps = [ts for ts in entry.timestamps if ts > cutoff]
            # Mark for removal if no recent attempts
            if not entry.timestamps:
                expired_ips.append(ip)

        # Remove empty entries
        for ip in expired_ips:
            del self._attempts[ip]

    def is_rate_limited(self, ip: str) -> tuple[bool, int]:
        """
        Check if an IP is currently rate limited.

        Args:
            ip: The client IP address

        Returns:
            Tuple of (is_limited, seconds_until_reset)
            - is_limited: True if the IP has exceeded the rate limit
            - seconds_until_reset: Seconds until the oldest attempt expires (for Retry-After header)
        """
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)

        with self._lock:
            # Clean up expired entries on each check
            self._cleanup_expired(cutoff)

            entry = self._attempts.get(ip)
            if not entry or not entry.timestamps:
                return False, 0

            if len(entry.timestamps) >= self.max_attempts:
                # Calculate when the oldest attempt will expire
                oldest = min(entry.timestamps)
                expires_at = oldest + timedelta(minutes=self.window_minutes)
                seconds_remaining = int((expires_at - datetime.utcnow()).total_seconds())
                return True, max(1, seconds_remaining)

            return False, 0

    def record_failed_attempt(self, ip: str) -> None:
        """
        Record a failed login attempt for an IP address.

        Args:
            ip: The client IP address
        """
        with self._lock:
            if ip not in self._attempts:
                self._attempts[ip] = RateLimitEntry()

            self._attempts[ip].timestamps.append(datetime.utcnow())

    def clear_attempts(self, ip: str) -> None:
        """
        Clear all failed attempts for an IP address (called on successful login).

        Args:
            ip: The client IP address
        """
        with self._lock:
            if ip in self._attempts:
                del self._attempts[ip]


# Global rate limiter instance for login attempts
# 5 failed attempts per 15 minutes
login_rate_limiter = RateLimiter(max_attempts=5, window_minutes=15)
