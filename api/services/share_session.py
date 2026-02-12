"""
Share Stream session management.

Allows a host to share a video stream with viewers via a shareable link.
Viewers see synced playback following the host's play/pause/seek state.

Architecture:
    Host creates session → gets token → viewers connect with token → SSE sync
"""

import atexit
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Registry of active share sessions
_active_sessions: Dict[str, 'ShareSession'] = {}

# Session expiry: 4 hours
SESSION_EXPIRY_SECONDS = 4 * 60 * 60


@dataclass
class HostState:
    """Current playback state of the host."""
    playing: bool = False
    position: float = 0.0
    speed: float = 1.0
    updated_at: float = field(default_factory=time.time)


@dataclass
class ShareSession:
    """A share stream session."""
    token: str
    image_id: int
    video_path: str
    original_filename: str
    host_state: HostState = field(default_factory=HostState)
    transcode_stream_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + SESSION_EXPIRY_SECONDS

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


def create_session(image_id: int, video_path: str, original_filename: str) -> ShareSession:
    """Create a new share session."""
    # Clean up expired sessions first
    _cleanup_expired()

    token = secrets.token_hex(6)  # 12-char hex token
    session = ShareSession(
        token=token,
        image_id=image_id,
        video_path=video_path,
        original_filename=original_filename,
    )
    _active_sessions[token] = session
    logger.info(f"[Share] Created session {token} for image {image_id}")
    return session


def get_session(token: str) -> Optional[ShareSession]:
    """Get a session by token. Returns None if not found or expired."""
    session = _active_sessions.get(token)
    if session and session.is_expired:
        destroy_session(token)
        return None
    return session


def update_host_state(token: str, playing: bool = None, position: float = None, speed: float = None) -> Optional[HostState]:
    """Update host playback state."""
    session = get_session(token)
    if not session:
        return None

    if playing is not None:
        session.host_state.playing = playing
    if position is not None:
        session.host_state.position = position
    if speed is not None:
        session.host_state.speed = speed
    session.host_state.updated_at = time.time()

    return session.host_state


def destroy_session(token: str):
    """Destroy a share session."""
    session = _active_sessions.pop(token, None)
    if session:
        logger.info(f"[Share] Destroyed session {token}")

    # Clean up per-session event broadcaster
    from .events import share_sync_events
    share_sync_events.pop(token, None)


def _cleanup_expired():
    """Remove expired sessions."""
    expired = [t for t, s in _active_sessions.items() if s.is_expired]
    for token in expired:
        destroy_session(token)


def _cleanup_on_exit():
    """Clean up all sessions on process exit."""
    logger.info("[Share] Cleaning up all sessions on exit...")
    for token in list(_active_sessions.keys()):
        destroy_session(token)


atexit.register(_cleanup_on_exit)
