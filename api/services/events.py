"""
Server-Sent Events (SSE) for real-time updates
"""
import asyncio
import json
from typing import AsyncGenerator
from datetime import datetime


class EventBroadcaster:
    """Simple event broadcaster for SSE"""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Subscribe to events"""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            self._subscribers.remove(queue)

    async def broadcast(self, event_type: str, data: dict = None):
        """Broadcast an event to all subscribers"""
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        message = f"data: {json.dumps(event)}\n\n"

        for queue in self._subscribers:
            await queue.put(message)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Global event broadcaster instance
library_events = EventBroadcaster()

# Migration event broadcaster instance
migration_events = EventBroadcaster()

# Subtitle event broadcaster instance
subtitle_events = EventBroadcaster()

# Per-session share sync event broadcasters
# Key: share session token, Value: EventBroadcaster
share_sync_events: dict[str, EventBroadcaster] = {}


def get_share_broadcaster(token: str) -> EventBroadcaster:
    """Get or create a broadcaster for a share session."""
    if token not in share_sync_events:
        share_sync_events[token] = EventBroadcaster()
    return share_sync_events[token]


# Event types
class EventType:
    IMAGE_ADDED = "image_added"
    IMAGE_UPDATED = "image_updated"
    IMAGE_DELETED = "image_deleted"
    TASK_COMPLETED = "task_completed"


class MigrationEventType:
    STARTED = "migration_started"
    PROGRESS = "migration_progress"
    COMPLETED = "migration_completed"
    ERROR = "migration_error"


class SubtitleEventType:
    STARTED = "subtitle_started"
    CUE = "subtitle_cue"
    PROGRESS = "subtitle_progress"
    COMPLETED = "subtitle_completed"
    ERROR = "subtitle_error"


# Cast event broadcaster instance
cast_events = EventBroadcaster()


class CastEventType:
    STATUS = "cast_status"
    CONNECTED = "cast_connected"
    DISCONNECTED = "cast_disconnected"
    ERROR = "cast_error"
