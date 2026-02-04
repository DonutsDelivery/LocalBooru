"""
Server-Sent Events streaming for library updates
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ...services.events import library_events

router = APIRouter()


@router.get("/events")
async def library_events_stream():
    """Server-Sent Events stream for real-time library updates"""
    async def event_generator():
        # Send initial connection message
        yield "data: {\"type\": \"connected\"}\n\n"
        # Stream events
        async for event in library_events.subscribe():
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
