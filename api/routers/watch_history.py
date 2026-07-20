"""
Watch history endpoints - track video playback position for resume functionality.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..models import WatchHistory, Image, ImageFile

router = APIRouter()


class PlaybackPositionUpdate(BaseModel):
    position: float
    duration: float


@router.post("/{image_id}")
async def save_playback_position(image_id: int, body: PlaybackPositionUpdate, db: AsyncSession = Depends(get_db)):
    """Upsert playback position for a video."""
    # Check if entry exists
    result = await db.execute(
        select(WatchHistory).where(WatchHistory.image_id == image_id)
    )
    entry = result.scalar_one_or_none()

    completed = body.duration > 0 and (body.position / body.duration) >= 0.9

    if entry:
        entry.playback_position = body.position
        entry.duration = body.duration
        entry.completed = completed
    else:
        entry = WatchHistory(
            image_id=image_id,
            playback_position=body.position,
            duration=body.duration,
            completed=completed,
        )
        db.add(entry)

    await db.commit()
    return {"success": True}


@router.get("/continue-watching")
async def get_continue_watching(db: AsyncSession = Depends(get_db)):
    """Get list of videos with partial watch progress for 'Continue Watching' row."""
    result = await db.execute(
        select(WatchHistory, Image)
        .join(Image, WatchHistory.image_id == Image.id)
        .where(WatchHistory.completed == False)
        .where(WatchHistory.playback_position > 10)
        .order_by(WatchHistory.last_watched.desc())
        .limit(20)
    )
    rows = result.all()

    items = []
    for watch, image in rows:
        # Get file path for the image
        file_result = await db.execute(
            select(ImageFile.original_path).where(ImageFile.image_id == image.id).limit(1)
        )
        file_path = file_result.scalar_one_or_none()

        items.append({
            "id": image.id,
            "filename": image.filename,
            "original_filename": image.original_filename,
            "thumbnail_url": image.thumbnail_url,
            "url": image.url,
            "duration": image.duration,
            "file_path": file_path,
            "playback_position": watch.playback_position,
            "watch_duration": watch.duration,
            "progress": watch.playback_position / watch.duration if watch.duration > 0 else 0,
            "last_watched": watch.last_watched.isoformat() if watch.last_watched else None,
        })

    return {"items": items}


@router.get("/{image_id}")
async def get_playback_position(image_id: int, db: AsyncSession = Depends(get_db)):
    """Get playback position for a specific video."""
    result = await db.execute(
        select(WatchHistory).where(WatchHistory.image_id == image_id)
    )
    entry = result.scalar_one_or_none()

    if not entry:
        return {"position": 0, "duration": 0, "completed": False}

    return {
        "position": entry.playback_position,
        "duration": entry.duration,
        "completed": entry.completed,
    }


@router.delete("/{image_id}")
async def delete_watch_history(image_id: int, db: AsyncSession = Depends(get_db)):
    """Remove a video from watch history."""
    await db.execute(
        delete(WatchHistory).where(WatchHistory.image_id == image_id)
    )
    await db.commit()
    return {"success": True}


@router.delete("")
async def clear_all_watch_history(db: AsyncSession = Depends(get_db)):
    """Clear all watch history."""
    await db.execute(delete(WatchHistory))
    await db.commit()
    return {"success": True}
