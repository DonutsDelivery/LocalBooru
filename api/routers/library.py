"""
Library router - library-wide operations and statistics
"""
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import (
    Image, Tag, ImageFile, WatchDirectory, TaskQueue,
    TaskStatus, TaskType, Rating
)

router = APIRouter()


@router.get("/stats")
async def library_stats(db: AsyncSession = Depends(get_db)):
    """Get library statistics"""
    # Total images
    total_images = await db.execute(select(func.count(Image.id)))
    total = total_images.scalar()

    # Favorites count
    favorites_count = await db.execute(
        select(func.count(Image.id)).where(Image.is_favorite == True)
    )
    favorites = favorites_count.scalar()

    # Total tags
    total_tags = await db.execute(select(func.count(Tag.id)))
    tags = total_tags.scalar()

    # Watch directories
    total_dirs = await db.execute(select(func.count(WatchDirectory.id)))
    directories = total_dirs.scalar()

    # Missing files
    missing_files = await db.execute(
        select(func.count(ImageFile.id)).where(ImageFile.file_exists == False)
    )
    missing = missing_files.scalar()

    # By rating
    rating_query = (
        select(Image.rating, func.count(Image.id))
        .group_by(Image.rating)
    )
    rating_result = await db.execute(rating_query)
    by_rating = {row[0].value: row[1] for row in rating_result}

    # Task queue status
    pending_tasks = await db.execute(
        select(func.count(TaskQueue.id)).where(TaskQueue.status == TaskStatus.pending)
    )
    processing_tasks = await db.execute(
        select(func.count(TaskQueue.id)).where(TaskQueue.status == TaskStatus.processing)
    )

    return {
        "total_images": total,
        "favorites": favorites,
        "total_tags": tags,
        "watch_directories": directories,
        "missing_files": missing,
        "by_rating": by_rating,
        "queue": {
            "pending": pending_tasks.scalar(),
            "processing": processing_tasks.scalar()
        }
    }


@router.get("/queue")
async def queue_status(db: AsyncSession = Depends(get_db)):
    """Get background task queue status"""
    # Count by status
    status_query = (
        select(TaskQueue.status, func.count(TaskQueue.id))
        .group_by(TaskQueue.status)
    )
    status_result = await db.execute(status_query)
    by_status = {row[0].value: row[1] for row in status_result}

    # Count by type
    type_query = (
        select(TaskQueue.task_type, func.count(TaskQueue.id))
        .where(TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing]))
        .group_by(TaskQueue.task_type)
    )
    type_result = await db.execute(type_query)
    by_type = {row[0].value: row[1] for row in type_result}

    # Recent failed tasks
    failed_query = (
        select(TaskQueue)
        .where(TaskQueue.status == TaskStatus.failed)
        .order_by(TaskQueue.completed_at.desc())
        .limit(10)
    )
    failed_result = await db.execute(failed_query)
    failed_tasks = [
        {
            "id": t.id,
            "type": t.task_type.value,
            "error": t.error_message,
            "attempts": t.attempts
        }
        for t in failed_result.scalars().all()
    ]

    return {
        "by_status": by_status,
        "pending_by_type": by_type,
        "recent_failures": failed_tasks
    }


@router.post("/queue/retry-failed")
async def retry_failed_tasks(db: AsyncSession = Depends(get_db)):
    """Retry all failed tasks"""
    from sqlalchemy import update

    result = await db.execute(
        update(TaskQueue)
        .where(TaskQueue.status == TaskStatus.failed)
        .values(status=TaskStatus.pending, attempts=0, error_message=None)
    )
    await db.commit()

    return {"retried": result.rowcount}


@router.delete("/queue/pending")
async def clear_pending_tasks(db: AsyncSession = Depends(get_db)):
    """Clear all pending tasks from the queue"""
    from sqlalchemy import delete

    result = await db.execute(
        delete(TaskQueue).where(TaskQueue.status == TaskStatus.pending)
    )
    await db.commit()

    return {"cleared": result.rowcount}


@router.post("/verify-files")
async def verify_all_files(db: AsyncSession = Depends(get_db)):
    """Queue a file verification task"""
    from ..services.task_queue import enqueue_task

    task = await enqueue_task(
        TaskType.verify_files,
        {},
        priority=0,  # Low priority
        db=db
    )

    return {"message": "File verification queued", "task_id": task.id}


@router.get("/untagged")
async def list_untagged(db: AsyncSession = Depends(get_db)):
    """Get count of images without tags"""
    from ..models import image_tags

    # Images with no tags
    subq = select(image_tags.c.image_id).distinct()
    untagged_query = select(func.count(Image.id)).where(Image.id.not_in(subq))
    result = await db.execute(untagged_query)
    count = result.scalar()

    return {"untagged_count": count}


@router.post("/detect-ages")
async def detect_ages_on_realistic(db: AsyncSession = Depends(get_db)):
    """Queue age detection for realistic images that don't have it yet"""
    import json
    from ..models import image_tags
    from ..services.task_queue import enqueue_task
    from ..services.age_detector import REALISTIC_TAGS

    # Get images with realistic tags that don't have age detection
    realistic_tag_query = select(Tag.id).where(Tag.name.in_(REALISTIC_TAGS))
    realistic_tag_result = await db.execute(realistic_tag_query)
    realistic_tag_ids = [r[0] for r in realistic_tag_result.all()]

    if not realistic_tag_ids:
        return {"queued": 0, "message": "No realistic tags found in database"}

    # Find images with those tags that have no age detection (num_faces IS NULL)
    images_query = (
        select(Image)
        .join(image_tags, Image.id == image_tags.c.image_id)
        .where(
            image_tags.c.tag_id.in_(realistic_tag_ids),
            Image.num_faces.is_(None)
        )
        .distinct()
    )
    result = await db.execute(images_query)
    images = result.scalars().all()

    # Get IDs of images that already have pending age_detect tasks
    pending_query = (
        select(TaskQueue.payload)
        .where(
            TaskQueue.task_type == TaskType.age_detect,
            TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing])
        )
    )
    pending_result = await db.execute(pending_query)
    already_queued = set()
    for row in pending_result.scalars().all():
        try:
            payload = json.loads(row)
            already_queued.add(payload.get('image_id'))
        except:
            pass

    queued = 0
    skipped = 0
    for image in images:
        if image.id in already_queued:
            skipped += 1
            continue

        # Get file path
        file_query = select(ImageFile).where(
            ImageFile.image_id == image.id,
            ImageFile.file_exists == True
        ).limit(1)
        file_result = await db.execute(file_query)
        image_file = file_result.scalar_one_or_none()

        if image_file:
            await enqueue_task(
                TaskType.age_detect,
                {
                    'image_id': image.id,
                    'image_path': image_file.original_path
                },
                priority=0,
                db=db
            )
            queued += 1

    return {"queued": queued, "skipped_already_queued": skipped}


@router.post("/tag-untagged")
async def tag_untagged_images(db: AsyncSession = Depends(get_db)):
    """Queue tagging for all untagged images"""
    import json
    from ..models import image_tags
    from ..services.task_queue import enqueue_task

    # Get untagged images
    subq = select(image_tags.c.image_id).distinct()
    untagged_query = (
        select(Image)
        .where(Image.id.not_in(subq))
    )
    result = await db.execute(untagged_query)
    images = result.scalars().all()

    # Get IDs of images that already have pending tag tasks
    pending_query = (
        select(TaskQueue.payload)
        .where(
            TaskQueue.task_type == TaskType.tag,
            TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing])
        )
    )
    pending_result = await db.execute(pending_query)
    already_queued = set()
    for row in pending_result.scalars().all():
        try:
            payload = json.loads(row)
            already_queued.add(payload.get('image_id'))
        except:
            pass

    queued = 0
    skipped = 0
    for image in images:
        # Skip if already queued
        if image.id in already_queued:
            skipped += 1
            continue

        # Get file path
        file_query = select(ImageFile).where(
            ImageFile.image_id == image.id,
            ImageFile.file_exists == True
        ).limit(1)
        file_result = await db.execute(file_query)
        image_file = file_result.scalar_one_or_none()

        if image_file:
            await enqueue_task(
                TaskType.tag,
                {
                    'image_id': image.id,
                    'image_path': image_file.original_path
                },
                priority=0,
                db=db
            )
            queued += 1

    return {"queued": queued, "skipped_already_queued": skipped}


# Endpoints for directory watcher integration

from pydantic import BaseModel


class ImportFileRequest(BaseModel):
    file_path: str
    watch_directory_id: int = None
    auto_tag: bool = True


class FileMissingRequest(BaseModel):
    file_path: str


@router.post("/import-file")
async def import_file(request: ImportFileRequest, db: AsyncSession = Depends(get_db)):
    """Import a single file (called by directory watcher)"""
    from ..services.importer import import_image

    result = await import_image(
        request.file_path,
        db,
        watch_directory_id=request.watch_directory_id,
        auto_tag=request.auto_tag
    )
    return result


@router.post("/file-missing")
async def mark_file_missing(request: FileMissingRequest, db: AsyncSession = Depends(get_db)):
    """Mark a file as missing (called by directory watcher)"""
    from ..services.file_tracker import mark_file_missing

    await mark_file_missing(request.file_path, db)
    return {"marked_missing": True}
