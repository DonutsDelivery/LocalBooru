"""
Library router - library-wide operations and statistics
"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db, directory_db_manager
from ..models import (
    Image, Tag, ImageFile, WatchDirectory, TaskQueue,
    TaskStatus, TaskType, Rating,
    DirectoryImage, DirectoryImageFile
)
from ..services.events import library_events

router = APIRouter()


@router.get("/stats")
async def library_stats(db: AsyncSession = Depends(get_db)):
    """Get library statistics"""
    total = 0
    favorites = 0
    by_rating = {}

    # Check per-directory databases first (new architecture)
    all_dir_ids = directory_db_manager.get_all_directory_ids()
    if all_dir_ids:
        for dir_id in all_dir_ids:
            if not directory_db_manager.db_exists(dir_id):
                continue
            dir_db = await directory_db_manager.get_session(dir_id)
            try:
                # Count images in this directory
                count_result = await dir_db.execute(select(func.count(DirectoryImage.id)))
                total += count_result.scalar() or 0

                # Count favorites in this directory
                fav_result = await dir_db.execute(
                    select(func.count(DirectoryImage.id)).where(DirectoryImage.is_favorite == True)
                )
                favorites += fav_result.scalar() or 0

                # Count by rating in this directory
                rating_query = (
                    select(DirectoryImage.rating, func.count(DirectoryImage.id))
                    .group_by(DirectoryImage.rating)
                )
                rating_result = await dir_db.execute(rating_query)
                for row in rating_result:
                    rating_val = row[0].value if row[0] else 'unrated'
                    by_rating[rating_val] = by_rating.get(rating_val, 0) + row[1]
            finally:
                await dir_db.close()
    else:
        # Fall back to legacy main database
        total_images = await db.execute(select(func.count(Image.id)))
        total = total_images.scalar()

        favorites_count = await db.execute(
            select(func.count(Image.id)).where(Image.is_favorite == True)
        )
        favorites = favorites_count.scalar()

        rating_query = (
            select(Image.rating, func.count(Image.id))
            .group_by(Image.rating)
        )
        rating_result = await db.execute(rating_query)
        by_rating = {row[0].value: row[1] for row in rating_result}

    # Total tags (always from main DB)
    total_tags = await db.execute(select(func.count(Tag.id)))
    tags = total_tags.scalar()

    # Watch directories
    total_dirs = await db.execute(select(func.count(WatchDirectory.id)))
    directories = total_dirs.scalar()

    # Missing files - check per-directory DBs if available
    missing = 0
    if all_dir_ids:
        from ..models import FileStatus
        for dir_id in all_dir_ids:
            if not directory_db_manager.db_exists(dir_id):
                continue
            dir_db = await directory_db_manager.get_session(dir_id)
            try:
                missing_result = await dir_db.execute(
                    select(func.count(DirectoryImageFile.id)).where(
                        DirectoryImageFile.file_status == FileStatus.missing
                    )
                )
                missing += missing_result.scalar() or 0
            finally:
                await dir_db.close()
    else:
        missing_files = await db.execute(
            select(func.count(ImageFile.id)).where(ImageFile.file_exists == False)
        )
        missing = missing_files.scalar()

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


@router.get("/queue/paused")
async def get_queue_paused():
    """Check if the task queue is paused"""
    from ..services.task_queue import task_queue
    return {"paused": task_queue.paused}


@router.post("/queue/pause")
async def pause_queue():
    """Pause the task queue"""
    from ..services.task_queue import task_queue
    task_queue.pause()
    return {"paused": True}


@router.post("/queue/resume")
async def resume_queue():
    """Resume the task queue"""
    from ..services.task_queue import task_queue
    task_queue.resume()
    return {"paused": False}


@router.delete("/queue/pending")
async def clear_pending_tasks(db: AsyncSession = Depends(get_db)):
    """Clear all pending tasks from the queue"""
    from sqlalchemy import delete

    result = await db.execute(
        delete(TaskQueue).where(TaskQueue.status == TaskStatus.pending)
    )
    await db.commit()

    return {"cleared": result.rowcount}


@router.post("/clean-missing")
async def clean_missing_files(db: AsyncSession = Depends(get_db)):
    """Remove all images with missing files from the library"""
    import json
    from pathlib import Path
    from ..database import get_data_dir

    cleaned = 0
    tasks_cleaned = 0

    # Find all ImageFile entries
    query = select(ImageFile)
    result = await db.execute(query)
    all_files = result.scalars().all()

    for image_file in all_files:
        if not Path(image_file.original_path).exists():
            # Check if image has other valid file references
            other_query = select(ImageFile).where(
                ImageFile.image_id == image_file.image_id,
                ImageFile.id != image_file.id
            )
            other_result = await db.execute(other_query)
            other_files = other_result.scalars().all()

            # Check if any other reference exists
            has_valid_ref = any(Path(f.original_path).exists() for f in other_files)

            if has_valid_ref:
                # Just delete this stale reference
                await db.delete(image_file)
            else:
                # No valid references - delete the image too
                image = await db.get(Image, image_file.image_id)
                if image:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)

            cleaned += 1

    # Also clean up stuck tasks with missing files
    from sqlalchemy import delete
    task_query = select(TaskQueue).where(
        TaskQueue.task_type.in_([TaskType.tag, TaskType.age_detect]),
        TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing, TaskStatus.failed])
    )
    task_result = await db.execute(task_query)
    tasks = task_result.scalars().all()

    for task in tasks:
        try:
            payload = json.loads(task.payload)
            file_path = payload.get('image_path') or payload.get('file_path')
            if file_path and not Path(file_path).exists():
                await db.delete(task)
                tasks_cleaned += 1
        except Exception:
            pass

    await db.commit()
    return {"cleaned": cleaned, "tasks_cleaned": tasks_cleaned}


@router.post("/clean-orphans")
async def clean_orphan_files(db: AsyncSession = Depends(get_db)):
    """Remove files from DB that have no watch_directory_id and aren't under any watched directory"""
    from pathlib import Path
    from ..database import get_data_dir

    # Get all watched directory paths
    dir_query = select(WatchDirectory)
    dir_result = await db.execute(dir_query)
    watched_dirs = dir_result.scalars().all()
    watched_paths = [d.path.rstrip('/') + '/' for d in watched_dirs]

    # Find orphaned files (no watch_directory_id)
    orphan_query = select(ImageFile).where(ImageFile.watch_directory_id == None)
    orphan_result = await db.execute(orphan_query)
    orphan_files = orphan_result.scalars().all()

    cleaned = 0
    for image_file in orphan_files:
        # Check if file path is under any watched directory
        is_watched = any(image_file.original_path.startswith(wp) for wp in watched_paths)

        if not is_watched:
            # File is orphaned - remove from DB
            image = await db.get(Image, image_file.image_id)
            if image:
                # Check if image has other file references
                other_query = select(ImageFile).where(
                    ImageFile.image_id == image_file.image_id,
                    ImageFile.id != image_file.id
                )
                other_result = await db.execute(other_query)
                has_other_refs = other_result.scalar_one_or_none() is not None

                if not has_other_refs:
                    # No other references - delete image and thumbnail
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)
                else:
                    # Just delete this file reference
                    await db.delete(image_file)
            cleaned += 1

    await db.commit()
    return {"cleaned": cleaned}


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


@router.post("/clean-video-thumbnails")
async def clean_video_thumbnails_endpoint(db: AsyncSession = Depends(get_db)):
    """Remove video thumbnail files (e.g., video.mp4.png) from the database.

    These are auto-generated thumbnails that shouldn't have been imported.
    """
    import traceback
    from ..services.file_tracker import clean_video_thumbnails

    try:
        stats = await clean_video_thumbnails()
        return stats
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@router.post("/regenerate-thumbnails")
async def regenerate_missing_thumbnails(
    directory_id: int = None,
    force_videos: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Find images without thumbnails and regenerate them.

    This is useful after server restarts during import, or if thumbnails
    were accidentally deleted.

    Args:
        directory_id: Optional - only check this directory. If not specified,
                      checks all directories.
        force_videos: If True, regenerate all video thumbnails even if they exist.
                     Useful for updating to middle-frame thumbnails.
    """
    import asyncio
    from ..services.importer import (
        generate_thumbnail_async,
        generate_video_thumbnail_async,
        is_video_file
    )
    from ..config import get_settings

    settings = get_settings()
    thumbnails_dir = Path(settings.thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    regenerated = 0
    failed = 0
    videos_updated = 0

    # Get directory IDs to check
    if directory_id:
        dir_ids = [directory_id]
    else:
        dir_ids = directory_db_manager.get_all_directory_ids()

    for dir_id in dir_ids:
        if not directory_db_manager.db_exists(dir_id):
            continue

        dir_db = await directory_db_manager.get_session(dir_id)
        try:
            # Get all images with their file paths
            query = (
                select(DirectoryImage.file_hash, DirectoryImageFile.original_path)
                .join(DirectoryImageFile, DirectoryImageFile.image_id == DirectoryImage.id)
                .where(DirectoryImageFile.file_exists == True)
            )
            try:
                result = await dir_db.execute(query)
                rows = result.all()
            except Exception as db_err:
                print(f"[Thumbnails] Skipping directory {dir_id}: {db_err}")
                continue

            for file_hash, original_path in rows:
                if not file_hash:
                    continue

                thumbnail_path = thumbnails_dir / f"{file_hash[:16]}.webp"
                is_video = is_video_file(original_path)
                thumbnail_exists = thumbnail_path.exists()

                # Decide if we need to regenerate
                needs_regen = False
                if not thumbnail_exists:
                    missing += 1
                    needs_regen = True
                elif force_videos and is_video:
                    needs_regen = True

                if not needs_regen:
                    continue

                # Check if source file exists
                if not Path(original_path).exists():
                    failed += 1
                    continue

                # Regenerate thumbnail
                try:
                    if is_video:
                        success = await generate_video_thumbnail_async(
                            original_path, str(thumbnail_path)
                        )
                    else:
                        success = await generate_thumbnail_async(
                            original_path, str(thumbnail_path)
                        )

                    if success:
                        if thumbnail_exists and force_videos:
                            videos_updated += 1
                        else:
                            regenerated += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[Thumbnails] Failed to regenerate for {original_path}: {e}")
                    failed += 1

        finally:
            await dir_db.close()

    message = f"Found {missing} missing thumbnails, regenerated {regenerated}, failed {failed}"
    if force_videos:
        message += f", updated {videos_updated} video thumbnails"

    return {
        "missing": missing,
        "regenerated": regenerated,
        "failed": failed,
        "videos_updated": videos_updated,
        "message": message
    }


@router.post("/detect-ages")
async def detect_ages_retrospective(db: AsyncSession = Depends(get_db)):
    """Queue age detection for images in directories with auto_age_detect enabled"""
    from ..services.age_detector import is_age_detection_enabled

    if not is_age_detection_enabled():
        return {"error": "Age detection is not enabled", "queued": 0}

    # Find images in directories with auto_age_detect enabled that don't have age data
    images_query = (
        select(Image)
        .join(ImageFile, ImageFile.image_id == Image.id)
        .join(WatchDirectory, WatchDirectory.id == ImageFile.watch_directory_id)
        .where(
            WatchDirectory.auto_age_detect == True,
            Image.age_detection_data.is_(None)  # No age data yet
        )
        .distinct()
    )

    result = await db.execute(images_query)
    images = result.scalars().all()

    if not images:
        return {"message": "No images need age detection (check directory settings)", "queued": 0}

    # Queue age detection tasks
    from ..services.task_queue import enqueue_task

    queued = 0
    for image in images:
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
                {"image_id": image.id, "image_path": image_file.original_path},
                priority=0,  # Low priority
                db=db
            )
            queued += 1

    return {"message": f"Queued age detection for {queued} images", "queued": queued}


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
async def tag_untagged_images(
    directory_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Queue tagging for all untagged images.

    Optionally filter by directory_id.
    """
    import json
    from ..models import image_tags, FileStatus
    from ..services.task_queue import enqueue_task

    # Debug: count total images
    total_images = await db.execute(select(func.count(Image.id)))
    total = total_images.scalar()

    # Debug: count images with tags
    tagged_subq = select(image_tags.c.image_id).distinct()
    tagged_count = await db.execute(
        select(func.count(Image.id)).where(Image.id.in_(tagged_subq))
    )
    tagged = tagged_count.scalar()

    # Get untagged images - simpler query without file_status filter first
    untagged_query = (
        select(Image)
        .where(Image.id.not_in(tagged_subq))
    )

    # Filter by directory if specified
    if directory_id is not None:
        untagged_query = (
            untagged_query
            .join(ImageFile, ImageFile.image_id == Image.id)
            .where(ImageFile.watch_directory_id == directory_id)
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
    skipped_queued = 0
    skipped_no_file = 0
    for image in images:
        # Skip if already queued
        if image.id in already_queued:
            skipped_queued += 1
            continue

        # Get file path - any existing file
        file_query = select(ImageFile).where(
            ImageFile.image_id == image.id,
            ImageFile.file_exists == True
        ).limit(1)
        file_result = await db.execute(file_query)
        image_file = file_result.scalar_one_or_none()

        if image_file:
            task = await enqueue_task(
                TaskType.tag,
                {
                    'image_id': image.id,
                    'image_path': image_file.original_path
                },
                priority=0,
                db=db,
                dedupe_key=image.id  # Prevent duplicate tasks for same image
            )
            if task:
                queued += 1
            else:
                skipped_queued += 1  # Was duplicate
        else:
            skipped_no_file += 1

    return {
        "queued": queued,
        "skipped_already_queued": skipped_queued,
        "skipped_no_file": skipped_no_file,
        "debug": {
            "total_images": total,
            "tagged_images": tagged,
            "untagged_found": len(images)
        }
    }


@router.post("/clear-duplicate-tasks")
async def clear_duplicate_tasks_endpoint(db: AsyncSession = Depends(get_db)):
    """Remove duplicate pending tasks from the queue.

    Keeps the oldest task for each image_id and marks duplicates as failed.
    """
    from ..services.task_queue import clear_duplicate_tasks
    removed = await clear_duplicate_tasks(db)
    return {"duplicates_removed": removed}


@router.delete("/clear-pending-tasks")
async def clear_pending_tasks(
    task_type: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Clear all pending tasks, optionally filtered by type.

    WARNING: This will cancel all queued work!
    """
    from ..models import TaskQueue, TaskStatus, TaskType as TT

    query = (
        update(TaskQueue)
        .where(TaskQueue.status == TaskStatus.pending)
    )

    if task_type:
        try:
            tt = TT(task_type)
            query = query.where(TaskQueue.task_type == tt)
        except ValueError:
            raise HTTPException(400, f"Invalid task type: {task_type}")

    query = query.values(status=TaskStatus.failed, error_message="Cancelled by user")

    result = await db.execute(query)
    await db.commit()

    return {"cancelled": result.rowcount}


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
