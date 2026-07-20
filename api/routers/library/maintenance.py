"""
Library maintenance operations - cleanup, regeneration, file management
"""
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...database import get_db, directory_db_manager, get_data_dir
from ...models import (
    Image, ImageFile, WatchDirectory, TaskQueue,
    TaskStatus, TaskType,
    DirectoryImage, DirectoryImageFile
)
from .models import ImportFileRequest, FileMissingRequest

router = APIRouter()


@router.post("/clean-missing")
async def clean_missing_files(db: AsyncSession = Depends(get_db)):
    """Remove all images with missing files from the library.

    Checks both the main database (legacy) and all per-directory databases.
    """
    cleaned = 0
    dir_cleaned = 0
    tasks_cleaned = 0

    # --- Clean per-directory databases ---
    for dir_id in directory_db_manager.get_all_directory_ids():
        if not directory_db_manager.db_exists(dir_id):
            continue

        dir_db = await directory_db_manager.get_session(dir_id)
        try:
            query = select(DirectoryImageFile)
            result = await dir_db.execute(query)
            file_records = result.scalars().all()

            for file_record in file_records:
                if Path(file_record.original_path).exists():
                    continue

                # Check for other valid file references
                other_query = select(DirectoryImageFile).where(
                    DirectoryImageFile.image_id == file_record.image_id,
                    DirectoryImageFile.id != file_record.id
                )
                other_result = await dir_db.execute(other_query)
                other_files = other_result.scalars().all()
                has_valid_ref = any(Path(f.original_path).exists() for f in other_files)

                if has_valid_ref:
                    await dir_db.delete(file_record)
                else:
                    image = await dir_db.get(DirectoryImage, file_record.image_id)
                    await dir_db.delete(file_record)
                    if image:
                        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                        if thumbnail_path.exists():
                            thumbnail_path.unlink()
                        await dir_db.delete(image)

                dir_cleaned += 1

            await dir_db.commit()
        finally:
            await dir_db.close()

    # --- Clean main database (legacy) ---
    query = select(ImageFile)
    result = await db.execute(query)
    all_files = result.scalars().all()

    for image_file in all_files:
        if not Path(image_file.original_path).exists():
            other_query = select(ImageFile).where(
                ImageFile.image_id == image_file.image_id,
                ImageFile.id != image_file.id
            )
            other_result = await db.execute(other_query)
            other_files = other_result.scalars().all()

            has_valid_ref = any(Path(f.original_path).exists() for f in other_files)

            if has_valid_ref:
                await db.delete(image_file)
            else:
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
    return {"cleaned": cleaned, "directory_cleaned": dir_cleaned, "tasks_cleaned": tasks_cleaned}


@router.post("/clean-orphans")
async def clean_orphan_files(db: AsyncSession = Depends(get_db)):
    """Remove files from DB that have no watch_directory_id and aren't under any watched directory"""
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
    from ...services.task_queue import enqueue_task

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
    from ...services.file_tracker import clean_video_thumbnails

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
    from ...services.importer import (
        generate_thumbnail_async,
        generate_video_thumbnail_async,
        is_video_file
    )
    from ...config import get_settings

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
    from ...services.age_detector import is_age_detection_enabled
    from ...services.task_queue import enqueue_task

    if not is_age_detection_enabled():
        return {"error": "Age detection is not enabled", "queued": 0}

    # Get directories with auto_age_detect enabled
    dir_query = select(WatchDirectory).where(WatchDirectory.auto_age_detect == True)
    dir_result = await db.execute(dir_query)
    enabled_dirs = dir_result.scalars().all()

    if not enabled_dirs:
        return {"message": "No directories have age detection enabled", "queued": 0}

    queued = 0

    # Query each per-directory database for images without age detection
    for watch_dir in enabled_dirs:
        if not directory_db_manager.db_exists(watch_dir.id):
            continue

        dir_db = await directory_db_manager.get_session(watch_dir.id)
        try:
            # Find images without age detection data in this directory
            images_query = (
                select(DirectoryImage)
                .options(selectinload(DirectoryImage.files))
                .where(DirectoryImage.age_detection_data.is_(None))
            )
            result = await dir_db.execute(images_query)
            images = result.scalars().all()

            for image in images:
                # Get file path from the first existing file
                file_path = None
                for f in image.files:
                    if f.file_exists:
                        file_path = f.original_path
                        break

                if file_path:
                    await enqueue_task(
                        TaskType.age_detect,
                        {
                            "image_id": image.id,
                            "directory_id": watch_dir.id,
                            "image_path": file_path
                        },
                        priority=0,  # Low priority
                        db=db
                    )
                    queued += 1
        finally:
            await dir_db.close()

    if queued == 0:
        return {"message": "No images need age detection (check directory settings)", "queued": 0}

    return {"message": f"Queued age detection for {queued} images", "queued": queued}


@router.post("/detect-ages-realistic")
async def detect_ages_on_realistic(db: AsyncSession = Depends(get_db)):
    """Queue age detection for realistic-tagged images that don't have it yet"""
    from ...models import image_tags
    from ...services.task_queue import enqueue_task
    from ...services.age_detector import REALISTIC_TAGS

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
    """Queue tagging for all untagged images in a directory.

    Uses per-directory databases.
    """
    from ...models import directory_image_tags
    from ...services.task_queue import enqueue_task

    if directory_id is None:
        return {"error": "directory_id is required", "queued": 0}

    # Check if directory database exists
    if not directory_db_manager.db_exists(directory_id):
        return {"error": "Directory database not found", "queued": 0}

    # Get IDs of images that already have pending tag tasks (from main db)
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

    # Query per-directory database
    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        # Count total images
        total_result = await dir_db.execute(select(func.count(DirectoryImage.id)))
        total = total_result.scalar() or 0

        # Get tagged image IDs
        tagged_subq = select(directory_image_tags.c.image_id).distinct()

        # Count tagged
        tagged_result = await dir_db.execute(
            select(func.count(DirectoryImage.id)).where(DirectoryImage.id.in_(tagged_subq))
        )
        tagged = tagged_result.scalar() or 0

        # Get untagged images with their file paths
        untagged_query = (
            select(DirectoryImage, DirectoryImageFile)
            .join(DirectoryImageFile, DirectoryImageFile.image_id == DirectoryImage.id)
            .where(DirectoryImage.id.not_in(tagged_subq))
            .where(DirectoryImageFile.file_exists == True)
        )
        result = await dir_db.execute(untagged_query)
        rows = result.all()
    finally:
        await dir_db.close()

    queued = 0
    skipped_queued = 0
    for image, image_file in rows:
        # Skip if already queued
        if image.id in already_queued:
            skipped_queued += 1
            continue

        task = await enqueue_task(
            TaskType.tag,
            {
                'image_id': image.id,
                'image_path': image_file.original_path,
                'directory_id': directory_id
            },
            priority=5,  # Higher priority for user-initiated tagging
            db=db,
            dedupe_key=f"{directory_id}:{image.id}"  # Prevent duplicate tasks
        )
        if task:
            queued += 1
        else:
            skipped_queued += 1  # Was duplicate

    return {
        "queued": queued,
        "skipped_already_queued": skipped_queued,
        "debug": {
            "total_images": total,
            "tagged_images": tagged,
            "untagged_found": len(rows)
        }
    }


@router.post("/reextract-video-metadata")
async def reextract_video_metadata(
    directory_id: int = None,
    force: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-extract video metadata (duration, dimensions) for videos missing this data.

    Args:
        directory_id: Specific directory to process, or None for all directories
        force: If True, re-extract even for videos that already have metadata
    """
    from ...services.video_preview import get_video_duration, is_video_file
    from ...services.importer import get_image_dimensions
    from pathlib import Path
    import asyncio

    updated = 0
    skipped = 0
    failed = 0
    not_found = 0

    # Get directory IDs to process
    if directory_id:
        dir_ids = [directory_id]
    else:
        dir_ids = directory_db_manager.get_all_directory_ids()

    for dir_id in dir_ids:
        if not directory_db_manager.db_exists(dir_id):
            continue

        dir_db = await directory_db_manager.get_session(dir_id)
        try:
            # Get videos - either all or just those missing metadata
            if force:
                query = (
                    select(DirectoryImage, DirectoryImageFile)
                    .join(DirectoryImageFile, DirectoryImageFile.image_id == DirectoryImage.id)
                    .where(DirectoryImageFile.file_exists == True)
                )
            else:
                # Only videos missing duration
                query = (
                    select(DirectoryImage, DirectoryImageFile)
                    .join(DirectoryImageFile, DirectoryImageFile.image_id == DirectoryImage.id)
                    .where(DirectoryImageFile.file_exists == True)
                    .where(DirectoryImage.duration == None)
                )

            result = await dir_db.execute(query)
            rows = result.all()

            for image, image_file in rows:
                file_path = image_file.original_path

                # Skip non-video files
                if not is_video_file(file_path):
                    skipped += 1
                    continue

                # Check file exists
                if not Path(file_path).exists():
                    not_found += 1
                    continue

                try:
                    # Extract duration
                    loop = asyncio.get_event_loop()
                    duration = await loop.run_in_executor(None, get_video_duration, file_path)

                    # Extract dimensions if missing
                    if image.width is None or image.height is None:
                        dimensions = await loop.run_in_executor(None, get_image_dimensions, file_path)
                        if dimensions:
                            image.width = dimensions[0]
                            image.height = dimensions[1]

                    if duration is not None:
                        image.duration = duration
                        updated += 1
                    else:
                        failed += 1

                except Exception as e:
                    print(f"[VideoMetadata] Failed for {file_path}: {e}")
                    failed += 1

            await dir_db.commit()

        finally:
            await dir_db.close()

    return {
        "updated": updated,
        "skipped_non_video": skipped,
        "failed": failed,
        "file_not_found": not_found,
        "message": f"Updated {updated} videos, {failed} failed, {not_found} files not found"
    }


@router.post("/import-file")
async def import_file(request: ImportFileRequest, db: AsyncSession = Depends(get_db)):
    """Import a single file (called by directory watcher)"""
    from ...services.importer import import_image

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
    from ...services.file_tracker import mark_file_missing

    await mark_file_missing(request.file_path, db)
    return {"marked_missing": True}


# Import Tag model for detect_ages_on_realistic
from ...models import Tag
