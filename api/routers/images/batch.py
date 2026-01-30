"""
Batch operations for images: delete, retag, age detect, metadata extract, move.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import os

from ...database import get_db, directory_db_manager
from ...models import (
    Image, ImageFile, image_tags, TaskType, WatchDirectory,
    DirectoryImage
)
from ...services.task_queue import enqueue_task
from .models import (
    BatchDeleteRequest, BatchRetagRequest, BatchAgeDetectRequest,
    BatchMetadataExtractRequest, BatchMoveRequest
)


router = APIRouter()


@router.post("/batch/delete")
async def batch_delete_images(
    request: BatchDeleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Delete multiple images from the library (optionally delete files too)"""
    from ...database import get_data_dir
    from ...services.video_preview import delete_preview_frames

    # Get all watch directory IDs to try their databases
    query = select(WatchDirectory.id)
    result = await db.execute(query)
    directory_ids = [row[0] for row in result.fetchall()]

    deleted = 0
    errors = []

    for image_id in request.image_ids:
        try:
            image = None
            file_hash = None

            # Try all directory databases first
            for dir_id in directory_ids:
                if directory_db_manager.db_exists(dir_id):
                    dir_db = await directory_db_manager.get_session(dir_id)
                    try:
                        query = select(DirectoryImage).options(selectinload(DirectoryImage.files)).where(DirectoryImage.id == image_id)
                        result = await dir_db.execute(query)
                        image = result.scalar_one_or_none()

                        if image:
                            file_hash = image.file_hash

                            # Optionally delete the actual file(s)
                            if request.delete_files and image.files:
                                for f in image.files:
                                    if os.path.exists(f.original_path):
                                        os.remove(f.original_path)

                            # Delete from directory database
                            await dir_db.delete(image)
                            await dir_db.commit()
                            break
                    finally:
                        await dir_db.close()

                if image:
                    break

            # If not found in directory databases, try main database
            if not image:
                query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
                result = await db.execute(query)
                image = result.scalar_one_or_none()

                if not image:
                    errors.append({"id": image_id, "error": "Image not found"})
                    continue

                file_hash = image.file_hash

                # Optionally delete the actual file(s)
                if request.delete_files:
                    for f in image.files:
                        if os.path.exists(f.original_path):
                            os.remove(f.original_path)

                # Delete from database (cascades to files, tags, etc.)
                await db.delete(image)

            # Delete thumbnail and preview frames (same for both databases)
            if file_hash:
                thumbnail_path = get_data_dir() / 'thumbnails' / f"{file_hash[:16]}.webp"
                if thumbnail_path.exists():
                    thumbnail_path.unlink()

                # Delete video preview frames
                delete_preview_frames(file_hash)

            deleted += 1

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    await db.commit()

    return {
        "deleted": deleted,
        "errors": errors,
        "total_requested": len(request.image_ids)
    }


@router.post("/batch/retag")
async def batch_retag_images(
    request: BatchRetagRequest,
    db: AsyncSession = Depends(get_db)
):
    """Queue retagging for multiple images"""
    queued = 0
    errors = []

    for image_id in request.image_ids:
        try:
            # Get image with files
            query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if not image:
                errors.append({"id": image_id, "error": "Image not found"})
                continue

            # Find a valid file path
            image_file = None
            for f in image.files:
                if f.file_exists and os.path.exists(f.original_path):
                    image_file = f
                    break

            if not image_file:
                errors.append({"id": image_id, "error": "No valid file found"})
                continue

            # Clear existing tags first
            from sqlalchemy import delete as sql_delete
            await db.execute(
                sql_delete(image_tags).where(image_tags.c.image_id == image_id)
            )

            # Queue tagging task
            await enqueue_task(
                TaskType.tag,
                {"image_id": image_id, "image_path": image_file.original_path},
                priority=5,  # Higher priority for user-initiated
                db=db
            )
            queued += 1

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    await db.commit()

    return {
        "queued": queued,
        "errors": errors,
        "total_requested": len(request.image_ids)
    }


@router.post("/batch/age-detect")
async def batch_age_detect_images(
    request: BatchAgeDetectRequest,
    db: AsyncSession = Depends(get_db)
):
    """Queue age detection for multiple images"""
    from ...services.age_detector import is_age_detection_enabled

    if not is_age_detection_enabled():
        raise HTTPException(status_code=400, detail="Age detection is not enabled")

    queued = 0
    errors = []

    for image_id in request.image_ids:
        try:
            # Get image with files
            query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if not image:
                errors.append({"id": image_id, "error": "Image not found"})
                continue

            # Find a valid file path
            image_file = None
            for f in image.files:
                if f.file_exists and os.path.exists(f.original_path):
                    image_file = f
                    break

            if not image_file:
                errors.append({"id": image_id, "error": "No valid file found"})
                continue

            # Clear existing age detection data
            image.num_faces = None
            image.min_detected_age = None
            image.max_detected_age = None
            image.detected_ages = None
            image.age_detection_data = None

            # Queue age detection task
            await enqueue_task(
                TaskType.age_detect,
                {"image_id": image_id, "image_path": image_file.original_path},
                priority=5,  # Higher priority for user-initiated
                db=db
            )
            queued += 1

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    await db.commit()

    return {
        "queued": queued,
        "errors": errors,
        "total_requested": len(request.image_ids)
    }


@router.post("/batch/extract-metadata")
async def batch_extract_metadata(
    request: BatchMetadataExtractRequest,
    db: AsyncSession = Depends(get_db)
):
    """Queue metadata extraction for multiple images"""
    queued = 0
    errors = []

    for image_id in request.image_ids:
        try:
            # Get image with files
            query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if not image:
                errors.append({"id": image_id, "error": "Image not found"})
                continue

            # Find a valid file path
            image_file = None
            for f in image.files:
                if f.file_exists and os.path.exists(f.original_path):
                    image_file = f
                    break

            if not image_file:
                errors.append({"id": image_id, "error": "No valid file found"})
                continue

            # Queue metadata extraction task
            await enqueue_task(
                TaskType.extract_metadata,
                {
                    "image_id": image_id,
                    "image_path": image_file.original_path,
                    "comfyui_prompt_node_ids": [],
                    "comfyui_negative_node_ids": [],
                    "format_hint": "auto"
                },
                priority=5,  # Higher priority for user-initiated
                db=db
            )
            queued += 1

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    await db.commit()

    return {
        "queued": queued,
        "errors": errors,
        "total_requested": len(request.image_ids)
    }


@router.post("/batch/move")
async def batch_move_images(
    request: BatchMoveRequest,
    db: AsyncSession = Depends(get_db)
):
    """Move multiple images to a different directory"""
    import shutil
    from ...models import WatchDirectory

    # Get the target directory
    dir_query = select(WatchDirectory).where(WatchDirectory.id == request.target_directory_id)
    dir_result = await db.execute(dir_query)
    target_dir = dir_result.scalar_one_or_none()

    if not target_dir:
        raise HTTPException(status_code=404, detail="Target directory not found")

    if not os.path.isdir(target_dir.path):
        raise HTTPException(status_code=400, detail="Target directory path does not exist on disk")

    moved = 0
    errors = []

    for image_id in request.image_ids:
        try:
            # Get image with files
            query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if not image:
                errors.append({"id": image_id, "error": "Image not found"})
                continue

            # Skip if already in target directory
            if image.directory_id == request.target_directory_id:
                errors.append({"id": image_id, "error": "Already in target directory"})
                continue

            # Move each file associated with this image
            for f in image.files:
                if not os.path.exists(f.original_path):
                    continue

                # Determine new path
                filename = os.path.basename(f.original_path)
                new_path = os.path.join(target_dir.path, filename)

                # Handle filename conflicts
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join(target_dir.path, f"{base}_{counter}{ext}")
                        counter += 1

                # Move the file
                shutil.move(f.original_path, new_path)

                # Update the file path in database
                f.original_path = new_path

            # Update the image's directory reference
            image.directory_id = request.target_directory_id
            moved += 1

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    await db.commit()

    return {
        "moved": moved,
        "errors": errors,
        "total_requested": len(request.image_ids)
    }
