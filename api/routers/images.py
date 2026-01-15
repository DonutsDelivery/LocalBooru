"""
Image router - simplified for local single-user library
"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import FileResponse
from sqlalchemy import select, func, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import os
import tempfile

from ..database import get_db
from ..models import Image, Tag, ImageFile, image_tags, Rating, TagCategory, FileStatus, TaskType, WatchDirectory
from ..services.importer import import_image
from ..services.file_tracker import check_file_availability
from ..services.task_queue import enqueue_task


# Request models for batch operations
class BatchDeleteRequest(BaseModel):
    image_ids: List[int]
    delete_files: bool = False


class BatchRetagRequest(BaseModel):
    image_ids: List[int]


class BatchAgeDetectRequest(BaseModel):
    image_ids: List[int]


class BatchMetadataExtractRequest(BaseModel):
    image_ids: List[int]


class BatchMoveRequest(BaseModel):
    image_ids: List[int]
    target_directory_id: int


class ImageAdjustmentRequest(BaseModel):
    # Gwenview-style ranges (all -100 to +100, 0 = no change)
    brightness: int = 0
    contrast: int = 0
    gamma: int = 0


router = APIRouter()


async def check_image_public_access(image_id: int, request: Request, db: AsyncSession) -> bool:
    """
    Check if an image is accessible for the current request.
    Returns True if accessible, False if blocked by public_access settings.
    """
    access_level = getattr(request.state, 'access_level', 'localhost')

    # Localhost and local_network (same WiFi) have full access
    if access_level in ('localhost', 'local_network'):
        return True

    # For public internet IPs, check if the image is in a public directory
    query = (
        select(ImageFile)
        .options(selectinload(ImageFile.watch_directory))
        .where(ImageFile.image_id == image_id)
        .limit(1)
    )
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file or not image_file.watch_directory:
        return False

    return image_file.watch_directory.public_access == True


async def get_image_file_status(image: Image, db: AsyncSession) -> dict:
    """Get the file availability status for an image.

    If file is confirmed deleted (parent exists but file doesn't),
    deletes the database entry.
    """
    from ..database import get_data_dir

    # Get the primary file for this image
    query = select(ImageFile).where(ImageFile.image_id == image.id).limit(1)
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        return {"status": "missing", "path": None}

    # Check actual file availability
    status = check_file_availability(image_file.original_path)

    if status == FileStatus.missing:
        # File confirmed deleted (parent dir exists) - remove from database
        # Check for other file references first
        other_query = select(ImageFile).where(
            ImageFile.image_id == image.id,
            ImageFile.id != image_file.id
        )
        other_result = await db.execute(other_query)
        has_other_refs = other_result.scalar_one_or_none() is not None

        await db.delete(image_file)

        if not has_other_refs:
            # No other references - delete the image and thumbnail
            thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            await db.delete(image)

        await db.commit()
        return {"status": "deleted", "path": image_file.original_path}

    # Update database if status changed (for drive_offline)
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    return {
        "status": status.value,
        "path": image_file.original_path
    }


@router.get("")
async def list_images(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    tags: Optional[str] = None,
    exclude_tags: Optional[str] = None,
    rating: Optional[str] = None,
    favorites_only: bool = False,
    directory_id: Optional[int] = Query(None, description="Filter to images from a specific directory"),
    min_age: Optional[int] = Query(None, ge=0, le=120, description="Minimum detected age"),
    max_age: Optional[int] = Query(None, ge=0, le=120, description="Maximum detected age"),
    has_faces: Optional[bool] = Query(None, description="Filter to images with detected faces"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe: today, week, month, year"),
    sort: str = "newest",
    db: AsyncSession = Depends(get_db)
):
    """List images with filtering and pagination"""
    query = select(Image).options(
        selectinload(Image.tags),
        selectinload(Image.files).selectinload(ImageFile.watch_directory)
    )

    filters = []

    # Network access filtering: public internet IPs can only see images from public directories
    # Local network (same WiFi) gets full access like localhost
    access_level = getattr(request.state, 'access_level', 'localhost')
    if access_level == 'public':
        # Get directories with public_access=True
        public_dirs_subq = select(WatchDirectory.id).where(WatchDirectory.public_access == True)
        # Filter to images in those directories
        public_images_subq = select(ImageFile.image_id).where(
            ImageFile.watch_directory_id.in_(public_dirs_subq)
        )
        filters.append(Image.id.in_(public_images_subq))

    # Exclude images where all files are confirmed missing
    # (drive_offline is OK to show, only exclude "missing" status)
    has_non_missing_file = select(ImageFile.image_id).where(
        ImageFile.file_status != FileStatus.missing
    )
    filters.append(Image.id.in_(has_non_missing_file))

    # Favorites filter
    if favorites_only:
        filters.append(Image.is_favorite == True)

    # Rating filter
    if rating:
        ratings = rating.split(",")
        valid_ratings = [r for r in ratings if r in [e.value for e in Rating]]
        if valid_ratings:
            filters.append(Image.rating.in_([Rating(r) for r in valid_ratings]))

    # Age filters
    if min_age is not None:
        # Include images where max_detected_age >= min_age (at least one face is old enough)
        filters.append(Image.max_detected_age >= min_age)
    if max_age is not None:
        # Include images where min_detected_age <= max_age (at least one face is young enough)
        filters.append(Image.min_detected_age <= max_age)
    if has_faces is not None:
        if has_faces:
            filters.append(Image.num_faces > 0)
        else:
            filters.append(or_(Image.num_faces == 0, Image.num_faces.is_(None)))

    # Timeframe filter (based on when image was added to library)
    if timeframe:
        from datetime import datetime, timedelta
        now = datetime.now()
        if timeframe == 'today':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            filters.append(Image.created_at >= start)
        elif timeframe == 'week':
            start = now - timedelta(days=7)
            filters.append(Image.created_at >= start)
        elif timeframe == 'month':
            start = now - timedelta(days=30)
            filters.append(Image.created_at >= start)
        elif timeframe == 'year':
            start = now - timedelta(days=365)
            filters.append(Image.created_at >= start)

    # Directory filter
    if directory_id is not None:
        dir_subq = select(ImageFile.image_id).where(ImageFile.watch_directory_id == directory_id)
        filters.append(Image.id.in_(dir_subq))

    # Tag inclusion filter
    if tags:
        tag_names = [t.strip().lower().replace(" ", "_") for t in tags.split(",") if t.strip()]
        for tag_name in tag_names:
            tag_subq = (
                select(image_tags.c.image_id)
                .join(Tag)
                .where(Tag.name == tag_name)
            )
            filters.append(Image.id.in_(tag_subq))

    # Tag exclusion filter
    if exclude_tags:
        exclude_names = [t.strip().lower().replace(" ", "_") for t in exclude_tags.split(",") if t.strip()]
        for tag_name in exclude_names:
            tag_subq = (
                select(image_tags.c.image_id)
                .join(Tag)
                .where(Tag.name == tag_name)
            )
            filters.append(Image.id.not_in(tag_subq))

    if filters:
        query = query.where(and_(*filters))

    # Sorting (with secondary sort by ID for consistent ordering)
    # Use file_modified_at for date sorting (actual file date), fall back to created_at
    if sort == "newest":
        query = query.order_by(
            desc(func.coalesce(Image.file_modified_at, Image.created_at)),
            desc(Image.id)
        )
    elif sort == "oldest":
        query = query.order_by(
            asc(func.coalesce(Image.file_modified_at, Image.created_at)),
            asc(Image.id)
        )
    elif sort == "random":
        query = query.order_by(func.random())

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    images = result.scalars().all()

    def get_file_status(img):
        """Get file status from loaded files relationship"""
        if not img.files:
            return "missing"
        return img.files[0].file_status.value if img.files[0].file_status else "unknown"

    return {
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "original_filename": img.original_filename,
                "width": img.width,
                "height": img.height,
                "rating": img.rating.value,
                "is_favorite": img.is_favorite,
                "thumbnail_url": img.thumbnail_url,
                "url": img.url,
                "file_status": get_file_status(img),
                "tags": [{"name": t.name, "category": t.category.value} for t in img.tags],
                "num_faces": img.num_faces,
                "min_age": img.min_detected_age,
                "max_age": img.max_detected_age,
                "created_at": img.created_at.isoformat() if img.created_at else None,
                # File metadata
                "file_size": img.file_size,
                "file_path": img.files[0].original_path if img.files else None,
                "directory_name": img.files[0].watch_directory.name if img.files and img.files[0].watch_directory else None,
                # AI generation metadata
                "prompt": img.prompt,
                "negative_prompt": img.negative_prompt,
                "model_name": img.model_name,
                "sampler": img.sampler,
                "seed": img.seed,
                "steps": img.steps,
                "cfg_scale": img.cfg_scale
            }
            for img in images
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }


@router.get("/{image_id}")
async def get_image(request: Request, image_id: int, db: AsyncSession = Depends(get_db)):
    """Get single image details"""
    # Check public access for non-localhost
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    query = (
        select(Image)
        .options(selectinload(Image.tags), selectinload(Image.files))
        .where(Image.id == image_id)
    )
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Increment view count
    image.view_count += 1
    await db.commit()

    return {
        "id": image.id,
        "filename": image.filename,
        "original_filename": image.original_filename,
        "file_hash": image.file_hash,
        "width": image.width,
        "height": image.height,
        "file_size": image.file_size,
        "rating": image.rating.value,
        "is_favorite": image.is_favorite,
        "view_count": image.view_count,
        "thumbnail_url": image.thumbnail_url,
        "url": image.url,
        "tags": [
            {"name": t.name, "category": t.category.value}
            for t in sorted(image.tags, key=lambda t: (t.category.value, t.name))
        ],
        "files": [
            {
                "path": f.original_path,
                "exists": f.file_exists,
                "status": f.file_status.value if f.file_status else "unknown"
            }
            for f in image.files
        ],
        "file_status": image.files[0].file_status.value if image.files and image.files[0].file_status else "unknown",
        "prompt": image.prompt,
        "negative_prompt": image.negative_prompt,
        "model_name": image.model_name,
        "seed": image.seed,
        "source_url": image.source_url,
        "import_source": image.import_source,
        "num_faces": image.num_faces,
        "min_age": image.min_detected_age,
        "max_age": image.max_detected_age,
        "detected_ages": image.detected_ages,
        "age_detection_data": image.age_detection_data,
        "created_at": image.created_at.isoformat() if image.created_at else None
    }


@router.get("/{image_id}/file")
async def get_image_file(request: Request, image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the original image file"""
    # Check public access for non-localhost
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        raise HTTPException(status_code=404, detail="Image file not found")

    # Check file availability with drive detection
    status = check_file_availability(image_file.original_path)

    # Update database if status changed
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    if status == FileStatus.available:
        return FileResponse(
            image_file.original_path,
            media_type="image/webp"  # Will be overridden by actual type
        )
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - the storage device containing this file is not connected"
        )
    else:
        raise HTTPException(status_code=404, detail="File is missing or was deleted")


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(request: Request, image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the thumbnail"""
    # Check public access for non-localhost
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    from ..database import get_data_dir

    query = select(Image).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"

    if thumbnail_path.exists():
        return FileResponse(str(thumbnail_path), media_type="image/webp")

    # Thumbnail doesn't exist - try to regenerate from original
    file_query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
    file_result = await db.execute(file_query)
    image_file = file_result.scalar_one_or_none()

    if not image_file:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Check file availability
    status = check_file_availability(image_file.original_path)

    # Update database if status changed
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    if status == FileStatus.available:
        from ..services.importer import generate_thumbnail
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        generate_thumbnail(image_file.original_path, str(thumbnail_path))
        return FileResponse(str(thumbnail_path), media_type="image/webp")
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - cannot generate thumbnail"
        )
    else:
        raise HTTPException(status_code=404, detail="Source file missing - cannot generate thumbnail")


@router.post("/{image_id}/favorite")
async def toggle_favorite(image_id: int, db: AsyncSession = Depends(get_db)):
    """Toggle favorite status"""
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.is_favorite = not image.is_favorite
    await db.commit()

    return {"is_favorite": image.is_favorite}


@router.patch("/{image_id}/rating")
async def update_rating(image_id: int, rating: str, db: AsyncSession = Depends(get_db)):
    """Update image rating"""
    if rating not in [r.value for r in Rating]:
        raise HTTPException(status_code=400, detail=f"Invalid rating: {rating}")

    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.rating = Rating(rating)
    await db.commit()

    return {"rating": image.rating.value}


@router.delete("/{image_id}")
async def delete_image(
    image_id: int,
    delete_file: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Delete an image from the library (optionally delete the file too)"""
    query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Optionally delete the actual file(s)
    if delete_file:
        for f in image.files:
            if os.path.exists(f.original_path):
                os.remove(f.original_path)

    # Delete thumbnail
    from ..database import get_data_dir
    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
    if thumbnail_path.exists():
        thumbnail_path.unlink()

    # Delete from database (cascades to files, tags, etc.)
    await db.delete(image)
    await db.commit()

    return {"deleted": True}


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    auto_tag: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Upload a new image manually"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await import_image(tmp_path, db, auto_tag=auto_tag)
        return result
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@router.post("/batch/delete")
async def batch_delete_images(
    request: BatchDeleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Delete multiple images from the library (optionally delete files too)"""
    from ..database import get_data_dir

    deleted = 0
    errors = []

    for image_id in request.image_ids:
        try:
            query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if not image:
                errors.append({"id": image_id, "error": "Image not found"})
                continue

            # Optionally delete the actual file(s)
            if request.delete_files:
                for f in image.files:
                    if os.path.exists(f.original_path):
                        os.remove(f.original_path)

            # Delete thumbnail
            thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
            if thumbnail_path.exists():
                thumbnail_path.unlink()

            # Delete from database (cascades to files, tags, etc.)
            await db.delete(image)
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
    from ..services.age_detector import is_age_detection_enabled

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
    from ..models import WatchDirectory

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


@router.post("/{image_id}/adjust")
async def apply_image_adjustments(
    image_id: int,
    adjustments: ImageAdjustmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Apply brightness, contrast, and gamma adjustments using Gwenview's exact algorithms"""
    import numpy as np
    from PIL import Image as PILImage
    from ..database import get_data_dir

    # Validate adjustment values (all -100 to +100)
    if not (-100 <= adjustments.brightness <= 100):
        raise HTTPException(status_code=400, detail="Brightness must be between -100 and +100")
    if not (-100 <= adjustments.contrast <= 100):
        raise HTTPException(status_code=400, detail="Contrast must be between -100 and +100")
    if not (-100 <= adjustments.gamma <= 100):
        raise HTTPException(status_code=400, detail="Gamma must be between -100 and +100")

    # Check if any adjustment is needed
    if adjustments.brightness == 0 and adjustments.contrast == 0 and adjustments.gamma == 0:
        return {"adjusted": False, "message": "No adjustments needed"}

    # Get image and file path
    query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Find a valid file path
    image_file = None
    for f in image.files:
        if os.path.exists(f.original_path):
            image_file = f
            break

    if not image_file:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    file_path = image_file.original_path
    file_ext = Path(file_path).suffix.lower()

    # Check if it's an editable image format
    editable_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']
    if file_ext not in editable_formats:
        raise HTTPException(status_code=400, detail=f"Cannot adjust {file_ext} files")

    try:
        # Open the image
        img = PILImage.open(file_path)

        # Preserve EXIF data if present
        exif_data = img.info.get('exif')

        # Convert to RGB if necessary for processing
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P':
                img = img.convert('RGBA')
            alpha = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
            img = img.convert('RGB')
        else:
            alpha = None
            if img.mode != 'RGB':
                img = img.convert('RGB')

        # Convert to numpy array for processing
        img_array = np.array(img, dtype=np.float32)

        # Gwenview formulas (applied in order: brightness -> contrast -> gamma)

        # Brightness: value + brightness * 255 / 100
        if adjustments.brightness != 0:
            img_array = img_array + (adjustments.brightness * 255 / 100)

        # Contrast: ((value - 127) * (contrast + 100) / 100) + 127
        if adjustments.contrast != 0:
            contrast_factor = (adjustments.contrast + 100) / 100
            img_array = ((img_array - 127) * contrast_factor) + 127

        # Gamma: pow(value / 255, 100 / (gamma + 100)) * 255
        if adjustments.gamma != 0:
            gamma_value = adjustments.gamma + 100
            if gamma_value > 0:  # Prevent division by zero
                img_array = np.clip(img_array, 0, 255)  # Clamp before gamma
                img_array = np.power(img_array / 255.0, 100.0 / gamma_value) * 255

        # Clamp final values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_array)

        # Restore alpha channel if present
        if alpha is not None:
            img = img.convert('RGBA')
            r, g, b, _ = img.split()
            img = PILImage.merge('RGBA', (r, g, b, alpha))

        # Save the image back to the same file
        save_kwargs = {}
        if file_ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = 95
            save_kwargs['optimize'] = True
            if exif_data:
                save_kwargs['exif'] = exif_data
        elif file_ext == '.webp':
            save_kwargs['quality'] = 95
            save_kwargs['method'] = 6
        elif file_ext == '.png':
            save_kwargs['optimize'] = True

        img.save(file_path, **save_kwargs)

        # Regenerate thumbnail
        from ..services.importer import generate_thumbnail
        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
        if thumbnail_path.exists():
            thumbnail_path.unlink()
        generate_thumbnail(file_path, str(thumbnail_path))

        return {
            "adjusted": True,
            "brightness": adjustments.brightness,
            "contrast": adjustments.contrast,
            "gamma": adjustments.gamma
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply adjustments: {str(e)}")
