"""
Image router - simplified for local single-user library

Architecture:
- Images can be in per-directory databases (directories/{id}.db) or legacy main DB
- directory_id parameter specifies which directory DB to query
- For cross-directory queries, we aggregate results from multiple DBs
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

from ..database import get_db, directory_db_manager
from ..models import (
    Image, Tag, ImageFile, image_tags, Rating, TagCategory, FileStatus, TaskType, WatchDirectory,
    DirectoryImage, DirectoryImageFile, directory_image_tags
)
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
    # Adjustment ranges (0 = no change)
    # Brightness: -200 to +200 (extended range for more control)
    # Contrast/Gamma: -100 to +100
    brightness: int = 0
    contrast: int = 0
    gamma: int = 0


router = APIRouter()


# =============================================================================
# Helper functions for per-directory database queries
# =============================================================================

async def get_image_from_directory(image_id: int, directory_id: int):
    """Get an image from a specific directory database."""
    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        query = (
            select(DirectoryImage)
            .options(selectinload(DirectoryImage.files))
            .where(DirectoryImage.id == image_id)
        )
        result = await dir_db.execute(query)
        return result.scalar_one_or_none()
    finally:
        await dir_db.close()


async def get_image_tags_from_directory(image_id: int, directory_id: int, main_db: AsyncSession) -> list:
    """Get tags for an image in a directory database."""
    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        # Get tag IDs from directory DB
        tag_ids_query = select(directory_image_tags.c.tag_id).where(
            directory_image_tags.c.image_id == image_id
        )
        result = await dir_db.execute(tag_ids_query)
        tag_ids = [row[0] for row in result.all()]

        if not tag_ids:
            return []

        # Get tag details from main DB
        tags_query = select(Tag).where(Tag.id.in_(tag_ids))
        tags_result = await main_db.execute(tags_query)
        return list(tags_result.scalars().all())
    finally:
        await dir_db.close()


async def find_image_directory(image_id: int, file_hash: str = None) -> int | None:
    """Find which directory contains an image by checking all directory DBs."""
    for dir_id in directory_db_manager.get_all_directory_ids():
        dir_db = await directory_db_manager.get_session(dir_id)
        try:
            if file_hash:
                query = select(DirectoryImage.id).where(DirectoryImage.file_hash == file_hash).limit(1)
            else:
                query = select(DirectoryImage.id).where(DirectoryImage.id == image_id).limit(1)
            result = await dir_db.execute(query)
            if result.scalar_one_or_none():
                return dir_id
        finally:
            await dir_db.close()
    return None


async def query_directory_images(
    directory_id: int,
    main_db: AsyncSession,
    tags: list[str] = None,
    exclude_tags: list[str] = None,
    rating: list[str] = None,
    favorites_only: bool = False,
    min_age: int = None,
    max_age: int = None,
    has_faces: bool = None,
    timeframe: str = None,
    sort: str = "newest",
    limit: int = 100,
    offset: int = 0
) -> tuple[list[dict], int]:
    """
    Query images from a single directory database.
    Returns (images_list, total_count).
    """
    from datetime import datetime, timedelta

    if not directory_db_manager.db_exists(directory_id):
        return [], 0

    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        query = select(DirectoryImage).options(selectinload(DirectoryImage.files))
        filters = []

        # Exclude missing files
        has_non_missing_file = select(DirectoryImageFile.image_id).where(
            DirectoryImageFile.file_status != FileStatus.missing
        )
        filters.append(DirectoryImage.id.in_(has_non_missing_file))

        # Favorites filter
        if favorites_only:
            filters.append(DirectoryImage.is_favorite == True)

        # Rating filter
        if rating:
            valid_ratings = [r for r in rating if r in [e.value for e in Rating]]
            if valid_ratings:
                filters.append(DirectoryImage.rating.in_([Rating(r) for r in valid_ratings]))

        # Age filters
        if min_age is not None:
            filters.append(DirectoryImage.max_detected_age >= min_age)
        if max_age is not None:
            filters.append(DirectoryImage.min_detected_age <= max_age)
        if has_faces is not None:
            if has_faces:
                filters.append(DirectoryImage.num_faces > 0)
            else:
                filters.append(or_(DirectoryImage.num_faces == 0, DirectoryImage.num_faces.is_(None)))

        # Timeframe filter
        if timeframe:
            now = datetime.now()
            if timeframe == 'today':
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                filters.append(DirectoryImage.created_at >= start)
            elif timeframe == 'week':
                start = now - timedelta(days=7)
                filters.append(DirectoryImage.created_at >= start)
            elif timeframe == 'month':
                start = now - timedelta(days=30)
                filters.append(DirectoryImage.created_at >= start)
            elif timeframe == 'year':
                start = now - timedelta(days=365)
                filters.append(DirectoryImage.created_at >= start)

        # Tag filters (need to query main DB for tag IDs)
        if tags:
            for tag_name in tags:
                tag_query = select(Tag.id).where(Tag.name == tag_name)
                tag_result = await main_db.execute(tag_query)
                tag_id = tag_result.scalar_one_or_none()
                if tag_id:
                    tag_subq = select(directory_image_tags.c.image_id).where(
                        directory_image_tags.c.tag_id == tag_id
                    )
                    filters.append(DirectoryImage.id.in_(tag_subq))
                else:
                    # Tag doesn't exist, no images will match
                    return [], 0

        if exclude_tags:
            for tag_name in exclude_tags:
                tag_query = select(Tag.id).where(Tag.name == tag_name)
                tag_result = await main_db.execute(tag_query)
                tag_id = tag_result.scalar_one_or_none()
                if tag_id:
                    tag_subq = select(directory_image_tags.c.image_id).where(
                        directory_image_tags.c.tag_id == tag_id
                    )
                    filters.append(DirectoryImage.id.not_in(tag_subq))

        if filters:
            query = query.where(and_(*filters))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await dir_db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        if sort == "newest":
            query = query.order_by(
                desc(func.coalesce(DirectoryImage.file_modified_at, DirectoryImage.created_at)),
                desc(DirectoryImage.id)
            )
        elif sort == "oldest":
            query = query.order_by(
                asc(func.coalesce(DirectoryImage.file_modified_at, DirectoryImage.created_at)),
                asc(DirectoryImage.id)
            )
        elif sort == "random":
            query = query.order_by(func.random())

        # Apply pagination
        query = query.offset(offset).limit(limit)

        result = await dir_db.execute(query)
        images = result.scalars().all()

        # Get directory name once (not per image)
        dir_query = select(WatchDirectory.name).where(WatchDirectory.id == directory_id)
        dir_result = await main_db.execute(dir_query)
        dir_name = dir_result.scalar_one_or_none()

        # Batch fetch all tags for these images (much faster than N+1 queries)
        image_ids = [img.id for img in images]
        tags_by_image = {}
        if image_ids:
            # Get all tag associations for these images in one query
            tag_ids_query = select(
                directory_image_tags.c.image_id,
                directory_image_tags.c.tag_id
            ).where(directory_image_tags.c.image_id.in_(image_ids))
            tag_assoc_result = await dir_db.execute(tag_ids_query)
            tag_associations = tag_assoc_result.all()

            # Collect unique tag IDs
            all_tag_ids = list(set(t[1] for t in tag_associations))

            # Fetch all tag details in one query from main DB
            if all_tag_ids:
                tags_query = select(Tag).where(Tag.id.in_(all_tag_ids))
                tags_result = await main_db.execute(tags_query)
                tags_by_id = {t.id: t for t in tags_result.scalars().all()}

                # Build tags_by_image mapping
                for image_id, tag_id in tag_associations:
                    if image_id not in tags_by_image:
                        tags_by_image[image_id] = []
                    if tag_id in tags_by_id:
                        tags_by_image[image_id].append(tags_by_id[tag_id])

        images_data = []
        for img in images:
            tags_list = tags_by_image.get(img.id, [])
            images_data.append({
                "id": img.id,
                "directory_id": directory_id,
                "filename": img.filename,
                "original_filename": img.original_filename,
                "width": img.width,
                "height": img.height,
                "rating": img.rating.value,
                "is_favorite": img.is_favorite,
                "thumbnail_url": f"/api/images/{img.id}/thumbnail?directory_id={directory_id}",
                "url": f"/api/images/{img.id}/file?directory_id={directory_id}",
                "file_status": img.files[0].file_status.value if img.files and img.files[0].file_status else "unknown",
                "tags": [{"name": t.name, "category": t.category.value} for t in tags_list],
                "num_faces": img.num_faces,
                "min_age": img.min_detected_age,
                "max_age": img.max_detected_age,
                "created_at": img.created_at.isoformat() if img.created_at else None,
                "file_size": img.file_size,
                "file_path": img.files[0].original_path if img.files else None,
                "directory_name": dir_name,
                "prompt": img.prompt,
                "negative_prompt": img.negative_prompt,
                "model_name": img.model_name,
                "sampler": img.sampler,
                "seed": img.seed,
                "steps": img.steps,
                "cfg_scale": img.cfg_scale,
                "duration": img.duration
            })

        return images_data, total

    finally:
        await dir_db.close()


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
    """
    List images with filtering and pagination.

    If directory_id is provided, queries that specific directory database.
    Otherwise, queries all directory databases and merges results.
    """
    # Parse tag filters
    tag_names = [t.strip().lower().replace(" ", "_") for t in (tags or "").split(",") if t.strip()]
    exclude_names = [t.strip().lower().replace(" ", "_") for t in (exclude_tags or "").split(",") if t.strip()]
    rating_list = [r for r in (rating or "").split(",") if r in [e.value for e in Rating]]

    offset = (page - 1) * per_page

    # Check if we should query per-directory databases
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        # Query single directory database
        images_data, total = await query_directory_images(
            directory_id=directory_id,
            main_db=db,
            tags=tag_names if tag_names else None,
            exclude_tags=exclude_names if exclude_names else None,
            rating=rating_list if rating_list else None,
            favorites_only=favorites_only,
            min_age=min_age,
            max_age=max_age,
            has_faces=has_faces,
            timeframe=timeframe,
            sort=sort,
            limit=per_page,
            offset=offset
        )

        return {
            "images": images_data,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }

    # No specific directory - check if any per-directory DBs exist
    all_dir_ids = directory_db_manager.get_all_directory_ids()

    if all_dir_ids and directory_id is None:
        # "All Directories" view - query the first directory that has matching content
        # This keeps the response fast while still showing images
        for dir_id in all_dir_ids:
            if not directory_db_manager.db_exists(dir_id):
                continue
            try:
                dir_images, dir_total = await query_directory_images(
                    directory_id=dir_id,
                    main_db=db,
                    tags=tag_names if tag_names else None,
                    exclude_tags=exclude_names if exclude_names else None,
                    rating=rating_list if rating_list else None,
                    favorites_only=favorites_only,
                    min_age=min_age,
                    max_age=max_age,
                    has_faces=has_faces,
                    timeframe=timeframe,
                    sort=sort,
                    limit=per_page,
                    offset=offset
                )
                if dir_images:
                    return {
                        "images": dir_images,
                        "total": dir_total,
                        "page": page,
                        "per_page": per_page,
                        "total_pages": (dir_total + per_page - 1) // per_page
                    }
            except Exception as e:
                print(f"[Images] Error querying directory {dir_id}: {e}")
                continue

        # No images found in any directory
        return {
            "images": [],
            "total": 0,
            "page": 1,
            "per_page": per_page,
            "total_pages": 0
        }

    # Fall back to legacy main database query
    query = select(Image).options(
        selectinload(Image.tags),
        selectinload(Image.files).selectinload(ImageFile.watch_directory)
    )

    filters = []

    # Network access filtering
    access_level = getattr(request.state, 'access_level', 'localhost')
    if access_level == 'public':
        public_dirs_subq = select(WatchDirectory.id).where(WatchDirectory.public_access == True)
        public_images_subq = select(ImageFile.image_id).where(
            ImageFile.watch_directory_id.in_(public_dirs_subq)
        )
        filters.append(Image.id.in_(public_images_subq))

    # Exclude missing files
    has_non_missing_file = select(ImageFile.image_id).where(
        ImageFile.file_status != FileStatus.missing
    )
    filters.append(Image.id.in_(has_non_missing_file))

    if favorites_only:
        filters.append(Image.is_favorite == True)

    if rating_list:
        filters.append(Image.rating.in_([Rating(r) for r in rating_list]))

    if min_age is not None:
        filters.append(Image.max_detected_age >= min_age)
    if max_age is not None:
        filters.append(Image.min_detected_age <= max_age)
    if has_faces is not None:
        if has_faces:
            filters.append(Image.num_faces > 0)
        else:
            filters.append(or_(Image.num_faces == 0, Image.num_faces.is_(None)))

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

    if directory_id is not None:
        dir_subq = select(ImageFile.image_id).where(ImageFile.watch_directory_id == directory_id)
        filters.append(Image.id.in_(dir_subq))

    for tag_name in tag_names:
        tag_subq = select(image_tags.c.image_id).join(Tag).where(Tag.name == tag_name)
        filters.append(Image.id.in_(tag_subq))

    for tag_name in exclude_names:
        tag_subq = select(image_tags.c.image_id).join(Tag).where(Tag.name == tag_name)
        filters.append(Image.id.not_in(tag_subq))

    if filters:
        query = query.where(and_(*filters))

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

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    images = result.scalars().all()

    def get_file_status(img):
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
                "file_size": img.file_size,
                "file_path": img.files[0].original_path if img.files else None,
                "directory_name": img.files[0].watch_directory.name if img.files and img.files[0].watch_directory else None,
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
async def get_image_file(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve the original image file"""
    original_path = None

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImageFile.original_path).where(
                DirectoryImageFile.image_id == image_id
            ).limit(1)
            result = await dir_db.execute(query)
            original_path = result.scalar_one_or_none()
        finally:
            await dir_db.close()

        if original_path:
            status = check_file_availability(original_path)
            if status == FileStatus.available:
                return FileResponse(original_path)  # Auto-detect media type from file
            elif status == FileStatus.drive_offline:
                raise HTTPException(
                    status_code=503,
                    detail="Drive is offline - the storage device containing this file is not connected"
                )
            else:
                raise HTTPException(status_code=404, detail="File is missing or was deleted")

    # Check public access for non-localhost (legacy path)
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    # Legacy: query main database
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
        return FileResponse(image_file.original_path)  # Auto-detect media type
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - the storage device containing this file is not connected"
        )
    else:
        raise HTTPException(status_code=404, detail="File is missing or was deleted")


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve the thumbnail"""
    from ..database import get_data_dir

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage).where(DirectoryImage.id == image_id)
            result = await dir_db.execute(query)
            image = result.scalar_one_or_none()

            if image and image.file_hash:
                thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"

                if thumbnail_path.exists():
                    return FileResponse(str(thumbnail_path), media_type="image/webp")

                # Try to regenerate from original
                file_query = select(DirectoryImageFile).where(
                    DirectoryImageFile.image_id == image_id
                ).limit(1)
                file_result = await dir_db.execute(file_query)
                image_file = file_result.scalar_one_or_none()

                if image_file and Path(image_file.original_path).exists():
                    from ..services.importer import generate_thumbnail, generate_video_thumbnail, is_video_file
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                    if is_video_file(image_file.original_path):
                        generate_video_thumbnail(image_file.original_path, str(thumbnail_path))
                    else:
                        generate_thumbnail(image_file.original_path, str(thumbnail_path))
                    if thumbnail_path.exists():
                        return FileResponse(str(thumbnail_path), media_type="image/webp")

                raise HTTPException(status_code=404, detail="Thumbnail not found")
        finally:
            await dir_db.close()

    # Check public access for non-localhost (legacy path)
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    # Legacy: query main database
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


@router.get("/{image_id}/preview-frames")
async def get_preview_frames(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get list of video preview frame URLs for an image.

    Returns frame URLs for videos with preview frames available.
    On-demand generation: if frames don't exist and source file is available,
    generation will be triggered and an empty array returned (check again later).
    """
    try:
        from ..services.video_preview import get_preview_frames as get_frames, generate_video_previews

        file_hash = None
        filename = None
        original_path = None

        # Try directory database first if directory_id provided
        if directory_id is not None and directory_db_manager.db_exists(directory_id):
            dir_db = await directory_db_manager.get_session(directory_id)
            try:
                query = (
                    select(DirectoryImage)
                    .options(selectinload(DirectoryImage.files))
                    .where(DirectoryImage.id == image_id)
                )
                result = await dir_db.execute(query)
                image = result.scalar_one_or_none()

                if image:
                    file_hash = image.file_hash
                    filename = image.filename
                    if image.files:
                        original_path = image.files[0].original_path
            finally:
                await dir_db.close()
        else:
            # Check public access for non-localhost (legacy path)
            if not await check_image_public_access(image_id, request, db):
                raise HTTPException(status_code=403, detail="This image is not available for remote access")

            # Legacy: query main database
            query = select(Image).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if image:
                file_hash = image.file_hash
                filename = image.filename

                file_query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
                file_result = await db.execute(file_query)
                image_file = file_result.scalar_one_or_none()
                if image_file:
                    original_path = image_file.original_path

        if not file_hash:
            raise HTTPException(status_code=404, detail="Image not found")

        # Check if this is a video
        ext = filename.lower().split('.')[-1] if filename else ''
        if ext not in ['webm', 'mp4', 'mov', 'avi', 'mkv']:
            return {"frames": [], "is_video": False}

        # Check for existing preview frames
        existing_frames = get_frames(file_hash)
        if existing_frames:
            # Return URLs to the frames
            dir_param = f"?directory_id={directory_id}" if directory_id else ""
            frame_urls = [
                f"/api/images/{image_id}/preview-frame/{i}{dir_param}"
                for i in range(len(existing_frames))
            ]
            return {"frames": frame_urls, "is_video": True, "count": len(existing_frames)}

        # No frames exist - try to generate them on-demand
        if original_path:
            status = check_file_availability(original_path)
            if status == FileStatus.available:
                # Trigger generation in background
                import asyncio
                from ..config import get_settings
                settings = get_settings()
                asyncio.create_task(
                    generate_video_previews(original_path, file_hash, settings.video_preview_frames)
                )

        # Return empty for now - client should retry after a moment
        return {"frames": [], "is_video": True, "generating": True}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PreviewFrames] Error for image {image_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting preview frames: {str(e)}")


@router.get("/{image_id}/preview-frame/{frame_index}")
async def get_preview_frame(
    request: Request,
    image_id: int,
    frame_index: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve a specific video preview frame image"""
    from ..services.video_preview import get_preview_dir

    if frame_index < 0 or frame_index >= 8:
        raise HTTPException(status_code=400, detail="Invalid frame index (must be 0-7)")

    file_hash = None

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage.file_hash).where(DirectoryImage.id == image_id)
            result = await dir_db.execute(query)
            file_hash = result.scalar_one_or_none()
        finally:
            await dir_db.close()
    else:
        # Check public access for non-localhost (legacy path)
        if not await check_image_public_access(image_id, request, db):
            raise HTTPException(status_code=403, detail="This image is not available for remote access")

        # Legacy: query main database
        query = select(Image.file_hash).where(Image.id == image_id)
        result = await db.execute(query)
        file_hash = result.scalar_one_or_none()

    if not file_hash:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get the frame file
    preview_dir = get_preview_dir(file_hash)
    frame_path = preview_dir / f"frame_{frame_index}.webp"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Preview frame not found")

    return FileResponse(str(frame_path), media_type="image/webp")


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

    # Delete video preview frames
    from ..services.video_preview import delete_preview_frames
    delete_preview_frames(image.file_hash)

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
    from ..services.video_preview import delete_preview_frames

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

            # Delete video preview frames
            delete_preview_frames(image.file_hash)

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


@router.post("/{image_id}/preview-adjust")
async def preview_image_adjustments(
    image_id: int,
    adjustments: ImageAdjustmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate a preview of image adjustments without modifying the original file.

    Returns a URL to a cached preview image that shows the adjustments with dithering.
    The preview is stored in a cache directory and cleaned up when a new preview is
    generated for the same image or when explicitly discarded.
    """
    import numpy as np
    from PIL import Image as PILImage
    from ..database import get_data_dir
    import hashlib

    # Validate adjustment values
    if not (-200 <= adjustments.brightness <= 200):
        raise HTTPException(status_code=400, detail="Brightness must be between -200 and +200")
    if not (-100 <= adjustments.contrast <= 100):
        raise HTTPException(status_code=400, detail="Contrast must be between -100 and +100")
    if not (-100 <= adjustments.gamma <= 100):
        raise HTTPException(status_code=400, detail="Gamma must be between -100 and +100")

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

    # Create preview cache directory
    preview_cache_dir = get_data_dir() / 'preview_cache'
    preview_cache_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing preview for this image
    for old_preview in preview_cache_dir.glob(f"{image_id}_*.webp"):
        old_preview.unlink()

    # Generate unique filename based on adjustments
    adj_hash = hashlib.md5(f"{adjustments.brightness}_{adjustments.contrast}_{adjustments.gamma}".encode()).hexdigest()[:8]
    preview_filename = f"{image_id}_{adj_hash}.webp"
    preview_path = preview_cache_dir / preview_filename

    try:
        # Open the image
        img = PILImage.open(file_path)

        # Convert to RGB for processing
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P':
                img = img.convert('RGBA')
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array for processing
        img_array = np.array(img, dtype=np.float32)

        # Apply adjustments (same as apply_image_adjustments)
        # Brightness: multiplicative (matches CSS brightness filter)
        # slider -100 to +100 maps to 0.0 to 2.0 multiplier
        if adjustments.brightness != 0:
            brightness_factor = 1 + (adjustments.brightness / 100)
            img_array = img_array * max(0, brightness_factor)

        if adjustments.contrast != 0:
            contrast_factor = (adjustments.contrast + 100) / 100
            img_array = ((img_array - 127) * contrast_factor) + 127

        if adjustments.gamma != 0:
            import math
            exponent = math.pow(3.0, -adjustments.gamma / 100.0)
            img_array = np.clip(img_array, 0, 255)
            img_array = np.power(img_array / 255.0, exponent) * 255

        # Apply dithering
        dither_strength = 0.5
        if adjustments.gamma != 0:
            dither_strength = 0.5 + (abs(adjustments.gamma) / 100.0) * 0.5
        dither_noise = np.random.uniform(-dither_strength, dither_strength, img_array.shape)
        img_array = img_array + dither_noise

        # Clamp and convert back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_array)

        # Save as WebP for efficient serving
        img.save(preview_path, format='WEBP', quality=90, method=4)

        return {
            "preview_url": f"/api/images/{image_id}/preview",
            "adjustments": {
                "brightness": adjustments.brightness,
                "contrast": adjustments.contrast,
                "gamma": adjustments.gamma
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


@router.get("/{image_id}/preview")
async def get_preview_image(image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the cached preview image for an image."""
    from ..database import get_data_dir

    preview_cache_dir = get_data_dir() / 'preview_cache'

    # Find the preview file for this image
    previews = list(preview_cache_dir.glob(f"{image_id}_*.webp"))

    if not previews:
        raise HTTPException(status_code=404, detail="No preview found for this image")

    # Return the most recent preview
    preview_path = previews[0]

    return FileResponse(str(preview_path), media_type="image/webp")


@router.delete("/{image_id}/preview")
async def discard_preview(image_id: int, db: AsyncSession = Depends(get_db)):
    """Discard the cached preview for an image."""
    from ..database import get_data_dir

    preview_cache_dir = get_data_dir() / 'preview_cache'

    deleted = 0
    for preview in preview_cache_dir.glob(f"{image_id}_*.webp"):
        preview.unlink()
        deleted += 1

    return {"deleted": deleted}


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

    # Validate adjustment values
    if not (-200 <= adjustments.brightness <= 200):
        raise HTTPException(status_code=400, detail="Brightness must be between -200 and +200")
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

        # Adjustments applied in order: brightness -> contrast -> gamma

        # Brightness: multiplicative (matches CSS brightness filter)
        # slider -100 to +100 maps to 0.0 to 2.0 multiplier
        if adjustments.brightness != 0:
            brightness_factor = 1 + (adjustments.brightness / 100)
            img_array = img_array * max(0, brightness_factor)

        # Contrast: ((value - 127) * (contrast + 100) / 100) + 127
        if adjustments.contrast != 0:
            contrast_factor = (adjustments.contrast + 100) / 100
            img_array = ((img_array - 127) * contrast_factor) + 127

        # Gamma: exponential mapping for proper gamma curve
        # slider -100 to +100 maps to exponent 3.0 to 0.33
        # At 0: exponent = 1.0 (no change)
        # Positive = brighter midtones (exponent < 1, lifts curve)
        # Negative = darker midtones (exponent > 1, lowers curve)
        if adjustments.gamma != 0:
            import math
            exponent = math.pow(3.0, -adjustments.gamma / 100.0)
            img_array = np.clip(img_array, 0, 255)  # Clamp before gamma
            img_array = np.power(img_array / 255.0, exponent) * 255

        # Apply dithering to reduce banding artifacts (especially visible after gamma)
        # Uses random noise scaled to the adjustments applied - more aggressive for gamma
        # which stretches limited dark values across wider ranges
        # Standard dithering uses +/-0.5, but gamma lifting needs up to +/-1.0
        dither_strength = 0.5
        if adjustments.gamma != 0:
            # Scale dithering with gamma intensity - more gamma lift = more dithering needed
            dither_strength = 0.5 + (abs(adjustments.gamma) / 100.0) * 0.5  # 0.5 to 1.0
        dither_noise = np.random.uniform(-dither_strength, dither_strength, img_array.shape)
        img_array = img_array + dither_noise

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
