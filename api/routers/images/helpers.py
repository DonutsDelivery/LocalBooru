"""
Shared helper functions for per-directory database queries.

Architecture:
- Images can be in per-directory databases (directories/{id}.db) or legacy main DB
- directory_id parameter specifies which directory DB to query
- For cross-directory queries, we aggregate results from multiple DBs
"""
from fastapi import Request
from sqlalchemy import select, func, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from ...database import directory_db_manager
from ...models import (
    Tag, ImageFile, Rating, FileStatus, WatchDirectory,
    DirectoryImage, DirectoryImageFile, directory_image_tags
)
from ...services.file_tracker import check_file_availability


# File extension sets for media type filtering
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.webm', '.mp4', '.mov', '.avi', '.mkv'}


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
    filename: str = None,
    min_width: int = None,
    min_height: int = None,
    orientation: str = None,
    min_duration: int = None,
    max_duration: int = None,
    sort: str = "newest",
    limit: int = 100,
    offset: int = 0,
    show_images: bool = True,
    show_videos: bool = True
) -> tuple[list[dict], int]:
    """
    Query images from a single directory database.
    Returns (images_list, total_count).
    """
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

        # Media type filtering (images vs videos)
        if not show_images and not show_videos:
            # Nothing to show
            return [], 0
        elif not show_images:
            # Only videos - filter by video extensions
            video_filter_conditions = []
            for ext in VIDEO_EXTENSIONS:
                video_filter_conditions.append(
                    DirectoryImageFile.original_path.ilike(f'%{ext}')
                )
            has_video_ext = select(DirectoryImageFile.image_id).where(
                or_(*video_filter_conditions)
            )
            filters.append(DirectoryImage.id.in_(has_video_ext))
        elif not show_videos:
            # Only images - filter by image extensions
            image_filter_conditions = []
            for ext in IMAGE_EXTENSIONS:
                image_filter_conditions.append(
                    DirectoryImageFile.original_path.ilike(f'%{ext}')
                )
            has_image_ext = select(DirectoryImageFile.image_id).where(
                or_(*image_filter_conditions)
            )
            filters.append(DirectoryImage.id.in_(has_image_ext))
        # If both are True, no filtering needed - show all

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

        # Filename filter - search in file paths (case-insensitive)
        if filename:
            # Search for filename in any of the image's file paths
            filename_pattern = f"%{filename}%"
            filename_subq = select(DirectoryImageFile.image_id).where(
                DirectoryImageFile.original_path.ilike(filename_pattern)
            )
            filters.append(DirectoryImage.id.in_(filename_subq))

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

        # Resolution filter - check that the shorter dimension meets the minimum
        # This handles both landscape and portrait correctly (e.g., 1440p means shorter side >= 1440)
        if min_height is not None:
            filters.append(func.least(DirectoryImage.width, DirectoryImage.height) >= min_height)

        # Orientation filter
        if orientation:
            if orientation == 'landscape':
                filters.append(DirectoryImage.width > DirectoryImage.height)
            elif orientation == 'portrait':
                filters.append(DirectoryImage.height > DirectoryImage.width)
            elif orientation == 'square':
                filters.append(DirectoryImage.width == DirectoryImage.height)

        # Duration filter (for videos)
        if min_duration is not None:
            filters.append(DirectoryImage.duration >= min_duration)
        if max_duration is not None:
            filters.append(DirectoryImage.duration <= max_duration)

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


async def get_image_file_status(image, db: AsyncSession) -> dict:
    """Get the file availability status for an image.

    If file is confirmed deleted (parent exists but file doesn't),
    deletes the database entry.
    """
    from ...database import get_data_dir
    from ...models import Image

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
