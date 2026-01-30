"""
Image listing, search, and filtering endpoints.
"""
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, func, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional

from ...database import get_db, directory_db_manager
from ...models import (
    Image, Tag, ImageFile, image_tags, Rating, FileStatus, WatchDirectory
)
from .helpers import query_directory_images, check_image_public_access


router = APIRouter()


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
    filename: Optional[str] = Query(None, description="Search by filename (case-insensitive partial match)"),
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
        # Get directory's media type settings
        dir_query = select(WatchDirectory).where(WatchDirectory.id == directory_id)
        dir_result = await db.execute(dir_query)
        directory = dir_result.scalar_one_or_none()

        show_images = directory.show_images if directory and hasattr(directory, 'show_images') else True
        show_videos = directory.show_videos if directory and hasattr(directory, 'show_videos') else True

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
            filename=filename,
            sort=sort,
            limit=per_page,
            offset=offset,
            show_images=show_images,
            show_videos=show_videos
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
                # Get directory's media type settings
                dir_query = select(WatchDirectory).where(WatchDirectory.id == dir_id)
                dir_result = await db.execute(dir_query)
                directory = dir_result.scalar_one_or_none()

                dir_show_images = directory.show_images if directory and hasattr(directory, 'show_images') else True
                dir_show_videos = directory.show_videos if directory and hasattr(directory, 'show_videos') else True

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
                    filename=filename,
                    sort=sort,
                    limit=per_page,
                    offset=offset,
                    show_images=dir_show_images,
                    show_videos=dir_show_videos
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
