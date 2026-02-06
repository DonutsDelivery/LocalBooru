"""
Image listing, search, and filtering endpoints.
"""
import os
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, func, desc, asc, or_, and_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional

from ...database import get_db, directory_db_manager
from ...models import (
    Image, Tag, ImageFile, image_tags, Rating, FileStatus, WatchDirectory,
    DirectoryImage, DirectoryImageFile, directory_image_tags
)
from .helpers import query_directory_images, check_image_public_access


router = APIRouter()


@router.get("/")
async def list_images(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=400),
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
    min_width: Optional[int] = Query(None, ge=1, description="Minimum width in pixels"),
    min_height: Optional[int] = Query(None, ge=1, description="Minimum height in pixels"),
    orientation: Optional[str] = Query(None, description="Filter by orientation: landscape, portrait, square"),
    min_duration: Optional[int] = Query(None, ge=0, description="Minimum video duration in seconds"),
    max_duration: Optional[int] = Query(None, ge=0, description="Maximum video duration in seconds"),
    import_source: Optional[str] = Query(None, description="Filter to images from a specific source folder"),
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
            min_width=min_width,
            min_height=min_height,
            orientation=orientation,
            min_duration=min_duration,
            max_duration=max_duration,
            import_source=import_source,
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
                    min_width=min_width,
                    min_height=min_height,
                    orientation=orientation,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    import_source=import_source,
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

    # Resolution filter - check that the shorter dimension meets the minimum
    # This handles both landscape and portrait correctly (e.g., 1440p means shorter side >= 1440)
    if min_height is not None:
        filters.append(func.least(Image.width, Image.height) >= min_height)

    # Orientation filter
    if orientation:
        if orientation == 'landscape':
            filters.append(Image.width > Image.height)
        elif orientation == 'portrait':
            filters.append(Image.height > Image.width)
        elif orientation == 'square':
            filters.append(Image.width == Image.height)

    # Duration filter (for videos)
    if min_duration is not None:
        filters.append(Image.duration >= min_duration)
    if max_duration is not None:
        filters.append(Image.duration <= max_duration)

    # Import source filter (for folder grouping)
    if import_source is not None:
        filters.append(Image.import_source == import_source)

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
    elif sort == "filename_asc":
        query = query.order_by(
            asc(func.lower(Image.original_filename)),
            asc(Image.id)
        )
    elif sort == "filename_desc":
        query = query.order_by(
            desc(func.lower(Image.original_filename)),
            desc(Image.id)
        )
    elif sort == "filesize_largest":
        query = query.order_by(
            desc(func.coalesce(Image.file_size, 0)),
            desc(Image.id)
        )
    elif sort == "filesize_smallest":
        query = query.order_by(
            asc(func.coalesce(Image.file_size, 0)),
            asc(Image.id)
        )
    elif sort == "resolution_high":
        query = query.order_by(
            desc(func.coalesce(Image.width, 0) * func.coalesce(Image.height, 0)),
            desc(Image.id)
        )
    elif sort == "resolution_low":
        query = query.order_by(
            asc(func.coalesce(Image.width, 0) * func.coalesce(Image.height, 0)),
            asc(Image.id)
        )
    elif sort == "duration_longest":
        query = query.order_by(
            desc(func.coalesce(Image.duration, 0)),
            desc(Image.id)
        )
    elif sort == "duration_shortest":
        query = query.order_by(
            asc(case((Image.duration.is_(None), 1), else_=0)),
            asc(func.coalesce(Image.duration, 0)),
            asc(Image.id)
        )
    elif sort == "folder_asc":
        query = query.order_by(
            asc(func.coalesce(func.lower(Image.import_source), '')),
            asc(Image.id)
        )
    elif sort == "folder_desc":
        query = query.order_by(
            desc(func.coalesce(func.lower(Image.import_source), '')),
            desc(Image.id)
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
                "directory_id": img.files[0].watch_directory_id if img.files else None,
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


@router.get("/folders")
async def list_folders(
    request: Request,
    directory_id: Optional[int] = Query(None, description="Filter to a specific directory"),
    rating: Optional[str] = None,
    favorites_only: bool = False,
    tags: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List folders grouped by import_source with counts and representative thumbnails.
    Respects the same filters as list_images so folder counts reflect active filters.
    """
    tag_names = [t.strip().lower().replace(" ", "_") for t in (tags or "").split(",") if t.strip()]
    rating_list = [r for r in (rating or "").split(",") if r in [e.value for e in Rating]]

    # Aggregate folder data across directory databases
    folders_map = {}  # path -> { count, thumbnail_url, width, height, created_at, directory_id }

    all_dir_ids = directory_db_manager.get_all_directory_ids()
    dir_ids_to_query = [directory_id] if directory_id is not None else all_dir_ids

    for dir_id in dir_ids_to_query:
        if not directory_db_manager.db_exists(dir_id):
            continue
        dir_db = await directory_db_manager.get_session(dir_id)
        try:
            # Build filters matching list_images logic
            filters = []

            # Exclude missing files
            has_non_missing_file = select(DirectoryImageFile.image_id).where(
                DirectoryImageFile.file_status != FileStatus.missing
            )
            filters.append(DirectoryImage.id.in_(has_non_missing_file))

            if favorites_only:
                filters.append(DirectoryImage.is_favorite == True)

            if rating_list:
                filters.append(DirectoryImage.rating.in_([Rating(r) for r in rating_list]))

            # Tag filters
            if tag_names:
                for tag_name in tag_names:
                    tag_query = select(Tag.id).where(Tag.name == tag_name)
                    tag_result = await db.execute(tag_query)
                    tag_id = tag_result.scalar_one_or_none()
                    if tag_id:
                        tag_subq = select(directory_image_tags.c.image_id).where(
                            directory_image_tags.c.tag_id == tag_id
                        )
                        filters.append(DirectoryImage.id.in_(tag_subq))
                    else:
                        filters = None
                        break

            if filters is None:
                # A tag didn't exist, skip this directory
                continue

            # Query: group by import_source, get count
            count_query = (
                select(
                    DirectoryImage.import_source,
                    func.count(DirectoryImage.id).label('count')
                )
                .where(and_(*filters) if filters else True)
                .where(DirectoryImage.import_source.isnot(None))
                .group_by(DirectoryImage.import_source)
            )
            count_result = await dir_db.execute(count_query)

            for row in count_result:
                path = row[0]
                count = row[1]
                if path in folders_map:
                    folders_map[path]['count'] += count
                else:
                    folders_map[path] = {
                        'count': count,
                        'thumbnail_url': None,
                        'width': None,
                        'height': None,
                        'created_at': None,
                        'directory_id': dir_id
                    }

            # For each folder path, get the newest image as representative thumbnail
            for path in list(folders_map.keys()):
                if folders_map[path]['thumbnail_url'] is not None:
                    continue  # Already have a thumbnail from a previous dir

                thumb_query = (
                    select(DirectoryImage)
                    .where(
                        and_(
                            DirectoryImage.import_source == path,
                            *filters
                        )
                    )
                    .order_by(desc(func.coalesce(DirectoryImage.file_modified_at, DirectoryImage.created_at)))
                    .limit(1)
                )
                thumb_result = await dir_db.execute(thumb_query)
                thumb_img = thumb_result.scalar_one_or_none()
                if thumb_img:
                    folders_map[path]['thumbnail_url'] = f"/api/images/{thumb_img.id}/thumbnail?directory_id={dir_id}"
                    folders_map[path]['width'] = thumb_img.width
                    folders_map[path]['height'] = thumb_img.height
                    folders_map[path]['created_at'] = thumb_img.created_at
                    folders_map[path]['directory_id'] = dir_id

        except Exception as e:
            print(f"[Folders] Error querying directory {dir_id}: {e}")
            continue
        finally:
            await dir_db.close()

    # Build sorted folder list (alphabetical by folder name)
    folders = []
    for path, data in folders_map.items():
        folders.append({
            'path': path,
            'name': os.path.basename(path) or path,
            'count': data['count'],
            'thumbnail_url': data['thumbnail_url'],
            'width': data['width'],
            'height': data['height'],
        })

    folders.sort(key=lambda f: f['name'].lower())

    return {
        "folders": folders,
        "total": len(folders)
    }
