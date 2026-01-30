"""
Library statistics endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db, directory_db_manager
from ...models import (
    Image, Tag, ImageFile, WatchDirectory, TaskQueue,
    TaskStatus, Rating,
    DirectoryImage, DirectoryImageFile
)

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
            try:
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
            except Exception as e:
                # Skip directories with corrupted/uninitialized databases
                print(f"[LibraryStats] Skipping directory {dir_id}: {e}")
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
        from ...models import FileStatus
        for dir_id in all_dir_ids:
            if not directory_db_manager.db_exists(dir_id):
                continue
            try:
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
            except Exception:
                # Skip directories with corrupted/uninitialized databases
                pass
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


@router.get("/untagged")
async def list_untagged(db: AsyncSession = Depends(get_db)):
    """Get count of images without tags"""
    from ...models import image_tags

    # Images with no tags
    subq = select(image_tags.c.image_id).distinct()
    untagged_query = select(func.count(Image.id)).where(Image.id.not_in(subq))
    result = await db.execute(untagged_query)
    count = result.scalar()

    return {"untagged_count": count}
