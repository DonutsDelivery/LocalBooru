"""
CRUD operations for watch directories - list, create, read, update, delete
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from typing import List
import asyncio
import json

from ...database import get_db, get_data_dir, directory_db_manager
from ...models import (
    WatchDirectory, ImageFile, TaskType, Image, image_tags, Tag,
    DirectoryImage, DirectoryImageFile, directory_image_tags, FileStatus,
    TaskQueue, TaskStatus
)
from ...services.task_queue import enqueue_task
from .models import DirectoryCreate, DirectoryUpdate, ParentDirectoryCreate, BulkDeleteRequest

router = APIRouter()


@router.get("/")
async def list_directories(request: Request, db: AsyncSession = Depends(get_db)):
    """List all watch directories"""
    access_level = getattr(request.state, 'access_level', 'localhost')

    query = select(WatchDirectory).order_by(WatchDirectory.created_at)

    # Only public internet IPs are filtered - local network gets full access
    if access_level == 'public':
        query = query.where(WatchDirectory.public_access == True)

    result = await db.execute(query)
    directories = result.scalars().all()

    # Get pending tag tasks per directory
    pending_tag_tasks = {}
    pending_query = select(TaskQueue.payload).where(
        TaskQueue.task_type == TaskType.tag,
        TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing])
    )
    pending_result = await db.execute(pending_query)
    for row in pending_result.scalars().all():
        try:
            payload = json.loads(row)
            dir_id = payload.get('directory_id')
            if dir_id:
                pending_tag_tasks[dir_id] = pending_tag_tasks.get(dir_id, 0) + 1
        except:
            pass

    # Get image counts and diagnostics per directory
    dir_data = []
    for d in directories:
        image_count = 0
        age_detected_count = 0
        tagged_count = 0
        favorited_count = 0

        # Check if this directory uses per-directory database
        if directory_db_manager.db_exists(d.id):
            # NEW ARCHITECTURE: Query per-directory database
            dir_db = await directory_db_manager.get_session(d.id)
            try:
                # Total image count
                count_result = await dir_db.execute(select(func.count(DirectoryImage.id)))
                image_count = count_result.scalar() or 0

                # Age detected count
                age_result = await dir_db.execute(
                    select(func.count(DirectoryImage.id)).where(DirectoryImage.num_faces.isnot(None))
                )
                age_detected_count = age_result.scalar() or 0

                # Tagged count (images with at least one tag)
                tagged_result = await dir_db.execute(
                    select(func.count(func.distinct(directory_image_tags.c.image_id)))
                )
                tagged_count = tagged_result.scalar() or 0

                # Favorited count
                fav_result = await dir_db.execute(
                    select(func.count(DirectoryImage.id)).where(DirectoryImage.is_favorite == True)
                )
                favorited_count = fav_result.scalar() or 0
            finally:
                await dir_db.close()
        else:
            # LEGACY: Query main database
            count_query = select(func.count(ImageFile.id)).where(
                ImageFile.watch_directory_id == d.id
            )
            count_result = await db.execute(count_query)
            image_count = count_result.scalar() or 0

            age_query = (
                select(func.count(Image.id))
                .join(ImageFile, ImageFile.image_id == Image.id)
                .where(
                    ImageFile.watch_directory_id == d.id,
                    Image.num_faces.isnot(None)
                )
            )
            age_result = await db.execute(age_query)
            age_detected_count = age_result.scalar() or 0

            tagged_query = (
                select(func.count(func.distinct(Image.id)))
                .join(ImageFile, ImageFile.image_id == Image.id)
                .join(image_tags, image_tags.c.image_id == Image.id)
                .where(ImageFile.watch_directory_id == d.id)
            )
            tagged_result = await db.execute(tagged_query)
            tagged_count = tagged_result.scalar() or 0

            fav_query = (
                select(func.count(Image.id))
                .join(ImageFile, ImageFile.image_id == Image.id)
                .where(
                    ImageFile.watch_directory_id == d.id,
                    Image.is_favorite == True
                )
            )
            fav_result = await db.execute(fav_query)
            favorited_count = fav_result.scalar() or 0

        # Check if path exists
        path_exists = Path(d.path).exists()

        dir_data.append({
            "id": d.id,
            "path": d.path,
            "name": d.name or Path(d.path).name,
            "enabled": d.enabled,
            "recursive": d.recursive,
            "auto_tag": d.auto_tag,
            "auto_age_detect": d.auto_age_detect,
            "image_count": image_count,
            "age_detected_count": age_detected_count,
            "age_detected_pct": round(age_detected_count / image_count * 100, 1) if image_count > 0 else 0,
            "tagged_count": tagged_count,
            "tagged_pct": round(tagged_count / image_count * 100, 1) if image_count > 0 else 0,
            "pending_tag_tasks": pending_tag_tasks.get(d.id, 0),
            "favorited_count": favorited_count,
            "favorited_pct": round(favorited_count / image_count * 100, 1) if image_count > 0 else 0,
            "path_exists": path_exists,
            "last_scanned_at": d.last_scanned_at.isoformat() if d.last_scanned_at else None,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            # ComfyUI metadata config
            "comfyui_prompt_node_ids": json.loads(d.comfyui_prompt_node_ids) if d.comfyui_prompt_node_ids else [],
            "comfyui_negative_node_ids": json.loads(d.comfyui_negative_node_ids) if d.comfyui_negative_node_ids else [],
            "metadata_format": d.metadata_format or "auto",
            # Network access
            "public_access": d.public_access if hasattr(d, 'public_access') else False,
            # Media type filtering
            "show_images": d.show_images if hasattr(d, 'show_images') else True,
            "show_videos": d.show_videos if hasattr(d, 'show_videos') else True,
            # Architecture indicator
            "uses_per_directory_db": directory_db_manager.db_exists(d.id)
        })

    return {"directories": dir_data}


@router.post("/")
async def add_directory(data: DirectoryCreate, db: AsyncSession = Depends(get_db)):
    """Add a new watch directory"""
    # Validate path exists
    path = Path(data.path).resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {data.path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {data.path}")

    # Check for duplicates
    existing = await db.execute(
        select(WatchDirectory).where(WatchDirectory.path == str(path))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Directory already added")

    # Create directory record in main DB
    directory = WatchDirectory(
        path=str(path),
        name=data.name or path.name,
        recursive=data.recursive,
        auto_tag=data.auto_tag,
        auto_age_detect=data.auto_age_detect,
        enabled=True
    )
    db.add(directory)
    await db.commit()
    await db.refresh(directory)

    # Create per-directory database
    await directory_db_manager.create_directory_db(directory.id)
    print(f"[Directory] Created per-directory database for {directory.id}")

    # Queue initial scan
    await enqueue_task(
        TaskType.scan_directory,
        {
            'directory_id': directory.id,
            'directory_path': str(path)
        },
        priority=2,
        db=db
    )

    # Start watching this directory
    from ...services.directory_watcher import directory_watcher
    await directory_watcher.add_directory(directory)

    return {
        "id": directory.id,
        "path": directory.path,
        "name": directory.name,
        "message": "Directory added, initial scan queued, watching for changes"
    }


@router.post("/add-parent")
async def add_parent_directory(data: ParentDirectoryCreate, db: AsyncSession = Depends(get_db)):
    """Add all immediate subdirectories of a parent folder as watch directories"""
    # Validate parent path exists
    parent_path = Path(data.path).resolve()
    if not parent_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {data.path}")
    if not parent_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {data.path}")

    # Get immediate subdirectories (not recursive)
    subdirs = [d for d in parent_path.iterdir() if d.is_dir()]

    if not subdirs:
        raise HTTPException(status_code=400, detail="No subdirectories found in the selected folder")

    added = []
    skipped = []

    for subdir in sorted(subdirs):
        # Check for duplicates
        existing = await db.execute(
            select(WatchDirectory).where(WatchDirectory.path == str(subdir))
        )
        existing_dir = existing.scalar_one_or_none()
        if existing_dir:
            # Update parent_path on existing directory so parent watch covers it
            if not existing_dir.parent_path:
                existing_dir.parent_path = str(parent_path)
                await db.commit()
            skipped.append(str(subdir))
            continue

        # Create directory record
        directory = WatchDirectory(
            path=str(subdir),
            name=subdir.name,
            recursive=data.recursive,
            auto_tag=data.auto_tag,
            auto_age_detect=data.auto_age_detect,
            parent_path=str(parent_path),
            enabled=True
        )
        db.add(directory)
        await db.commit()
        await db.refresh(directory)

        # Create per-directory database
        await directory_db_manager.create_directory_db(directory.id)

        # Queue initial scan
        await enqueue_task(
            TaskType.scan_directory,
            {
                'directory_id': directory.id,
                'directory_path': str(subdir)
            },
            priority=2,
            db=db
        )

        # Start watching this directory
        from ...services.directory_watcher import directory_watcher
        await directory_watcher.add_directory(directory)

        added.append({
            "id": directory.id,
            "path": directory.path,
            "name": directory.name
        })

    return {
        "added": added,
        "skipped": skipped,
        "message": f"Added {len(added)} directories, skipped {len(skipped)} existing"
    }


@router.get("/{directory_id}")
async def get_directory(directory_id: int, db: AsyncSession = Depends(get_db)):
    """Get directory details"""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    # Count images
    count_query = select(func.count(ImageFile.id)).where(
        ImageFile.watch_directory_id == directory_id
    )
    count_result = await db.execute(count_query)
    image_count = count_result.scalar()

    return {
        "id": directory.id,
        "path": directory.path,
        "name": directory.name or Path(directory.path).name,
        "enabled": directory.enabled,
        "recursive": directory.recursive,
        "auto_tag": directory.auto_tag,
        "auto_age_detect": directory.auto_age_detect,
        "image_count": image_count,
        "path_exists": Path(directory.path).exists(),
        "last_scanned_at": directory.last_scanned_at.isoformat() if directory.last_scanned_at else None,
        "created_at": directory.created_at.isoformat() if directory.created_at else None,
        # ComfyUI metadata config
        "comfyui_prompt_node_ids": json.loads(directory.comfyui_prompt_node_ids) if directory.comfyui_prompt_node_ids else [],
        "comfyui_negative_node_ids": json.loads(directory.comfyui_negative_node_ids) if directory.comfyui_negative_node_ids else [],
        "metadata_format": directory.metadata_format or "auto",
        # Network access
        "public_access": directory.public_access if hasattr(directory, 'public_access') else False,
        # Media type filtering
        "show_images": directory.show_images if hasattr(directory, 'show_images') else True,
        "show_videos": directory.show_videos if hasattr(directory, 'show_videos') else True
    }


@router.patch("/{directory_id}")
async def update_directory(
    directory_id: int,
    data: DirectoryUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update directory settings"""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if data.name is not None:
        directory.name = data.name
    if data.enabled is not None:
        directory.enabled = data.enabled
    if data.recursive is not None:
        directory.recursive = data.recursive
    if data.auto_tag is not None:
        directory.auto_tag = data.auto_tag
    if data.auto_age_detect is not None:
        directory.auto_age_detect = data.auto_age_detect
    if data.public_access is not None:
        directory.public_access = data.public_access
    if data.show_images is not None:
        directory.show_images = data.show_images
    if data.show_videos is not None:
        directory.show_videos = data.show_videos

    await db.commit()

    # Refresh watcher if enabled/recursive changed
    from ...services.directory_watcher import directory_watcher
    await directory_watcher.refresh()

    return {
        "id": directory.id,
        "name": directory.name,
        "enabled": directory.enabled,
        "recursive": directory.recursive,
        "auto_tag": directory.auto_tag,
        "auto_age_detect": directory.auto_age_detect,
        "public_access": directory.public_access if hasattr(directory, 'public_access') else False,
        "show_images": directory.show_images if hasattr(directory, 'show_images') else True,
        "show_videos": directory.show_videos if hasattr(directory, 'show_videos') else True
    }


@router.post("/bulk-delete")
async def bulk_delete_directories(
    data: BulkDeleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Remove multiple watch directories in one efficient operation."""
    return await _delete_directories(data.directory_ids, data.keep_images, db)


@router.delete("/{directory_id}")
async def remove_directory(
    directory_id: int,
    keep_images: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Remove a single watch directory. For bulk operations, use POST /directories/bulk-delete instead."""
    return await _delete_directories([directory_id], keep_images, db)


async def _delete_directories(
    directory_ids: List[int],
    keep_images: bool,
    db: AsyncSession
):
    """
    Internal: efficiently delete multiple directories.

    With per-directory databases, deletion is now simple:
    1. Get tag counts from directory DB and decrement in main DB
    2. Delete the directory database file (instant!)
    3. Remove the WatchDirectory record from main DB
    """
    if not directory_ids:
        return {"deleted": 0, "images_removed": False, "image_count": 0}

    # Pause task queue to prevent conflicts
    from ...services.task_queue import task_queue
    was_paused = task_queue.paused
    if not was_paused:
        task_queue.pause()
        await asyncio.sleep(0.5)

    try:
        # Verify directories exist
        dir_query = select(WatchDirectory).where(WatchDirectory.id.in_(directory_ids))
        dir_result = await db.execute(dir_query)
        directories = dir_result.scalars().all()

        if not directories:
            raise HTTPException(status_code=404, detail="No directories found")

        found_ids = [d.id for d in directories]
        print(f"[Directory] Bulk delete starting for {len(found_ids)} directories (keep_images={keep_images})")

        total_images = 0
        all_file_hashes = []
        images_removed = False

        for dir_id in found_ids:
            # Check if this directory uses the new per-directory database
            if directory_db_manager.db_exists(dir_id):
                # NEW ARCHITECTURE: Per-directory database
                print(f"[Directory] Deleting directory {dir_id} (per-directory DB)")

                # Get tag counts and file hashes before deletion
                dir_db = await directory_db_manager.get_session(dir_id)
                try:
                    # Get image count
                    count_result = await dir_db.execute(select(func.count(DirectoryImage.id)))
                    image_count = count_result.scalar() or 0
                    total_images += image_count

                    # Get file hashes for thumbnail cleanup
                    hash_result = await dir_db.execute(select(DirectoryImage.file_hash))
                    all_file_hashes.extend([row[0] for row in hash_result.all() if row[0]])

                    if not keep_images:
                        # Get tag counts to decrement in main DB
                        tag_counts_query = (
                            select(directory_image_tags.c.tag_id, func.count(directory_image_tags.c.image_id))
                            .group_by(directory_image_tags.c.tag_id)
                        )
                        tag_counts_result = await dir_db.execute(tag_counts_query)
                        tag_counts = {row[0]: row[1] for row in tag_counts_result.all()}

                        # Decrement tag post_counts in main DB
                        for tag_id, count in tag_counts.items():
                            tag = await db.get(Tag, tag_id)
                            if tag:
                                tag.post_count = max(0, tag.post_count - count)
                        await db.commit()

                        images_removed = True
                finally:
                    await dir_db.close()

                # DELETE THE DATABASE FILE - instant!
                await directory_db_manager.delete_directory_db(dir_id)

            else:
                # LEGACY: Images in main database
                print(f"[Directory] Deleting directory {dir_id} (legacy main DB)")

                if keep_images:
                    await db.execute(
                        update(ImageFile)
                        .where(ImageFile.watch_directory_id == dir_id)
                        .values(watch_directory_id=None)
                    )
                    await db.commit()
                else:
                    # Get image_ids and file_hashes
                    file_query = select(ImageFile.image_id, Image.file_hash).join(
                        Image, ImageFile.image_id == Image.id
                    ).where(ImageFile.watch_directory_id == dir_id)
                    file_result = await db.execute(file_query)
                    rows = file_result.all()

                    image_ids = list(set(row[0] for row in rows if row[0] is not None))
                    file_hashes = [row[1] for row in rows if row[1] is not None]
                    all_file_hashes.extend(file_hashes)
                    total_images += len(image_ids)

                    if image_ids:
                        await db.execute(
                            update(ImageFile)
                            .where(ImageFile.watch_directory_id == dir_id)
                            .values(watch_directory_id=None)
                        )
                        await db.commit()

                        # Delete images in batches
                        img_batch_size = 500
                        for i in range(0, len(image_ids), img_batch_size):
                            img_batch = image_ids[i:i + img_batch_size]
                            await db.execute(delete(Image).where(Image.id.in_(img_batch)))
                            await db.commit()

                    images_removed = True

            # Delete directory record from main DB
            await db.execute(delete(WatchDirectory).where(WatchDirectory.id == dir_id))
            await db.commit()

        print(f"[Directory] Deleted {len(found_ids)} directories, {total_images} images")

        # Stop watching deleted directories
        from ...services.directory_watcher import directory_watcher
        for dir_id in found_ids:
            try:
                await directory_watcher.remove_directory(dir_id)
            except Exception:
                pass

        # Delete thumbnails (best effort)
        if all_file_hashes:
            thumbnails_dir = get_data_dir() / 'thumbnails'
            for file_hash in set(all_file_hashes):
                if file_hash:
                    thumb_path = thumbnails_dir / f"{file_hash[:16]}.webp"
                    try:
                        if thumb_path.exists():
                            thumb_path.unlink()
                    except Exception:
                        pass
            print(f"[Directory] Cleaned up thumbnails")

        return {
            "deleted": len(found_ids),
            "images_removed": images_removed,
            "image_count": total_images
        }

    except Exception as e:
        await db.rollback()
        print(f"[Directory] Bulk delete failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if not was_paused:
            task_queue.resume()
