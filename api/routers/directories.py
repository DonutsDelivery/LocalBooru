"""
Watch directories router - manage directories to watch for new images
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pathlib import Path
from typing import Optional
import shutil
import os

from ..database import get_db, get_data_dir
from ..models import WatchDirectory, ImageFile, TaskType, Image, image_tags
from ..services.task_queue import enqueue_task

router = APIRouter()


class DirectoryCreate(BaseModel):
    path: str
    name: Optional[str] = None
    recursive: bool = True
    auto_tag: bool = True


class DirectoryUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    recursive: Optional[bool] = None
    auto_tag: Optional[bool] = None


@router.get("")
async def list_directories(db: AsyncSession = Depends(get_db)):
    """List all watch directories"""
    query = select(WatchDirectory).order_by(WatchDirectory.created_at)
    result = await db.execute(query)
    directories = result.scalars().all()

    # Get image counts and diagnostics per directory
    dir_data = []
    for d in directories:
        # Total image count
        count_query = select(func.count(ImageFile.id)).where(
            ImageFile.watch_directory_id == d.id
        )
        count_result = await db.execute(count_query)
        image_count = count_result.scalar() or 0

        # Age detected count (images with num_faces not null)
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

        # Tagged count (images with at least one tag)
        tagged_query = (
            select(func.count(func.distinct(Image.id)))
            .join(ImageFile, ImageFile.image_id == Image.id)
            .join(image_tags, image_tags.c.image_id == Image.id)
            .where(ImageFile.watch_directory_id == d.id)
        )
        tagged_result = await db.execute(tagged_query)
        tagged_count = tagged_result.scalar() or 0

        # Favorited count
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
            "image_count": image_count,
            "age_detected_count": age_detected_count,
            "age_detected_pct": round(age_detected_count / image_count * 100, 1) if image_count > 0 else 0,
            "tagged_count": tagged_count,
            "tagged_pct": round(tagged_count / image_count * 100, 1) if image_count > 0 else 0,
            "favorited_count": favorited_count,
            "favorited_pct": round(favorited_count / image_count * 100, 1) if image_count > 0 else 0,
            "path_exists": path_exists,
            "last_scanned_at": d.last_scanned_at.isoformat() if d.last_scanned_at else None,
            "created_at": d.created_at.isoformat() if d.created_at else None
        })

    return {"directories": dir_data}


@router.post("")
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

    # Create directory record
    directory = WatchDirectory(
        path=str(path),
        name=data.name or path.name,
        recursive=data.recursive,
        auto_tag=data.auto_tag,
        enabled=True
    )
    db.add(directory)
    await db.commit()
    await db.refresh(directory)

    # Queue initial scan
    await enqueue_task(
        TaskType.scan_directory,
        {
            'directory_id': directory.id,
            'directory_path': str(path)
        },
        priority=2,  # High priority for new directories
        db=db
    )

    # Start watching this directory
    from ..services.directory_watcher import directory_watcher
    await directory_watcher.add_directory(directory)

    return {
        "id": directory.id,
        "path": directory.path,
        "name": directory.name,
        "message": "Directory added, initial scan queued, watching for changes"
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
        "image_count": image_count,
        "path_exists": Path(directory.path).exists(),
        "last_scanned_at": directory.last_scanned_at.isoformat() if directory.last_scanned_at else None,
        "created_at": directory.created_at.isoformat() if directory.created_at else None
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

    await db.commit()

    # Refresh watcher if enabled/recursive changed
    from ..services.directory_watcher import directory_watcher
    await directory_watcher.refresh()

    return {
        "id": directory.id,
        "name": directory.name,
        "enabled": directory.enabled,
        "recursive": directory.recursive,
        "auto_tag": directory.auto_tag
    }


@router.delete("/{directory_id}")
async def remove_directory(
    directory_id: int,
    remove_images: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Remove a watch directory"""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if remove_images:
        # Delete all images from this directory
        # This will cascade to ImageFile records
        from ..models import Image
        file_query = select(ImageFile.image_id).where(
            ImageFile.watch_directory_id == directory_id
        )
        file_result = await db.execute(file_query)
        image_ids = [row[0] for row in file_result.all()]

        for image_id in image_ids:
            image = await db.get(Image, image_id)
            if image:
                await db.delete(image)
    else:
        # Just unlink the files from this directory
        from sqlalchemy import update
        await db.execute(
            update(ImageFile)
            .where(ImageFile.watch_directory_id == directory_id)
            .values(watch_directory_id=None)
        )

    await db.delete(directory)
    await db.commit()

    # Stop watching this directory
    from ..services.directory_watcher import directory_watcher
    await directory_watcher.remove_directory(directory_id)

    return {"deleted": True, "images_removed": remove_images}


@router.post("/{directory_id}/scan")
async def scan_directory(directory_id: int, db: AsyncSession = Depends(get_db)):
    """Manually trigger a directory scan"""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if not Path(directory.path).exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")

    # Queue scan task
    task = await enqueue_task(
        TaskType.scan_directory,
        {
            'directory_id': directory.id,
            'directory_path': directory.path
        },
        priority=2,
        db=db
    )

    return {
        "message": "Scan queued",
        "task_id": task.id
    }


class PruneRequest(BaseModel):
    dumpster_path: Optional[str] = None  # Custom dumpster path, defaults to ~/.localbooru/dumpster


@router.post("/{directory_id}/prune")
async def prune_directory(
    directory_id: int,
    data: PruneRequest = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Move all non-favorited images from this directory to a dumpster folder.
    Preserves folder structure to avoid filename conflicts.
    Removes pruned images from the library.
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if not Path(directory.path).exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")

    # Determine dumpster path
    if data and data.dumpster_path:
        dumpster_base = Path(data.dumpster_path)
    else:
        dumpster_base = get_data_dir() / "dumpster"

    # Create subdirectory named after the watch directory
    dir_name = Path(directory.path).name
    dumpster_dir = dumpster_base / dir_name
    dumpster_dir.mkdir(parents=True, exist_ok=True)

    # Get all non-favorited images from this directory
    query = (
        select(ImageFile)
        .join(Image, ImageFile.image_id == Image.id)
        .options(selectinload(ImageFile.image))
        .where(
            ImageFile.watch_directory_id == directory_id,
            Image.is_favorite == False
        )
    )
    result = await db.execute(query)
    image_files = result.scalars().all()

    moved_count = 0
    failed_count = 0
    removed_images = []

    for image_file in image_files:
        original_path = Path(image_file.original_path)

        if not original_path.exists():
            continue

        # Calculate relative path from watch directory
        try:
            relative_path = original_path.relative_to(directory.path)
        except ValueError:
            # File is not under the watch directory (shouldn't happen)
            relative_path = original_path.name

        # Create destination path preserving structure
        dest_path = dumpster_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle filename conflicts
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            # Move the file
            shutil.move(str(original_path), str(dest_path))
            moved_count += 1

            # Track image for removal
            removed_images.append(image_file.image_id)

            # Delete thumbnail
            thumbnail_path = get_data_dir() / 'thumbnails' / f"{image_file.image.file_hash[:16]}.webp"
            if thumbnail_path.exists():
                thumbnail_path.unlink()

        except Exception as e:
            print(f"[Prune] Failed to move {original_path}: {e}")
            failed_count += 1

    # Remove images from database (unique image IDs only)
    unique_image_ids = set(removed_images)
    for image_id in unique_image_ids:
        image = await db.get(Image, image_id)
        if image:
            await db.delete(image)

    await db.commit()

    return {
        "pruned": moved_count,
        "failed": failed_count,
        "removed_from_library": len(unique_image_ids),
        "dumpster_path": str(dumpster_dir)
    }
