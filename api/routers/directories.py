"""
Watch directories router - manage directories to watch for new images
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pathlib import Path
from typing import Optional, List
import shutil
import os
import json

from ..database import get_db, get_data_dir
from ..models import WatchDirectory, ImageFile, TaskType, Image, image_tags
from ..services.task_queue import enqueue_task

router = APIRouter()


class DirectoryCreate(BaseModel):
    path: str
    name: Optional[str] = None
    recursive: bool = True
    auto_tag: bool = True
    auto_age_detect: bool = False


class ParentDirectoryCreate(BaseModel):
    path: str
    recursive: bool = True
    auto_tag: bool = True
    auto_age_detect: bool = False


class DirectoryUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    recursive: Optional[bool] = None
    auto_tag: Optional[bool] = None
    auto_age_detect: Optional[bool] = None
    public_access: Optional[bool] = None  # Allow public network access to this directory


@router.get("")
async def list_directories(request: Request, db: AsyncSession = Depends(get_db)):
    """List all watch directories"""
    access_level = getattr(request.state, 'access_level', 'localhost')

    query = select(WatchDirectory).order_by(WatchDirectory.created_at)

    # Only public internet IPs are filtered - local network gets full access
    if access_level == 'public':
        query = query.where(WatchDirectory.public_access == True)

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
            "auto_age_detect": d.auto_age_detect,
            "image_count": image_count,
            "age_detected_count": age_detected_count,
            "age_detected_pct": round(age_detected_count / image_count * 100, 1) if image_count > 0 else 0,
            "tagged_count": tagged_count,
            "tagged_pct": round(tagged_count / image_count * 100, 1) if image_count > 0 else 0,
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
            "public_access": d.public_access if hasattr(d, 'public_access') else False
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
        auto_age_detect=data.auto_age_detect,
        enabled=True
    )
    db.add(directory)
    await db.commit()
    await db.refresh(directory)

    # Clear any stale ImageFile associations from a previously deleted directory
    # that had the same ID (SQLite can reuse IDs)
    from sqlalchemy import update
    stale_cleanup = await db.execute(
        update(ImageFile)
        .where(ImageFile.watch_directory_id == directory.id)
        .where(~ImageFile.original_path.like(str(path) + '%'))
        .values(watch_directory_id=None)
    )
    if stale_cleanup.rowcount > 0:
        await db.commit()
        print(f"[Directory] Cleared {stale_cleanup.rowcount} stale file associations from reused ID {directory.id}")

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
        if existing.scalar_one_or_none():
            skipped.append(str(subdir))
            continue

        # Create directory record
        directory = WatchDirectory(
            path=str(subdir),
            name=subdir.name,
            recursive=data.recursive,
            auto_tag=data.auto_tag,
            auto_age_detect=data.auto_age_detect,
            enabled=True
        )
        db.add(directory)
        await db.commit()
        await db.refresh(directory)

        # Clear any stale ImageFile associations from reused ID
        from sqlalchemy import update
        stale_cleanup = await db.execute(
            update(ImageFile)
            .where(ImageFile.watch_directory_id == directory.id)
            .where(~ImageFile.original_path.like(str(subdir) + '%'))
            .values(watch_directory_id=None)
        )
        if stale_cleanup.rowcount > 0:
            await db.commit()
            print(f"[Directory] Cleared {stale_cleanup.rowcount} stale associations from reused ID {directory.id}")

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
        from ..services.directory_watcher import directory_watcher
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
        "public_access": directory.public_access if hasattr(directory, 'public_access') else False
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

    await db.commit()

    # Refresh watcher if enabled/recursive changed
    from ..services.directory_watcher import directory_watcher
    await directory_watcher.refresh()

    return {
        "id": directory.id,
        "name": directory.name,
        "enabled": directory.enabled,
        "recursive": directory.recursive,
        "auto_tag": directory.auto_tag,
        "auto_age_detect": directory.auto_age_detect,
        "public_access": directory.public_access if hasattr(directory, 'public_access') else False
    }


@router.delete("/{directory_id}")
async def remove_directory(
    directory_id: int,
    keep_images: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Remove a watch directory. Images are removed from library by default (files on disk are never touched)."""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if keep_images:
        # Just unlink the files from this directory (keep in library)
        from sqlalchemy import update
        await db.execute(
            update(ImageFile)
            .where(ImageFile.watch_directory_id == directory_id)
            .values(watch_directory_id=None)
        )
        images_removed = False
    else:
        # Remove images from library (default) - files on disk are NOT touched
        from ..models import Image
        file_query = select(ImageFile.image_id).where(
            ImageFile.watch_directory_id == directory_id
        )
        file_result = await db.execute(file_query)
        image_ids = [row[0] for row in file_result.all()]

        for image_id in image_ids:
            image = await db.get(Image, image_id)
            if image:
                # Delete thumbnails
                from ..database import get_data_dir
                thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                await db.delete(image)
        images_removed = True

    await db.delete(directory)
    await db.commit()

    # Stop watching this directory
    from ..services.directory_watcher import directory_watcher
    await directory_watcher.remove_directory(directory_id)

    return {"deleted": True, "images_removed": images_removed}


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


class ComfyUIConfigUpdate(BaseModel):
    comfyui_prompt_node_ids: Optional[List[str]] = None
    comfyui_negative_node_ids: Optional[List[str]] = None
    metadata_format: Optional[str] = None  # auto, a1111, comfyui, none


@router.get("/{directory_id}/comfyui-nodes")
async def discover_comfyui_nodes(
    directory_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Scan sample images in directory to discover ComfyUI node types.
    Returns list of nodes with text content for configuration.
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    from ..services.metadata_extractor import discover_comfyui_nodes as discover_nodes

    path = Path(directory.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")

    discovered_nodes = []
    scanned_files = 0
    max_files = 10  # Scan up to 10 files to find examples

    # Scan for PNG/WebP files with ComfyUI metadata
    patterns = ['*.png', '*.PNG', '*.webp', '*.WEBP']
    files_to_scan = []

    for pattern in patterns:
        if directory.recursive:
            files_to_scan.extend(path.rglob(pattern))
        else:
            files_to_scan.extend(path.glob(pattern))

    for img_file in files_to_scan:
        if scanned_files >= max_files:
            break

        try:
            nodes = discover_nodes(str(img_file))
            if nodes:
                # Merge with existing nodes (avoid duplicates by node_id)
                existing_ids = {n['node_id'] for n in discovered_nodes}
                for node in nodes:
                    if node['node_id'] not in existing_ids:
                        discovered_nodes.append(node)
                        existing_ids.add(node['node_id'])
                scanned_files += 1
        except Exception:
            continue

    # Sort by node type for better UX
    discovered_nodes.sort(key=lambda n: (n['node_type'], n['node_id']))

    return {
        "nodes": discovered_nodes,
        "directory_id": directory_id,
        "files_scanned": scanned_files,
        "current_config": {
            "comfyui_prompt_node_ids": json.loads(directory.comfyui_prompt_node_ids) if directory.comfyui_prompt_node_ids else [],
            "comfyui_negative_node_ids": json.loads(directory.comfyui_negative_node_ids) if directory.comfyui_negative_node_ids else [],
            "metadata_format": directory.metadata_format or "auto"
        }
    }


@router.patch("/{directory_id}/comfyui-config")
async def update_comfyui_config(
    directory_id: int,
    config: ComfyUIConfigUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update ComfyUI metadata extraction configuration for a directory"""
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if config.comfyui_prompt_node_ids is not None:
        directory.comfyui_prompt_node_ids = json.dumps(config.comfyui_prompt_node_ids)
    if config.comfyui_negative_node_ids is not None:
        directory.comfyui_negative_node_ids = json.dumps(config.comfyui_negative_node_ids)
    if config.metadata_format is not None:
        directory.metadata_format = config.metadata_format

    await db.commit()

    return {
        "id": directory.id,
        "comfyui_prompt_node_ids": json.loads(directory.comfyui_prompt_node_ids) if directory.comfyui_prompt_node_ids else [],
        "comfyui_negative_node_ids": json.loads(directory.comfyui_negative_node_ids) if directory.comfyui_negative_node_ids else [],
        "metadata_format": directory.metadata_format or "auto"
    }


@router.post("/{directory_id}/reextract-metadata")
async def reextract_directory_metadata(
    directory_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-extract AI generation metadata for all images in this directory.
    Useful after updating ComfyUI node configuration.
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    # Get all image files in this directory
    query = (
        select(ImageFile)
        .options(selectinload(ImageFile.image))
        .where(ImageFile.watch_directory_id == directory_id)
    )
    result = await db.execute(query)
    image_files = result.scalars().all()

    # Get ComfyUI config
    comfyui_prompt_node_ids = []
    comfyui_negative_node_ids = []
    if directory.comfyui_prompt_node_ids:
        try:
            comfyui_prompt_node_ids = json.loads(directory.comfyui_prompt_node_ids)
        except Exception:
            pass
    if directory.comfyui_negative_node_ids:
        try:
            comfyui_negative_node_ids = json.loads(directory.comfyui_negative_node_ids)
        except Exception:
            pass

    queued = 0
    for image_file in image_files:
        if not image_file.image:
            continue

        if not Path(image_file.original_path).exists():
            continue

        await enqueue_task(
            TaskType.extract_metadata,
            {
                'image_id': image_file.image_id,
                'image_path': image_file.original_path,
                'comfyui_prompt_node_ids': comfyui_prompt_node_ids,
                'comfyui_negative_node_ids': comfyui_negative_node_ids,
                'format_hint': directory.metadata_format or 'auto'
            },
            priority=0,
            db=db
        )
        queued += 1

    await db.commit()

    return {
        "queued": queued,
        "directory_id": directory_id,
        "message": f"Queued metadata re-extraction for {queued} images"
    }
