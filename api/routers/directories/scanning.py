"""
Directory scanning and rescan endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pathlib import Path
import json

from ...database import get_db, directory_db_manager
from ...models import WatchDirectory, ImageFile, TaskType
from ...services.task_queue import enqueue_task
from .models import ScanOptions, ComfyUIConfigUpdate

router = APIRouter()


@router.post("/{directory_id}/scan")
async def scan_directory(
    directory_id: int,
    options: ScanOptions = None,
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger a directory scan.

    By default, this only imports new files. Set clean_deleted=true to also
    remove references to deleted files (slower).
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if not Path(directory.path).exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")

    clean_deleted = options.clean_deleted if options else False

    # Queue scan task
    task = await enqueue_task(
        TaskType.scan_directory,
        {
            'directory_id': directory.id,
            'directory_path': directory.path,
            'clean_deleted': clean_deleted
        },
        priority=2,
        db=db
    )

    return {
        "message": "Scan queued",
        "task_id": task.id,
        "clean_deleted": clean_deleted
    }


@router.post("/{directory_id}/clean-deleted")
async def clean_deleted_files(directory_id: int, db: AsyncSession = Depends(get_db)):
    """Remove references to files that no longer exist on disk.

    This is a separate operation from scanning because it requires checking
    every file in the database against the filesystem.
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    from ...services.file_tracker import _clean_deleted_files

    removed = await _clean_deleted_files(directory_id)

    return {
        "removed": removed,
        "directory_id": directory_id,
        "message": f"Removed {removed} references to deleted files"
    }


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

    from ...services.metadata_extractor import discover_comfyui_nodes as discover_nodes

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
