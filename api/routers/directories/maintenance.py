"""
Directory maintenance operations - prune, repair, relocate, verify
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pathlib import Path
import shutil
import os

from ...database import get_db, get_data_dir, directory_db_manager
from ...models import (
    WatchDirectory, ImageFile, Image,
    DirectoryImage, DirectoryImageFile, FileStatus
)
from .models import BulkVerifyRequest, PruneRequest, DirectoryPathUpdate

router = APIRouter()


@router.post("/{directory_id}/repair")
async def repair_directory_paths(directory_id: int, db: AsyncSession = Depends(get_db)):
    """Fix all file path issues in a directory.

    1. Builds filename->path mapping from filesystem (fast)
    2. Fixes paths for files that were moved (by filename match)
    3. Removes records for files that are truly gone
    """
    from ...services.file_tracker import is_media_file

    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    if not Path(directory.path).exists():
        raise HTTPException(status_code=400, detail="Directory path does not exist")

    if not directory_db_manager.db_exists(directory_id):
        raise HTTPException(status_code=400, detail="Directory database does not exist")

    # Step 1: Scan filesystem and build filename -> path mapping (fast - no hashing)
    # Use os.walk to include hidden directories (rglob skips them)
    dir_path = Path(directory.path)
    name_to_path = {}

    if directory.recursive:
        for root, dirs, files in os.walk(dir_path):
            for fname in files:
                fpath = Path(root) / fname
                if is_media_file(fpath):
                    name_to_path[fname] = str(fpath)
    else:
        for fpath in dir_path.iterdir():
            if fpath.is_file() and is_media_file(fpath):
                name_to_path[fpath.name] = str(fpath)

    # Step 2: Check each DB record and fix/remove
    dir_db = await directory_db_manager.get_session(directory_id)
    repaired = 0
    valid = 0
    removed = 0

    try:
        query = select(DirectoryImageFile)
        result = await dir_db.execute(query)
        file_records = result.scalars().all()

        for file_record in file_records:
            current_path = Path(file_record.original_path)

            if current_path.exists():
                file_record.file_exists = True
                file_record.file_status = FileStatus.available
                valid += 1
                continue

            # Path is invalid - try to find by filename
            filename = current_path.name
            if filename in name_to_path:
                file_record.original_path = name_to_path[filename]
                file_record.file_exists = True
                file_record.file_status = FileStatus.available
                repaired += 1
            else:
                # Can't find file - remove record
                image = await dir_db.get(DirectoryImage, file_record.image_id)

                other_files = await dir_db.execute(
                    select(DirectoryImageFile).where(
                        DirectoryImageFile.image_id == file_record.image_id,
                        DirectoryImageFile.id != file_record.id
                    )
                )
                has_other_refs = other_files.scalar_one_or_none() is not None

                await dir_db.delete(file_record)

                if not has_other_refs and image:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await dir_db.delete(image)

                removed += 1

        await dir_db.commit()
    finally:
        await dir_db.close()

    return {
        "directory_id": directory_id,
        "files_on_disk": len(name_to_path),
        "valid": valid,
        "repaired": repaired,
        "removed": removed,
        "message": f"Fixed {repaired} paths, {valid} already valid, removed {removed} missing"
    }


@router.post("/{directory_id}/verify")
async def verify_directory_files_endpoint(directory_id: int, db: AsyncSession = Depends(get_db)):
    """Verify all files in a directory still exist at their recorded paths.

    - Files that exist are marked as available
    - Missing files are deleted from the database
    - Offline drives are detected and files are marked accordingly
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    from ...services.file_tracker import verify_directory_files

    stats = await verify_directory_files(directory_id, db)

    return {
        "directory_id": directory_id,
        "directory_name": directory.name or Path(directory.path).name,
        **stats,
        "message": f"Verified {stats['verified']} files, deleted {stats['deleted']} missing"
    }


@router.post("/bulk-verify")
async def bulk_verify_directories(data: BulkVerifyRequest, db: AsyncSession = Depends(get_db)):
    """Verify files in multiple directories at once.

    For each directory:
    - Files that exist are marked as available
    - Missing files are deleted from the database
    """
    from ...services.file_tracker import verify_directory_files

    if not data.directory_ids:
        return {"results": [], "totals": {"verified": 0, "deleted": 0, "drive_offline": 0}}

    results = []
    totals = {"verified": 0, "deleted": 0, "drive_offline": 0}

    for directory_id in data.directory_ids:
        directory = await db.get(WatchDirectory, directory_id)
        if not directory:
            continue

        stats = await verify_directory_files(directory_id, db)

        results.append({
            "directory_id": directory_id,
            "directory_name": directory.name or Path(directory.path).name,
            **stats
        })

        for key in totals:
            totals[key] += stats.get(key, 0)

    return {
        "results": results,
        "totals": totals,
        "message": f"Verified {len(results)} directories: {totals['verified']} files OK, {totals['deleted']} deleted"
    }


@router.post("/bulk-repair")
async def bulk_repair_directories(data: BulkVerifyRequest, db: AsyncSession = Depends(get_db)):
    """Repair file paths in multiple directories at once.

    For each directory:
    - Builds filename->path mapping from filesystem
    - Fixes paths for files that were moved (by filename match)
    - Removes records for files that are truly gone
    """
    from ...services.file_tracker import is_media_file

    if not data.directory_ids:
        return {"results": [], "totals": {"valid": 0, "repaired": 0, "removed": 0}}

    results = []
    totals = {"valid": 0, "repaired": 0, "removed": 0}

    for directory_id in data.directory_ids:
        directory = await db.get(WatchDirectory, directory_id)
        if not directory:
            continue

        if not Path(directory.path).exists():
            continue

        if not directory_db_manager.db_exists(directory_id):
            continue

        # Scan filesystem and build filename -> path mapping
        # Use os.walk to include hidden directories (rglob skips them)
        dir_path = Path(directory.path)
        name_to_path = {}

        if directory.recursive:
            for root, dirs, files in os.walk(dir_path):
                for fname in files:
                    fpath = Path(root) / fname
                    if is_media_file(fpath):
                        name_to_path[fname] = str(fpath)
        else:
            for fpath in dir_path.iterdir():
                if fpath.is_file() and is_media_file(fpath):
                    name_to_path[fpath.name] = str(fpath)

        # Check each DB record and fix/remove
        dir_db = await directory_db_manager.get_session(directory_id)
        repaired = 0
        valid = 0
        removed = 0

        try:
            query = select(DirectoryImageFile)
            result = await dir_db.execute(query)
            file_records = result.scalars().all()

            for file_record in file_records:
                current_path = Path(file_record.original_path)

                if current_path.exists():
                    file_record.file_exists = True
                    file_record.file_status = FileStatus.available
                    valid += 1
                    continue

                # Path is invalid - try to find by filename
                filename = current_path.name
                if filename in name_to_path:
                    file_record.original_path = name_to_path[filename]
                    file_record.file_exists = True
                    file_record.file_status = FileStatus.available
                    repaired += 1
                else:
                    # Can't find file - remove record
                    image = await dir_db.get(DirectoryImage, file_record.image_id)

                    other_files = await dir_db.execute(
                        select(DirectoryImageFile).where(
                            DirectoryImageFile.image_id == file_record.image_id,
                            DirectoryImageFile.id != file_record.id
                        )
                    )
                    has_other_refs = other_files.scalar_one_or_none() is not None

                    await dir_db.delete(file_record)

                    if not has_other_refs and image:
                        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                        if thumbnail_path.exists():
                            thumbnail_path.unlink()
                        await dir_db.delete(image)

                    removed += 1

            await dir_db.commit()
        finally:
            await dir_db.close()

        results.append({
            "directory_id": directory_id,
            "directory_name": directory.name or Path(directory.path).name,
            "valid": valid,
            "repaired": repaired,
            "removed": removed
        })

        totals["valid"] += valid
        totals["repaired"] += repaired
        totals["removed"] += removed

    return {
        "results": results,
        "totals": totals,
        "message": f"Repaired {len(results)} directories: {totals['valid']} OK, {totals['repaired']} fixed, {totals['removed']} removed"
    }


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


@router.patch("/{directory_id}/path")
async def update_directory_path(
    directory_id: int,
    data: DirectoryPathUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a directory's path when it has been moved to a new location.
    Updates all associated ImageFile paths to point to the new location.
    """
    directory = await db.get(WatchDirectory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")

    # Validate new path exists
    new_path = Path(data.new_path).resolve()
    if not new_path.exists():
        raise HTTPException(status_code=400, detail=f"New path does not exist: {data.new_path}")
    if not new_path.is_dir():
        raise HTTPException(status_code=400, detail=f"New path is not a directory: {data.new_path}")

    # Check if new path is already used by another directory
    existing = await db.execute(
        select(WatchDirectory).where(
            WatchDirectory.path == str(new_path),
            WatchDirectory.id != directory_id
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="New path is already used by another directory")

    old_path = directory.path
    old_path_normalized = old_path.rstrip('/\\')
    new_path_str = str(new_path)

    # Update all ImageFile paths that belong to this directory
    # Get all image files in this directory
    file_query = select(ImageFile).where(ImageFile.watch_directory_id == directory_id)
    file_result = await db.execute(file_query)
    image_files = file_result.scalars().all()

    updated_count = 0
    for image_file in image_files:
        if image_file.original_path.startswith(old_path_normalized):
            # Replace old path prefix with new path
            relative_part = image_file.original_path[len(old_path_normalized):]
            image_file.original_path = new_path_str + relative_part
            updated_count += 1

    # Update directory path
    directory.path = new_path_str

    await db.commit()

    # Refresh filesystem watcher
    from ...services.directory_watcher import directory_watcher
    await directory_watcher.refresh()

    return {
        "id": directory.id,
        "old_path": old_path,
        "new_path": new_path_str,
        "files_updated": updated_count,
        "message": f"Directory path updated. {updated_count} file references updated."
    }
