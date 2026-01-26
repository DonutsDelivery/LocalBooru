"""
File tracker service - manages file locations and verification

Architecture:
- File tracking now uses per-directory databases
- Main database tracks WatchDirectory metadata
- Each directory's images are tracked in directories/{id}.db
"""
import os
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Image, ImageFile, WatchDirectory, TaskQueue, TaskType, TaskStatus, FileStatus,
    DirectoryImage, DirectoryImageFile
)
from ..config import get_settings
from ..database import AsyncSessionLocal, directory_db_manager

settings = get_settings()

# Concurrency for imports within a directory scan
SCAN_CONCURRENCY = 4  # Reduced to prevent disk I/O saturation during imports
BATCH_SIZE = 100  # Process N imports per batch

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.webm', '.mp4', '.mov', '.avi', '.mkv'}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def check_file_availability(file_path: str) -> FileStatus:
    """
    Check if a file is available, distinguishing between:
    - File exists and accessible
    - File deleted but parent directory exists (confirmed missing)
    - Parent directory/drive is unavailable (drive offline)
    """
    path = Path(file_path)

    if path.exists():
        return FileStatus.available

    # File doesn't exist - check if parent directory exists
    # Walk up the path to find the first existing parent
    parent = path.parent
    while parent != parent.parent:  # Stop at root
        if parent.exists():
            # Parent exists but file doesn't = file was deleted
            return FileStatus.missing
        parent = parent.parent

    # No parent directories exist = drive/mount point is offline
    return FileStatus.drive_offline


def is_drive_available(watch_directory_path: str) -> bool:
    """Check if a watch directory's drive/mount point is available"""
    path = Path(watch_directory_path)

    # Check if the path or any parent exists
    while path != path.parent:
        if path.exists():
            return True
        path = path.parent

    return False


def is_video_thumbnail(path: Path) -> bool:
    """Check if file is a video thumbnail (e.g., video.mp4.png, clip.webm.jpg).

    These are auto-generated thumbnail files that shouldn't be imported.
    Detects files with a video extension anywhere in the name followed by an image extension,
    or any file with 2+ dots that ends with an image extension.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()

    # Must end with an image extension
    if suffix not in IMAGE_EXTENSIONS:
        return False

    # Check if filename has 2+ dots (e.g., "video.mp4.png", "file.something.jpg")
    if name.count('.') >= 2:
        return True

    return False


def is_media_file(path: Path) -> bool:
    """Check if file is a supported media file"""
    if path.suffix.lower() not in ALL_EXTENSIONS:
        return False
    # Filter out video thumbnails (e.g., video.mp4.png)
    if is_video_thumbnail(path):
        return False
    return True


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _stream_media_files(path: Path, recursive: bool):
    """Generator that yields media files as they're discovered."""
    glob_iter = path.rglob('*') if recursive else path.glob('*')
    for file_path in glob_iter:
        if file_path.is_file() and is_media_file(file_path):
            yield file_path


async def scan_directory(
    directory_id: int,
    directory_path: str,
    db: AsyncSession,
    recursive: bool = True,
    clean_deleted: bool = False
) -> dict:
    """
    Scan a directory for media files and import them concurrently.

    Uses per-directory database for storing images.
    Uses streaming file discovery to start processing while scanning.

    Args:
        directory_id: ID of the watch directory
        directory_path: Path to the directory to scan
        db: Database session
        recursive: Whether to scan subdirectories
        clean_deleted: Whether to clean up deleted files (slower, deferred by default)
    """
    from .importer import import_image
    from ..database import get_data_dir

    stats = {
        'found': 0,
        'imported': 0,
        'duplicates': 0,
        'errors': 0,
        'cleaned': 0,
        'removed': 0
    }

    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Ensure the directory database exists
    await directory_db_manager.ensure_db_exists(directory_id)

    # Only clean deleted files if explicitly requested (slow operation)
    if clean_deleted:
        stats['removed'] = await _clean_deleted_files(directory_id)

    # Stream file discovery and batch processing
    semaphore = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def import_one(file_path: Path) -> str:
        async with semaphore:
            try:
                async with AsyncSessionLocal() as session:
                    result = await import_image(
                        str(file_path),
                        session,
                        watch_directory_id=directory_id,
                        auto_tag=False,  # Skip auto-tag during bulk import for speed
                        skip_commit=False
                    )
                    return result['status']
            except Exception as e:
                print(f"[Scan] Import error for {file_path.name}: {e}")
                return 'error'

    # Use streaming file discovery - start processing while finding files
    batch = []
    processed = 0

    for file_path in _stream_media_files(path, recursive):
        stats['found'] += 1
        batch.append(file_path)

        if len(batch) >= BATCH_SIZE:
            results = await asyncio.gather(*[import_one(f) for f in batch])

            for status in results:
                if status == 'imported':
                    stats['imported'] += 1
                elif status == 'duplicate':
                    stats['duplicates'] += 1
                elif status == 'error':
                    stats['errors'] += 1

            processed += len(batch)
            if stats['found'] > 100:
                print(f"[Scan] Progress: {processed} files processed, {stats['found']} found so far")

            batch = []

    # Process remaining files in final batch
    if batch:
        results = await asyncio.gather(*[import_one(f) for f in batch])

        for status in results:
            if status == 'imported':
                stats['imported'] += 1
            elif status == 'duplicate':
                stats['duplicates'] += 1
            elif status == 'error':
                stats['errors'] += 1

    # Update last scanned timestamp in main DB
    await db.execute(
        update(WatchDirectory)
        .where(WatchDirectory.id == directory_id)
        .values(last_scanned_at=datetime.now())
    )
    await db.commit()

    # After all imports complete, generate video preview frames in background
    # This runs at low priority so it doesn't interfere with the next scan
    if stats['imported'] > 0:
        asyncio.create_task(
            _generate_video_previews_for_directory(directory_id)
        )

    return stats


async def _generate_video_previews_for_directory(directory_id: int):
    """Generate preview frames for all videos in a directory that don't have them yet.

    Runs after import completes to avoid resource contention during import.
    """
    from .video_preview import generate_video_previews, get_preview_frames

    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        # Get all videos without preview frames
        query = select(DirectoryImage, DirectoryImageFile.original_path).join(
            DirectoryImageFile, DirectoryImageFile.image_id == DirectoryImage.id
        ).where(
            DirectoryImage.filename.ilike('%.webm') |
            DirectoryImage.filename.ilike('%.mp4') |
            DirectoryImage.filename.ilike('%.mov') |
            DirectoryImage.filename.ilike('%.avi') |
            DirectoryImage.filename.ilike('%.mkv')
        )

        result = await dir_db.execute(query)
        videos = result.all()

        generated = 0
        for image, file_path in videos:
            # Skip if preview frames already exist
            existing = get_preview_frames(image.file_hash)
            if existing:
                continue

            # Check if file still exists
            if not Path(file_path).exists():
                continue

            # Generate preview frames (semaphore inside limits concurrency)
            try:
                frames = await generate_video_previews(file_path, image.file_hash, num_frames=8)
                if frames:
                    generated += 1
            except Exception as e:
                print(f"[Scan] Preview generation error for {image.filename}: {e}")

        if generated > 0:
            print(f"[Scan] Generated preview frames for {generated} videos in directory {directory_id}")

    finally:
        await dir_db.close()


async def _clean_deleted_files(directory_id: int) -> int:
    """Clean up deleted files from the directory database.

    Returns the number of removed file references.
    """
    from ..database import get_data_dir

    removed = 0
    dir_db = await directory_db_manager.get_session(directory_id)

    try:
        # Get all file references in this directory DB
        existing_files_query = select(DirectoryImageFile)
        existing_result = await dir_db.execute(existing_files_query)
        existing_files = existing_result.scalars().all()

        for image_file in existing_files:
            if not Path(image_file.original_path).exists():
                # Get the image to delete thumbnail
                image = await dir_db.get(DirectoryImage, image_file.image_id)
                if image:
                    # Check if this is the only file for this image
                    files_count_query = select(func.count(DirectoryImageFile.id)).where(
                        DirectoryImageFile.image_id == image.id
                    )
                    files_count_result = await dir_db.execute(files_count_query)
                    files_count = files_count_result.scalar()

                    if files_count <= 1:
                        # Last file - delete the image and thumbnail
                        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                        if thumbnail_path.exists():
                            thumbnail_path.unlink()
                        await dir_db.delete(image)
                    else:
                        # Just delete this file reference
                        await dir_db.delete(image_file)
                    removed += 1

        if removed > 0:
            await dir_db.commit()
            print(f"[Scan] Removed {removed} deleted files from directory database")
    finally:
        await dir_db.close()

    return removed


async def clean_video_thumbnails() -> dict:
    """Remove video thumbnail files (e.g., video.mp4.png) from all directory databases.

    These are auto-generated thumbnails that shouldn't have been imported.
    Returns stats on how many were removed.
    """
    from ..database import get_data_dir

    stats = {'removed': 0, 'directories_checked': 0}

    all_dir_ids = directory_db_manager.get_all_directory_ids()
    print(f"[Cleanup] Found {len(all_dir_ids)} directories to check")

    for directory_id in all_dir_ids:
        stats['directories_checked'] += 1

        # Ensure database is properly initialized
        await directory_db_manager.ensure_db_exists(directory_id)
        dir_db = await directory_db_manager.get_session(directory_id)

        try:
            # Get all file references
            query = select(DirectoryImageFile)
            try:
                result = await dir_db.execute(query)
            except Exception as e:
                print(f"[Cleanup] Skipping directory {directory_id}: {e}")
                continue
            files = list(result.scalars().all())  # Convert to list to avoid iteration issues

            for image_file in files:
                path = Path(image_file.original_path)
                if is_video_thumbnail(path):
                    # Get the image to delete thumbnail
                    image = await dir_db.get(DirectoryImage, image_file.image_id)
                    if image:
                        # Check if this is the only file for this image
                        files_count_query = select(func.count(DirectoryImageFile.id)).where(
                            DirectoryImageFile.image_id == image.id
                        )
                        files_count_result = await dir_db.execute(files_count_query)
                        files_count = files_count_result.scalar()

                        # Always delete the file reference first
                        await dir_db.delete(image_file)

                        if files_count <= 1:
                            # Last file - also delete the image and thumbnail
                            if image.file_hash:
                                thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                                if thumbnail_path.exists():
                                    thumbnail_path.unlink()
                            await dir_db.delete(image)

                        stats['removed'] += 1
                        print(f"[Cleanup] Removed video thumbnail: {path.name}")

            await dir_db.commit()
        finally:
            await dir_db.close()

    print(f"[Cleanup] Removed {stats['removed']} video thumbnails from {stats['directories_checked']} directories")
    return stats


async def verify_file_locations(db: AsyncSession, batch_size: int = 100) -> dict:
    """Verify files still exist at their recorded locations.

    Checks both legacy main DB files and per-directory DB files.

    - If file exists: mark as available
    - If file deleted (parent exists): DELETE from DB
    - If drive offline: mark as drive_offline (keep in DB)
    """
    from ..database import get_data_dir

    stats = {
        'verified': 0,
        'deleted': 0,
        'drive_offline': 0,
        'relocated': 0
    }

    # First verify files in the legacy main database
    query = (
        select(ImageFile)
        .where(ImageFile.file_status != FileStatus.drive_offline)
        .limit(batch_size)
    )
    result = await db.execute(query)
    files = result.scalars().all()

    for image_file in files:
        status = check_file_availability(image_file.original_path)

        if status == FileStatus.available:
            image_file.file_status = FileStatus.available
            image_file.file_exists = True
            stats['verified'] += 1
        elif status == FileStatus.drive_offline:
            image_file.file_status = FileStatus.drive_offline
            image_file.last_verified_at = datetime.now()
            stats['drive_offline'] += 1
        else:
            image = await db.get(Image, image_file.image_id)
            relocated = False

            if image:
                new_path = await find_file_by_hash(image.file_hash, db)
                if new_path:
                    image_file.original_path = new_path
                    image_file.file_status = FileStatus.available
                    image_file.file_exists = True
                    image_file.last_verified_at = datetime.now()
                    stats['relocated'] += 1
                    relocated = True

            if not relocated:
                other_files = await db.execute(
                    select(ImageFile).where(
                        ImageFile.image_id == image_file.image_id,
                        ImageFile.id != image_file.id
                    )
                )
                has_other_refs = other_files.scalar_one_or_none() is not None

                await db.delete(image_file)

                if not has_other_refs and image:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)

                stats['deleted'] += 1

    await db.commit()

    # Now verify files in per-directory databases
    for directory_id in directory_db_manager.get_all_directory_ids():
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = (
                select(DirectoryImageFile)
                .where(DirectoryImageFile.file_status != FileStatus.drive_offline)
                .limit(batch_size)
            )
            result = await dir_db.execute(query)
            files = result.scalars().all()

            for image_file in files:
                status = check_file_availability(image_file.original_path)

                if status == FileStatus.available:
                    image_file.file_status = FileStatus.available
                    image_file.file_exists = True
                    stats['verified'] += 1
                elif status == FileStatus.drive_offline:
                    image_file.file_status = FileStatus.drive_offline
                    image_file.last_verified_at = datetime.now()
                    stats['drive_offline'] += 1
                else:
                    image = await dir_db.get(DirectoryImage, image_file.image_id)

                    other_files = await dir_db.execute(
                        select(DirectoryImageFile).where(
                            DirectoryImageFile.image_id == image_file.image_id,
                            DirectoryImageFile.id != image_file.id
                        )
                    )
                    has_other_refs = other_files.scalar_one_or_none() is not None

                    await dir_db.delete(image_file)

                    if not has_other_refs and image:
                        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                        if thumbnail_path.exists():
                            thumbnail_path.unlink()
                        await dir_db.delete(image)

                    stats['deleted'] += 1

            await dir_db.commit()
        finally:
            await dir_db.close()

    return stats


async def verify_directory_files(directory_id: int, db: AsyncSession) -> dict:
    """Verify files in a specific directory still exist at their recorded locations.

    - If file exists: mark as available
    - If file deleted (parent exists): DELETE from DB
    - If drive offline: mark as drive_offline (keep in DB)

    Returns stats on verified, deleted, and drive_offline files.
    """
    from ..database import get_data_dir

    stats = {
        'verified': 0,
        'deleted': 0,
        'drive_offline': 0
    }

    if not directory_db_manager.db_exists(directory_id):
        return stats

    dir_db = await directory_db_manager.get_session(directory_id)
    try:
        # Get all files in this directory
        query = select(DirectoryImageFile)
        result = await dir_db.execute(query)
        files = result.scalars().all()

        for image_file in files:
            status = check_file_availability(image_file.original_path)

            if status == FileStatus.available:
                image_file.file_status = FileStatus.available
                image_file.file_exists = True
                image_file.last_verified_at = datetime.now()
                stats['verified'] += 1
            elif status == FileStatus.drive_offline:
                image_file.file_status = FileStatus.drive_offline
                image_file.last_verified_at = datetime.now()
                stats['drive_offline'] += 1
            else:
                # File is missing - delete the record
                image = await dir_db.get(DirectoryImage, image_file.image_id)

                # Check for other file references
                other_files = await dir_db.execute(
                    select(DirectoryImageFile).where(
                        DirectoryImageFile.image_id == image_file.image_id,
                        DirectoryImageFile.id != image_file.id
                    )
                )
                has_other_refs = other_files.scalar_one_or_none() is not None

                await dir_db.delete(image_file)

                if not has_other_refs and image:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await dir_db.delete(image)

                stats['deleted'] += 1

        await dir_db.commit()
    finally:
        await dir_db.close()

    return stats


async def find_file_by_hash(file_hash: str, db: AsyncSession) -> str | None:
    """Try to find a file with matching hash in watch directories"""
    # Get all enabled watch directories
    query = select(WatchDirectory).where(WatchDirectory.enabled == True)
    result = await db.execute(query)
    directories = result.scalars().all()

    for watch_dir in directories:
        path = Path(watch_dir.path)
        if not path.exists():
            continue

        # Search for files
        if watch_dir.recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')

        for file_path in files:
            if not file_path.is_file():
                continue
            if not is_media_file(file_path):
                continue

            try:
                if calculate_file_hash(str(file_path)) == file_hash:
                    return str(file_path)
            except Exception:
                continue

    return None


async def mark_file_missing(file_path: str, db: AsyncSession, directory_id: int = None):
    """Handle file deletion (called when file watcher detects deletion).

    Deletes the ImageFile entry. If no other references exist, deletes the Image too.

    Args:
        file_path: Path to the deleted file
        db: Main database session
        directory_id: If provided, look in the directory database
    """
    from ..database import get_data_dir

    if directory_id:
        # Look in directory database
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImageFile).where(DirectoryImageFile.original_path == file_path)
            result = await dir_db.execute(query)
            image_file = result.scalar_one_or_none()

            if not image_file:
                return

            image_id = image_file.image_id

            other_query = select(DirectoryImageFile).where(
                DirectoryImageFile.image_id == image_id,
                DirectoryImageFile.id != image_file.id
            )
            other_result = await dir_db.execute(other_query)
            has_other_refs = other_result.scalar_one_or_none() is not None

            await dir_db.delete(image_file)

            if not has_other_refs:
                image = await dir_db.get(DirectoryImage, image_id)
                if image:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await dir_db.delete(image)

            await dir_db.commit()
        finally:
            await dir_db.close()
    else:
        # Legacy: Look in main database
        query = select(ImageFile).where(ImageFile.original_path == file_path)
        result = await db.execute(query)
        image_file = result.scalar_one_or_none()

        if not image_file:
            return

        image_id = image_file.image_id

        other_query = select(ImageFile).where(
            ImageFile.image_id == image_id,
            ImageFile.id != image_file.id
        )
        other_result = await db.execute(other_query)
        has_other_refs = other_result.scalar_one_or_none() is not None

        await db.delete(image_file)

        if not has_other_refs:
            image = await db.get(Image, image_id)
            if image:
                thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                await db.delete(image)

        await db.commit()


async def update_file_path(old_path: str, new_path: str, db: AsyncSession):
    """Update file path (called when file watcher detects move)"""
    query = (
        update(ImageFile)
        .where(ImageFile.original_path == old_path)
        .values(original_path=new_path, file_exists=True, last_verified_at=datetime.now())
    )
    await db.execute(query)
    await db.commit()
