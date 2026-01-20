"""
File tracker service - manages file locations and verification
"""
import os
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Image, ImageFile, WatchDirectory, TaskQueue, TaskType, TaskStatus, FileStatus
from ..config import get_settings
from ..database import AsyncSessionLocal

settings = get_settings()

# Concurrency for imports within a directory scan
SCAN_CONCURRENCY = 8
BATCH_SIZE = 100  # Commit every N imports

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


def is_media_file(path: Path) -> bool:
    """Check if file is a supported media file"""
    return path.suffix.lower() in ALL_EXTENSIONS


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def scan_directory(
    directory_id: int,
    directory_path: str,
    db: AsyncSession,
    recursive: bool = True
) -> dict:
    """Scan a directory for media files and import them concurrently"""
    from .importer import import_image

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

    # Clean up stale associations - files pointing to this directory_id but not in this path
    from ..database import get_data_dir
    stale_query = select(ImageFile).where(
        ImageFile.watch_directory_id == directory_id,
        ~ImageFile.original_path.like(str(path) + '%')
    )
    stale_result = await db.execute(stale_query)
    stale_files = stale_result.scalars().all()

    for stale_file in stale_files:
        if Path(stale_file.original_path).exists():
            # File exists but in wrong directory - just clear association
            stale_file.watch_directory_id = None
            stats['cleaned'] += 1
        else:
            # File doesn't exist - delete from DB
            image = await db.get(Image, stale_file.image_id)
            if image:
                files_count_query = select(func.count(ImageFile.id)).where(ImageFile.image_id == image.id)
                files_count_result = await db.execute(files_count_query)
                files_count = files_count_result.scalar()

                if files_count <= 1:
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)
                else:
                    await db.delete(stale_file)
                stats['removed'] += 1

    if stats['cleaned'] > 0 or stats['removed'] > 0:
        await db.commit()
        if stats['cleaned'] > 0:
            print(f"[Scan] Cleared {stats['cleaned']} stale file associations")
        if stats['removed'] > 0:
            print(f"[Scan] Removed {stats['removed']} non-existent stale files")

    # Remove deleted files - files in DB for this directory that no longer exist on disk
    existing_files_query = select(ImageFile).where(ImageFile.watch_directory_id == directory_id)
    existing_result = await db.execute(existing_files_query)
    existing_files = existing_result.scalars().all()

    for image_file in existing_files:
        if not Path(image_file.original_path).exists():
            # Get the image to delete thumbnail
            image = await db.get(Image, image_file.image_id)
            if image:
                # Check if this is the only file for this image
                files_count_query = select(func.count(ImageFile.id)).where(ImageFile.image_id == image.id)
                files_count_result = await db.execute(files_count_query)
                files_count = files_count_result.scalar()

                if files_count <= 1:
                    # Last file - delete the image and thumbnail
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)
                else:
                    # Just delete this file reference
                    await db.delete(image_file)
                stats['removed'] += 1

    if stats['removed'] > 0:
        await db.commit()
        print(f"[Scan] Removed {stats['removed']} deleted files from database")

    # Collect all media files first
    if recursive:
        files = list(path.rglob('*'))
    else:
        files = list(path.glob('*'))

    media_files = [f for f in files if f.is_file() and is_media_file(f)]
    stats['found'] = len(media_files)

    # Process with concurrent imports (each gets its own session to avoid race conditions)
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
                        skip_commit=False  # Each import commits individually
                    )
                    return result['status']
            except Exception as e:
                print(f"[Scan] Import error for {file_path.name}: {e}")
                return 'error'

    for i in range(0, len(media_files), BATCH_SIZE):
        batch = media_files[i:i + BATCH_SIZE]

        # Process batch concurrently (each import has its own session)
        results = await asyncio.gather(*[import_one(f) for f in batch])

        for status in results:
            if status == 'imported':
                stats['imported'] += 1
            elif status == 'duplicate':
                stats['duplicates'] += 1
            elif status == 'error':
                stats['errors'] += 1

        # Progress log every batch
        if stats['found'] > 100:
            print(f"[Scan] Progress: {i + len(batch)}/{stats['found']} files")

    # Update last scanned timestamp
    await db.execute(
        update(WatchDirectory)
        .where(WatchDirectory.id == directory_id)
        .values(last_scanned_at=datetime.now())
    )
    await db.commit()

    return stats


async def verify_file_locations(db: AsyncSession, batch_size: int = 100) -> dict:
    """Verify files still exist at their recorded locations.

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

    # Get files that aren't already marked as drive offline
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
            # Drive is offline - keep in DB, mark as offline
            image_file.file_status = FileStatus.drive_offline
            image_file.last_verified_at = datetime.now()
            stats['drive_offline'] += 1
        else:
            # File is confirmed deleted (parent dir exists but file doesn't)
            # Try to relocate by hash first
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
                # Can't relocate - delete entry (and image if no other references)
                other_files = await db.execute(
                    select(ImageFile).where(
                        ImageFile.image_id == image_file.image_id,
                        ImageFile.id != image_file.id
                    )
                )
                has_other_refs = other_files.scalar_one_or_none() is not None

                await db.delete(image_file)

                if not has_other_refs and image:
                    # No other file references - delete image and thumbnail
                    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                    await db.delete(image)

                stats['deleted'] += 1

    await db.commit()
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


async def mark_file_missing(file_path: str, db: AsyncSession):
    """Handle file deletion (called when file watcher detects deletion).

    Deletes the ImageFile entry. If no other references exist, deletes the Image too.
    """
    from ..database import get_data_dir

    # Find the ImageFile entry
    query = select(ImageFile).where(ImageFile.original_path == file_path)
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        return

    image_id = image_file.image_id

    # Check for other file references
    other_query = select(ImageFile).where(
        ImageFile.image_id == image_id,
        ImageFile.id != image_file.id
    )
    other_result = await db.execute(other_query)
    has_other_refs = other_result.scalar_one_or_none() is not None

    # Delete this file reference
    await db.delete(image_file)

    if not has_other_refs:
        # No other references - delete the image and thumbnail
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
