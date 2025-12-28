"""
File tracker service - manages file locations and verification
"""
import os
import hashlib
from pathlib import Path
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Image, ImageFile, WatchDirectory, TaskQueue, TaskType, TaskStatus, FileStatus
from ..config import get_settings

settings = get_settings()

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.webm', '.mp4', '.mov'}
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
    """Scan a directory for media files and import them"""
    from .importer import import_image

    stats = {
        'found': 0,
        'imported': 0,
        'duplicates': 0,
        'errors': 0
    }

    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Get iterator based on recursive setting
    if recursive:
        files = path.rglob('*')
    else:
        files = path.glob('*')

    for file_path in files:
        if not file_path.is_file():
            continue
        if not is_media_file(file_path):
            continue

        stats['found'] += 1

        try:
            result = await import_image(
                str(file_path),
                db,
                watch_directory_id=directory_id,
                auto_tag=True
            )
            if result['status'] == 'imported':
                stats['imported'] += 1
            elif result['status'] == 'duplicate':
                stats['duplicates'] += 1
        except Exception as e:
            print(f"Error importing {file_path}: {e}")
            stats['errors'] += 1

    # Update last scanned timestamp
    await db.execute(
        update(WatchDirectory)
        .where(WatchDirectory.id == directory_id)
        .values(last_scanned_at=datetime.now())
    )
    await db.commit()

    return stats


async def verify_file_locations(db: AsyncSession, batch_size: int = 100) -> dict:
    """Verify files still exist at their recorded locations"""
    stats = {
        'verified': 0,
        'missing': 0,
        'drive_offline': 0,
        'relocated': 0
    }

    # Get files that aren't already marked as missing (skip confirmed missing)
    query = (
        select(ImageFile)
        .where(ImageFile.file_status != FileStatus.missing)
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
            # Drive is offline - don't mark as missing, just update status
            image_file.file_status = FileStatus.drive_offline
            # Keep file_exists True since file might still exist when drive comes back
            stats['drive_offline'] += 1
        else:
            # File is confirmed missing - try to relocate by hash
            image = await db.get(Image, image_file.image_id)
            if image:
                new_path = await find_file_by_hash(image.file_hash, db)
                if new_path:
                    image_file.original_path = new_path
                    image_file.file_status = FileStatus.available
                    image_file.file_exists = True
                    stats['relocated'] += 1
                else:
                    image_file.file_status = FileStatus.missing
                    image_file.file_exists = False
                    stats['missing'] += 1
            else:
                image_file.file_status = FileStatus.missing
                image_file.file_exists = False
                stats['missing'] += 1

        image_file.last_verified_at = datetime.now()

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
    """Mark a file as missing (called when file watcher detects deletion)"""
    query = (
        update(ImageFile)
        .where(ImageFile.original_path == file_path)
        .values(file_exists=False, last_verified_at=datetime.now())
    )
    await db.execute(query)
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
