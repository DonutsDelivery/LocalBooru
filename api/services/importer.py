"""
Image importer service - imports images by reference (no copying)

Architecture:
- Images are stored in per-directory databases (directories/{id}.db)
- Tags remain in the main database (library.db) for global consistency
- Tagging creates associations in the directory database using global tag IDs
"""
import os
import hashlib
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage
import imagehash
import xxhash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError

from ..models import (
    Image, ImageFile, Tag, image_tags, TaskType, WatchDirectory,
    DirectoryImage, DirectoryImageFile, directory_image_tags
)
from ..database import directory_db_manager
from ..config import get_settings
from .task_queue import enqueue_task
from .events import library_events, EventType
from .video_preview import (
    check_ffmpeg_available,
    get_video_duration,
    get_video_duration_async,
    get_hwaccel_args
)

settings = get_settings()

# Video file extensions
VIDEO_EXTENSIONS = {'webm', 'mp4', 'mov', 'avi', 'mkv'}


def is_video_file(file_path: str) -> bool:
    """Check if the file is a video based on extension"""
    ext = Path(file_path).suffix.lower().lstrip('.')
    return ext in VIDEO_EXTENSIONS

# Thread pool for CPU-bound operations (thumbnails, hashing)
_executor = ThreadPoolExecutor(max_workers=4)  # Limited to prevent disk I/O saturation


async def safe_enqueue_task(task_type, payload, priority, db, max_retries=3):
    """Enqueue a task with retry logic for database locks."""
    # Use image_id as dedupe key to prevent duplicate tasks
    dedupe_key = payload.get('image_id')
    for attempt in range(max_retries):
        try:
            await enqueue_task(task_type, payload, priority=priority, db=db, dedupe_key=dedupe_key)
            return True
        except OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                await db.rollback()
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                # Skip task creation on persistent failure - not critical
                print(f"[Import] Skipping task creation after {max_retries} retries: {e}")
                try:
                    await db.rollback()
                except:
                    pass
                return False
        except Exception as e:
            print(f"[Import] Error enqueueing task: {e}")
            try:
                await db.rollback()
            except:
                pass
            return False
    return False


def calculate_file_hash(file_path: str) -> str:
    """Calculate xxhash of a file (10x faster than SHA256)"""
    hasher = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):  # 64KB chunks
            hasher.update(byte_block)
    return hasher.hexdigest()


def calculate_quick_hash(file_path: str) -> str:
    """Calculate a quick hash from file size + first/last 64KB.

    This is ~100x faster than full file hash for large files.
    Sufficient for duplicate detection in most cases.
    """
    hasher = xxhash.xxh64()
    file_size = os.path.getsize(file_path)
    chunk_size = 65536  # 64KB

    # Include file size in hash
    hasher.update(file_size.to_bytes(8, 'little'))

    with open(file_path, "rb") as f:
        # Read first chunk
        hasher.update(f.read(chunk_size))

        # Read last chunk (if file is large enough)
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)  # Seek from end
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()


async def calculate_file_hash_async(file_path: str) -> str:
    """Async wrapper for file hashing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, calculate_file_hash, file_path)


async def calculate_quick_hash_async(file_path: str) -> str:
    """Async wrapper for quick file hashing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, calculate_quick_hash, file_path)


def calculate_perceptual_hash(file_path: str) -> str | None:
    """Calculate perceptual hash for visual duplicate detection"""
    try:
        img = PILImage.open(file_path)
        phash = imagehash.phash(img)
        return str(phash)
    except Exception:
        return None


async def calculate_perceptual_hash_async(file_path: str) -> str | None:
    """Async wrapper for perceptual hash calculation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, calculate_perceptual_hash, file_path)


def get_image_dimensions(file_path: str) -> tuple[int, int] | None:
    """Get image width and height (supports both images and videos)"""
    # Check if it's a video file
    video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp'}
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in video_extensions:
        # Use ffprobe for video files
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                file_path
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                line = result.stdout.strip()
                if line:
                    width, height = map(int, line.split(','))
                    return (width, height)
        except Exception:
            pass

    # Try PIL for image files
    try:
        with PILImage.open(file_path) as img:
            return img.size
    except Exception:
        pass

    return None


async def get_image_dimensions_async(file_path: str) -> tuple[int, int] | None:
    """Async wrapper for getting image dimensions"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_image_dimensions, file_path)


def generate_thumbnail(file_path: str, output_path: str, size: int = 400) -> bool:
    """Generate a thumbnail for the image"""
    try:
        with PILImage.open(file_path) as img:
            # Use draft() for JPEG to reduce memory at load time
            if img.format == 'JPEG':
                img.draft(None, (size, size))

            # reducing_gap=3 for memory-efficient multi-step resize
            img.thumbnail((size, size), PILImage.Resampling.LANCZOS, reducing_gap=3)

            # Convert to RGB if necessary (for RGBA, P mode images)
            if img.mode in ('RGBA', 'P'):
                background = PILImage.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background

            # method=0 for fastest WebP encoding
            img.save(output_path, 'WEBP', quality=85, method=0)
            return True
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False


async def generate_thumbnail_async(file_path: str, output_path: str, size: int = 400) -> bool:
    """Async wrapper for thumbnail generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, generate_thumbnail, file_path, output_path, size)


def find_existing_thumbnail(video_path: str) -> str | None:
    """Find an existing thumbnail file for a video.

    Checks common thumbnail naming conventions used by media players:
    - video.mp4.jpg, video.mp4.png (full filename + image ext)
    - video.jpg, video.png (name without video ext)
    - video-poster.jpg, video.mp4-poster.jpg
    - video-thumb.jpg, video.mp4-thumb.jpg
    - folder.jpg, poster.jpg (in same directory)

    Returns the path to the existing thumbnail if found, None otherwise.
    """
    video = Path(video_path)
    video_dir = video.parent
    video_name = video.stem  # filename without extension
    video_full = video.name  # filename with extension

    image_exts = ['.jpg', '.jpeg', '.png', '.webp']
    suffixes = ['', '-poster', '-thumb', '-fanart']

    # Check patterns based on video filename
    for suffix in suffixes:
        for ext in image_exts:
            # Pattern: video.mp4.jpg, video.mp4-poster.jpg
            candidate = video_dir / f"{video_full}{suffix}{ext}"
            if candidate.exists():
                return str(candidate)

            # Pattern: video.jpg, video-poster.jpg
            candidate = video_dir / f"{video_name}{suffix}{ext}"
            if candidate.exists():
                return str(candidate)

    # Check folder-level thumbnails
    for name in ['folder', 'poster', 'thumb', 'cover']:
        for ext in image_exts:
            candidate = video_dir / f"{name}{ext}"
            if candidate.exists():
                return str(candidate)

    return None


def generate_video_thumbnail(video_path: str, output_path: str, size: int = 400) -> bool:
    """Generate a thumbnail for a video file.

    First checks for existing thumbnails (e.g., video.mp4.jpg) and uses those.
    Falls back to generating with ffmpeg if none found.

    Optimizations applied:
    - Uses ionice/nice for low I/O and CPU priority (won't slow down video playback)
    - Uses -skip_frame nokey to only decode keyframes (~110x faster)
    - Uses -ss before -i for fast seeking
    - Hardware acceleration when available
    - Seeks to middle of video for more representative thumbnail

    See research: https://github.com/jellyfin/jellyfin/issues/11336
    """
    import subprocess
    import shutil
    from .video_preview import get_low_priority_prefix

    # Check for existing thumbnail first
    existing = find_existing_thumbnail(video_path)
    if existing:
        try:
            # Resize existing thumbnail to our standard size
            with PILImage.open(existing) as img:
                img.thumbnail((size, size), PILImage.Resampling.LANCZOS)
                if img.mode in ('RGBA', 'P'):
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[3])
                    else:
                        background.paste(img)
                    img = background
                img.save(output_path, 'WEBP', quality=85, method=0)
            print(f"[Import] Used existing thumbnail: {Path(existing).name}")
            return True
        except Exception as e:
            print(f"[Import] Failed to use existing thumbnail {existing}: {e}")
            # Fall through to generate with ffmpeg

    if not check_ffmpeg_available():
        return False

    try:
        # Get video duration to seek to middle
        duration = get_video_duration(video_path)
        if duration and duration > 1.0:
            seek_time = duration / 2
        else:
            seek_time = 0.5  # Fallback for very short videos or unknown duration

        # Build command with low priority prefix for background processing
        low_priority = get_low_priority_prefix()
        hwaccel_args = get_hwaccel_args()

        cmd = low_priority + ['ffmpeg', '-y']

        # Add keyframe-only decoding for massive speedup (only decode I-frames)
        # This is safe for thumbnails since we just need any nearby frame
        cmd.extend(['-skip_frame', 'nokey'])

        cmd.extend(hwaccel_args)
        cmd.extend([
            '-ss', str(seek_time),  # Seek to middle of video (fast seek before -i)
            '-i', video_path,
            '-vframes', '1',
            '-vsync', 'passthrough',  # Don't duplicate frames when using skip_frame
            '-vf', f'scale={size}:-1',  # Scale to width, maintain aspect ratio
            '-c:v', 'libwebp',
            '-quality', '85',
            output_path
        ])
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        print(f"[Import] Error generating video thumbnail: {e}")
        return False


async def generate_video_thumbnail_async(video_path: str, output_path: str, size: int = 400) -> bool:
    """Async wrapper for video thumbnail generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, generate_video_thumbnail, video_path, output_path, size)


async def import_image(
    file_path: str,
    db: AsyncSession,
    watch_directory_id: int = None,
    auto_tag: bool = True,
    skip_commit: bool = False,
    fast_mode: bool = True
) -> dict:
    """
    Import an image by reference (stores path, doesn't copy file)

    Images are stored in per-directory databases. If watch_directory_id is provided,
    the image is stored in that directory's database. Otherwise, it uses the legacy
    main database (for manual uploads).

    Args:
        fast_mode: If True (default), adds to DB immediately and defers heavy work
                   (hashing, dimensions, thumbnails) to background tasks.
                   Images appear in UI instantly.

    Returns:
        dict with status: 'imported', 'duplicate', or 'error'
    """
    path = Path(file_path)

    if not path.exists():
        return {'status': 'error', 'message': 'File not found'}

    if not path.is_file():
        return {'status': 'error', 'message': 'Not a file'}

    # Determine which database to use
    if watch_directory_id:
        if fast_mode:
            # Fast path: add to DB immediately, defer heavy work
            return await _fast_import_to_directory_db(
                file_path, directory_id=watch_directory_id, auto_tag=auto_tag, main_db=db
            )
        else:
            # Slow path: calculate hash first (for manual uploads where we want immediate duplicates check)
            file_hash = await calculate_file_hash_async(file_path)
            return await _import_to_directory_db(
                file_path, file_hash, watch_directory_id, auto_tag, db
            )
    else:
        # Legacy: use main database for manual uploads (always slow mode for compat)
        file_hash = await calculate_file_hash_async(file_path)
        return await _import_to_main_db(
            file_path, file_hash, auto_tag, skip_commit, db
        )


async def _fast_import_to_directory_db(
    file_path: str,
    directory_id: int,
    auto_tag: bool,
    main_db: AsyncSession
) -> dict:
    """Fast import: Add to DB immediately, defer heavy work to background.

    This makes images appear in the UI instantly during bulk imports.
    Hash, dimensions, perceptual hash, and thumbnails are calculated later.
    """
    path = Path(file_path)

    await directory_db_manager.ensure_db_exists(directory_id)
    dir_db = await directory_db_manager.get_session(directory_id)

    try:
        # Quick check: path already imported?
        existing = await dir_db.execute(
            select(DirectoryImageFile).where(DirectoryImageFile.original_path == file_path)
        )
        if existing.scalar_one_or_none():
            return {'status': 'duplicate', 'directory_id': directory_id, 'message': 'Path already imported'}

        # Get minimal file info (fast - just stat call)
        stat = os.stat(file_path)
        file_size = stat.st_size
        file_modified_at = datetime.fromtimestamp(stat.st_mtime)
        file_created_at = datetime.fromtimestamp(getattr(stat, 'st_birthtime', stat.st_ctime))

        # Use quick hash (first+last 64KB + size) - fast but good for duplicates
        quick_hash = calculate_quick_hash(file_path)
        ext = path.suffix.lower()

        # Check for duplicate by quick hash
        existing_hash = await dir_db.execute(
            select(DirectoryImage).where(DirectoryImage.file_hash == quick_hash)
        )
        if existing_hash.scalar_one_or_none():
            return {'status': 'duplicate', 'directory_id': directory_id, 'message': 'Duplicate file detected'}

        image = DirectoryImage(
            filename=f"{quick_hash[:16]}{ext}",
            original_filename=path.name,
            file_hash=quick_hash,  # Quick hash - good enough for most duplicate detection
            file_size=file_size,
            import_source=str(path.parent),
            file_created_at=file_created_at,
            file_modified_at=file_modified_at
        )
        dir_db.add(image)
        await dir_db.flush()

        # Create file reference
        image_file = DirectoryImageFile(
            image_id=image.id,
            original_path=file_path,
            file_exists=True
        )
        dir_db.add(image_file)
        await dir_db.commit()

        image_id = image.id

        # Queue background task to complete the import (hash, dimensions, thumbnail)
        await safe_enqueue_task(
            TaskType.extract_metadata,  # Reuse metadata task type for now
            {
                'image_id': image_id,
                'directory_id': directory_id,
                'image_path': file_path,
                'complete_import': True,  # Flag to indicate this needs full processing
                'auto_tag': auto_tag
            },
            priority=2,  # Higher priority than tagging
            db=main_db
        )

        # Broadcast immediately so UI sees the image
        await library_events.broadcast(EventType.IMAGE_ADDED, {
            'image_id': image_id,
            'directory_id': directory_id,
            'filename': image.filename,
            'thumbnail': f"/api/images/{image_id}/thumbnail?directory_id={directory_id}"
        })

        return {
            'status': 'imported',
            'image_id': image_id,
            'directory_id': directory_id,
            'filename': image.filename
        }

    finally:
        await dir_db.close()


async def _import_to_directory_db(
    file_path: str,
    file_hash: str,
    directory_id: int,
    auto_tag: bool,
    main_db: AsyncSession
) -> dict:
    """Import an image to a directory-specific database.

    Fast path: If file_hash is None, creates record immediately and defers
    hash calculation to background. This allows images to appear in UI instantly.
    """
    path = Path(file_path)

    # Ensure the directory database exists
    await directory_db_manager.ensure_db_exists(directory_id)

    # Get a session for the directory database
    dir_db = await directory_db_manager.get_session(directory_id)

    try:
        # Check if this path already exists (fast check - no hash needed)
        existing_path = await dir_db.execute(
            select(DirectoryImageFile).where(DirectoryImageFile.original_path == file_path)
        )
        existing_file = existing_path.scalar_one_or_none()

        if existing_file:
            return {
                'status': 'duplicate',
                'image_id': existing_file.image_id,
                'directory_id': directory_id,
                'message': 'Path already imported'
            }

        # Check for exact duplicate by hash in this directory
        existing = await dir_db.execute(
            select(DirectoryImage).where(DirectoryImage.file_hash == file_hash)
        )
        existing_image = existing.scalar_one_or_none()

        if existing_image:
            # Image already exists with same hash but different path - add new file reference
            image_file = DirectoryImageFile(
                image_id=existing_image.id,
                original_path=file_path,
                file_exists=True
            )
            dir_db.add(image_file)
            await dir_db.commit()

            return {
                'status': 'duplicate',
                'image_id': existing_image.id,
                'directory_id': directory_id,
                'message': 'Image already exists (added new path reference)'
            }

        # Get file info
        file_size = os.path.getsize(file_path)

        # Check if this is a video file
        is_video = is_video_file(file_path)

        # For videos, skip PIL operations and use ffprobe instead
        # For images, run dimension/phash extraction in parallel
        if is_video:
            dimensions = None
            perceptual_hash = None
            video_duration = await get_video_duration_async(file_path)
        else:
            # Run both PIL operations concurrently on thread pool
            dimensions, perceptual_hash = await asyncio.gather(
                get_image_dimensions_async(file_path),
                calculate_perceptual_hash_async(file_path)
            )
            video_duration = None

        # Get file timestamps
        stat = os.stat(file_path)
        file_modified_at = datetime.fromtimestamp(stat.st_mtime)
        file_created_at = datetime.fromtimestamp(
            getattr(stat, 'st_birthtime', stat.st_ctime)
        )

        # Create filename from hash
        ext = path.suffix.lower()
        filename = f"{file_hash[:16]}{ext}"

        # Create image record in directory database
        image = DirectoryImage(
            filename=filename,
            original_filename=path.name,
            file_hash=file_hash,
            perceptual_hash=perceptual_hash,
            width=dimensions[0] if dimensions else None,
            height=dimensions[1] if dimensions else None,
            file_size=file_size,
            duration=video_duration,
            import_source=str(path.parent),
            file_created_at=file_created_at,
            file_modified_at=file_modified_at
        )
        dir_db.add(image)
        await dir_db.flush()

        # Create file reference in directory database
        image_file = DirectoryImageFile(
            image_id=image.id,
            original_path=file_path,
            file_exists=True
        )
        dir_db.add(image_file)
        await dir_db.commit()

        # Generate thumbnail
        thumbnails_dir = Path(settings.thumbnails_dir)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnails_dir / f"{file_hash[:16]}.webp"

        # Fire-and-forget thumbnail generation only (don't block import)
        # Preview frames are generated in batch AFTER all imports complete
        if is_video:
            asyncio.create_task(
                generate_video_thumbnail_async(file_path, str(thumbnail_path))
            )
        else:
            asyncio.create_task(
                generate_thumbnail_async(file_path, str(thumbnail_path))
            )

        # Queue tagging task if auto_tag is enabled (skip videos - tagger only works on images)
        if auto_tag and not is_video_file(file_path):
            await safe_enqueue_task(
                TaskType.tag,
                {
                    'image_id': image.id,
                    'directory_id': directory_id,
                    'image_path': file_path
                },
                priority=1,
                db=main_db
            )

        # Queue metadata extraction task
        comfyui_prompt_node_ids = []
        comfyui_negative_node_ids = []
        format_hint = 'auto'

        watch_dir = await main_db.get(WatchDirectory, directory_id)
        if watch_dir:
            import json as json_module
            if watch_dir.comfyui_prompt_node_ids:
                try:
                    comfyui_prompt_node_ids = json_module.loads(watch_dir.comfyui_prompt_node_ids)
                except Exception:
                    pass
            if watch_dir.comfyui_negative_node_ids:
                try:
                    comfyui_negative_node_ids = json_module.loads(watch_dir.comfyui_negative_node_ids)
                except Exception:
                    pass
            if watch_dir.metadata_format:
                format_hint = watch_dir.metadata_format

        await safe_enqueue_task(
            TaskType.extract_metadata,
            {
                'image_id': image.id,
                'directory_id': directory_id,
                'image_path': file_path,
                'comfyui_prompt_node_ids': comfyui_prompt_node_ids,
                'comfyui_negative_node_ids': comfyui_negative_node_ids,
                'format_hint': format_hint
            },
            priority=0,
            db=main_db
        )

        # Broadcast new image event
        await library_events.broadcast(EventType.IMAGE_ADDED, {
            'image_id': image.id,
            'directory_id': directory_id,
            'filename': filename,
            'thumbnail': f"/thumbnails/{file_hash[:16]}.webp"
        })

        return {
            'status': 'imported',
            'image_id': image.id,
            'directory_id': directory_id,
            'filename': filename
        }

    finally:
        await dir_db.close()


async def _import_to_main_db(
    file_path: str,
    file_hash: str,
    auto_tag: bool,
    skip_commit: bool,
    db: AsyncSession
) -> dict:
    """Legacy: Import an image to the main database (for manual uploads without a directory)."""
    path = Path(file_path)

    # Check if this path already exists in the database
    existing_path = await db.execute(
        select(ImageFile).where(ImageFile.original_path == file_path)
    )
    existing_file = existing_path.scalar_one_or_none()

    if existing_file:
        existing_image_check = await db.execute(
            select(Image).where(Image.id == existing_file.image_id)
        )
        existing_image = existing_image_check.scalar_one_or_none()

        if existing_image and existing_image.file_hash == file_hash:
            return {
                'status': 'duplicate',
                'image_id': existing_image.id,
                'message': 'Image already exists'
            }
        else:
            return {
                'status': 'duplicate',
                'image_id': existing_file.image_id,
                'message': 'Path already imported (file may have been modified)'
            }

    # Check for exact duplicate by hash
    existing = await db.execute(
        select(Image).where(Image.file_hash == file_hash)
    )
    existing_image = existing.scalar_one_or_none()

    if existing_image:
        image_file = ImageFile(
            image_id=existing_image.id,
            original_path=file_path,
            file_exists=True
        )
        db.add(image_file)
        await db.commit()

        return {
            'status': 'duplicate',
            'image_id': existing_image.id,
            'message': 'Image already exists (added new path reference)'
        }

    # Get file info
    file_size = os.path.getsize(file_path)
    is_video = is_video_file(file_path)

    if is_video:
        dimensions = None
        perceptual_hash = None
        video_duration = await get_video_duration_async(file_path)
    else:
        # Run both PIL operations concurrently on thread pool
        dimensions, perceptual_hash = await asyncio.gather(
            get_image_dimensions_async(file_path),
            calculate_perceptual_hash_async(file_path)
        )
        video_duration = None

    stat = os.stat(file_path)
    file_modified_at = datetime.fromtimestamp(stat.st_mtime)
    file_created_at = datetime.fromtimestamp(
        getattr(stat, 'st_birthtime', stat.st_ctime)
    )

    ext = path.suffix.lower()
    filename = f"{file_hash[:16]}{ext}"

    image = Image(
        filename=filename,
        original_filename=path.name,
        file_hash=file_hash,
        perceptual_hash=perceptual_hash,
        width=dimensions[0] if dimensions else None,
        height=dimensions[1] if dimensions else None,
        file_size=file_size,
        duration=video_duration,
        import_source=str(path.parent),
        file_created_at=file_created_at,
        file_modified_at=file_modified_at
    )
    db.add(image)
    await db.flush()

    image_file = ImageFile(
        image_id=image.id,
        original_path=file_path,
        file_exists=True
    )
    db.add(image_file)

    thumbnails_dir = Path(settings.thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnails_dir / f"{file_hash[:16]}.webp"

    # Fire-and-forget thumbnail generation only (don't block import)
    # Preview frames are generated in batch AFTER all imports complete
    if is_video:
        asyncio.create_task(
            generate_video_thumbnail_async(file_path, str(thumbnail_path))
        )
    else:
        asyncio.create_task(
            generate_thumbnail_async(file_path, str(thumbnail_path))
        )

    if not skip_commit:
        await db.commit()

    if auto_tag:
        await safe_enqueue_task(
            TaskType.tag,
            {
                'image_id': image.id,
                'image_path': file_path
            },
            priority=1,
            db=db
        )

    await safe_enqueue_task(
        TaskType.extract_metadata,
        {
            'image_id': image.id,
            'image_path': file_path,
            'comfyui_prompt_node_ids': [],
            'comfyui_negative_node_ids': [],
            'format_hint': 'auto'
        },
        priority=0,
        db=db
    )

    await library_events.broadcast(EventType.IMAGE_ADDED, {
        'image_id': image.id,
        'filename': filename,
        'thumbnail': f"/thumbnails/{file_hash[:16]}.webp"
    })

    return {
        'status': 'imported',
        'image_id': image.id,
        'filename': filename
    }


def shutdown():
    """Shutdown the importer service and cleanup thread pool"""
    print("[Importer] Shutting down...")
    _executor.shutdown(wait=True, cancel_futures=True)
    print("[Importer] Shutdown complete")
