"""
Image importer service - imports images by reference (no copying)
"""
import os
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
import imagehash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Image, ImageFile, Tag, image_tags, TaskType, WatchDirectory
from ..config import get_settings
from .task_queue import enqueue_task
from .events import library_events, EventType

settings = get_settings()


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_perceptual_hash(file_path: str) -> str | None:
    """Calculate perceptual hash for visual duplicate detection"""
    try:
        img = PILImage.open(file_path)
        phash = imagehash.phash(img)
        return str(phash)
    except Exception:
        return None


def get_image_dimensions(file_path: str) -> tuple[int, int] | None:
    """Get image width and height"""
    try:
        with PILImage.open(file_path) as img:
            return img.size
    except Exception:
        return None


def generate_thumbnail(file_path: str, output_path: str, size: int = 400) -> bool:
    """Generate a thumbnail for the image"""
    try:
        with PILImage.open(file_path) as img:
            img.thumbnail((size, size), PILImage.Resampling.LANCZOS)

            # Convert to RGB if necessary (for RGBA, P mode images)
            if img.mode in ('RGBA', 'P'):
                background = PILImage.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background

            img.save(output_path, 'WEBP', quality=85)
            return True
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False


async def import_image(
    file_path: str,
    db: AsyncSession,
    watch_directory_id: int = None,
    auto_tag: bool = True
) -> dict:
    """
    Import an image by reference (stores path, doesn't copy file)

    Returns:
        dict with status: 'imported', 'duplicate', or 'error'
    """
    path = Path(file_path)

    if not path.exists():
        return {'status': 'error', 'message': 'File not found'}

    if not path.is_file():
        return {'status': 'error', 'message': 'Not a file'}

    # Calculate file hash
    file_hash = calculate_file_hash(file_path)

    # Check if this path already exists in the database
    existing_path = await db.execute(
        select(ImageFile).where(ImageFile.original_path == file_path)
    )
    existing_file = existing_path.scalar_one_or_none()

    if existing_file:
        # Path already exists - check if it's the same image (same hash)
        existing_image_check = await db.execute(
            select(Image).where(Image.id == existing_file.image_id)
        )
        existing_image = existing_image_check.scalar_one_or_none()

        if existing_image and existing_image.file_hash == file_hash:
            # Same file - update directory association if needed
            if watch_directory_id and existing_file.watch_directory_id != watch_directory_id:
                existing_file.watch_directory_id = watch_directory_id
                existing_file.file_exists = True
                await db.commit()
            return {
                'status': 'duplicate',
                'image_id': existing_image.id,
                'message': 'Image already exists'
            }
        else:
            # File was modified - update the hash and re-process
            # For now, just skip it to avoid complexity
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
        # Image already exists with same hash but different path - add new file reference
        image_file = ImageFile(
            image_id=existing_image.id,
            original_path=file_path,
            watch_directory_id=watch_directory_id,
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
    dimensions = get_image_dimensions(file_path)
    perceptual_hash = calculate_perceptual_hash(file_path)

    # Get file timestamps
    stat = os.stat(file_path)
    file_modified_at = datetime.fromtimestamp(stat.st_mtime)
    # Use birth time on macOS/Windows, fall back to ctime on Linux
    file_created_at = datetime.fromtimestamp(
        getattr(stat, 'st_birthtime', stat.st_ctime)
    )

    # Create filename from hash
    ext = path.suffix.lower()
    filename = f"{file_hash[:16]}{ext}"

    # Create image record
    image = Image(
        filename=filename,
        original_filename=path.name,
        file_hash=file_hash,
        perceptual_hash=perceptual_hash,
        width=dimensions[0] if dimensions else None,
        height=dimensions[1] if dimensions else None,
        file_size=file_size,
        import_source=str(path.parent),
        file_created_at=file_created_at,
        file_modified_at=file_modified_at
    )
    db.add(image)
    await db.flush()

    # Create file reference
    image_file = ImageFile(
        image_id=image.id,
        original_path=file_path,
        watch_directory_id=watch_directory_id,
        file_exists=True
    )
    db.add(image_file)

    # Generate thumbnail
    thumbnails_dir = Path(settings.thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnails_dir / f"{file_hash[:16]}.webp"
    generate_thumbnail(file_path, str(thumbnail_path))

    await db.commit()

    # Queue tagging task if auto_tag is enabled
    if auto_tag:
        await enqueue_task(
            TaskType.tag,
            {
                'image_id': image.id,
                'image_path': file_path
            },
            priority=1,  # New imports get priority
            db=db
        )

    # Queue metadata extraction task (extracts AI generation prompts)
    # Get ComfyUI config from watch directory if available
    comfyui_prompt_node_ids = []
    comfyui_negative_node_ids = []
    format_hint = 'auto'

    if watch_directory_id:
        watch_dir = await db.get(WatchDirectory, watch_directory_id)
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

    await enqueue_task(
        TaskType.extract_metadata,
        {
            'image_id': image.id,
            'image_path': file_path,
            'comfyui_prompt_node_ids': comfyui_prompt_node_ids,
            'comfyui_negative_node_ids': comfyui_negative_node_ids,
            'format_hint': format_hint
        },
        priority=0,  # Lower priority than tagging
        db=db
    )

    # Broadcast new image event
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
