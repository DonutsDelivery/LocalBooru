"""
Single image CRUD operations: get, file, thumbnail, preview frames, favorite, rating, delete, upload.

IMPORTANT: Route ordering matters in FastAPI!
Static routes like /media/file-info and /upload MUST be defined BEFORE
dynamic routes like /{image_id} to prevent FastAPI from matching "media" or "upload"
as an image_id.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional
from pathlib import Path
import os
import tempfile

from ...database import get_db, directory_db_manager
from ...models import (
    Image, ImageFile, Rating, FileStatus, WatchDirectory,
    DirectoryImage, DirectoryImageFile
)
from ...services.importer import import_image
from ...services.file_tracker import check_file_availability
from .helpers import check_image_public_access, find_image_directory


router = APIRouter()


# =============================================================================
# Static routes (MUST be defined before /{image_id} routes)
# =============================================================================

@router.get("/media/file-info")
async def get_file_info(path: str = Query(...)):
    """Get file information (size) for a file path"""
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        size = file_path.stat().st_size
        return {"size": size, "path": str(file_path)}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    auto_tag: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Upload a new image manually"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await import_image(tmp_path, db, auto_tag=auto_tag)
        return result
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


# =============================================================================
# Dynamic routes (/{image_id}/...)
# =============================================================================

@router.get("/{image_id}")
async def get_image(request: Request, image_id: int, db: AsyncSession = Depends(get_db)):
    """Get single image details"""
    # Check public access for non-localhost
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    query = (
        select(Image)
        .options(selectinload(Image.tags), selectinload(Image.files))
        .where(Image.id == image_id)
    )
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Increment view count
    image.view_count += 1
    await db.commit()

    return {
        "id": image.id,
        "filename": image.filename,
        "original_filename": image.original_filename,
        "file_hash": image.file_hash,
        "width": image.width,
        "height": image.height,
        "file_size": image.file_size,
        "rating": image.rating.value,
        "is_favorite": image.is_favorite,
        "view_count": image.view_count,
        "thumbnail_url": image.thumbnail_url,
        "url": image.url,
        "tags": [
            {"name": t.name, "category": t.category.value}
            for t in sorted(image.tags, key=lambda t: (t.category.value, t.name))
        ],
        "files": [
            {
                "path": f.original_path,
                "exists": f.file_exists,
                "status": f.file_status.value if f.file_status else "unknown"
            }
            for f in image.files
        ],
        "file_status": image.files[0].file_status.value if image.files and image.files[0].file_status else "unknown",
        "prompt": image.prompt,
        "negative_prompt": image.negative_prompt,
        "model_name": image.model_name,
        "seed": image.seed,
        "source_url": image.source_url,
        "import_source": image.import_source,
        "num_faces": image.num_faces,
        "min_age": image.min_detected_age,
        "max_age": image.max_detected_age,
        "detected_ages": image.detected_ages,
        "age_detection_data": image.age_detection_data,
        "created_at": image.created_at.isoformat() if image.created_at else None
    }


@router.get("/{image_id}/file")
async def get_image_file(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve the original image file"""
    original_path = None

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImageFile.original_path).where(
                DirectoryImageFile.image_id == image_id
            ).limit(1)
            result = await dir_db.execute(query)
            original_path = result.scalar_one_or_none()
        finally:
            await dir_db.close()

        if original_path:
            status = check_file_availability(original_path)
            if status == FileStatus.available:
                return FileResponse(original_path)  # Auto-detect media type from file
            elif status == FileStatus.drive_offline:
                raise HTTPException(
                    status_code=503,
                    detail="Drive is offline - the storage device containing this file is not connected"
                )
            else:
                # File missing - try to find by filename in the watched directory
                # Use os.walk to include hidden directories
                import os as os_module
                from ...services.file_tracker import is_media_file
                filename = Path(original_path).name
                directory = await db.get(WatchDirectory, directory_id)
                if directory and Path(directory.path).exists():
                    dir_path = Path(directory.path)
                    found_path = None
                    if directory.recursive:
                        for root, dirs, files in os_module.walk(dir_path):
                            if filename in files:
                                found_path = Path(root) / filename
                                break
                    else:
                        candidate = dir_path / filename
                        if candidate.exists():
                            found_path = candidate

                    if found_path and found_path.is_file() and is_media_file(found_path):
                        # Found it - update the DB record and serve
                        new_path = str(found_path)
                        dir_db = await directory_db_manager.get_session(directory_id)
                        try:
                            update_query = select(DirectoryImageFile).where(
                                DirectoryImageFile.image_id == image_id
                            ).limit(1)
                            result = await dir_db.execute(update_query)
                            file_record = result.scalar_one_or_none()
                            if file_record:
                                file_record.original_path = new_path
                                file_record.file_exists = True
                                file_record.file_status = FileStatus.available
                                await dir_db.commit()
                        finally:
                            await dir_db.close()
                        return FileResponse(new_path)
                raise HTTPException(status_code=404, detail="File is missing or was deleted")

    # Check public access for non-localhost (legacy path)
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    # Legacy: query main database
    query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        raise HTTPException(status_code=404, detail="Image file not found")

    # Check file availability with drive detection
    status = check_file_availability(image_file.original_path)

    # Update database if status changed
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    if status == FileStatus.available:
        return FileResponse(image_file.original_path)  # Auto-detect media type
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - the storage device containing this file is not connected"
        )
    else:
        raise HTTPException(status_code=404, detail="File is missing or was deleted")


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve the thumbnail"""
    from ...database import get_data_dir

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage).where(DirectoryImage.id == image_id)
            result = await dir_db.execute(query)
            image = result.scalar_one_or_none()

            if image and image.file_hash:
                thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"

                if thumbnail_path.exists():
                    return FileResponse(str(thumbnail_path), media_type="image/webp")

                # Try to regenerate from original
                file_query = select(DirectoryImageFile).where(
                    DirectoryImageFile.image_id == image_id
                ).limit(1)
                file_result = await dir_db.execute(file_query)
                image_file = file_result.scalar_one_or_none()

                if image_file and Path(image_file.original_path).exists():
                    from ...services.importer import generate_thumbnail, generate_video_thumbnail, is_video_file
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                    if is_video_file(image_file.original_path):
                        generate_video_thumbnail(image_file.original_path, str(thumbnail_path))
                    else:
                        generate_thumbnail(image_file.original_path, str(thumbnail_path))
                    if thumbnail_path.exists():
                        return FileResponse(str(thumbnail_path), media_type="image/webp")

                raise HTTPException(status_code=404, detail="Thumbnail not found")
        finally:
            await dir_db.close()

    # Check public access for non-localhost (legacy path)
    if not await check_image_public_access(image_id, request, db):
        raise HTTPException(status_code=403, detail="This image is not available for remote access")

    # Legacy: query main database
    query = select(Image).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"

    if thumbnail_path.exists():
        return FileResponse(str(thumbnail_path), media_type="image/webp")

    # Thumbnail doesn't exist - try to regenerate from original
    file_query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
    file_result = await db.execute(file_query)
    image_file = file_result.scalar_one_or_none()

    if not image_file:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Check file availability
    status = check_file_availability(image_file.original_path)

    # Update database if status changed
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    if status == FileStatus.available:
        from ...services.importer import generate_thumbnail
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        generate_thumbnail(image_file.original_path, str(thumbnail_path))
        return FileResponse(str(thumbnail_path), media_type="image/webp")
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - cannot generate thumbnail"
        )
    else:
        raise HTTPException(status_code=404, detail="Source file missing - cannot generate thumbnail")


@router.get("/{image_id}/preview-frames")
async def get_preview_frames(
    request: Request,
    image_id: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get list of video preview frame URLs for an image.

    Returns frame URLs for videos with preview frames available.
    On-demand generation: if frames don't exist and source file is available,
    generation will be triggered and an empty array returned (check again later).
    """
    try:
        from ...services.video_preview import get_preview_frames as get_frames, generate_video_previews

        file_hash = None
        filename = None
        original_path = None

        # Try directory database first if directory_id provided
        if directory_id is not None and directory_db_manager.db_exists(directory_id):
            dir_db = await directory_db_manager.get_session(directory_id)
            try:
                query = (
                    select(DirectoryImage)
                    .options(selectinload(DirectoryImage.files))
                    .where(DirectoryImage.id == image_id)
                )
                result = await dir_db.execute(query)
                image = result.scalar_one_or_none()

                if image:
                    file_hash = image.file_hash
                    filename = image.filename
                    if image.files:
                        original_path = image.files[0].original_path
            finally:
                await dir_db.close()
        elif directory_id is None and directory_db_manager.get_all_directory_ids():
            # No directory_id provided but per-directory DBs exist - search them
            found_dir_id = await find_image_directory(image_id)
            if found_dir_id:
                directory_id = found_dir_id  # Update for URL generation later
                dir_db = await directory_db_manager.get_session(found_dir_id)
                try:
                    query = (
                        select(DirectoryImage)
                        .options(selectinload(DirectoryImage.files))
                        .where(DirectoryImage.id == image_id)
                    )
                    result = await dir_db.execute(query)
                    image = result.scalar_one_or_none()

                    if image:
                        file_hash = image.file_hash
                        filename = image.filename
                        if image.files:
                            original_path = image.files[0].original_path
                finally:
                    await dir_db.close()

        # Fall back to legacy main database if not found in directory DBs
        if not file_hash:
            # Check public access for non-localhost (legacy path)
            if not await check_image_public_access(image_id, request, db):
                raise HTTPException(status_code=403, detail="This image is not available for remote access")

            # Legacy: query main database
            query = select(Image).where(Image.id == image_id)
            result = await db.execute(query)
            image = result.scalar_one_or_none()

            if image:
                file_hash = image.file_hash
                filename = image.filename

                file_query = select(ImageFile).where(ImageFile.image_id == image_id).limit(1)
                file_result = await db.execute(file_query)
                image_file = file_result.scalar_one_or_none()
                if image_file:
                    original_path = image_file.original_path

        if not file_hash:
            raise HTTPException(status_code=404, detail="Image not found")

        # Check if this is a video
        ext = filename.lower().split('.')[-1] if filename else ''
        if ext not in ['webm', 'mp4', 'mov', 'avi', 'mkv']:
            return {"frames": [], "is_video": False}

        # Check for existing preview frames
        existing_frames = get_frames(file_hash)
        if existing_frames:
            # Return URLs to the frames
            dir_param = f"?directory_id={directory_id}" if directory_id else ""
            frame_urls = [
                f"/api/images/{image_id}/preview-frame/{i}{dir_param}"
                for i in range(len(existing_frames))
            ]
            return {"frames": frame_urls, "is_video": True, "count": len(existing_frames)}

        # No frames exist - try to generate them on-demand
        if original_path:
            status = check_file_availability(original_path)
            if status == FileStatus.available:
                # Trigger generation in background
                import asyncio
                from ...config import get_settings
                settings = get_settings()
                asyncio.create_task(
                    generate_video_previews(original_path, file_hash, settings.video_preview_frames)
                )

        # Return empty for now - client should retry after a moment
        return {"frames": [], "is_video": True, "generating": True}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PreviewFrames] Error for image {image_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting preview frames: {str(e)}")


@router.get("/{image_id}/preview-frame/{frame_index}")
async def get_preview_frame(
    request: Request,
    image_id: int,
    frame_index: int,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Serve a specific video preview frame image"""
    from ...services.video_preview import get_preview_dir

    if frame_index < 0 or frame_index >= 8:
        raise HTTPException(status_code=400, detail="Invalid frame index (must be 0-7)")

    file_hash = None

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage.file_hash).where(DirectoryImage.id == image_id)
            result = await dir_db.execute(query)
            file_hash = result.scalar_one_or_none()
        finally:
            await dir_db.close()
    elif directory_id is None and directory_db_manager.get_all_directory_ids():
        # No directory_id provided but per-directory DBs exist - search them
        found_dir_id = await find_image_directory(image_id)
        if found_dir_id:
            dir_db = await directory_db_manager.get_session(found_dir_id)
            try:
                query = select(DirectoryImage.file_hash).where(DirectoryImage.id == image_id)
                result = await dir_db.execute(query)
                file_hash = result.scalar_one_or_none()
            finally:
                await dir_db.close()

    # Fall back to legacy main database if not found
    if not file_hash:
        # Check public access for non-localhost (legacy path)
        if not await check_image_public_access(image_id, request, db):
            raise HTTPException(status_code=403, detail="This image is not available for remote access")

        # Legacy: query main database
        query = select(Image.file_hash).where(Image.id == image_id)
        result = await db.execute(query)
        file_hash = result.scalar_one_or_none()

    if not file_hash:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get the frame file
    preview_dir = get_preview_dir(file_hash)
    frame_path = preview_dir / f"frame_{frame_index}.webp"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Preview frame not found")

    return FileResponse(str(frame_path), media_type="image/webp")


@router.post("/{image_id}/favorite")
async def toggle_favorite(image_id: int, db: AsyncSession = Depends(get_db)):
    """Toggle favorite status"""
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.is_favorite = not image.is_favorite
    await db.commit()

    return {"is_favorite": image.is_favorite}


@router.patch("/{image_id}/rating")
async def update_rating(image_id: int, rating: str, db: AsyncSession = Depends(get_db)):
    """Update image rating"""
    if rating not in [r.value for r in Rating]:
        raise HTTPException(status_code=400, detail=f"Invalid rating: {rating}")

    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.rating = Rating(rating)
    await db.commit()

    return {"rating": image.rating.value}


@router.delete("/{image_id}")
async def delete_image(
    image_id: int,
    delete_file: bool = False,
    directory_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Delete an image from the library (optionally delete the file too)"""
    from ...database import get_data_dir
    from ...services.video_preview import delete_preview_frames

    image = None
    file_hash = None
    original_path = None

    # Try directory database first if directory_id provided
    if directory_id is not None and directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage).options(selectinload(DirectoryImage.files)).where(DirectoryImage.id == image_id)
            result = await dir_db.execute(query)
            image = result.scalar_one_or_none()

            if image:
                file_hash = image.file_hash
                # Get the original path from files
                if image.files:
                    original_path = image.files[0].original_path

                # Optionally delete the actual file(s)
                if delete_file and image.files:
                    for f in image.files:
                        if os.path.exists(f.original_path):
                            os.remove(f.original_path)

                # Delete from directory database
                await dir_db.delete(image)
                await dir_db.commit()
        finally:
            await dir_db.close()

    # If not found in directory DB, try main database
    if not image:
        query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
        result = await db.execute(query)
        image = result.scalar_one_or_none()

        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        file_hash = image.file_hash
        if image.files:
            original_path = image.files[0].original_path

        # Optionally delete the actual file(s)
        if delete_file:
            for f in image.files:
                if os.path.exists(f.original_path):
                    os.remove(f.original_path)

        # Delete from main database (cascades to files, tags, etc.)
        await db.delete(image)
        await db.commit()

    # Delete thumbnail
    if file_hash:
        thumbnail_path = get_data_dir() / 'thumbnails' / f"{file_hash[:16]}.webp"
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        # Delete video preview frames
        delete_preview_frames(file_hash)

    return {"deleted": True}
