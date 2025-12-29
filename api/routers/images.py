"""
Image router - simplified for local single-user library
"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import select, func, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional
from pathlib import Path
import os
import tempfile

from ..database import get_db
from ..models import Image, Tag, ImageFile, image_tags, Rating, TagCategory, FileStatus
from ..services.importer import import_image
from ..services.file_tracker import check_file_availability

router = APIRouter()


async def get_image_file_status(image: Image, db: AsyncSession) -> dict:
    """Get the file availability status for an image.

    If file is confirmed deleted (parent exists but file doesn't),
    deletes the database entry.
    """
    from ..database import get_data_dir

    # Get the primary file for this image
    query = select(ImageFile).where(ImageFile.image_id == image.id).limit(1)
    result = await db.execute(query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        return {"status": "missing", "path": None}

    # Check actual file availability
    status = check_file_availability(image_file.original_path)

    if status == FileStatus.missing:
        # File confirmed deleted (parent dir exists) - remove from database
        # Check for other file references first
        other_query = select(ImageFile).where(
            ImageFile.image_id == image.id,
            ImageFile.id != image_file.id
        )
        other_result = await db.execute(other_query)
        has_other_refs = other_result.scalar_one_or_none() is not None

        await db.delete(image_file)

        if not has_other_refs:
            # No other references - delete the image and thumbnail
            thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            await db.delete(image)

        await db.commit()
        return {"status": "deleted", "path": image_file.original_path}

    # Update database if status changed (for drive_offline)
    if image_file.file_status != status:
        image_file.file_status = status
        image_file.file_exists = (status == FileStatus.available)
        await db.commit()

    return {
        "status": status.value,
        "path": image_file.original_path
    }


@router.get("")
async def list_images(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    tags: Optional[str] = None,
    exclude_tags: Optional[str] = None,
    rating: Optional[str] = None,
    favorites_only: bool = False,
    directory_id: Optional[int] = Query(None, description="Filter to images from a specific directory"),
    min_age: Optional[int] = Query(None, ge=0, le=120, description="Minimum detected age"),
    max_age: Optional[int] = Query(None, ge=0, le=120, description="Maximum detected age"),
    has_faces: Optional[bool] = Query(None, description="Filter to images with detected faces"),
    sort: str = "newest",
    db: AsyncSession = Depends(get_db)
):
    """List images with filtering and pagination"""
    query = select(Image).options(selectinload(Image.tags), selectinload(Image.files))

    filters = []

    # Exclude images where all files are confirmed missing
    # (drive_offline is OK to show, only exclude "missing" status)
    has_non_missing_file = select(ImageFile.image_id).where(
        ImageFile.file_status != FileStatus.missing
    )
    filters.append(Image.id.in_(has_non_missing_file))

    # Favorites filter
    if favorites_only:
        filters.append(Image.is_favorite == True)

    # Rating filter
    if rating:
        ratings = rating.split(",")
        valid_ratings = [r for r in ratings if r in [e.value for e in Rating]]
        if valid_ratings:
            filters.append(Image.rating.in_([Rating(r) for r in valid_ratings]))

    # Age filters
    if min_age is not None:
        # Include images where max_detected_age >= min_age (at least one face is old enough)
        filters.append(Image.max_detected_age >= min_age)
    if max_age is not None:
        # Include images where min_detected_age <= max_age (at least one face is young enough)
        filters.append(Image.min_detected_age <= max_age)
    if has_faces is not None:
        if has_faces:
            filters.append(Image.num_faces > 0)
        else:
            filters.append(or_(Image.num_faces == 0, Image.num_faces.is_(None)))

    # Directory filter
    if directory_id is not None:
        dir_subq = select(ImageFile.image_id).where(ImageFile.watch_directory_id == directory_id)
        filters.append(Image.id.in_(dir_subq))

    # Tag inclusion filter
    if tags:
        tag_names = [t.strip().lower().replace(" ", "_") for t in tags.split(",") if t.strip()]
        for tag_name in tag_names:
            tag_subq = (
                select(image_tags.c.image_id)
                .join(Tag)
                .where(Tag.name == tag_name)
            )
            filters.append(Image.id.in_(tag_subq))

    # Tag exclusion filter
    if exclude_tags:
        exclude_names = [t.strip().lower().replace(" ", "_") for t in exclude_tags.split(",") if t.strip()]
        for tag_name in exclude_names:
            tag_subq = (
                select(image_tags.c.image_id)
                .join(Tag)
                .where(Tag.name == tag_name)
            )
            filters.append(Image.id.not_in(tag_subq))

    if filters:
        query = query.where(and_(*filters))

    # Sorting (with secondary sort by ID for consistent ordering)
    # Use file_modified_at for date sorting (actual file date), fall back to created_at
    if sort == "newest":
        query = query.order_by(
            desc(func.coalesce(Image.file_modified_at, Image.created_at)),
            desc(Image.id)
        )
    elif sort == "oldest":
        query = query.order_by(
            asc(func.coalesce(Image.file_modified_at, Image.created_at)),
            asc(Image.id)
        )
    elif sort == "random":
        query = query.order_by(func.random())

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    images = result.scalars().all()

    def get_file_status(img):
        """Get file status from loaded files relationship"""
        if not img.files:
            return "missing"
        return img.files[0].file_status.value if img.files[0].file_status else "unknown"

    return {
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "original_filename": img.original_filename,
                "width": img.width,
                "height": img.height,
                "rating": img.rating.value,
                "is_favorite": img.is_favorite,
                "thumbnail_url": img.thumbnail_url,
                "url": img.url,
                "file_status": get_file_status(img),
                "tags": [{"name": t.name, "category": t.category.value} for t in img.tags],
                "num_faces": img.num_faces,
                "min_age": img.min_detected_age,
                "max_age": img.max_detected_age,
                "created_at": img.created_at.isoformat() if img.created_at else None
            }
            for img in images
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }


@router.get("/{image_id}")
async def get_image(image_id: int, db: AsyncSession = Depends(get_db)):
    """Get single image details"""
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
async def get_image_file(image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the original image file"""
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
        return FileResponse(
            image_file.original_path,
            media_type="image/webp"  # Will be overridden by actual type
        )
    elif status == FileStatus.drive_offline:
        raise HTTPException(
            status_code=503,
            detail="Drive is offline - the storage device containing this file is not connected"
        )
    else:
        raise HTTPException(status_code=404, detail="File is missing or was deleted")


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the thumbnail"""
    from ..database import get_data_dir

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
        from ..services.importer import generate_thumbnail
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
    db: AsyncSession = Depends(get_db)
):
    """Delete an image from the library (optionally delete the file too)"""
    query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Optionally delete the actual file(s)
    if delete_file:
        for f in image.files:
            if os.path.exists(f.original_path):
                os.remove(f.original_path)

    # Delete thumbnail
    from ..database import get_data_dir
    thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
    if thumbnail_path.exists():
        thumbnail_path.unlink()

    # Delete from database (cascades to files, tags, etc.)
    await db.delete(image)
    await db.commit()

    return {"deleted": True}


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
