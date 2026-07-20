"""
Collections/Albums endpoints - user-created groupings of images.
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, delete, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional, List

from ..database import get_db
from ..models import Collection, CollectionItem, Image, ImageFile

router = APIRouter()


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cover_image_id: Optional[int] = None


class CollectionItemsAdd(BaseModel):
    image_ids: List[int]


class CollectionItemsReorder(BaseModel):
    image_ids: List[int]


def serialize_image(image, file_path=None):
    """Serialize an Image model to dict."""
    return {
        "id": image.id,
        "filename": image.filename,
        "original_filename": image.original_filename,
        "file_hash": image.file_hash,
        "width": image.width,
        "height": image.height,
        "file_size": image.file_size,
        "duration": image.duration,
        "rating": image.rating.value if image.rating else None,
        "is_favorite": image.is_favorite,
        "view_count": image.view_count,
        "url": image.url,
        "thumbnail_url": image.thumbnail_url,
        "file_path": file_path,
        "created_at": image.created_at.isoformat() if image.created_at else None,
    }


@router.get("")
async def list_collections(db: AsyncSession = Depends(get_db)):
    """List all collections with cover image and item count."""
    result = await db.execute(
        select(Collection).order_by(Collection.updated_at.desc().nullslast(), Collection.created_at.desc())
    )
    collections = result.scalars().all()

    items = []
    for c in collections:
        cover_thumb = None
        if c.cover_image_id:
            img_result = await db.execute(select(Image).where(Image.id == c.cover_image_id))
            cover_img = img_result.scalar_one_or_none()
            if cover_img:
                cover_thumb = cover_img.thumbnail_url

        items.append({
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "cover_image_id": c.cover_image_id,
            "cover_thumbnail_url": cover_thumb,
            "item_count": c.item_count,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        })

    return {"collections": items}


@router.post("")
async def create_collection(body: CollectionCreate, db: AsyncSession = Depends(get_db)):
    """Create a new collection."""
    collection = Collection(name=body.name, description=body.description)
    db.add(collection)
    await db.commit()
    await db.refresh(collection)
    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "item_count": 0,
    }


@router.get("/{collection_id}")
async def get_collection(
    collection_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get a collection with its items paginated."""
    # Get collection info
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()
    if not collection:
        return {"error": "Collection not found"}, 404

    # Get items with pagination
    offset = (page - 1) * per_page
    items_result = await db.execute(
        select(CollectionItem, Image)
        .join(Image, CollectionItem.image_id == Image.id)
        .where(CollectionItem.collection_id == collection_id)
        .order_by(CollectionItem.sort_order)
        .offset(offset)
        .limit(per_page)
    )
    rows = items_result.all()

    images = []
    for ci, image in rows:
        file_result = await db.execute(
            select(ImageFile.original_path).where(ImageFile.image_id == image.id).limit(1)
        )
        file_path = file_result.scalar_one_or_none()
        images.append(serialize_image(image, file_path))

    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "cover_image_id": collection.cover_image_id,
        "item_count": collection.item_count,
        "images": images,
        "page": page,
        "per_page": per_page,
        "has_more": len(images) == per_page,
    }


@router.patch("/{collection_id}")
async def update_collection(collection_id: int, body: CollectionUpdate, db: AsyncSession = Depends(get_db)):
    """Update collection name, description, or cover image."""
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()
    if not collection:
        return {"error": "Collection not found"}, 404

    if body.name is not None:
        collection.name = body.name
    if body.description is not None:
        collection.description = body.description
    if body.cover_image_id is not None:
        collection.cover_image_id = body.cover_image_id

    await db.commit()
    return {"success": True}


@router.delete("/{collection_id}")
async def delete_collection(collection_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a collection (not the images in it)."""
    await db.execute(
        delete(Collection).where(Collection.id == collection_id)
    )
    await db.commit()
    return {"success": True}


@router.post("/{collection_id}/items")
async def add_items(collection_id: int, body: CollectionItemsAdd, db: AsyncSession = Depends(get_db)):
    """Add images to a collection."""
    # Get current max sort order
    result = await db.execute(
        select(func.max(CollectionItem.sort_order)).where(CollectionItem.collection_id == collection_id)
    )
    max_order = result.scalar() or 0

    added = 0
    for i, image_id in enumerate(body.image_ids):
        # Check if already in collection
        existing = await db.execute(
            select(CollectionItem).where(
                CollectionItem.collection_id == collection_id,
                CollectionItem.image_id == image_id,
            )
        )
        if existing.scalar_one_or_none():
            continue

        item = CollectionItem(
            collection_id=collection_id,
            image_id=image_id,
            sort_order=max_order + i + 1,
        )
        db.add(item)
        added += 1

    # Update item count
    if added > 0:
        result = await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        collection = result.scalar_one_or_none()
        if collection:
            collection.item_count = (collection.item_count or 0) + added
            # Auto-set cover if none set
            if not collection.cover_image_id and body.image_ids:
                collection.cover_image_id = body.image_ids[0]

    await db.commit()
    return {"success": True, "added": added}


@router.delete("/{collection_id}/items")
async def remove_items(collection_id: int, body: CollectionItemsAdd, db: AsyncSession = Depends(get_db)):
    """Remove images from a collection."""
    for image_id in body.image_ids:
        await db.execute(
            delete(CollectionItem).where(
                CollectionItem.collection_id == collection_id,
                CollectionItem.image_id == image_id,
            )
        )

    # Update item count
    count_result = await db.execute(
        select(func.count()).where(CollectionItem.collection_id == collection_id)
    )
    new_count = count_result.scalar()
    await db.execute(
        update(Collection).where(Collection.id == collection_id).values(item_count=new_count)
    )

    await db.commit()
    return {"success": True}


@router.patch("/{collection_id}/items/reorder")
async def reorder_items(collection_id: int, body: CollectionItemsReorder, db: AsyncSession = Depends(get_db)):
    """Reorder items in a collection. Body contains image_ids in desired order."""
    for i, image_id in enumerate(body.image_ids):
        await db.execute(
            update(CollectionItem)
            .where(
                CollectionItem.collection_id == collection_id,
                CollectionItem.image_id == image_id,
            )
            .values(sort_order=i)
        )
    await db.commit()
    return {"success": True}
