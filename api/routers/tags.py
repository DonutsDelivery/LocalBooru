"""
Tags router - manage tags and search
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..models import Tag, TagCategory, image_tags

router = APIRouter()


@router.get("")
async def list_tags(
    q: Optional[str] = None,
    category: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    sort: str = "count",
    db: AsyncSession = Depends(get_db)
):
    """List tags with optional search and filtering"""
    query = select(Tag)

    filters = []

    # Search filter
    if q:
        search_term = q.lower().replace(" ", "_")
        filters.append(Tag.name.ilike(f"%{search_term}%"))

    # Category filter
    if category:
        if category not in [c.value for c in TagCategory]:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        filters.append(Tag.category == TagCategory(category))

    if filters:
        query = query.where(*filters)

    # Sorting
    if sort == "count":
        query = query.order_by(desc(Tag.post_count))
    elif sort == "name":
        query = query.order_by(Tag.name)
    elif sort == "newest":
        query = query.order_by(desc(Tag.created_at))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    tags = result.scalars().all()

    return {
        "tags": [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category.value,
                "post_count": t.post_count
            }
            for t in tags
        ],
        "total": total,
        "page": page,
        "per_page": per_page
    }


@router.get("/autocomplete")
async def autocomplete_tags(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Autocomplete tag search - optimized single query with prefix priority"""
    search_term = q.lower().replace(" ", "_")

    # Single query with UNION: prefix matches first (priority 0), then contains (priority 1)
    # This avoids two separate queries and leverages index on tag name
    from sqlalchemy import literal, union_all

    prefix_query = (
        select(Tag.id, Tag.name, Tag.category, Tag.post_count, literal(0).label('priority'))
        .where(Tag.name.ilike(f"{search_term}%"))
    )

    contains_query = (
        select(Tag.id, Tag.name, Tag.category, Tag.post_count, literal(1).label('priority'))
        .where(
            Tag.name.ilike(f"%{search_term}%"),
            ~Tag.name.ilike(f"{search_term}%")  # Exclude prefix matches
        )
    )

    combined = union_all(prefix_query, contains_query).subquery()

    final_query = (
        select(combined)
        .order_by(combined.c.priority, combined.c.post_count.desc())
        .limit(limit)
    )

    result = await db.execute(final_query)
    rows = result.all()

    return [
        {
            "name": row.name,
            "category": row.category.value if hasattr(row.category, 'value') else row.category,
            "post_count": row.post_count
        }
        for row in rows
    ]


@router.get("/{tag_name}")
async def get_tag(tag_name: str, db: AsyncSession = Depends(get_db)):
    """Get tag details"""
    normalized = tag_name.lower().replace(" ", "_")

    query = select(Tag).where(Tag.name == normalized)
    result = await db.execute(query)
    tag = result.scalar_one_or_none()

    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    return {
        "id": tag.id,
        "name": tag.name,
        "category": tag.category.value,
        "post_count": tag.post_count,
        "created_at": tag.created_at.isoformat() if tag.created_at else None
    }


@router.patch("/{tag_name}/category")
async def update_tag_category(
    tag_name: str,
    category: str,
    db: AsyncSession = Depends(get_db)
):
    """Update tag category"""
    if category not in [c.value for c in TagCategory]:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    normalized = tag_name.lower().replace(" ", "_")

    query = select(Tag).where(Tag.name == normalized)
    result = await db.execute(query)
    tag = result.scalar_one_or_none()

    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    tag.category = TagCategory(category)
    await db.commit()

    return {"name": tag.name, "category": tag.category.value}


@router.get("/stats/overview")
async def tag_stats(db: AsyncSession = Depends(get_db)):
    """Get tag statistics"""
    total_query = select(func.count(Tag.id))
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    # Count by category
    category_query = (
        select(Tag.category, func.count(Tag.id))
        .group_by(Tag.category)
    )
    category_result = await db.execute(category_query)
    by_category = {row[0].value: row[1] for row in category_result}

    # Top tags
    top_query = (
        select(Tag)
        .order_by(desc(Tag.post_count))
        .limit(10)
    )
    top_result = await db.execute(top_query)
    top_tags = [
        {"name": t.name, "count": t.post_count}
        for t in top_result.scalars().all()
    ]

    return {
        "total": total,
        "by_category": by_category,
        "top_tags": top_tags
    }
