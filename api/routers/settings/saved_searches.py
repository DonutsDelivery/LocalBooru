"""
Saved searches endpoints - store and retrieve named filter presets.
"""
from fastapi import APIRouter
import uuid
from datetime import datetime

from .models import get_saved_searches, save_saved_searches, SavedSearchCreate

router = APIRouter()


@router.get("/saved-searches")
async def list_saved_searches():
    """List all saved searches."""
    return {"searches": get_saved_searches()}


@router.post("/saved-searches")
async def create_saved_search(body: SavedSearchCreate):
    """Create a new saved search."""
    searches = get_saved_searches()
    search = {
        "id": uuid.uuid4().hex[:12],
        "name": body.name,
        "filters": body.filters,
        "created_at": datetime.now().isoformat(),
    }
    searches.append(search)
    save_saved_searches(searches)
    return {"success": True, "search": search}


@router.delete("/saved-searches/{search_id}")
async def delete_saved_search(search_id: str):
    """Delete a saved search by ID."""
    searches = get_saved_searches()
    searches = [s for s in searches if s.get("id") != search_id]
    save_saved_searches(searches)
    return {"success": True}
