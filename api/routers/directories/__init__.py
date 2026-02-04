"""
Watch directories router - manage directories to watch for new images

Architecture:
- Each directory has its own database file at directories/{id}.db
- Deleting a directory = deleting its database file (instant!)
- Tag counts in main DB must be decremented before deletion

This module is split into focused sub-routers:
- crud.py: Create, read, update, delete directory endpoints
- scanning.py: Directory scanning and rescan endpoints
- maintenance.py: Prune, repair, relocate operations
- models.py: Shared Pydantic models
"""
from fastapi import APIRouter

from .crud import router as crud_router, list_directories, add_directory
from .scanning import router as scanning_router
from .maintenance import router as maintenance_router

# Create main router that combines all sub-routers
router = APIRouter()

# Add root routes explicitly to handle both with and without trailing slash
# This is needed because FastAPI's redirect_slashes doesn't work well with catch-all routes
router.add_api_route("", list_directories, methods=["GET"])
router.add_api_route("", add_directory, methods=["POST"])

# Include all sub-routers (they have their own specific paths)
router.include_router(crud_router)
router.include_router(scanning_router)
router.include_router(maintenance_router)
