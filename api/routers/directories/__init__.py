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

from .crud import router as crud_router
from .scanning import router as scanning_router
from .maintenance import router as maintenance_router

# Create main router that combines all sub-routers
router = APIRouter()

# Include all sub-routers (no prefix - they define their own routes)
router.include_router(crud_router)
router.include_router(scanning_router)
router.include_router(maintenance_router)
