"""
Image router - aggregates all image-related endpoints.

This package splits the monolithic images.py into focused modules:
- list.py: Image listing, search, filtering endpoints
- single.py: Single image CRUD operations
- batch.py: Batch operations (delete, retag, move)
- adjustments.py: Image adjustment endpoints (brightness, contrast, etc.)
- models.py: Shared Pydantic models
- helpers.py: Shared helper functions for per-directory database queries

Architecture:
- Images can be in per-directory databases (directories/{id}.db) or legacy main DB
- directory_id parameter specifies which directory DB to query
- For cross-directory queries, we aggregate results from multiple DBs
"""
from fastapi import APIRouter

from .list import router as list_router
from .single import router as single_router
from .batch import router as batch_router
from .adjustments import router as adjustments_router


# Main router that includes all sub-routers
router = APIRouter()

# Include list endpoints (GET "")
router.include_router(list_router)

# Include single image endpoints (GET/POST/PATCH/DELETE "/{image_id}/...")
# Note: The order matters! More specific routes like "/media/file-info" and "/upload"
# need to be included before the generic "/{image_id}" routes.
# Since single_router has both, we need to ensure proper ordering in single.py
router.include_router(single_router)

# Include batch endpoints (POST "/batch/...")
router.include_router(batch_router)

# Include adjustment endpoints (POST "/{image_id}/preview-adjust", etc.)
router.include_router(adjustments_router)
