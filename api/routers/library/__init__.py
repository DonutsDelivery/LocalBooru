"""
Library router - library-wide operations and statistics

This package combines several sub-routers:
- stats: Library statistics and counts
- queue: Task queue management
- maintenance: Cleanup and regeneration operations
- events: Server-Sent Events streaming
"""
from fastapi import APIRouter

from .stats import router as stats_router
from .queue import router as queue_router
from .maintenance import router as maintenance_router
from .events import router as events_router

# Create the main library router that combines all sub-routers
router = APIRouter()

# Include all sub-routers without prefix (routes already defined at correct paths)
router.include_router(stats_router)
router.include_router(queue_router)
router.include_router(maintenance_router)
router.include_router(events_router)
