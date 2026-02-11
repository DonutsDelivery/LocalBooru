"""
LocalBooru API - Simplified single-user local image library
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os

from .config import get_settings
from .database import init_db, close_db, get_data_dir

settings = get_settings()

# Frontend dist directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("Starting LocalBooru API...")

    # Add persistent packages directory to path (for age detection deps that survive updates)
    from .routers.settings import (
        ensure_packages_in_path, set_setting, AGE_DETECTION_INSTALLING,
        get_packages_dir, patch_mivolo_for_timm_compat
    )
    ensure_packages_in_path()

    # Apply mivolo patches for timm compatibility (for existing installs)
    patch_mivolo_for_timm_compat(get_packages_dir())

    # Clear stuck "installing" flag from previous crash (thread won't survive restart)
    set_setting(AGE_DETECTION_INSTALLING, "false")

    # Generate or load TLS certificate for HTTPS
    from .services.certificate import get_or_create_certificate, get_certificate_fingerprint
    try:
        cert_path, key_path = get_or_create_certificate()
        fingerprint = get_certificate_fingerprint()
        if fingerprint:
            print(f"[Startup] TLS Certificate fingerprint: {fingerprint}")
    except Exception as e:
        print(f"[Startup] Warning: Could not create TLS certificate: {e}")

    await init_db()

    # Ensure directories exist
    data_dir = get_data_dir()
    os.makedirs(data_dir / 'thumbnails', exist_ok=True)

    # Start background task worker
    from .services.task_queue import task_queue
    await task_queue.start()

    # Start directory watcher
    from .services.directory_watcher import directory_watcher
    await directory_watcher.start()

    # Kill any orphaned SVP processes from previous runs
    print("[Startup] Cleaning up orphaned SVP processes...")
    from .services.svp_stream import kill_orphaned_svp_processes
    kill_orphaned_svp_processes()

    yield

    # Shutdown - cleanup all resources gracefully
    print("\n" + "="*50)
    print("Shutting down LocalBooru API...")
    print("="*50)

    # Stop directory watcher first (prevents new imports)
    print("[Shutdown] Stopping directory watcher...")
    await directory_watcher.stop()

    # Stop background task queue
    print("[Shutdown] Stopping task queue...")
    await task_queue.stop()

    # Stop SVP streams
    print("[Shutdown] Stopping SVP streams...")
    from .services.svp_stream import stop_all_svp_streams
    stop_all_svp_streams()

    # Stop optical flow streams and cleanup thread pool
    print("[Shutdown] Stopping optical flow streams...")
    from .services.optical_flow_stream import shutdown as shutdown_optical_flow
    shutdown_optical_flow()

    # Cleanup video preview thread pool
    print("[Shutdown] Stopping video preview service...")
    from .services.video_preview import shutdown as shutdown_video_preview
    shutdown_video_preview()

    # Cleanup importer thread pool
    print("[Shutdown] Stopping importer service...")
    from .services.importer import shutdown as shutdown_importer
    shutdown_importer()

    # Close database connections
    print("[Shutdown] Closing database connections...")
    await close_db()

    print("="*50)
    print("LocalBooru shutdown complete.")
    print("="*50 + "\n")


app = FastAPI(
    title="LocalBooru",
    description="Local image library with auto-tagging",
    version="0.1.0",
    lifespan=lifespan
)

# Access control middleware - must be added before CORS
from .middleware.access_control import AccessControlMiddleware
app.add_middleware(AccessControlMiddleware)

# CORS middleware - allow network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (access control handled by middleware)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for thumbnails
thumbnails_dir = get_data_dir() / 'thumbnails'
thumbnails_dir.mkdir(exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(thumbnails_dir)), name="thumbnails")

# Include routers - all under /api prefix to avoid conflicts with frontend SPA routes
from .routers import images, tags, directories, library, network, users, app_update
from .routers import settings as settings_router

app.include_router(images.router, prefix="/api/images", tags=["Images"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(directories.router, prefix="/api/directories", tags=["Watch Directories"])
app.include_router(library.router, prefix="/api/library", tags=["Library"])
app.include_router(settings_router.router, prefix="/api/settings", tags=["Settings"])
app.include_router(network.router, prefix="/api/network", tags=["Network"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(app_update.router, prefix="/api/app/update", tags=["App Update"])


@app.get("/api")
async def api_root():
    return {
        "name": "LocalBooru",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths"""
    import os
    from pathlib import Path

    cwd = os.getcwd()
    tagger_base = settings.tagger_base_path
    tagger_full = Path(cwd) / tagger_base / "vit-v3"
    model_path = tagger_full / "model.onnx"
    tags_path = tagger_full / "selected_tags.csv"

    return {
        "cwd": cwd,
        "tagger_base_path": tagger_base,
        "tagger_full_path": str(tagger_full),
        "model_exists": model_path.exists(),
        "tags_exists": tags_path.exists(),
        "tagger_dir_contents": os.listdir(str(tagger_full)) if tagger_full.exists() else "DIR NOT FOUND",
        "resources_contents": os.listdir(cwd) if os.path.isdir(cwd) else "NOT A DIR"
    }


# Serve frontend static files
if FRONTEND_DIR.exists():
    # Mount assets directory
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")

    # Serve static files from frontend root (icon.png, etc)
    @app.get("/icon.png")
    async def serve_icon():
        icon_path = FRONTEND_DIR / "icon.png"
        if icon_path.exists():
            return FileResponse(icon_path)
        return {"error": "not found"}

    # Catch-all route for SPA - must be last
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        # Don't serve SPA for API routes or static assets
        if full_path.startswith(("api/", "thumbnails/")):
            return {"error": "not found"}

        # Return index.html for SPA routing
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"error": "frontend not built"}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
