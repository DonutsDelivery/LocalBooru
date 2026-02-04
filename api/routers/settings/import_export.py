"""
Settings import/export endpoints - data migration, import/export.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio

from .models import MigrationRequest

router = APIRouter()


# Track ongoing migration state
_migration_state = {
    "running": False,
    "progress": None,
    "result": None
}


# =============================================================================
# Data Migration Endpoints
# =============================================================================

@router.get("/migration")
async def get_migration_info_endpoint():
    """Get information about current mode and migration options."""
    from ...migration import get_migration_info
    info = await get_migration_info()

    # Include current migration state
    info["migration_running"] = _migration_state["running"]
    if _migration_state["progress"]:
        info["migration_progress"] = {
            "phase": _migration_state["progress"].phase,
            "percent": _migration_state["progress"].percent,
            "current_file": _migration_state["progress"].current_file,
            "files_copied": _migration_state["progress"].files_copied,
            "total_files": _migration_state["progress"].total_files,
        }
    if _migration_state["result"]:
        info["last_result"] = {
            "success": _migration_state["result"].success,
            "error": _migration_state["result"].error,
            "files_copied": _migration_state["result"].files_copied,
            "bytes_copied": _migration_state["result"].bytes_copied,
        }

    return info


@router.get("/migration/directories")
async def get_migration_directories(mode: str):
    """Get watch directories available for selective migration.

    Returns list of directories with metadata (path, image count, size).
    """
    from ...migration import get_watch_directories_for_migration, MigrationMode

    try:
        migration_mode = MigrationMode(mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Must be 'system_to_portable' or 'portable_to_system'",
            "directories": []
        }

    directories = await get_watch_directories_for_migration(migration_mode)

    return {
        "success": True,
        "directories": directories,
        "total_count": len(directories),
        "total_images": sum(d["image_count"] for d in directories),
        "total_thumbnail_size": sum(d["thumbnail_size"] for d in directories)
    }


@router.post("/migration/validate")
async def validate_migration(request: MigrationRequest):
    """Validate migration can proceed (dry run).

    If directory_ids is provided, validates selective migration.
    If directory_ids is None or empty, validates full migration.
    """
    from ...migration import (
        migrate_data, migrate_data_selective, MigrationMode,
        get_migration_paths, calculate_selective_migration_size
    )

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "valid": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Use selective migration if directory_ids provided
    if request.directory_ids is not None and len(request.directory_ids) > 0:
        result = await migrate_data_selective(mode, request.directory_ids, dry_run=True)

        # Also get image count for selected directories
        try:
            source, _ = get_migration_paths(mode)
            _, _, thumb_bytes = calculate_selective_migration_size(source, request.directory_ids)
        except:
            thumb_bytes = 0

        return {
            "valid": result.success,
            "error": result.error,
            "source_path": result.source_path,
            "dest_path": result.dest_path,
            "files_to_copy": result.files_copied,
            "bytes_to_copy": result.bytes_copied,
            "size_mb": round(result.bytes_copied / 1024 / 1024, 1) if result.bytes_copied else 0,
            "thumbnail_size_mb": round(thumb_bytes / 1024 / 1024, 1) if thumb_bytes else 0,
            "selective": True,
            "directory_count": len(request.directory_ids)
        }
    else:
        result = await migrate_data(mode, dry_run=True)

        return {
            "valid": result.success,
            "error": result.error,
            "source_path": result.source_path,
            "dest_path": result.dest_path,
            "files_to_copy": result.files_copied,
            "bytes_to_copy": result.bytes_copied,
            "size_mb": round(result.bytes_copied / 1024 / 1024, 1) if result.bytes_copied else 0,
            "selective": False
        }


@router.post("/migration/start")
async def start_migration(request: MigrationRequest):
    """Start data migration (runs in background).

    If directory_ids is provided, performs selective migration.
    If directory_ids is None or empty, performs full migration.
    """
    from ...migration import migrate_data, migrate_data_selective, MigrationMode
    from ...services.events import migration_events, MigrationEventType

    if _migration_state["running"]:
        return {"success": False, "error": "Migration already in progress"}

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Determine if selective migration
    is_selective = request.directory_ids is not None and len(request.directory_ids) > 0
    directory_ids = request.directory_ids if is_selective else []

    # First validate
    if is_selective:
        validation = await migrate_data_selective(mode, directory_ids, dry_run=True)
    else:
        validation = await migrate_data(mode, dry_run=True)

    if not validation.success:
        return {"success": False, "error": validation.error}

    # Reset state
    _migration_state["running"] = True
    _migration_state["progress"] = None
    _migration_state["result"] = None

    def progress_callback(progress):
        _migration_state["progress"] = progress
        # Broadcast progress via SSE (fire and forget)
        asyncio.create_task(migration_events.broadcast(
            MigrationEventType.PROGRESS,
            {
                "phase": progress.phase,
                "percent": round(progress.percent, 1),
                "current_file": progress.current_file,
                "files_copied": progress.files_copied,
                "total_files": progress.total_files,
                "bytes_copied": progress.bytes_copied,
                "total_bytes": progress.total_bytes,
                "error": progress.error
            }
        ))

    async def run_migration():
        try:
            # Broadcast start event
            await migration_events.broadcast(MigrationEventType.STARTED, {
                "mode": mode.value,
                "selective": is_selective,
                "directory_count": len(directory_ids) if is_selective else None
            })

            if is_selective:
                result = await migrate_data_selective(mode, directory_ids, progress_callback=progress_callback)
            else:
                result = await migrate_data(mode, progress_callback=progress_callback)

            _migration_state["result"] = result

            # Broadcast completion/error event
            if result.success:
                await migration_events.broadcast(MigrationEventType.COMPLETED, {
                    "files_copied": result.files_copied,
                    "bytes_copied": result.bytes_copied,
                    "source_path": result.source_path,
                    "dest_path": result.dest_path,
                    "selective": is_selective
                })
            else:
                await migration_events.broadcast(MigrationEventType.ERROR, {
                    "error": result.error,
                    "files_copied": result.files_copied
                })
        except Exception as e:
            from ...migration import MigrationResult
            _migration_state["result"] = MigrationResult(
                success=False,
                mode=mode,
                source_path="",
                dest_path="",
                files_copied=0,
                bytes_copied=0,
                error=str(e)
            )
            await migration_events.broadcast(MigrationEventType.ERROR, {"error": str(e)})
        finally:
            _migration_state["running"] = False

    # Start background task
    asyncio.create_task(run_migration())

    return {
        "success": True,
        "selective": is_selective,
        "directory_count": len(directory_ids) if is_selective else None,
        "message": "Migration started. Subscribe to /api/settings/migration/events for real-time progress."
    }


@router.get("/migration/status")
async def get_migration_status():
    """Get current migration progress."""
    from ...migration import ImportResult

    response = {
        "running": _migration_state["running"],
        "progress": None,
        "result": None
    }

    if _migration_state["progress"]:
        p = _migration_state["progress"]
        response["progress"] = {
            "phase": p.phase,
            "percent": round(p.percent, 1),
            "current_file": p.current_file,
            "files_copied": p.files_copied,
            "total_files": p.total_files,
            "bytes_copied": p.bytes_copied,
            "total_bytes": p.total_bytes,
            "error": p.error
        }

    if _migration_state["result"]:
        r = _migration_state["result"]
        result_data = {
            "success": r.success,
            "mode": r.mode.value if hasattr(r.mode, 'value') else r.mode,
            "source_path": r.source_path,
            "dest_path": r.dest_path,
            "files_copied": r.files_copied,
            "bytes_copied": r.bytes_copied,
            "error": r.error
        }

        # Add import-specific fields if this was an import operation
        if isinstance(r, ImportResult):
            result_data["import"] = True
            result_data["directories_imported"] = r.directories_imported
            result_data["images_imported"] = r.images_imported
            result_data["images_skipped"] = r.images_skipped
            result_data["tags_created"] = r.tags_created
            result_data["tags_reused"] = r.tags_reused
        else:
            result_data["import"] = False

        response["result"] = result_data

    return response


@router.post("/migration/cleanup")
async def cleanup_migration(request: MigrationRequest):
    """Clean up partially copied data from a failed migration.

    Use this if migration failed and you want to remove incomplete data
    from the destination before retrying.
    """
    from ...migration import cleanup_partial_migration, get_migration_paths, MigrationMode
    from pathlib import Path

    if _migration_state["running"]:
        return {"success": False, "error": "Migration is currently running"}

    try:
        mode = MigrationMode(request.mode)
        _, dest = get_migration_paths(mode)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    success, message = cleanup_partial_migration(dest)
    return {"success": success, "message": message}


@router.post("/migration/delete-source")
async def delete_migration_source(request: MigrationRequest):
    """Delete source data after successful migration.

    WARNING: This permanently deletes data. Only use after verifying
    migration completed successfully.
    """
    from ...migration import delete_source_data, verify_migration, MigrationMode

    if _migration_state["running"]:
        return {"success": False, "error": "Migration is currently running"}

    try:
        mode = MigrationMode(request.mode)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # First verify migration succeeded
    verified, issues = await verify_migration(mode)
    if not verified:
        return {
            "success": False,
            "error": "Migration verification failed. Cannot delete source.",
            "issues": issues
        }

    success, message = await delete_source_data(mode)
    return {"success": success, "message": message}


@router.post("/migration/verify")
async def verify_migration_endpoint(request: MigrationRequest):
    """Verify that migration completed successfully."""
    from ...migration import verify_migration, MigrationMode

    try:
        mode = MigrationMode(request.mode)
    except ValueError as e:
        return {"success": False, "error": str(e), "issues": []}

    success, issues = await verify_migration(mode)
    return {"success": success, "issues": issues}


@router.get("/migration/events")
async def migration_events_stream():
    """Server-Sent Events stream for real-time migration progress.

    Events:
    - migration_started: Migration has begun
    - migration_progress: Progress update (phase, percent, current_file, etc.)
    - migration_completed: Migration finished successfully
    - migration_error: Migration failed with error
    """
    from ...services.events import migration_events

    async def event_generator():
        # Send initial connection message
        yield "data: {\"type\": \"connected\"}\n\n"
        # Stream events
        async for event in migration_events.subscribe():
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# Directory Import Endpoints (import into existing database)
# =============================================================================

@router.post("/migration/import/validate")
async def validate_import_endpoint(request: MigrationRequest):
    """Validate import can proceed (dry run).

    Import differs from migration:
    - Import ADDS directories to an existing destination database
    - Migration REQUIRES an empty destination

    directory_ids is required for import.
    """
    from ...migration import (
        import_directories, validate_import, calculate_import_size,
        MigrationMode, get_migration_paths
    )

    if not request.directory_ids or len(request.directory_ids) == 0:
        return {
            "valid": False,
            "error": "directory_ids is required for import"
        }

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "valid": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Validate import
    errors = validate_import(mode, request.directory_ids)
    if errors:
        return {
            "valid": False,
            "error": "; ".join(errors)
        }

    # Calculate size
    try:
        source, dest = get_migration_paths(mode)
        total_files, total_bytes, images_to_import, images_to_skip, total_db_records, total_tags = calculate_import_size(
            source, request.directory_ids, dest
        )
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    return {
        "valid": True,
        "error": None,
        "source_path": str(source),
        "dest_path": str(dest),
        "files_to_copy": total_files,
        "bytes_to_copy": total_bytes,
        "size_mb": round(total_bytes / 1024 / 1024, 1) if total_bytes else 0,
        "images_to_import": images_to_import,
        "images_to_skip": images_to_skip,
        "directory_count": len(request.directory_ids),
        "total_db_records": total_db_records,
        "total_tags": total_tags
    }


@router.post("/migration/import/start")
async def start_import(request: MigrationRequest):
    """Start directory import (runs in background).

    Import differs from migration:
    - Import ADDS directories to an existing destination database
    - Migration REQUIRES an empty destination

    directory_ids is required for import.
    """
    from ...migration import import_directories, validate_import, MigrationMode, ImportResult
    from ...services.events import migration_events, MigrationEventType

    if _migration_state["running"]:
        return {"success": False, "error": "Migration/import already in progress"}

    if not request.directory_ids or len(request.directory_ids) == 0:
        return {
            "success": False,
            "error": "directory_ids is required for import"
        }

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Validate
    errors = validate_import(mode, request.directory_ids)
    if errors:
        return {"success": False, "error": "; ".join(errors)}

    directory_ids = request.directory_ids

    # Reset state
    _migration_state["running"] = True
    _migration_state["progress"] = None
    _migration_state["result"] = None

    def progress_callback(progress):
        _migration_state["progress"] = progress
        # Broadcast progress via SSE
        asyncio.create_task(migration_events.broadcast(
            MigrationEventType.PROGRESS,
            {
                "phase": progress.phase,
                "percent": round(progress.percent, 1),
                "current_file": progress.current_file,
                "files_copied": progress.files_copied,
                "total_files": progress.total_files,
                "bytes_copied": progress.bytes_copied,
                "total_bytes": progress.total_bytes,
                "error": progress.error
            }
        ))

    async def run_import():
        try:
            # Broadcast start event
            await migration_events.broadcast(MigrationEventType.STARTED, {
                "mode": mode.value,
                "import": True,
                "directory_count": len(directory_ids)
            })

            result = await import_directories(mode, directory_ids, progress_callback=progress_callback)
            _migration_state["result"] = result

            # Broadcast completion/error event
            if result.success:
                await migration_events.broadcast(MigrationEventType.COMPLETED, {
                    "import": True,
                    "directories_imported": result.directories_imported,
                    "images_imported": result.images_imported,
                    "images_skipped": result.images_skipped,
                    "tags_created": result.tags_created,
                    "tags_reused": result.tags_reused,
                    "files_copied": result.files_copied,
                    "bytes_copied": result.bytes_copied,
                    "source_path": result.source_path,
                    "dest_path": result.dest_path
                })
            else:
                await migration_events.broadcast(MigrationEventType.ERROR, {
                    "error": result.error,
                    "files_copied": result.files_copied
                })
        except Exception as e:
            _migration_state["result"] = ImportResult(
                success=False,
                mode=mode,
                source_path="",
                dest_path="",
                directories_imported=0,
                images_imported=0,
                images_skipped=0,
                tags_created=0,
                tags_reused=0,
                files_copied=0,
                bytes_copied=0,
                error=str(e)
            )
            await migration_events.broadcast(MigrationEventType.ERROR, {"error": str(e)})
        finally:
            _migration_state["running"] = False

    # Start background task
    asyncio.create_task(run_import())

    return {
        "success": True,
        "import": True,
        "directory_count": len(directory_ids),
        "message": "Import started. Subscribe to /api/settings/migration/events for real-time progress."
    }
