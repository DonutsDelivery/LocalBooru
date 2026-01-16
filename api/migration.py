r"""
LocalBooru data migration module.

Handles bidirectional migration between system and portable installations.
System locations:
  - Windows: %APPDATA%\.localbooru
  - Linux/Mac: ~/.localbooru
Portable location:
  - data/ folder next to the application

Migration includes: library.db (+ WAL files), thumbnails/, preview_cache/,
settings.json, packages/, models/
"""
import os
import shutil
import asyncio
from pathlib import Path
from typing import Callable, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from .config import get_data_dir, get_system_data_dir


class MigrationMode(str, Enum):
    SYSTEM_TO_PORTABLE = "system_to_portable"
    PORTABLE_TO_SYSTEM = "portable_to_system"


@dataclass
class MigrationProgress:
    phase: str
    current_file: str
    files_copied: int
    total_files: int
    bytes_copied: int
    total_bytes: int
    percent: float
    error: Optional[str] = None


@dataclass
class MigrationResult:
    success: bool
    mode: MigrationMode
    source_path: str
    dest_path: str
    files_copied: int
    bytes_copied: int
    error: Optional[str] = None


# Files/directories to migrate
MIGRATION_ITEMS = [
    "library.db",
    "library.db-shm",  # SQLite WAL shared memory (may not exist)
    "library.db-wal",  # SQLite WAL log (may not exist)
    "settings.json",
    "thumbnails",
    "preview_cache",
    "packages",
    "models",
]


def get_portable_data_dir() -> Optional[Path]:
    """Get portable data directory path from environment.

    Returns None if not in portable mode.
    """
    portable_path = os.environ.get('LOCALBOORU_PORTABLE_DATA')
    if portable_path:
        return Path(portable_path)
    return None


def is_portable_mode() -> bool:
    """Check if currently running in portable mode."""
    return get_portable_data_dir() is not None


def get_current_mode() -> Literal["system", "portable"]:
    """Get the current installation mode."""
    return "portable" if is_portable_mode() else "system"


def get_migration_paths(mode: MigrationMode) -> tuple[Path, Path]:
    """Get source and destination paths for the given migration mode.

    Returns:
        tuple of (source_path, dest_path)

    Raises:
        ValueError: If portable mode not configured but trying to migrate to/from it
    """
    system_dir = get_system_data_dir()
    portable_dir = get_portable_data_dir()

    if mode == MigrationMode.SYSTEM_TO_PORTABLE:
        if portable_dir is None:
            raise ValueError(
                "Cannot migrate to portable mode: LOCALBOORU_PORTABLE_DATA not set. "
                "Please run LocalBooru from a portable installation."
            )
        return system_dir, portable_dir
    else:  # PORTABLE_TO_SYSTEM
        if portable_dir is None:
            raise ValueError(
                "Cannot migrate from portable mode: LOCALBOORU_PORTABLE_DATA not set. "
                "You must be running in portable mode to migrate to system."
            )
        return portable_dir, system_dir


def calculate_migration_size(source: Path) -> tuple[int, int]:
    """Calculate total files and bytes to migrate.

    Follows symlinks to get actual file sizes.

    Returns:
        tuple of (total_files, total_bytes)
    """
    total_files = 0
    total_bytes = 0

    for item_name in MIGRATION_ITEMS:
        item_path = source / item_name
        if not item_path.exists():
            continue

        # Resolve symlinks to get actual path
        resolved = item_path.resolve() if item_path.is_symlink() else item_path

        if resolved.is_file():
            total_files += 1
            total_bytes += resolved.stat().st_size
        elif resolved.is_dir():
            # followlinks=True to count symlinked files/dirs
            for root, dirs, files in os.walk(resolved, followlinks=True):
                for f in files:
                    file_path = Path(root) / f
                    if file_path.is_file():  # Skip broken symlinks
                        total_files += 1
                        try:
                            total_bytes += file_path.stat().st_size
                        except OSError:
                            pass  # Skip files we can't stat

    return total_files, total_bytes


def check_disk_space(dest: Path, required_bytes: int) -> tuple[bool, int]:
    """Check if destination has enough disk space.

    Cross-platform: uses shutil.disk_usage which works on Windows, Linux, and macOS.

    Returns:
        tuple of (has_enough_space, available_bytes)
    """
    # Ensure parent directory exists for disk_usage check
    dest.mkdir(parents=True, exist_ok=True)

    # shutil.disk_usage works on all platforms (Windows, Linux, macOS)
    usage = shutil.disk_usage(str(dest))
    available = usage.free

    # Require 10% buffer
    required_with_buffer = int(required_bytes * 1.1)

    return available >= required_with_buffer, available


def validate_migration(
    mode: MigrationMode,
    source: Path,
    dest: Path,
    total_bytes: int
) -> list[str]:
    """Validate migration can proceed.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check source exists and has data
    if not source.exists():
        errors.append(f"Source directory does not exist: {source}")
        return errors

    db_path = source / "library.db"
    if not db_path.exists():
        errors.append(f"No database found at source: {db_path}")

    # Check destination doesn't already have data
    dest_db = dest / "library.db"
    if dest_db.exists():
        errors.append(
            f"Destination already has a database: {dest_db}. "
            "Please backup and remove existing data first."
        )

    # Check disk space
    has_space, available = check_disk_space(dest, total_bytes)
    if not has_space:
        errors.append(
            f"Insufficient disk space. Required: {total_bytes / 1024 / 1024:.1f} MB, "
            f"Available: {available / 1024 / 1024:.1f} MB"
        )

    return errors


async def migrate_data(
    mode: MigrationMode,
    progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
    dry_run: bool = False
) -> MigrationResult:
    """Migrate data between system and portable installations.

    Args:
        mode: Direction of migration
        progress_callback: Optional callback for progress updates
        dry_run: If True, only validate without copying

    Returns:
        MigrationResult with outcome details
    """
    try:
        source, dest = get_migration_paths(mode)
    except ValueError as e:
        return MigrationResult(
            success=False,
            mode=mode,
            source_path="",
            dest_path="",
            files_copied=0,
            bytes_copied=0,
            error=str(e)
        )

    # Calculate sizes
    total_files, total_bytes = calculate_migration_size(source)

    if total_files == 0:
        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=0,
            bytes_copied=0,
            error="No data found to migrate"
        )

    # Validate
    errors = validate_migration(mode, source, dest, total_bytes)
    if errors:
        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=0,
            bytes_copied=0,
            error="; ".join(errors)
        )

    if dry_run:
        return MigrationResult(
            success=True,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=total_files,
            bytes_copied=total_bytes,
            error=None
        )

    # Perform migration
    files_copied = 0
    bytes_copied = 0

    def report_progress(phase: str, current_file: str = ""):
        if progress_callback:
            progress = MigrationProgress(
                phase=phase,
                current_file=current_file,
                files_copied=files_copied,
                total_files=total_files,
                bytes_copied=bytes_copied,
                total_bytes=total_bytes,
                percent=(bytes_copied / total_bytes * 100) if total_bytes > 0 else 0
            )
            progress_callback(progress)

    try:
        # Ensure destination exists
        dest.mkdir(parents=True, exist_ok=True)

        report_progress("starting")

        for item_name in MIGRATION_ITEMS:
            source_item = source / item_name
            dest_item = dest / item_name

            if not source_item.exists():
                continue

            # Resolve symlinks to copy actual content (ensures portability)
            resolved_item = source_item.resolve() if source_item.is_symlink() else source_item

            if resolved_item.is_file():
                report_progress("copying", item_name)
                # follow_symlinks=True copies the actual file content
                shutil.copy2(resolved_item, dest_item, follow_symlinks=True)
                files_copied += 1
                bytes_copied += resolved_item.stat().st_size
                report_progress("copied", item_name)

                # Yield to event loop periodically
                await asyncio.sleep(0)

            elif resolved_item.is_dir():
                # Copy directory recursively, following symlinks
                for root, dirs, files in os.walk(resolved_item, followlinks=True):
                    root_path = Path(root)
                    # Calculate relative path from the resolved source
                    rel_from_resolved = root_path.relative_to(resolved_item)
                    rel_path = Path(item_name) / rel_from_resolved
                    dest_root = dest / rel_path
                    dest_root.mkdir(parents=True, exist_ok=True)

                    for f in files:
                        src_file = root_path / f
                        dst_file = dest_root / f
                        rel_file = str(rel_path / f)

                        # Skip broken symlinks
                        if not src_file.exists():
                            continue

                        try:
                            report_progress("copying", rel_file)
                            shutil.copy2(src_file, dst_file, follow_symlinks=True)
                            files_copied += 1
                            bytes_copied += src_file.stat().st_size
                            report_progress("copied", rel_file)
                        except (OSError, PermissionError) as e:
                            # Log but continue on individual file errors
                            print(f"[Migration] Warning: Could not copy {rel_file}: {e}")

                        # Yield to event loop every 10 files
                        if files_copied % 10 == 0:
                            await asyncio.sleep(0)

        report_progress("complete")

        return MigrationResult(
            success=True,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=files_copied,
            bytes_copied=bytes_copied
        )

    except Exception as e:
        # Report error through progress
        if progress_callback:
            progress_callback(MigrationProgress(
                phase="error",
                current_file="",
                files_copied=files_copied,
                total_files=total_files,
                bytes_copied=bytes_copied,
                total_bytes=total_bytes,
                percent=(bytes_copied / total_bytes * 100) if total_bytes > 0 else 0,
                error=str(e)
            ))

        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=files_copied,
            bytes_copied=bytes_copied,
            error=str(e)
        )


async def get_migration_info() -> dict:
    """Get information about current mode and migration options.

    Returns dict with:
        - current_mode: "system" or "portable"
        - system_path: Path to system data directory
        - portable_path: Path to portable data directory (if available)
        - system_has_data: Whether system directory has existing data
        - portable_has_data: Whether portable directory has existing data
        - system_data_size: Size of system data in bytes
        - portable_data_size: Size of portable data in bytes
    """
    current_mode = get_current_mode()
    system_path = get_system_data_dir()
    portable_path = get_portable_data_dir()

    def get_data_size(path: Path) -> int:
        if not path or not path.exists():
            return 0
        total = 0
        for item_name in MIGRATION_ITEMS:
            item = path / item_name
            if item.is_file():
                total += item.stat().st_size
            elif item.is_dir():
                for root, dirs, files in os.walk(item):
                    for f in files:
                        total += (Path(root) / f).stat().st_size
        return total

    system_has_data = (system_path / "library.db").exists()
    portable_has_data = portable_path and (portable_path / "library.db").exists()

    return {
        "current_mode": current_mode,
        "system_path": str(system_path),
        "portable_path": str(portable_path) if portable_path else None,
        "system_has_data": system_has_data,
        "portable_has_data": portable_has_data,
        "system_data_size": get_data_size(system_path) if system_has_data else 0,
        "portable_data_size": get_data_size(portable_path) if portable_has_data else 0,
    }


def cleanup_partial_migration(dest: Path) -> tuple[bool, str]:
    """Clean up partially copied migration data.

    Called when migration fails to remove incomplete data from destination.

    Args:
        dest: Destination directory to clean up

    Returns:
        tuple of (success, message)
    """
    if not dest.exists():
        return True, "Nothing to clean up"

    cleaned = []
    errors = []

    for item_name in MIGRATION_ITEMS:
        item_path = dest / item_name
        if not item_path.exists():
            continue

        try:
            if item_path.is_file():
                item_path.unlink()
                cleaned.append(item_name)
            elif item_path.is_dir():
                shutil.rmtree(item_path)
                cleaned.append(item_name)
        except Exception as e:
            errors.append(f"{item_name}: {e}")

    if errors:
        return False, f"Cleaned {len(cleaned)} items, but errors occurred: {'; '.join(errors)}"

    return True, f"Cleaned up {len(cleaned)} items"


async def delete_source_data(mode: MigrationMode) -> tuple[bool, str]:
    """Delete source data after successful migration.

    WARNING: This permanently deletes data. Only call after verifying migration succeeded.

    Args:
        mode: The migration mode that was used (determines which location to delete)

    Returns:
        tuple of (success, message)
    """
    try:
        source, _ = get_migration_paths(mode)
    except ValueError as e:
        return False, str(e)

    if not source.exists():
        return True, "Source already removed"

    # Safety check: ensure the other location has data before deleting
    system_path = get_system_data_dir()
    portable_path = get_portable_data_dir()

    if mode == MigrationMode.SYSTEM_TO_PORTABLE:
        # Migrated TO portable, so portable should have data
        if not portable_path or not (portable_path / "library.db").exists():
            return False, "Safety check failed: Portable location has no database"
    else:
        # Migrated TO system, so system should have data
        if not (system_path / "library.db").exists():
            return False, "Safety check failed: System location has no database"

    deleted = []
    errors = []

    for item_name in MIGRATION_ITEMS:
        item_path = source / item_name
        if not item_path.exists():
            continue

        try:
            if item_path.is_file():
                item_path.unlink()
                deleted.append(item_name)
            elif item_path.is_dir():
                shutil.rmtree(item_path)
                deleted.append(item_name)
        except Exception as e:
            errors.append(f"{item_name}: {e}")

    if errors:
        return False, f"Deleted {len(deleted)} items, but errors occurred: {'; '.join(errors)}"

    return True, f"Successfully deleted {len(deleted)} items from source"


async def verify_migration(mode: MigrationMode) -> tuple[bool, list[str]]:
    """Verify that migration completed successfully.

    Checks that all critical files exist at the destination.

    Args:
        mode: The migration mode to verify

    Returns:
        tuple of (success, list of issues found)
    """
    try:
        _, dest = get_migration_paths(mode)
    except ValueError as e:
        return False, [str(e)]

    issues = []

    # Critical: database must exist
    db_path = dest / "library.db"
    if not db_path.exists():
        issues.append("Database file (library.db) not found at destination")

    # Check for WAL files if they exist at source (indicates incomplete checkpoint)
    wal_path = dest / "library.db-wal"
    shm_path = dest / "library.db-shm"
    if wal_path.exists() and wal_path.stat().st_size > 0:
        # WAL exists but this is normal - SQLite will handle it
        pass

    # Optional but important: thumbnails directory
    thumb_dir = dest / "thumbnails"
    if not thumb_dir.exists():
        issues.append("Thumbnails directory not found (will be regenerated)")

    # Settings file
    settings_path = dest / "settings.json"
    if not settings_path.exists():
        issues.append("Settings file not found (will use defaults)")

    return len(issues) == 0 or (len(issues) == 2 and "thumbnails" in issues[0] and "settings" in issues[1]), issues
