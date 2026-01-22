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


async def get_watch_directories_for_migration(mode: MigrationMode) -> list[dict]:
    """Get watch directories from source database with metadata for selective migration.

    Returns list of dicts with:
        - id: directory ID
        - path: directory path
        - name: user-friendly name
        - image_count: number of images in this directory
        - thumbnail_size: size of thumbnails for these images in bytes
        - path_accessible: whether the path would be accessible at destination
    """
    import sqlite3

    try:
        source, dest = get_migration_paths(mode)
    except ValueError:
        return []

    source_db = source / "library.db"
    if not source_db.exists():
        return []

    directories = []

    # Use synchronous sqlite3 to read from source (not our active database)
    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Get all watch directories with image counts
        cursor.execute("""
            SELECT
                wd.id,
                wd.path,
                wd.name,
                COUNT(DISTINCT if2.image_id) as image_count
            FROM watch_directories wd
            LEFT JOIN image_files if2 ON if2.watch_directory_id = wd.id
            GROUP BY wd.id
            ORDER BY wd.path
        """)

        rows = cursor.fetchall()

        # Get thumbnails directory to calculate sizes
        thumb_dir = source / "thumbnails"

        for row in rows:
            dir_id = row['id']
            dir_path = row['path']
            dir_name = row['name'] or Path(dir_path).name

            # Get image hashes for this directory to find thumbnails
            cursor.execute("""
                SELECT DISTINCT i.file_hash
                FROM images i
                JOIN image_files if2 ON if2.image_id = i.id
                WHERE if2.watch_directory_id = ?
            """, (dir_id,))
            hashes = [r[0] for r in cursor.fetchall()]

            # Calculate thumbnail size for these images
            thumb_size = 0
            if thumb_dir.exists():
                for file_hash in hashes:
                    # Thumbnails are stored as hash.webp
                    thumb_path = thumb_dir / f"{file_hash}.webp"
                    if thumb_path.exists():
                        try:
                            thumb_size += thumb_path.stat().st_size
                        except OSError:
                            pass

            # Check if path would be accessible at destination
            # For portable -> system, warn if path is on portable drive
            path_accessible = True
            warning = None

            if mode == MigrationMode.PORTABLE_TO_SYSTEM:
                # Check if directory path is on a removable/portable drive
                dir_path_obj = Path(dir_path)
                if not dir_path_obj.exists():
                    path_accessible = False
                    warning = "Path does not currently exist"
                else:
                    # Simple heuristic: if path is on same drive as portable data, warn
                    portable_path = get_portable_data_dir()
                    if portable_path:
                        try:
                            # On Windows, compare drive letters
                            # On Linux, compare mount points
                            if os.name == 'nt':
                                portable_drive = str(portable_path.resolve())[:3].upper()
                                dir_drive = str(dir_path_obj.resolve())[:3].upper()
                                if portable_drive == dir_drive and portable_drive not in ('C:\\', 'C:/'):
                                    warning = "Path may be on portable drive"
                            else:
                                # Linux: check if on same mount point as portable
                                import subprocess
                                try:
                                    portable_mount = subprocess.run(
                                        ['df', '--output=target', str(portable_path)],
                                        capture_output=True, text=True
                                    ).stdout.strip().split('\n')[-1]
                                    dir_mount = subprocess.run(
                                        ['df', '--output=target', str(dir_path_obj)],
                                        capture_output=True, text=True
                                    ).stdout.strip().split('\n')[-1]
                                    if portable_mount == dir_mount and portable_mount != '/':
                                        warning = "Path may be on portable drive"
                                except:
                                    pass
                        except:
                            pass

            directories.append({
                "id": dir_id,
                "path": dir_path,
                "name": dir_name,
                "image_count": row['image_count'],
                "thumbnail_size": thumb_size,
                "path_accessible": path_accessible,
                "warning": warning
            })

    finally:
        conn.close()

    return directories


def calculate_selective_migration_size(
    source: Path,
    directory_ids: list[int]
) -> tuple[int, int, int]:
    """Calculate size of selective migration.

    Returns:
        tuple of (total_files, total_bytes, thumbnail_bytes)
    """
    import sqlite3

    source_db = source / "library.db"
    if not source_db.exists():
        return 0, 0, 0

    total_files = 0
    total_bytes = 0
    thumbnail_bytes = 0

    # Count non-selective items (settings, packages, models)
    non_db_items = ["settings.json", "packages", "models"]
    for item_name in non_db_items:
        item_path = source / item_name
        if not item_path.exists():
            continue
        resolved = item_path.resolve() if item_path.is_symlink() else item_path
        if resolved.is_file():
            total_files += 1
            total_bytes += resolved.stat().st_size
        elif resolved.is_dir():
            for root, dirs, files in os.walk(resolved, followlinks=True):
                for f in files:
                    file_path = Path(root) / f
                    if file_path.is_file():
                        total_files += 1
                        try:
                            total_bytes += file_path.stat().st_size
                        except OSError:
                            pass

    # Database file size (new DB will be smaller, but estimate full size)
    # In practice, the new DB will be similar size if migrating all directories
    if source_db.exists():
        total_files += 1
        total_bytes += source_db.stat().st_size

    # Calculate thumbnail sizes for selected directories
    conn = sqlite3.connect(str(source_db))
    cursor = conn.cursor()

    try:
        if directory_ids:
            placeholders = ','.join('?' * len(directory_ids))
            cursor.execute(f"""
                SELECT DISTINCT i.file_hash
                FROM images i
                JOIN image_files if2 ON if2.image_id = i.id
                WHERE if2.watch_directory_id IN ({placeholders})
            """, directory_ids)
        else:
            cursor.execute("""
                SELECT DISTINCT i.file_hash
                FROM images i
                JOIN image_files if2 ON if2.image_id = i.id
            """)

        hashes = [r[0] for r in cursor.fetchall()]

        thumb_dir = source / "thumbnails"
        preview_dir = source / "preview_cache"

        for file_hash in hashes:
            # Thumbnails
            thumb_path = thumb_dir / f"{file_hash}.webp"
            if thumb_path.exists():
                total_files += 1
                size = thumb_path.stat().st_size
                total_bytes += size
                thumbnail_bytes += size

            # Preview cache (if exists)
            preview_path = preview_dir / file_hash
            if preview_path.exists() and preview_path.is_dir():
                for f in preview_path.iterdir():
                    if f.is_file():
                        total_files += 1
                        total_bytes += f.stat().st_size

    finally:
        conn.close()

    return total_files, total_bytes, thumbnail_bytes


async def migrate_data_selective(
    mode: MigrationMode,
    directory_ids: list[int],
    progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
    dry_run: bool = False
) -> MigrationResult:
    """Migrate data selectively, only including specified watch directories.

    This creates a new database at the destination with only the selected
    directories' data, then copies relevant thumbnails and other files.

    Args:
        mode: Direction of migration
        directory_ids: List of watch_directory IDs to include (empty = all)
        progress_callback: Optional callback for progress updates
        dry_run: If True, only validate without copying

    Returns:
        MigrationResult with outcome details
    """
    import sqlite3
    import json

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

    source_db = source / "library.db"
    if not source_db.exists():
        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=0,
            bytes_copied=0,
            error="No database found at source"
        )

    # Calculate sizes
    total_files, total_bytes, _ = calculate_selective_migration_size(source, directory_ids)

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

    # Validate destination
    dest_db = dest / "library.db"
    if dest_db.exists():
        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=0,
            bytes_copied=0,
            error=f"Destination already has a database: {dest_db}. Please backup and remove existing data first."
        )

    # Check disk space
    has_space, available = check_disk_space(dest, total_bytes)
    if not has_space:
        return MigrationResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            files_copied=0,
            bytes_copied=0,
            error=f"Insufficient disk space. Required: {total_bytes / 1024 / 1024:.1f} MB, Available: {available / 1024 / 1024:.1f} MB"
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

    # Perform selective migration
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
        dest.mkdir(parents=True, exist_ok=True)
        report_progress("starting")

        # Step 1: Create new database with selected data
        report_progress("creating_database", "library.db")

        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row
        dest_conn = sqlite3.connect(str(dest_db))

        try:
            # Copy schema from source
            source_cursor = source_conn.cursor()
            dest_cursor = dest_conn.cursor()

            # Get all table creation SQL
            source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
            for (sql,) in source_cursor.fetchall():
                if sql and not sql.startswith('CREATE TABLE sqlite_'):
                    dest_cursor.execute(sql)

            # Get all index creation SQL
            source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
            for (sql,) in source_cursor.fetchall():
                if sql:
                    try:
                        dest_cursor.execute(sql)
                    except sqlite3.OperationalError:
                        pass  # Index might already exist from table creation

            dest_conn.commit()

            # Build list of image IDs to migrate
            if directory_ids:
                placeholders = ','.join('?' * len(directory_ids))
                source_cursor.execute(f"""
                    SELECT DISTINCT if2.image_id
                    FROM image_files if2
                    WHERE if2.watch_directory_id IN ({placeholders})
                """, directory_ids)
            else:
                source_cursor.execute("SELECT DISTINCT image_id FROM image_files")

            image_ids = [r[0] for r in source_cursor.fetchall()]
            image_id_set = set(image_ids)

            # Copy watch_directories (only selected)
            report_progress("copying_data", "watch_directories")
            if directory_ids:
                source_cursor.execute(f"SELECT * FROM watch_directories WHERE id IN ({placeholders})", directory_ids)
            else:
                source_cursor.execute("SELECT * FROM watch_directories")

            rows = source_cursor.fetchall()
            if rows:
                cols = [desc[0] for desc in source_cursor.description]
                placeholders_insert = ','.join('?' * len(cols))
                for row in rows:
                    dest_cursor.execute(
                        f"INSERT INTO watch_directories ({','.join(cols)}) VALUES ({placeholders_insert})",
                        tuple(row)
                    )

            # Copy images (only those with files in selected directories)
            report_progress("copying_data", "images")
            if image_ids:
                # Copy in batches to avoid SQL limits
                batch_size = 500
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT * FROM images WHERE id IN ({batch_placeholders})", batch)
                    rows = source_cursor.fetchall()
                    if rows:
                        cols = [desc[0] for desc in source_cursor.description]
                        placeholders_insert = ','.join('?' * len(cols))
                        for row in rows:
                            dest_cursor.execute(
                                f"INSERT INTO images ({','.join(cols)}) VALUES ({placeholders_insert})",
                                tuple(row)
                            )
                    await asyncio.sleep(0)

            # Copy image_files (only for selected directories)
            report_progress("copying_data", "image_files")
            if directory_ids:
                source_cursor.execute(f"SELECT * FROM image_files WHERE watch_directory_id IN ({','.join('?' * len(directory_ids))})", directory_ids)
            else:
                source_cursor.execute("SELECT * FROM image_files")

            rows = source_cursor.fetchall()
            if rows:
                cols = [desc[0] for desc in source_cursor.description]
                placeholders_insert = ','.join('?' * len(cols))
                for row in rows:
                    dest_cursor.execute(
                        f"INSERT INTO image_files ({','.join(cols)}) VALUES ({placeholders_insert})",
                        tuple(row)
                    )

            # Copy image_tags (only for selected images)
            report_progress("copying_data", "image_tags")
            if image_ids:
                batch_size = 500
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT * FROM image_tags WHERE image_id IN ({batch_placeholders})", batch)
                    rows = source_cursor.fetchall()
                    if rows:
                        cols = [desc[0] for desc in source_cursor.description]
                        placeholders_insert = ','.join('?' * len(cols))
                        for row in rows:
                            try:
                                dest_cursor.execute(
                                    f"INSERT INTO image_tags ({','.join(cols)}) VALUES ({placeholders_insert})",
                                    tuple(row)
                                )
                            except sqlite3.IntegrityError:
                                pass  # Skip duplicates
                    await asyncio.sleep(0)

            # Get tag IDs used by migrated images
            tag_ids = set()
            if image_ids:
                batch_size = 500
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT DISTINCT tag_id FROM image_tags WHERE image_id IN ({batch_placeholders})", batch)
                    tag_ids.update(r[0] for r in source_cursor.fetchall())

            # Copy tags (only those used by migrated images)
            report_progress("copying_data", "tags")
            if tag_ids:
                tag_id_list = list(tag_ids)
                batch_size = 500
                for i in range(0, len(tag_id_list), batch_size):
                    batch = tag_id_list[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT * FROM tags WHERE id IN ({batch_placeholders})", batch)
                    rows = source_cursor.fetchall()
                    if rows:
                        cols = [desc[0] for desc in source_cursor.description]
                        placeholders_insert = ','.join('?' * len(cols))
                        for row in rows:
                            dest_cursor.execute(
                                f"INSERT INTO tags ({','.join(cols)}) VALUES ({placeholders_insert})",
                                tuple(row)
                            )

            # Copy tag_aliases for migrated tags
            report_progress("copying_data", "tag_aliases")
            if tag_ids:
                tag_id_list = list(tag_ids)
                batch_size = 500
                for i in range(0, len(tag_id_list), batch_size):
                    batch = tag_id_list[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT * FROM tag_aliases WHERE target_tag_id IN ({batch_placeholders})", batch)
                    rows = source_cursor.fetchall()
                    if rows:
                        cols = [desc[0] for desc in source_cursor.description]
                        placeholders_insert = ','.join('?' * len(cols))
                        for row in rows:
                            try:
                                dest_cursor.execute(
                                    f"INSERT INTO tag_aliases ({','.join(cols)}) VALUES ({placeholders_insert})",
                                    tuple(row)
                                )
                            except sqlite3.IntegrityError:
                                pass

            # Copy settings table
            report_progress("copying_data", "settings")
            source_cursor.execute("SELECT * FROM settings")
            rows = source_cursor.fetchall()
            if rows:
                cols = [desc[0] for desc in source_cursor.description]
                placeholders_insert = ','.join('?' * len(cols))
                for row in rows:
                    try:
                        dest_cursor.execute(
                            f"INSERT INTO settings ({','.join(cols)}) VALUES ({placeholders_insert})",
                            tuple(row)
                        )
                    except sqlite3.IntegrityError:
                        pass

            # Copy users table
            report_progress("copying_data", "users")
            source_cursor.execute("SELECT * FROM users")
            rows = source_cursor.fetchall()
            if rows:
                cols = [desc[0] for desc in source_cursor.description]
                placeholders_insert = ','.join('?' * len(cols))
                for row in rows:
                    try:
                        dest_cursor.execute(
                            f"INSERT INTO users ({','.join(cols)}) VALUES ({placeholders_insert})",
                            tuple(row)
                        )
                    except sqlite3.IntegrityError:
                        pass

            dest_conn.commit()
            files_copied += 1
            bytes_copied += dest_db.stat().st_size
            report_progress("copied_database", "library.db")

        finally:
            source_conn.close()
            dest_conn.close()

        await asyncio.sleep(0)

        # Step 2: Get file hashes for selected images (for thumbnail copying)
        source_conn = sqlite3.connect(str(source_db))
        source_cursor = source_conn.cursor()

        try:
            if directory_ids:
                placeholders = ','.join('?' * len(directory_ids))
                source_cursor.execute(f"""
                    SELECT DISTINCT i.file_hash
                    FROM images i
                    JOIN image_files if2 ON if2.image_id = i.id
                    WHERE if2.watch_directory_id IN ({placeholders})
                """, directory_ids)
            else:
                source_cursor.execute("""
                    SELECT DISTINCT i.file_hash
                    FROM images i
                    JOIN image_files if2 ON if2.image_id = i.id
                """)

            selected_hashes = set(r[0] for r in source_cursor.fetchall())
        finally:
            source_conn.close()

        # Step 3: Copy thumbnails for selected images
        source_thumb_dir = source / "thumbnails"
        dest_thumb_dir = dest / "thumbnails"

        if source_thumb_dir.exists():
            dest_thumb_dir.mkdir(parents=True, exist_ok=True)

            for file_hash in selected_hashes:
                thumb_file = f"{file_hash}.webp"
                source_thumb = source_thumb_dir / thumb_file
                dest_thumb = dest_thumb_dir / thumb_file

                if source_thumb.exists():
                    report_progress("copying", f"thumbnails/{thumb_file}")
                    shutil.copy2(source_thumb, dest_thumb)
                    files_copied += 1
                    bytes_copied += source_thumb.stat().st_size
                    report_progress("copied", f"thumbnails/{thumb_file}")

                if files_copied % 50 == 0:
                    await asyncio.sleep(0)

        # Step 4: Copy preview cache for selected images
        source_preview_dir = source / "preview_cache"
        dest_preview_dir = dest / "preview_cache"

        if source_preview_dir.exists():
            dest_preview_dir.mkdir(parents=True, exist_ok=True)

            for file_hash in selected_hashes:
                source_preview = source_preview_dir / file_hash
                dest_preview = dest_preview_dir / file_hash

                if source_preview.exists() and source_preview.is_dir():
                    dest_preview.mkdir(parents=True, exist_ok=True)
                    for f in source_preview.iterdir():
                        if f.is_file():
                            report_progress("copying", f"preview_cache/{file_hash}/{f.name}")
                            shutil.copy2(f, dest_preview / f.name)
                            files_copied += 1
                            bytes_copied += f.stat().st_size

                if files_copied % 50 == 0:
                    await asyncio.sleep(0)

        # Step 5: Copy other files (settings.json, packages/, models/)
        non_selective_items = ["settings.json", "packages", "models"]

        for item_name in non_selective_items:
            source_item = source / item_name
            dest_item = dest / item_name

            if not source_item.exists():
                continue

            resolved_item = source_item.resolve() if source_item.is_symlink() else source_item

            if resolved_item.is_file():
                report_progress("copying", item_name)
                shutil.copy2(resolved_item, dest_item, follow_symlinks=True)
                files_copied += 1
                bytes_copied += resolved_item.stat().st_size
                report_progress("copied", item_name)
                await asyncio.sleep(0)

            elif resolved_item.is_dir():
                for root, dirs, files in os.walk(resolved_item, followlinks=True):
                    root_path = Path(root)
                    rel_from_resolved = root_path.relative_to(resolved_item)
                    rel_path = Path(item_name) / rel_from_resolved
                    dest_root = dest / rel_path
                    dest_root.mkdir(parents=True, exist_ok=True)

                    for f in files:
                        src_file = root_path / f
                        dst_file = dest_root / f
                        rel_file = str(rel_path / f)

                        if not src_file.exists():
                            continue

                        try:
                            report_progress("copying", rel_file)
                            shutil.copy2(src_file, dst_file, follow_symlinks=True)
                            files_copied += 1
                            bytes_copied += src_file.stat().st_size
                            report_progress("copied", rel_file)
                        except (OSError, PermissionError) as e:
                            print(f"[Migration] Warning: Could not copy {rel_file}: {e}")

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


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    mode: MigrationMode
    source_path: str
    dest_path: str
    directories_imported: int
    images_imported: int
    images_skipped: int  # Duplicates by file_hash
    tags_created: int
    tags_reused: int
    files_copied: int
    bytes_copied: int
    error: Optional[str] = None


def validate_import(
    mode: MigrationMode,
    directory_ids: list[int]
) -> list[str]:
    """Validate that import can proceed.

    Unlike migration, import REQUIRES destination to have existing data.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        source, dest = get_migration_paths(mode)
    except ValueError as e:
        errors.append(str(e))
        return errors

    # Source must have database
    source_db = source / "library.db"
    if not source_db.exists():
        errors.append(f"No database found at source: {source_db}")

    # Destination must have database (opposite of migration!)
    dest_db = dest / "library.db"
    if not dest_db.exists():
        errors.append(
            f"No database at destination: {dest_db}. "
            "Use migration instead of import for empty destinations."
        )

    if not directory_ids:
        errors.append("No directories selected for import")

    # Check for watch directory path conflicts
    if source_db.exists() and dest_db.exists() and directory_ids:
        import sqlite3

        source_conn = sqlite3.connect(str(source_db))
        dest_conn = sqlite3.connect(str(dest_db))

        try:
            source_cursor = source_conn.cursor()
            dest_cursor = dest_conn.cursor()

            # Get paths of directories being imported
            placeholders = ','.join('?' * len(directory_ids))
            source_cursor.execute(
                f"SELECT path FROM watch_directories WHERE id IN ({placeholders})",
                directory_ids
            )
            source_paths = set(r[0] for r in source_cursor.fetchall())

            # Get existing paths in destination
            dest_cursor.execute("SELECT path FROM watch_directories")
            dest_paths = set(r[0] for r in dest_cursor.fetchall())

            # Check for conflicts
            conflicts = source_paths & dest_paths
            if conflicts:
                for path in conflicts:
                    errors.append(f"Watch directory already exists in destination: {path}")

        finally:
            source_conn.close()
            dest_conn.close()

    return errors


def calculate_import_size(
    source: Path,
    directory_ids: list[int],
    dest: Path
) -> tuple[int, int, int, int]:
    """Calculate size of import operation.

    Returns:
        tuple of (total_files, total_bytes, images_to_import, images_to_skip)
    """
    import sqlite3

    source_db = source / "library.db"
    dest_db = dest / "library.db"

    if not source_db.exists() or not dest_db.exists():
        return 0, 0, 0, 0

    total_files = 0
    total_bytes = 0
    images_to_import = 0
    images_to_skip = 0

    source_conn = sqlite3.connect(str(source_db))
    dest_conn = sqlite3.connect(str(dest_db))

    try:
        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()

        # Get file hashes from destination for duplicate detection
        dest_cursor.execute("SELECT file_hash FROM images")
        dest_hashes = set(r[0] for r in dest_cursor.fetchall())

        # Get file hashes from selected directories
        placeholders = ','.join('?' * len(directory_ids))
        source_cursor.execute(f"""
            SELECT DISTINCT i.file_hash
            FROM images i
            JOIN image_files if2 ON if2.image_id = i.id
            WHERE if2.watch_directory_id IN ({placeholders})
        """, directory_ids)

        source_hashes = []
        for (file_hash,) in source_cursor.fetchall():
            if file_hash in dest_hashes:
                images_to_skip += 1
            else:
                images_to_import += 1
                source_hashes.append(file_hash)

        # Calculate thumbnail sizes for non-duplicate images
        thumb_dir = source / "thumbnails"
        preview_dir = source / "preview_cache"

        for file_hash in source_hashes:
            # Thumbnails
            thumb_path = thumb_dir / f"{file_hash}.webp"
            if thumb_path.exists():
                total_files += 1
                total_bytes += thumb_path.stat().st_size

            # Preview cache
            preview_path = preview_dir / file_hash
            if preview_path.exists() and preview_path.is_dir():
                for f in preview_path.iterdir():
                    if f.is_file():
                        total_files += 1
                        total_bytes += f.stat().st_size

    finally:
        source_conn.close()
        dest_conn.close()

    return total_files, total_bytes, images_to_import, images_to_skip


async def import_directories(
    mode: MigrationMode,
    directory_ids: list[int],
    progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
    dry_run: bool = False
) -> ImportResult:
    """Import watch directories from source into existing destination database.

    Unlike migration, this:
    - Requires destination to have existing data
    - Remaps IDs to avoid conflicts
    - Deduplicates tags by name
    - Skips images that already exist (by file_hash)

    Args:
        mode: Direction of import
        directory_ids: List of watch_directory IDs to import
        progress_callback: Optional callback for progress updates
        dry_run: If True, only validate without copying

    Returns:
        ImportResult with outcome details
    """
    import sqlite3

    try:
        source, dest = get_migration_paths(mode)
    except ValueError as e:
        return ImportResult(
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

    source_db = source / "library.db"
    dest_db = dest / "library.db"

    # Validate
    errors = validate_import(mode, directory_ids)
    if errors:
        return ImportResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            directories_imported=0,
            images_imported=0,
            images_skipped=0,
            tags_created=0,
            tags_reused=0,
            files_copied=0,
            bytes_copied=0,
            error="; ".join(errors)
        )

    # Calculate sizes
    total_files, total_bytes, images_to_import, images_to_skip = calculate_import_size(
        source, directory_ids, dest
    )

    if dry_run:
        return ImportResult(
            success=True,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            directories_imported=len(directory_ids),
            images_imported=images_to_import,
            images_skipped=images_to_skip,
            tags_created=0,  # Unknown until actual import
            tags_reused=0,
            files_copied=total_files,
            bytes_copied=total_bytes,
            error=None
        )

    # Perform import
    files_copied = 0
    bytes_copied = 0
    directories_imported = 0
    images_imported = 0
    actual_images_skipped = 0
    tags_created = 0
    tags_reused = 0

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
        report_progress("starting")

        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row
        dest_conn = sqlite3.connect(str(dest_db))
        dest_conn.row_factory = sqlite3.Row

        try:
            source_cursor = source_conn.cursor()
            dest_cursor = dest_conn.cursor()

            # === Phase 1: Build ID mappings ===
            report_progress("analyzing", "Building ID mappings")

            # Get max IDs from destination
            dest_cursor.execute("SELECT COALESCE(MAX(id), 0) FROM watch_directories")
            max_watch_dir_id = dest_cursor.fetchone()[0]

            dest_cursor.execute("SELECT COALESCE(MAX(id), 0) FROM images")
            max_image_id = dest_cursor.fetchone()[0]

            dest_cursor.execute("SELECT COALESCE(MAX(id), 0) FROM image_files")
            max_image_file_id = dest_cursor.fetchone()[0]

            dest_cursor.execute("SELECT COALESCE(MAX(id), 0) FROM tags")
            max_tag_id = dest_cursor.fetchone()[0]

            dest_cursor.execute("SELECT COALESCE(MAX(id), 0) FROM tag_aliases")
            max_alias_id = dest_cursor.fetchone()[0]

            # Get existing tags in destination (for deduplication)
            dest_cursor.execute("SELECT id, name FROM tags")
            dest_tags_by_name = {row['name']: row['id'] for row in dest_cursor.fetchall()}

            # Get existing file hashes in destination (for duplicate detection)
            dest_cursor.execute("SELECT file_hash FROM images")
            dest_hashes = set(r[0] for r in dest_cursor.fetchall())

            # ID mapping tables
            watch_dir_id_map = {}  # old_id -> new_id
            image_id_map = {}  # old_id -> new_id
            image_file_id_map = {}  # old_id -> new_id
            tag_id_map = {}  # old_id -> new_id

            # Track which source images to skip (duplicates)
            images_to_skip_ids = set()

            # === Phase 2: Import watch directories ===
            report_progress("importing", "watch_directories")

            placeholders = ','.join('?' * len(directory_ids))
            source_cursor.execute(
                f"SELECT * FROM watch_directories WHERE id IN ({placeholders})",
                directory_ids
            )
            watch_dir_rows = source_cursor.fetchall()
            watch_dir_cols = [desc[0] for desc in source_cursor.description]

            for row in watch_dir_rows:
                old_id = row['id']
                new_id = max_watch_dir_id + 1
                max_watch_dir_id = new_id
                watch_dir_id_map[old_id] = new_id

                # Build new row with remapped ID
                new_row = list(row)
                id_idx = watch_dir_cols.index('id')
                new_row[id_idx] = new_id

                placeholders_insert = ','.join('?' * len(watch_dir_cols))
                dest_cursor.execute(
                    f"INSERT INTO watch_directories ({','.join(watch_dir_cols)}) VALUES ({placeholders_insert})",
                    new_row
                )
                directories_imported += 1

            # === Phase 3: Identify images to import (non-duplicates) ===
            report_progress("importing", "Identifying images")

            placeholders = ','.join('?' * len(directory_ids))
            source_cursor.execute(f"""
                SELECT DISTINCT i.*
                FROM images i
                JOIN image_files if2 ON if2.image_id = i.id
                WHERE if2.watch_directory_id IN ({placeholders})
            """, directory_ids)
            image_rows = source_cursor.fetchall()
            image_cols = [desc[0] for desc in source_cursor.description]

            # First pass: identify which images to skip
            for row in image_rows:
                if row['file_hash'] in dest_hashes:
                    images_to_skip_ids.add(row['id'])

            # === Phase 4: Import images (non-duplicates only) ===
            report_progress("importing", "images")

            for row in image_rows:
                old_id = row['id']

                if old_id in images_to_skip_ids:
                    actual_images_skipped += 1
                    continue

                new_id = max_image_id + 1
                max_image_id = new_id
                image_id_map[old_id] = new_id

                new_row = list(row)
                id_idx = image_cols.index('id')
                new_row[id_idx] = new_id

                placeholders_insert = ','.join('?' * len(image_cols))
                dest_cursor.execute(
                    f"INSERT INTO images ({','.join(image_cols)}) VALUES ({placeholders_insert})",
                    new_row
                )
                images_imported += 1

            await asyncio.sleep(0)

            # === Phase 5: Import image_files ===
            report_progress("importing", "image_files")

            source_cursor.execute(
                f"SELECT * FROM image_files WHERE watch_directory_id IN ({','.join('?' * len(directory_ids))})",
                directory_ids
            )
            image_file_rows = source_cursor.fetchall()
            image_file_cols = [desc[0] for desc in source_cursor.description]

            for row in image_file_rows:
                old_image_id = row['image_id']

                # Skip if parent image was skipped (duplicate)
                if old_image_id in images_to_skip_ids:
                    continue

                old_id = row['id']
                new_id = max_image_file_id + 1
                max_image_file_id = new_id
                image_file_id_map[old_id] = new_id

                new_row = list(row)
                id_idx = image_file_cols.index('id')
                image_id_idx = image_file_cols.index('image_id')
                watch_dir_idx = image_file_cols.index('watch_directory_id')

                new_row[id_idx] = new_id
                new_row[image_id_idx] = image_id_map[old_image_id]
                new_row[watch_dir_idx] = watch_dir_id_map[row['watch_directory_id']]

                placeholders_insert = ','.join('?' * len(image_file_cols))
                dest_cursor.execute(
                    f"INSERT INTO image_files ({','.join(image_file_cols)}) VALUES ({placeholders_insert})",
                    new_row
                )

            await asyncio.sleep(0)

            # === Phase 6: Import tags (deduplicate by name) ===
            report_progress("importing", "tags")

            # Get tag IDs used by imported images
            imported_image_ids = list(image_id_map.keys())
            source_tag_ids = set()

            if imported_image_ids:
                batch_size = 500
                for i in range(0, len(imported_image_ids), batch_size):
                    batch = imported_image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(
                        f"SELECT DISTINCT tag_id FROM image_tags WHERE image_id IN ({batch_placeholders})",
                        batch
                    )
                    source_tag_ids.update(r[0] for r in source_cursor.fetchall())

            # Get tag data and build mapping
            if source_tag_ids:
                tag_id_list = list(source_tag_ids)
                batch_size = 500
                for i in range(0, len(tag_id_list), batch_size):
                    batch = tag_id_list[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(
                        f"SELECT * FROM tags WHERE id IN ({batch_placeholders})",
                        batch
                    )
                    tag_rows = source_cursor.fetchall()
                    tag_cols = [desc[0] for desc in source_cursor.description]

                    for row in tag_rows:
                        old_id = row['id']
                        tag_name = row['name']

                        if tag_name in dest_tags_by_name:
                            # Tag exists in destination - reuse it
                            tag_id_map[old_id] = dest_tags_by_name[tag_name]
                            tags_reused += 1
                        else:
                            # New tag - create it
                            new_id = max_tag_id + 1
                            max_tag_id = new_id
                            tag_id_map[old_id] = new_id
                            dest_tags_by_name[tag_name] = new_id

                            new_row = list(row)
                            id_idx = tag_cols.index('id')
                            new_row[id_idx] = new_id
                            # Reset post_count - will be recalculated
                            if 'post_count' in tag_cols:
                                post_count_idx = tag_cols.index('post_count')
                                new_row[post_count_idx] = 0

                            placeholders_insert = ','.join('?' * len(tag_cols))
                            dest_cursor.execute(
                                f"INSERT INTO tags ({','.join(tag_cols)}) VALUES ({placeholders_insert})",
                                new_row
                            )
                            tags_created += 1

            await asyncio.sleep(0)

            # === Phase 7: Import image_tags ===
            report_progress("importing", "image_tags")

            if imported_image_ids:
                batch_size = 500
                for i in range(0, len(imported_image_ids), batch_size):
                    batch = imported_image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(
                        f"SELECT * FROM image_tags WHERE image_id IN ({batch_placeholders})",
                        batch
                    )
                    image_tag_rows = source_cursor.fetchall()
                    image_tag_cols = [desc[0] for desc in source_cursor.description]

                    for row in image_tag_rows:
                        old_image_id = row['image_id']
                        old_tag_id = row['tag_id']

                        if old_image_id not in image_id_map or old_tag_id not in tag_id_map:
                            continue

                        new_row = list(row)
                        image_id_idx = image_tag_cols.index('image_id')
                        tag_id_idx = image_tag_cols.index('tag_id')
                        new_row[image_id_idx] = image_id_map[old_image_id]
                        new_row[tag_id_idx] = tag_id_map[old_tag_id]

                        placeholders_insert = ','.join('?' * len(image_tag_cols))
                        try:
                            dest_cursor.execute(
                                f"INSERT INTO image_tags ({','.join(image_tag_cols)}) VALUES ({placeholders_insert})",
                                new_row
                            )
                        except sqlite3.IntegrityError:
                            pass  # Skip duplicates

                    await asyncio.sleep(0)

            # === Phase 8: Update tag post counts ===
            report_progress("importing", "Updating tag counts")

            # Recalculate post_count for all affected tags
            affected_tag_ids = list(set(tag_id_map.values()))
            if affected_tag_ids:
                batch_size = 100
                for i in range(0, len(affected_tag_ids), batch_size):
                    batch = affected_tag_ids[i:i + batch_size]
                    for tag_id in batch:
                        dest_cursor.execute(
                            "UPDATE tags SET post_count = (SELECT COUNT(*) FROM image_tags WHERE tag_id = ?) WHERE id = ?",
                            (tag_id, tag_id)
                        )

            dest_conn.commit()
            report_progress("database_complete", "library.db")

        finally:
            source_conn.close()
            dest_conn.close()

        await asyncio.sleep(0)

        # === Phase 9: Copy thumbnails for imported images ===
        report_progress("copying", "thumbnails")

        # Get file hashes for imported images
        source_conn = sqlite3.connect(str(source_db))
        source_cursor = source_conn.cursor()

        try:
            imported_image_ids_list = list(image_id_map.keys())
            imported_hashes = []

            if imported_image_ids_list:
                batch_size = 500
                for i in range(0, len(imported_image_ids_list), batch_size):
                    batch = imported_image_ids_list[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(
                        f"SELECT file_hash FROM images WHERE id IN ({batch_placeholders})",
                        batch
                    )
                    imported_hashes.extend(r[0] for r in source_cursor.fetchall())
        finally:
            source_conn.close()

        source_thumb_dir = source / "thumbnails"
        dest_thumb_dir = dest / "thumbnails"

        if source_thumb_dir.exists() and imported_hashes:
            dest_thumb_dir.mkdir(parents=True, exist_ok=True)

            for file_hash in imported_hashes:
                thumb_file = f"{file_hash}.webp"
                source_thumb = source_thumb_dir / thumb_file
                dest_thumb = dest_thumb_dir / thumb_file

                if source_thumb.exists() and not dest_thumb.exists():
                    report_progress("copying", f"thumbnails/{thumb_file}")
                    shutil.copy2(source_thumb, dest_thumb)
                    files_copied += 1
                    bytes_copied += source_thumb.stat().st_size

                if files_copied % 50 == 0:
                    await asyncio.sleep(0)

        # === Phase 10: Copy preview cache for imported images ===
        report_progress("copying", "preview_cache")

        source_preview_dir = source / "preview_cache"
        dest_preview_dir = dest / "preview_cache"

        if source_preview_dir.exists() and imported_hashes:
            dest_preview_dir.mkdir(parents=True, exist_ok=True)

            for file_hash in imported_hashes:
                source_preview = source_preview_dir / file_hash
                dest_preview = dest_preview_dir / file_hash

                if source_preview.exists() and source_preview.is_dir():
                    dest_preview.mkdir(parents=True, exist_ok=True)
                    for f in source_preview.iterdir():
                        if f.is_file():
                            dest_file = dest_preview / f.name
                            if not dest_file.exists():
                                report_progress("copying", f"preview_cache/{file_hash}/{f.name}")
                                shutil.copy2(f, dest_file)
                                files_copied += 1
                                bytes_copied += f.stat().st_size

                if files_copied % 50 == 0:
                    await asyncio.sleep(0)

        report_progress("complete")

        return ImportResult(
            success=True,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            directories_imported=directories_imported,
            images_imported=images_imported,
            images_skipped=actual_images_skipped,
            tags_created=tags_created,
            tags_reused=tags_reused,
            files_copied=files_copied,
            bytes_copied=bytes_copied
        )

    except Exception as e:
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

        return ImportResult(
            success=False,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            directories_imported=directories_imported,
            images_imported=images_imported,
            images_skipped=actual_images_skipped,
            tags_created=tags_created,
            tags_reused=tags_reused,
            files_copied=files_copied,
            bytes_copied=bytes_copied,
            error=str(e)
        )
