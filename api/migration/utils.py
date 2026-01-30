"""
Data migration utility functions.

Provides path helpers, disk space checks, size calculations, and validation.
"""
import os
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..config import get_data_dir, get_system_data_dir
from .types import MigrationMode, MIGRATION_ITEMS


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
