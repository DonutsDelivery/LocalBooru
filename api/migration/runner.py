"""
Data migration runner.

Handles full and selective migration between system and portable installations.
"""
import os
import shutil
import asyncio
import sqlite3
from pathlib import Path
from typing import Callable, Optional

from .types import MigrationMode, MigrationProgress, MigrationResult, MIGRATION_ITEMS
from .utils import (
    get_migration_paths,
    calculate_migration_size,
    validate_migration,
    calculate_selective_migration_size,
    check_disk_space,
)


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
