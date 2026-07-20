"""
Data import runner.

Handles importing watch directories from another installation into an existing database.
"""
import os
import shutil
import asyncio
import sqlite3
from pathlib import Path
from typing import Callable, Optional

from .types import MigrationMode, MigrationProgress, ImportResult
from .utils import get_migration_paths, check_disk_space


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
) -> tuple[int, int, int, int, int, int]:
    """Calculate size of import operation.

    Returns:
        tuple of (total_files, total_bytes, images_to_import, images_to_skip, total_db_records, total_tags)
    """
    source_db = source / "library.db"
    dest_db = dest / "library.db"

    if not source_db.exists() or not dest_db.exists():
        return 0, 0, 0, 0, 0, 0

    total_files = 0
    total_bytes = 0
    images_to_import = 0
    images_to_skip = 0
    total_db_records = 0
    total_tags = 0

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

        # Count DB records to be inserted
        total_db_records += len(directory_ids)  # watch_directories
        total_db_records += images_to_import  # images

        # Count image_files for non-duplicate images
        if source_hashes:
            # Get image IDs for non-duplicate hashes
            batch_size = 500
            image_ids = []
            for i in range(0, len(source_hashes), batch_size):
                batch = source_hashes[i:i + batch_size]
                batch_placeholders = ','.join('?' * len(batch))
                source_cursor.execute(f"SELECT id FROM images WHERE file_hash IN ({batch_placeholders})", batch)
                image_ids.extend(r[0] for r in source_cursor.fetchall())

            # Count image_files
            if image_ids:
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"""
                        SELECT COUNT(*) FROM image_files
                        WHERE image_id IN ({batch_placeholders}) AND watch_directory_id IN ({','.join('?' * len(directory_ids))})
                    """, batch + directory_ids)
                    total_db_records += source_cursor.fetchone()[0]

                # Count image_tags
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT COUNT(*) FROM image_tags WHERE image_id IN ({batch_placeholders})", batch)
                    total_db_records += source_cursor.fetchone()[0]

                # Count unique tags
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    source_cursor.execute(f"SELECT COUNT(DISTINCT tag_id) FROM image_tags WHERE image_id IN ({batch_placeholders})", batch)
                    total_tags += source_cursor.fetchone()[0]

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

    return total_files, total_bytes, images_to_import, images_to_skip, total_db_records, total_tags


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
    - Uses batch inserts for better performance

    Args:
        mode: Direction of import
        directory_ids: List of watch_directory IDs to import
        progress_callback: Optional callback for progress updates
        dry_run: If True, only validate without copying

    Returns:
        ImportResult with outcome details
    """
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

    # Calculate sizes including DB records for accurate progress
    total_files, total_bytes, images_to_import, images_to_skip, total_db_records, total_tags = calculate_import_size(
        source, directory_ids, dest
    )

    # Total operations = DB records + file copies
    # Weight DB operations as 1 unit each, file copies proportional to bytes
    # For progress, we'll track: records_done + (bytes_copied / total_bytes * total_files)
    total_operations = total_db_records + total_files

    if dry_run:
        return ImportResult(
            success=True,
            mode=mode,
            source_path=str(source),
            dest_path=str(dest),
            directories_imported=len(directory_ids),
            images_imported=images_to_import,
            images_skipped=images_to_skip,
            tags_created=total_tags,  # Estimate
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
    records_done = 0

    def report_progress(phase: str, current_file: str = ""):
        if progress_callback:
            # Calculate overall progress based on DB records + file copies
            if total_operations > 0:
                file_progress = (files_copied / total_files) if total_files > 0 else 0
                db_weight = total_db_records / total_operations
                file_weight = total_files / total_operations
                percent = (records_done / total_db_records * db_weight * 100 if total_db_records > 0 else 0) + \
                         (file_progress * file_weight * 100)
            else:
                percent = 100 if phase == "complete" else 0

            progress = MigrationProgress(
                phase=phase,
                current_file=current_file,
                files_copied=files_copied,
                total_files=total_files,
                bytes_copied=bytes_copied,
                total_bytes=total_bytes,
                percent=min(percent, 100)
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

            # === Phase 2: Import watch directories (batch) ===
            report_progress("importing", f"watch_directories (0/{len(directory_ids)})")

            placeholders = ','.join('?' * len(directory_ids))
            source_cursor.execute(
                f"SELECT * FROM watch_directories WHERE id IN ({placeholders})",
                directory_ids
            )
            watch_dir_rows = source_cursor.fetchall()
            watch_dir_cols = [desc[0] for desc in source_cursor.description]
            id_idx = watch_dir_cols.index('id')

            # Build all new rows for batch insert
            watch_dir_batch = []
            for row in watch_dir_rows:
                old_id = row['id']
                new_id = max_watch_dir_id + 1
                max_watch_dir_id = new_id
                watch_dir_id_map[old_id] = new_id

                new_row = list(row)
                new_row[id_idx] = new_id
                watch_dir_batch.append(new_row)

            # Batch insert all directories at once
            if watch_dir_batch:
                placeholders_insert = ','.join('?' * len(watch_dir_cols))
                dest_cursor.executemany(
                    f"INSERT INTO watch_directories ({','.join(watch_dir_cols)}) VALUES ({placeholders_insert})",
                    watch_dir_batch
                )
                directories_imported = len(watch_dir_batch)
                records_done += directories_imported

            report_progress("importing", f"watch_directories ({directories_imported}/{len(directory_ids)})")

            # === Phase 3: Identify and import images ===
            report_progress("importing", "Analyzing images...")

            placeholders = ','.join('?' * len(directory_ids))
            source_cursor.execute(f"""
                SELECT DISTINCT i.*
                FROM images i
                JOIN image_files if2 ON if2.image_id = i.id
                WHERE if2.watch_directory_id IN ({placeholders})
            """, directory_ids)
            image_rows = source_cursor.fetchall()
            image_cols = [desc[0] for desc in source_cursor.description]
            id_idx = image_cols.index('id')
            hash_idx = image_cols.index('file_hash')

            # Identify duplicates and build batch for non-duplicates
            image_batch = []
            total_source_images = len(image_rows)
            for row in image_rows:
                old_id = row['id']
                if row['file_hash'] in dest_hashes:
                    images_to_skip_ids.add(old_id)
                    actual_images_skipped += 1
                else:
                    new_id = max_image_id + 1
                    max_image_id = new_id
                    image_id_map[old_id] = new_id

                    new_row = list(row)
                    new_row[id_idx] = new_id
                    image_batch.append(new_row)

            # Batch insert images
            report_progress("importing", f"images (0/{len(image_batch)})")
            if image_batch:
                placeholders_insert = ','.join('?' * len(image_cols))
                # Insert in batches of 1000 for memory efficiency
                batch_size = 1000
                for i in range(0, len(image_batch), batch_size):
                    batch = image_batch[i:i + batch_size]
                    dest_cursor.executemany(
                        f"INSERT INTO images ({','.join(image_cols)}) VALUES ({placeholders_insert})",
                        batch
                    )
                    images_imported += len(batch)
                    records_done += len(batch)
                    report_progress("importing", f"images ({images_imported}/{len(image_batch)})")
                    await asyncio.sleep(0)

            # === Phase 4: Import image_files (batch) ===
            report_progress("importing", "image_files...")

            source_cursor.execute(
                f"SELECT * FROM image_files WHERE watch_directory_id IN ({','.join('?' * len(directory_ids))})",
                directory_ids
            )
            image_file_rows = source_cursor.fetchall()
            image_file_cols = [desc[0] for desc in source_cursor.description]
            id_idx = image_file_cols.index('id')
            image_id_idx = image_file_cols.index('image_id')
            watch_dir_idx = image_file_cols.index('watch_directory_id')

            image_file_batch = []
            for row in image_file_rows:
                old_image_id = row['image_id']
                if old_image_id in images_to_skip_ids:
                    continue

                old_id = row['id']
                new_id = max_image_file_id + 1
                max_image_file_id = new_id
                image_file_id_map[old_id] = new_id

                new_row = list(row)
                new_row[id_idx] = new_id
                new_row[image_id_idx] = image_id_map[old_image_id]
                new_row[watch_dir_idx] = watch_dir_id_map[row['watch_directory_id']]
                image_file_batch.append(new_row)

            # Batch insert
            if image_file_batch:
                placeholders_insert = ','.join('?' * len(image_file_cols))
                batch_size = 1000
                for i in range(0, len(image_file_batch), batch_size):
                    batch = image_file_batch[i:i + batch_size]
                    dest_cursor.executemany(
                        f"INSERT INTO image_files ({','.join(image_file_cols)}) VALUES ({placeholders_insert})",
                        batch
                    )
                    records_done += len(batch)
                    report_progress("importing", f"image_files ({min(i + batch_size, len(image_file_batch))}/{len(image_file_batch)})")
                    await asyncio.sleep(0)

            # === Phase 5: Import tags (deduplicate by name, batch) ===
            report_progress("importing", "tags...")

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

            # Get all tag data at once and build batch
            tag_batch = []
            tag_cols = None
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
                    if tag_rows:
                        if tag_cols is None:
                            tag_cols = [desc[0] for desc in source_cursor.description]
                        id_idx = tag_cols.index('id')
                        name_idx = tag_cols.index('name')
                        post_count_idx = tag_cols.index('post_count') if 'post_count' in tag_cols else None

                        for row in tag_rows:
                            old_id = row['id']
                            tag_name = row['name']

                            if tag_name in dest_tags_by_name:
                                tag_id_map[old_id] = dest_tags_by_name[tag_name]
                                tags_reused += 1
                            else:
                                new_id = max_tag_id + 1
                                max_tag_id = new_id
                                tag_id_map[old_id] = new_id
                                dest_tags_by_name[tag_name] = new_id

                                new_row = list(row)
                                new_row[id_idx] = new_id
                                if post_count_idx is not None:
                                    new_row[post_count_idx] = 0
                                tag_batch.append(new_row)
                                tags_created += 1

            # Batch insert new tags
            if tag_batch and tag_cols:
                placeholders_insert = ','.join('?' * len(tag_cols))
                dest_cursor.executemany(
                    f"INSERT INTO tags ({','.join(tag_cols)}) VALUES ({placeholders_insert})",
                    tag_batch
                )
                records_done += len(tag_batch)

            report_progress("importing", f"tags ({tags_created} created, {tags_reused} reused)")
            await asyncio.sleep(0)

            # === Phase 6: Import image_tags (batch) ===
            report_progress("importing", "image_tags...")

            image_tag_batch = []
            image_tag_cols = None
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

                    if image_tag_rows and not image_tag_cols:
                        image_tag_cols = [desc[0] for desc in source_cursor.description]

                    if image_tag_rows:
                        image_id_idx = image_tag_cols.index('image_id')
                        tag_id_idx = image_tag_cols.index('tag_id')

                        for row in image_tag_rows:
                            old_image_id = row['image_id']
                            old_tag_id = row['tag_id']

                            if old_image_id not in image_id_map or old_tag_id not in tag_id_map:
                                continue

                            new_row = list(row)
                            new_row[image_id_idx] = image_id_map[old_image_id]
                            new_row[tag_id_idx] = tag_id_map[old_tag_id]
                            image_tag_batch.append(tuple(new_row))

            # Batch insert image_tags (with conflict ignore)
            if image_tag_batch and image_tag_cols:
                placeholders_insert = ','.join('?' * len(image_tag_cols))
                batch_size = 1000
                for i in range(0, len(image_tag_batch), batch_size):
                    batch = image_tag_batch[i:i + batch_size]
                    # Use INSERT OR IGNORE to skip duplicates
                    dest_cursor.executemany(
                        f"INSERT OR IGNORE INTO image_tags ({','.join(image_tag_cols)}) VALUES ({placeholders_insert})",
                        batch
                    )
                    records_done += len(batch)
                    report_progress("importing", f"image_tags ({min(i + batch_size, len(image_tag_batch))}/{len(image_tag_batch)})")
                    await asyncio.sleep(0)

            # === Phase 7: Update tag post counts (batch) ===
            report_progress("importing", "Updating tag counts...")

            affected_tag_ids = list(set(tag_id_map.values()))
            if affected_tag_ids:
                # Update all tag counts in one query using a subquery
                batch_size = 500
                for i in range(0, len(affected_tag_ids), batch_size):
                    batch = affected_tag_ids[i:i + batch_size]
                    batch_placeholders = ','.join('?' * len(batch))
                    dest_cursor.execute(f"""
                        UPDATE tags SET post_count = (
                            SELECT COUNT(*) FROM image_tags WHERE image_tags.tag_id = tags.id
                        ) WHERE id IN ({batch_placeholders})
                    """, batch)

            dest_conn.commit()
            report_progress("database_complete", "library.db")

        finally:
            source_conn.close()
            dest_conn.close()

        await asyncio.sleep(0)

        # === Phase 8: Copy thumbnails for imported images ===
        report_progress("copying", "thumbnails...")

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
            total_thumbs = len(imported_hashes)

            for idx, file_hash in enumerate(imported_hashes):
                thumb_file = f"{file_hash}.webp"
                source_thumb = source_thumb_dir / thumb_file
                dest_thumb = dest_thumb_dir / thumb_file

                if source_thumb.exists() and not dest_thumb.exists():
                    shutil.copy2(source_thumb, dest_thumb)
                    files_copied += 1
                    bytes_copied += source_thumb.stat().st_size

                # Report progress every 100 files
                if (idx + 1) % 100 == 0 or idx == total_thumbs - 1:
                    report_progress("copying", f"thumbnails ({idx + 1}/{total_thumbs})")
                    await asyncio.sleep(0)

        # === Phase 9: Copy preview cache for imported images ===
        source_preview_dir = source / "preview_cache"
        dest_preview_dir = dest / "preview_cache"

        if source_preview_dir.exists() and imported_hashes:
            dest_preview_dir.mkdir(parents=True, exist_ok=True)
            preview_count = 0

            for file_hash in imported_hashes:
                source_preview = source_preview_dir / file_hash
                dest_preview = dest_preview_dir / file_hash

                if source_preview.exists() and source_preview.is_dir():
                    dest_preview.mkdir(parents=True, exist_ok=True)
                    for f in source_preview.iterdir():
                        if f.is_file():
                            dest_file = dest_preview / f.name
                            if not dest_file.exists():
                                shutil.copy2(f, dest_file)
                                files_copied += 1
                                bytes_copied += f.stat().st_size
                                preview_count += 1

                if preview_count % 50 == 0:
                    report_progress("copying", f"preview_cache ({preview_count} files)")
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
        import traceback
        traceback.print_exc()

        if progress_callback:
            progress_callback(MigrationProgress(
                phase="error",
                current_file="",
                files_copied=files_copied,
                total_files=total_files,
                bytes_copied=bytes_copied,
                total_bytes=total_bytes,
                percent=0,
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
