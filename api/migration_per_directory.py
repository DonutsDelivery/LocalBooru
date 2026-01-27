"""
Migration script: Convert from single library.db to per-directory databases

This script migrates existing Image, ImageFile, and image_tags data from the
main library.db into per-directory SQLite databases (directories/{id}.db).

Usage:
    python -m api.migration_per_directory [--dry-run] [--verbose]

Options:
    --dry-run    Show what would be migrated without making changes
    --verbose    Print detailed progress information
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import AsyncSessionLocal, directory_db_manager
from api.models import (
    WatchDirectory, Image, ImageFile, image_tags, Tag,
    DirectoryImage, DirectoryImageFile, directory_image_tags
)


async def count_items_to_migrate(db: AsyncSession) -> dict:
    """Count items that need to be migrated."""
    # Get directories that don't yet have per-directory databases
    dir_query = select(WatchDirectory)
    dir_result = await db.execute(dir_query)
    directories = dir_result.scalars().all()

    stats = {
        "directories_total": len(directories),
        "directories_to_migrate": 0,
        "images_to_migrate": 0,
        "files_to_migrate": 0,
        "tags_to_migrate": 0
    }

    for directory in directories:
        if directory_db_manager.db_exists(directory.id):
            continue  # Already migrated

        stats["directories_to_migrate"] += 1

        # Count images in this directory
        images_query = (
            select(func.count(func.distinct(ImageFile.image_id)))
            .where(ImageFile.watch_directory_id == directory.id)
        )
        images_result = await db.execute(images_query)
        image_count = images_result.scalar() or 0
        stats["images_to_migrate"] += image_count

        # Count file references
        files_query = (
            select(func.count(ImageFile.id))
            .where(ImageFile.watch_directory_id == directory.id)
        )
        files_result = await db.execute(files_query)
        stats["files_to_migrate"] += files_result.scalar() or 0

        # Count tag associations (approximate)
        if image_count > 0:
            image_ids_query = (
                select(ImageFile.image_id)
                .where(ImageFile.watch_directory_id == directory.id)
            )
            image_ids_result = await db.execute(image_ids_query)
            image_ids = [row[0] for row in image_ids_result.all()]

            if image_ids:
                tags_query = (
                    select(func.count(image_tags.c.image_id))
                    .where(image_tags.c.image_id.in_(image_ids))
                )
                tags_result = await db.execute(tags_query)
                stats["tags_to_migrate"] += tags_result.scalar() or 0

    return stats


async def migrate_directory(
    directory: WatchDirectory,
    main_db: AsyncSession,
    verbose: bool = False
) -> dict:
    """Migrate a single directory's data to a per-directory database."""
    stats = {
        "images_migrated": 0,
        "files_migrated": 0,
        "tags_migrated": 0,
        "errors": []
    }

    if verbose:
        print(f"  Migrating directory {directory.id}: {directory.name}")

    # Create the per-directory database
    await directory_db_manager.create_directory_db(directory.id)

    # Get directory database session
    dir_db = await directory_db_manager.get_session(directory.id)

    try:
        # Get all image files for this directory
        files_query = (
            select(ImageFile)
            .where(ImageFile.watch_directory_id == directory.id)
        )
        files_result = await main_db.execute(files_query)
        image_files = files_result.scalars().all()

        # Group by image_id
        image_ids = set(f.image_id for f in image_files)

        if verbose:
            print(f"    Found {len(image_ids)} images, {len(image_files)} file references")

        # Migrate each image
        for image_id in image_ids:
            try:
                # Get the image from main DB
                image = await main_db.get(Image, image_id)
                if not image:
                    stats["errors"].append(f"Image {image_id} not found")
                    continue

                # Create DirectoryImage in directory DB
                dir_image = DirectoryImage(
                    id=image.id,  # Keep same ID for simplicity
                    filename=image.filename,
                    original_filename=image.original_filename,
                    file_hash=image.file_hash,
                    perceptual_hash=image.perceptual_hash,
                    width=image.width,
                    height=image.height,
                    file_size=image.file_size,
                    duration=image.duration,
                    rating=image.rating,
                    prompt=image.prompt,
                    negative_prompt=image.negative_prompt,
                    model_name=image.model_name,
                    sampler=image.sampler,
                    seed=image.seed,
                    steps=image.steps,
                    cfg_scale=image.cfg_scale,
                    source_url=image.source_url,
                    num_faces=image.num_faces,
                    min_detected_age=image.min_detected_age,
                    max_detected_age=image.max_detected_age,
                    detected_ages=image.detected_ages,
                    age_detection_data=image.age_detection_data,
                    is_favorite=image.is_favorite,
                    import_source=image.import_source,
                    view_count=image.view_count,
                    created_at=image.created_at,
                    updated_at=image.updated_at,
                    file_created_at=image.file_created_at,
                    file_modified_at=image.file_modified_at
                )
                dir_db.add(dir_image)
                await dir_db.flush()
                stats["images_migrated"] += 1

                # Get and migrate file references
                for image_file in [f for f in image_files if f.image_id == image_id]:
                    dir_file = DirectoryImageFile(
                        image_id=dir_image.id,
                        original_path=image_file.original_path,
                        file_exists=image_file.file_exists,
                        file_status=image_file.file_status,
                        last_verified_at=image_file.last_verified_at,
                        created_at=image_file.created_at
                    )
                    dir_db.add(dir_file)
                    stats["files_migrated"] += 1

                # Get and migrate tag associations
                tags_query = (
                    select(image_tags)
                    .where(image_tags.c.image_id == image_id)
                )
                tags_result = await main_db.execute(tags_query)
                tag_assocs = tags_result.all()

                for assoc in tag_assocs:
                    await dir_db.execute(
                        directory_image_tags.insert().values(
                            image_id=dir_image.id,
                            tag_id=assoc.tag_id,
                            confidence=assoc.confidence,
                            is_manual=assoc.is_manual
                        )
                    )
                    stats["tags_migrated"] += 1

            except Exception as e:
                stats["errors"].append(f"Error migrating image {image_id}: {str(e)}")

        await dir_db.commit()

        if verbose:
            print(f"    Migrated: {stats['images_migrated']} images, {stats['files_migrated']} files, {stats['tags_migrated']} tags")
            if stats["errors"]:
                print(f"    Errors: {len(stats['errors'])}")

    except Exception as e:
        stats["errors"].append(f"Directory migration failed: {str(e)}")
        # Clean up failed migration
        await directory_db_manager.delete_directory_db(directory.id)

    finally:
        await dir_db.close()

    return stats


async def verify_migration(directory_id: int, main_db: AsyncSession) -> dict:
    """Verify that migration was successful by comparing counts."""
    verification = {
        "success": True,
        "main_db_images": 0,
        "dir_db_images": 0,
        "main_db_files": 0,
        "dir_db_files": 0,
        "main_db_tags": 0,
        "dir_db_tags": 0
    }

    # Count in main DB
    images_query = (
        select(func.count(func.distinct(ImageFile.image_id)))
        .where(ImageFile.watch_directory_id == directory_id)
    )
    result = await main_db.execute(images_query)
    verification["main_db_images"] = result.scalar() or 0

    files_query = (
        select(func.count(ImageFile.id))
        .where(ImageFile.watch_directory_id == directory_id)
    )
    result = await main_db.execute(files_query)
    verification["main_db_files"] = result.scalar() or 0

    # Count in directory DB
    if directory_db_manager.db_exists(directory_id):
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            result = await dir_db.execute(select(func.count(DirectoryImage.id)))
            verification["dir_db_images"] = result.scalar() or 0

            result = await dir_db.execute(select(func.count(DirectoryImageFile.id)))
            verification["dir_db_files"] = result.scalar() or 0

            result = await dir_db.execute(select(func.count(directory_image_tags.c.image_id)))
            verification["dir_db_tags"] = result.scalar() or 0
        finally:
            await dir_db.close()

    # Check if counts match
    if verification["main_db_images"] != verification["dir_db_images"]:
        verification["success"] = False
    if verification["main_db_files"] != verification["dir_db_files"]:
        verification["success"] = False

    return verification


async def run_migration(dry_run: bool = False, verbose: bool = False):
    """Run the full migration."""
    print("=" * 60)
    print("Per-Directory Database Migration")
    print("=" * 60)
    print()

    async with AsyncSessionLocal() as db:
        # Count items to migrate
        stats = await count_items_to_migrate(db)

        print("Migration Statistics:")
        print(f"  Total directories: {stats['directories_total']}")
        print(f"  Directories to migrate: {stats['directories_to_migrate']}")
        print(f"  Images to migrate: {stats['images_to_migrate']}")
        print(f"  File references to migrate: {stats['files_to_migrate']}")
        print(f"  Tag associations to migrate: {stats['tags_to_migrate']}")
        print()

        if stats["directories_to_migrate"] == 0:
            print("Nothing to migrate. All directories already use per-directory databases.")
            return

        if dry_run:
            print("DRY RUN - No changes will be made.")
            return

        # Confirm
        print("This migration will:")
        print("  1. Create per-directory databases for each watch directory")
        print("  2. Copy image data from main DB to directory DBs")
        print("  3. Verify migration integrity")
        print()
        print("The original data in library.db will NOT be deleted.")
        print("You can run cleanup manually after verifying the migration.")
        print()

        response = input("Continue with migration? [y/N] ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return

        print()
        print("Starting migration...")
        print()

        # Get directories to migrate
        dir_query = select(WatchDirectory)
        dir_result = await db.execute(dir_query)
        directories = dir_result.scalars().all()

        total_stats = {
            "directories_migrated": 0,
            "images_migrated": 0,
            "files_migrated": 0,
            "tags_migrated": 0,
            "errors": []
        }

        for directory in directories:
            if directory_db_manager.db_exists(directory.id):
                if verbose:
                    print(f"Skipping directory {directory.id} (already migrated)")
                continue

            dir_stats = await migrate_directory(directory, db, verbose)
            total_stats["directories_migrated"] += 1
            total_stats["images_migrated"] += dir_stats["images_migrated"]
            total_stats["files_migrated"] += dir_stats["files_migrated"]
            total_stats["tags_migrated"] += dir_stats["tags_migrated"]
            total_stats["errors"].extend(dir_stats["errors"])

            # Verify migration
            verification = await verify_migration(directory.id, db)
            if not verification["success"]:
                total_stats["errors"].append(
                    f"Verification failed for directory {directory.id}: "
                    f"main={verification['main_db_images']} vs dir={verification['dir_db_images']}"
                )

        print()
        print("=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print()
        print("Results:")
        print(f"  Directories migrated: {total_stats['directories_migrated']}")
        print(f"  Images migrated: {total_stats['images_migrated']}")
        print(f"  File references migrated: {total_stats['files_migrated']}")
        print(f"  Tag associations migrated: {total_stats['tags_migrated']}")

        if total_stats["errors"]:
            print()
            print(f"Errors ({len(total_stats['errors'])}):")
            for error in total_stats["errors"][:10]:
                print(f"  - {error}")
            if len(total_stats["errors"]) > 10:
                print(f"  ... and {len(total_stats['errors']) - 10} more errors")

        print()
        print("Next steps:")
        print("  1. Verify the application works correctly")
        print("  2. Test gallery browsing, searching, and filtering")
        print("  3. Test directory deletion (should be instant now!)")
        print("  4. Once verified, you can optionally remove old Image/ImageFile")
        print("     tables from library.db to save space (backup first!)")


def main():
    parser = argparse.ArgumentParser(description="Migrate to per-directory databases")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    asyncio.run(run_migration(dry_run=args.dry_run, verbose=args.verbose))


if __name__ == "__main__":
    main()
