"""
Main Tagger class and database operations for WD-Tagger-V3.
Handles the core tagging workflow and database integration.
"""
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ...models import (
    Tag, Image as ImageModel, TagCategory, Rating, image_tags, TaggerModel,
    DirectoryImage, directory_image_tags
)
from ...database import directory_db_manager
from .models import ensure_model_downloaded, DEFAULT_MODEL
from .preprocessing import preprocess_image
from .postprocessing import get_tags_from_probs, run_inference

logger = logging.getLogger(__name__)

# Video extensions to skip (tagger only works on images)
VIDEO_EXTENSIONS = {'.webm', '.mp4', '.mov', '.avi', '.mkv'}


async def tag_image(
    image_path: str,
    db: AsyncSession,
    image_id: int = None,
    directory_id: int = None,
    model_type: TaggerModel = None
) -> dict:
    """
    Tag an image using WD-Tagger-V3.
    Optionally updates database if image_id is provided.

    Args:
        image_path: Path to the image file
        db: Database session (main database)
        image_id: Optional image ID to update in database
        directory_id: If provided, image is in a directory database
        model_type: Which tagger model to use (vit-v3, eva02-large-v3, swinv2-v3)
    """
    # Skip video files - tagger only works on images
    ext = Path(image_path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return {
            "tags": [],
            "rating": "pg",
            "rating_scores": {},
            "general_count": 0,
            "character_count": 0,
            "model_used": "skipped_video",
            "skipped": True
        }

    if model_type is None:
        model_type = DEFAULT_MODEL

    # Ensure model is downloaded (auto-download if needed)
    await ensure_model_downloaded(model_type)

    # Preprocess and run inference
    image_array = preprocess_image(image_path)
    probs = run_inference(image_array, model_type)

    # Get tags from probabilities
    tag_results = get_tags_from_probs(probs, model_type)

    all_tags = tag_results["general_tags"] + tag_results["character_tags"]
    tag_names = []

    if image_id:
        if directory_id:
            # New architecture: image is in a directory database
            tag_names = await _tag_image_in_directory_db(
                image_id, directory_id, tag_results, all_tags, image_path, db
            )
        else:
            # Legacy: image is in main database
            tag_names = await _tag_image_in_main_db(
                image_id, tag_results, all_tags, image_path, db
            )
    else:
        tag_names = [t["name"] for t in all_tags]

    return {
        "tags": tag_names,
        "rating": tag_results["rating"].value if tag_results["rating"] else "pg",
        "rating_scores": tag_results["rating_scores"],
        "general_count": len(tag_results["general_tags"]),
        "character_count": len(tag_results["character_tags"]),
        "model_used": model_type.value
    }


async def _tag_image_in_directory_db(
    image_id: int,
    directory_id: int,
    tag_results: dict,
    all_tags: list,
    image_path: str,
    main_db: AsyncSession
) -> list[str]:
    """
    Tag an image stored in a directory database.

    - Tags are created/fetched from main DB
    - Associations are stored in directory DB
    - Post counts are updated in main DB
    """
    # Get directory database session
    dir_db = await directory_db_manager.get_session(directory_id)

    try:
        # Get image from directory database
        img_result = await dir_db.execute(
            select(DirectoryImage).where(DirectoryImage.id == image_id)
        )
        image = img_result.scalar_one_or_none()

        if not image:
            return [t["name"] for t in all_tags]

        # Update rating in directory DB
        if tag_results["rating"]:
            image.rating = tag_results["rating"]

        # Batch tag processing
        tag_names = [t["name"] for t in all_tags]
        tag_confidence_map = {t["name"]: t["confidence"] for t in all_tags}
        tag_category_map = {t["name"]: t["category"] for t in all_tags}

        # Fetch all existing tags from main DB
        existing_tags_result = await main_db.execute(
            select(Tag).where(Tag.name.in_(tag_names))
        )
        existing_tags = {t.name: t for t in existing_tags_result.scalars().all()}

        # Create missing tags in main DB
        missing_tag_names = set(tag_names) - set(existing_tags.keys())
        for tag_name in missing_tag_names:
            try:
                async with main_db.begin_nested():
                    tag = Tag(
                        name=tag_name,
                        category=tag_category_map[tag_name]
                    )
                    main_db.add(tag)
                    await main_db.flush()
                    existing_tags[tag_name] = tag
            except IntegrityError:
                tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
                existing_tags[tag_name] = tag_result.scalar_one()

        # Get existing associations from directory DB
        tag_ids = [t.id for t in existing_tags.values()]
        existing_assocs_result = await dir_db.execute(
            select(directory_image_tags.c.tag_id).where(
                directory_image_tags.c.image_id == image_id,
                directory_image_tags.c.tag_id.in_(tag_ids)
            )
        )
        existing_assoc_tag_ids = {row[0] for row in existing_assocs_result.all()}

        # Batch insert new associations into directory DB
        new_associations = []
        tags_to_increment = []
        for tag_name in tag_names:
            tag = existing_tags[tag_name]
            if tag.id not in existing_assoc_tag_ids:
                new_associations.append({
                    "image_id": image_id,
                    "tag_id": tag.id,
                    "confidence": tag_confidence_map[tag_name],
                    "is_manual": False
                })
                tags_to_increment.append(tag)

        if new_associations:
            await dir_db.execute(directory_image_tags.insert(), new_associations)
            # Update post counts in main DB
            for tag in tags_to_increment:
                tag.post_count += 1

        await dir_db.commit()
        await main_db.commit()

        # Run age detection
        age_result = await _run_age_detection_if_realistic_directory(
            image_path, image, directory_id, tag_names, main_db, dir_db
        )
        if age_result:
            tag_names.extend(age_result.get("age_tags", []))

        return tag_names

    finally:
        await dir_db.close()


async def _tag_image_in_main_db(
    image_id: int,
    tag_results: dict,
    all_tags: list,
    image_path: str,
    db: AsyncSession
) -> list[str]:
    """Legacy: Tag an image stored in the main database."""
    # Get image from database
    img_result = await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    image = img_result.scalar_one_or_none()

    if not image:
        return [t["name"] for t in all_tags]

    # Update rating
    if tag_results["rating"]:
        image.rating = tag_results["rating"]

    # Batch tag processing for better performance
    tag_names = [t["name"] for t in all_tags]
    tag_confidence_map = {t["name"]: t["confidence"] for t in all_tags}
    tag_category_map = {t["name"]: t["category"] for t in all_tags}

    # Fetch all existing tags in one query
    existing_tags_result = await db.execute(
        select(Tag).where(Tag.name.in_(tag_names))
    )
    existing_tags = {t.name: t for t in existing_tags_result.scalars().all()}

    # Find tags that need to be created
    missing_tag_names = set(tag_names) - set(existing_tags.keys())

    # Batch create missing tags
    for tag_name in missing_tag_names:
        try:
            async with db.begin_nested():
                tag = Tag(
                    name=tag_name,
                    category=tag_category_map[tag_name]
                )
                db.add(tag)
                await db.flush()
                existing_tags[tag_name] = tag
        except IntegrityError:
            tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
            existing_tags[tag_name] = tag_result.scalar_one()

    # Get existing associations in one query
    tag_ids = [t.id for t in existing_tags.values()]
    existing_assocs_result = await db.execute(
        select(image_tags.c.tag_id).where(
            image_tags.c.image_id == image_id,
            image_tags.c.tag_id.in_(tag_ids)
        )
    )
    existing_assoc_tag_ids = {row[0] for row in existing_assocs_result.all()}

    # Batch insert new associations
    new_associations = []
    tags_to_increment = []
    for tag_name in tag_names:
        tag = existing_tags[tag_name]
        if tag.id not in existing_assoc_tag_ids:
            new_associations.append({
                "image_id": image_id,
                "tag_id": tag.id,
                "confidence": tag_confidence_map[tag_name],
                "is_manual": False
            })
            tags_to_increment.append(tag)

    if new_associations:
        await db.execute(image_tags.insert(), new_associations)
        # Update post counts
        for tag in tags_to_increment:
            tag.post_count += 1

    await db.commit()

    # Run age detection for realistic/photorealistic images
    age_result = await _run_age_detection_if_realistic(
        image_path, image, tag_names, db
    )
    if age_result:
        tag_names.extend(age_result.get("age_tags", []))

    return tag_names


async def _run_age_detection_if_realistic(
    image_path: str,
    image: 'ImageModel',
    tag_names: list[str],
    db: AsyncSession
) -> dict | None:
    """Run age detection if directory has auto_age_detect enabled (legacy main DB)."""
    import json
    from ..age_detector import detect_ages, is_age_detection_enabled

    # Check if age detection is enabled globally first
    if not is_age_detection_enabled():
        return None

    # Check if the image's directory has auto_age_detect enabled
    from ...models import ImageFile, WatchDirectory
    file_query = select(ImageFile).where(ImageFile.image_id == image.id).limit(1)
    file_result = await db.execute(file_query)
    image_file = file_result.scalar_one_or_none()

    if not image_file or not image_file.watch_directory_id:
        return None

    dir_query = select(WatchDirectory).where(WatchDirectory.id == image_file.watch_directory_id)
    dir_result = await db.execute(dir_query)
    directory = dir_result.scalar_one_or_none()

    if not directory or not directory.auto_age_detect:
        return None

    try:
        result = await detect_ages(image_path)
        if result is None:
            return None

        # Store age detection results on image
        image.num_faces = result.num_faces
        image.min_detected_age = result.min_age
        image.max_detected_age = result.max_age
        image.detected_ages = json.dumps(result.ages) if result.ages else None
        image.age_detection_data = json.dumps(result.to_dict())

        # Add age-related tags
        age_tags = result.get_age_tags()
        for tag_name in age_tags:
            # Find or create tag
            tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
            tag = tag_result.scalar_one_or_none()

            if not tag:
                try:
                    async with db.begin_nested():
                        # Determine category
                        category = TagCategory.meta
                        tag = Tag(name=tag_name, category=category)
                        db.add(tag)
                        await db.flush()
                except IntegrityError:
                    tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
                    tag = tag_result.scalar_one()

            # Check if association exists
            existing = await db.execute(
                select(image_tags).where(
                    image_tags.c.image_id == image.id,
                    image_tags.c.tag_id == tag.id
                )
            )

            if not existing.first():
                await db.execute(
                    image_tags.insert().values(
                        image_id=image.id,
                        tag_id=tag.id,
                        confidence=0.95,  # High confidence for face detection
                        is_manual=False
                    )
                )
                tag.post_count += 1

        await db.commit()

        logger.info(f"Age detection for image {image.id}: {result.num_faces} faces, ages {result.ages}")
        return {"age_tags": age_tags, "result": result.to_dict()}

    except Exception as e:
        logger.error(f"Age detection failed for image {image.id}: {e}")
        return None


async def _run_age_detection_if_realistic_directory(
    image_path: str,
    image: 'DirectoryImage',
    directory_id: int,
    tag_names: list[str],
    main_db: AsyncSession,
    dir_db: AsyncSession
) -> dict | None:
    """Run age detection for an image in a directory database."""
    import json
    from ..age_detector import detect_ages, is_age_detection_enabled
    from ...models import WatchDirectory

    # Check if age detection is enabled globally
    if not is_age_detection_enabled():
        return None

    # Check if the directory has auto_age_detect enabled
    dir_query = select(WatchDirectory).where(WatchDirectory.id == directory_id)
    dir_result = await main_db.execute(dir_query)
    directory = dir_result.scalar_one_or_none()

    if not directory or not directory.auto_age_detect:
        return None

    try:
        result = await detect_ages(image_path)
        if result is None:
            return None

        # Store age detection results on image in directory DB
        image.num_faces = result.num_faces
        image.min_detected_age = result.min_age
        image.max_detected_age = result.max_age
        image.detected_ages = json.dumps(result.ages) if result.ages else None
        image.age_detection_data = json.dumps(result.to_dict())

        # Add age-related tags
        age_tags = result.get_age_tags()
        for tag_name in age_tags:
            # Find or create tag in main DB
            tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
            tag = tag_result.scalar_one_or_none()

            if not tag:
                try:
                    async with main_db.begin_nested():
                        category = TagCategory.meta
                        tag = Tag(name=tag_name, category=category)
                        main_db.add(tag)
                        await main_db.flush()
                except IntegrityError:
                    tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
                    tag = tag_result.scalar_one()

            # Check if association exists in directory DB
            existing = await dir_db.execute(
                select(directory_image_tags).where(
                    directory_image_tags.c.image_id == image.id,
                    directory_image_tags.c.tag_id == tag.id
                )
            )

            if not existing.first():
                await dir_db.execute(
                    directory_image_tags.insert().values(
                        image_id=image.id,
                        tag_id=tag.id,
                        confidence=0.95,
                        is_manual=False
                    )
                )
                tag.post_count += 1

        await dir_db.commit()
        await main_db.commit()

        logger.info(f"Age detection for directory image {image.id}: {result.num_faces} faces, ages {result.ages}")
        return {"age_tags": age_tags, "result": result.to_dict()}

    except Exception as e:
        logger.error(f"Age detection failed for directory image {image.id}: {e}")
        return None


# Fallback tagger for when model isn't loaded
async def tag_image_fallback(image_path: str) -> dict:
    """Fallback when model isn't available - returns empty tags."""
    return {
        "tags": [],
        "rating": "pg",
        "rating_scores": {},
        "general_count": 0,
        "character_count": 0,
        "error": "Tagger model not loaded"
    }
