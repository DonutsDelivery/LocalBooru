"""
External upload service - uploads images to federated boorus (Phase 5/6)
"""
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Image, ImageFile, BooruInstance, ExternalUpload, UploadStatus


async def upload_to_booru(image_id: int, booru_id: int, db: AsyncSession) -> dict:
    """
    Upload an image to an external booru.

    This is a placeholder for Phase 5/6 implementation.
    """
    # Get image
    image = await db.get(Image, image_id)
    if not image:
        raise ValueError(f"Image {image_id} not found")

    # Get booru
    booru = await db.get(BooruInstance, booru_id)
    if not booru:
        raise ValueError(f"Booru {booru_id} not found")

    # Get file path
    file_query = select(ImageFile).where(
        ImageFile.image_id == image_id,
        ImageFile.file_exists == True
    ).limit(1)
    result = await db.execute(file_query)
    image_file = result.scalar_one_or_none()

    if not image_file:
        raise ValueError("No valid file found for image")

    # TODO: Implement actual upload logic based on booru.instance_type
    # For now, just create a placeholder record

    upload = ExternalUpload(
        image_id=image_id,
        booru_id=booru_id,
        status=UploadStatus.pending,
        error_message="Upload not yet implemented"
    )
    db.add(upload)
    await db.commit()

    return {
        'status': 'pending',
        'message': 'Upload functionality coming in Phase 5/6',
        'upload_id': upload.id
    }
