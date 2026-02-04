"""
Image adjustment endpoints: preview and apply brightness, contrast, gamma adjustments.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pathlib import Path
import os

from ...database import get_db
from ...models import Image
from .models import ImageAdjustmentRequest


router = APIRouter()


@router.post("/{image_id}/preview-adjust")
async def preview_image_adjustments(
    image_id: int,
    adjustments: ImageAdjustmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate a preview of image adjustments without modifying the original file.

    Returns a URL to a cached preview image that shows the adjustments with dithering.
    The preview is stored in a cache directory and cleaned up when a new preview is
    generated for the same image or when explicitly discarded.
    """
    import numpy as np
    from PIL import Image as PILImage
    from ...database import get_data_dir
    import hashlib

    # Validate adjustment values
    if not (-200 <= adjustments.brightness <= 200):
        raise HTTPException(status_code=400, detail="Brightness must be between -200 and +200")
    if not (-100 <= adjustments.contrast <= 100):
        raise HTTPException(status_code=400, detail="Contrast must be between -100 and +100")
    if not (-100 <= adjustments.gamma <= 100):
        raise HTTPException(status_code=400, detail="Gamma must be between -100 and +100")

    # Get image and file path
    query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Find a valid file path
    image_file = None
    for f in image.files:
        if os.path.exists(f.original_path):
            image_file = f
            break

    if not image_file:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    file_path = image_file.original_path

    # Create preview cache directory
    preview_cache_dir = get_data_dir() / 'preview_cache'
    preview_cache_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing preview for this image
    for old_preview in preview_cache_dir.glob(f"{image_id}_*.webp"):
        old_preview.unlink()

    # Generate unique filename based on adjustments
    adj_hash = hashlib.md5(f"{adjustments.brightness}_{adjustments.contrast}_{adjustments.gamma}".encode()).hexdigest()[:8]
    preview_filename = f"{image_id}_{adj_hash}.webp"
    preview_path = preview_cache_dir / preview_filename

    try:
        # Open the image
        img = PILImage.open(file_path)

        # Convert to RGB for processing
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P':
                img = img.convert('RGBA')
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array for processing
        img_array = np.array(img, dtype=np.float32)

        # Apply adjustments (same as apply_image_adjustments)
        # Brightness: multiplicative (matches CSS brightness filter)
        # slider -100 to +100 maps to 0.0 to 2.0 multiplier
        if adjustments.brightness != 0:
            brightness_factor = 1 + (adjustments.brightness / 100)
            img_array = img_array * max(0, brightness_factor)

        if adjustments.contrast != 0:
            contrast_factor = (adjustments.contrast + 100) / 100
            img_array = ((img_array - 127) * contrast_factor) + 127

        if adjustments.gamma != 0:
            import math
            exponent = math.pow(3.0, -adjustments.gamma / 100.0)
            img_array = np.clip(img_array, 0, 255)
            img_array = np.power(img_array / 255.0, exponent) * 255

        # Apply dithering
        dither_strength = 0.5
        if adjustments.gamma != 0:
            dither_strength = 0.5 + (abs(adjustments.gamma) / 100.0) * 0.5
        dither_noise = np.random.uniform(-dither_strength, dither_strength, img_array.shape)
        img_array = img_array + dither_noise

        # Clamp and convert back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_array)

        # Save as WebP for efficient serving
        img.save(preview_path, format='WEBP', quality=90, method=4)

        return {
            "preview_url": f"/api/images/{image_id}/preview",
            "adjustments": {
                "brightness": adjustments.brightness,
                "contrast": adjustments.contrast,
                "gamma": adjustments.gamma
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


@router.get("/{image_id}/preview")
async def get_preview_image(image_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the cached preview image for an image."""
    from ...database import get_data_dir

    preview_cache_dir = get_data_dir() / 'preview_cache'

    # Find the preview file for this image
    previews = list(preview_cache_dir.glob(f"{image_id}_*.webp"))

    if not previews:
        raise HTTPException(status_code=404, detail="No preview found for this image")

    # Return the most recent preview
    preview_path = previews[0]

    return FileResponse(str(preview_path), media_type="image/webp")


@router.delete("/{image_id}/preview")
async def discard_preview(image_id: int, db: AsyncSession = Depends(get_db)):
    """Discard the cached preview for an image."""
    from ...database import get_data_dir

    preview_cache_dir = get_data_dir() / 'preview_cache'

    deleted = 0
    for preview in preview_cache_dir.glob(f"{image_id}_*.webp"):
        preview.unlink()
        deleted += 1

    return {"deleted": deleted}


@router.post("/{image_id}/adjust")
async def apply_image_adjustments(
    image_id: int,
    adjustments: ImageAdjustmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Apply brightness, contrast, and gamma adjustments using Gwenview's exact algorithms"""
    import numpy as np
    from PIL import Image as PILImage
    from ...database import get_data_dir

    # Validate adjustment values
    if not (-200 <= adjustments.brightness <= 200):
        raise HTTPException(status_code=400, detail="Brightness must be between -200 and +200")
    if not (-100 <= adjustments.contrast <= 100):
        raise HTTPException(status_code=400, detail="Contrast must be between -100 and +100")
    if not (-100 <= adjustments.gamma <= 100):
        raise HTTPException(status_code=400, detail="Gamma must be between -100 and +100")

    # Check if any adjustment is needed
    if adjustments.brightness == 0 and adjustments.contrast == 0 and adjustments.gamma == 0:
        return {"adjusted": False, "message": "No adjustments needed"}

    # Get image and file path
    query = select(Image).options(selectinload(Image.files)).where(Image.id == image_id)
    result = await db.execute(query)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Find a valid file path
    image_file = None
    for f in image.files:
        if os.path.exists(f.original_path):
            image_file = f
            break

    if not image_file:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    file_path = image_file.original_path
    file_ext = Path(file_path).suffix.lower()

    # Check if it's an editable image format
    editable_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']
    if file_ext not in editable_formats:
        raise HTTPException(status_code=400, detail=f"Cannot adjust {file_ext} files")

    try:
        # Open the image
        img = PILImage.open(file_path)

        # Preserve EXIF data if present
        exif_data = img.info.get('exif')

        # Convert to RGB if necessary for processing
        if img.mode in ('RGBA', 'LA', 'P'):
            if img.mode == 'P':
                img = img.convert('RGBA')
            alpha = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
            img = img.convert('RGB')
        else:
            alpha = None
            if img.mode != 'RGB':
                img = img.convert('RGB')

        # Convert to numpy array for processing
        img_array = np.array(img, dtype=np.float32)

        # Adjustments applied in order: brightness -> contrast -> gamma

        # Brightness: multiplicative (matches CSS brightness filter)
        # slider -100 to +100 maps to 0.0 to 2.0 multiplier
        if adjustments.brightness != 0:
            brightness_factor = 1 + (adjustments.brightness / 100)
            img_array = img_array * max(0, brightness_factor)

        # Contrast: ((value - 127) * (contrast + 100) / 100) + 127
        if adjustments.contrast != 0:
            contrast_factor = (adjustments.contrast + 100) / 100
            img_array = ((img_array - 127) * contrast_factor) + 127

        # Gamma: exponential mapping for proper gamma curve
        # slider -100 to +100 maps to exponent 3.0 to 0.33
        # At 0: exponent = 1.0 (no change)
        # Positive = brighter midtones (exponent < 1, lifts curve)
        # Negative = darker midtones (exponent > 1, lowers curve)
        if adjustments.gamma != 0:
            import math
            exponent = math.pow(3.0, -adjustments.gamma / 100.0)
            img_array = np.clip(img_array, 0, 255)  # Clamp before gamma
            img_array = np.power(img_array / 255.0, exponent) * 255

        # Apply dithering to reduce banding artifacts (especially visible after gamma)
        # Uses random noise scaled to the adjustments applied - more aggressive for gamma
        # which stretches limited dark values across wider ranges
        # Standard dithering uses +/-0.5, but gamma lifting needs up to +/-1.0
        dither_strength = 0.5
        if adjustments.gamma != 0:
            # Scale dithering with gamma intensity - more gamma lift = more dithering needed
            dither_strength = 0.5 + (abs(adjustments.gamma) / 100.0) * 0.5  # 0.5 to 1.0
        dither_noise = np.random.uniform(-dither_strength, dither_strength, img_array.shape)
        img_array = img_array + dither_noise

        # Clamp final values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_array)

        # Restore alpha channel if present
        if alpha is not None:
            img = img.convert('RGBA')
            r, g, b, _ = img.split()
            img = PILImage.merge('RGBA', (r, g, b, alpha))

        # Save the image back to the same file
        save_kwargs = {}
        if file_ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = 95
            save_kwargs['optimize'] = True
            if exif_data:
                save_kwargs['exif'] = exif_data
        elif file_ext == '.webp':
            save_kwargs['quality'] = 95
            save_kwargs['method'] = 6
        elif file_ext == '.png':
            save_kwargs['optimize'] = True

        img.save(file_path, **save_kwargs)

        # Regenerate thumbnail
        from ...services.importer import generate_thumbnail
        thumbnail_path = get_data_dir() / 'thumbnails' / f"{image.file_hash[:16]}.webp"
        if thumbnail_path.exists():
            thumbnail_path.unlink()
        generate_thumbnail(file_path, str(thumbnail_path))

        return {
            "adjusted": True,
            "brightness": adjustments.brightness,
            "contrast": adjustments.contrast,
            "gamma": adjustments.gamma
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply adjustments: {str(e)}")
