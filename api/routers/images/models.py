"""
Shared Pydantic models for image router endpoints.
"""
from pydantic import BaseModel
from typing import List


class BatchDeleteRequest(BaseModel):
    image_ids: List[int]
    delete_files: bool = False


class BatchRetagRequest(BaseModel):
    image_ids: List[int]


class BatchAgeDetectRequest(BaseModel):
    image_ids: List[int]


class BatchMetadataExtractRequest(BaseModel):
    image_ids: List[int]


class BatchMoveRequest(BaseModel):
    image_ids: List[int]
    target_directory_id: int


class ImageAdjustmentRequest(BaseModel):
    # Adjustment ranges (0 = no change)
    # Brightness: -200 to +200 (extended range for more control)
    # Contrast/Gamma: -100 to +100
    brightness: int = 0
    contrast: int = 0
    gamma: int = 0
