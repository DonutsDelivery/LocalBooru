"""
Pydantic models for directory endpoints
"""
from pydantic import BaseModel
from typing import Optional, List


class DirectoryCreate(BaseModel):
    path: str
    name: Optional[str] = None
    recursive: bool = True
    auto_tag: bool = True
    auto_age_detect: bool = False


class BulkDeleteRequest(BaseModel):
    directory_ids: List[int]
    keep_images: bool = False


class BulkVerifyRequest(BaseModel):
    directory_ids: List[int]


class ParentDirectoryCreate(BaseModel):
    path: str
    recursive: bool = True
    auto_tag: bool = True
    auto_age_detect: bool = False


class DirectoryUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    recursive: Optional[bool] = None
    auto_tag: Optional[bool] = None
    auto_age_detect: Optional[bool] = None
    public_access: Optional[bool] = None  # Allow public network access to this directory
    show_images: Optional[bool] = None  # Show image files in gallery
    show_videos: Optional[bool] = None  # Show video files in gallery


class ScanOptions(BaseModel):
    clean_deleted: bool = False  # Whether to clean up deleted files before scanning


class PruneRequest(BaseModel):
    dumpster_path: Optional[str] = None  # Custom dumpster path, defaults to ~/.localbooru/dumpster


class DirectoryPathUpdate(BaseModel):
    new_path: str


class ComfyUIConfigUpdate(BaseModel):
    comfyui_prompt_node_ids: Optional[List[str]] = None
    comfyui_negative_node_ids: Optional[List[str]] = None
    metadata_format: Optional[str] = None  # auto, a1111, comfyui, none
