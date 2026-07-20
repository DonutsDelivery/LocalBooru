"""
Shared Pydantic models for library router
"""
from pydantic import BaseModel


class ImportFileRequest(BaseModel):
    file_path: str
    watch_directory_id: int = None
    auto_tag: bool = True


class FileMissingRequest(BaseModel):
    file_path: str
