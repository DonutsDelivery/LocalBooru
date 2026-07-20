"""
Data migration types and constants.

Types for bidirectional migration between system and portable installations.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MigrationMode(str, Enum):
    SYSTEM_TO_PORTABLE = "system_to_portable"
    PORTABLE_TO_SYSTEM = "portable_to_system"


@dataclass
class MigrationProgress:
    phase: str
    current_file: str
    files_copied: int
    total_files: int
    bytes_copied: int
    total_bytes: int
    percent: float
    error: Optional[str] = None


@dataclass
class MigrationResult:
    success: bool
    mode: MigrationMode
    source_path: str
    dest_path: str
    files_copied: int
    bytes_copied: int
    error: Optional[str] = None


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    mode: MigrationMode
    source_path: str
    dest_path: str
    directories_imported: int
    images_imported: int
    images_skipped: int  # Duplicates by file_hash
    tags_created: int
    tags_reused: int
    files_copied: int
    bytes_copied: int
    error: Optional[str] = None


# Files/directories to migrate
MIGRATION_ITEMS = [
    "library.db",
    "library.db-shm",  # SQLite WAL shared memory (may not exist)
    "library.db-wal",  # SQLite WAL log (may not exist)
    "settings.json",
    "thumbnails",
    "preview_cache",
    "packages",
    "models",
]
