r"""
LocalBooru data migration module.

Handles bidirectional migration between system and portable installations.
System locations:
  - Windows: %APPDATA%\.localbooru
  - Linux/Mac: ~/.localbooru
Portable location:
  - data/ folder next to the application

Migration includes: library.db (+ WAL files), thumbnails/, preview_cache/,
settings.json, packages/, models/
"""

# Types and constants
from .types import (
    MigrationMode,
    MigrationProgress,
    MigrationResult,
    ImportResult,
    MIGRATION_ITEMS,
)

# Utility functions
from .utils import (
    get_portable_data_dir,
    is_portable_mode,
    get_current_mode,
    get_migration_paths,
    calculate_migration_size,
    check_disk_space,
    validate_migration,
    cleanup_partial_migration,
    delete_source_data,
    verify_migration,
    get_migration_info,
    get_watch_directories_for_migration,
    calculate_selective_migration_size,
)

# Migration runners
from .runner import (
    migrate_data,
    migrate_data_selective,
)

# Import runner
from .import_runner import (
    validate_import,
    calculate_import_size,
    import_directories,
)

__all__ = [
    # Types
    "MigrationMode",
    "MigrationProgress",
    "MigrationResult",
    "ImportResult",
    "MIGRATION_ITEMS",
    # Utils
    "get_portable_data_dir",
    "is_portable_mode",
    "get_current_mode",
    "get_migration_paths",
    "calculate_migration_size",
    "check_disk_space",
    "validate_migration",
    "cleanup_partial_migration",
    "delete_source_data",
    "verify_migration",
    "get_migration_info",
    "get_watch_directories_for_migration",
    "calculate_selective_migration_size",
    # Migration
    "migrate_data",
    "migrate_data_selective",
    # Import
    "validate_import",
    "calculate_import_size",
    "import_directories",
]
