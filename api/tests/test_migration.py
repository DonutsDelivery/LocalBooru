"""
Tests for LocalBooru data migration module.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from api.migration import (
    MigrationMode,
    MigrationProgress,
    MigrationResult,
    MIGRATION_ITEMS,
    get_portable_data_dir,
    is_portable_mode,
    get_current_mode,
    get_migration_paths,
    calculate_migration_size,
    check_disk_space,
    validate_migration,
    migrate_data,
    get_migration_info,
    cleanup_partial_migration,
    delete_source_data,
    verify_migration,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    system_dir = Path(tempfile.mkdtemp())
    portable_dir = Path(tempfile.mkdtemp())

    yield system_dir, portable_dir

    # Cleanup
    shutil.rmtree(system_dir, ignore_errors=True)
    shutil.rmtree(portable_dir, ignore_errors=True)


@pytest.fixture
def populated_source(temp_dirs):
    """Create a populated source directory with test data."""
    system_dir, portable_dir = temp_dirs

    # Create test database file
    db_file = system_dir / "library.db"
    db_file.write_bytes(b"SQLite test database content" * 100)

    # Create settings file
    settings_file = system_dir / "settings.json"
    settings_file.write_text('{"test": "settings"}')

    # Create thumbnails directory with files
    thumb_dir = system_dir / "thumbnails"
    thumb_dir.mkdir()
    for i in range(5):
        (thumb_dir / f"thumb_{i}.jpg").write_bytes(b"thumbnail" * 10)

    # Create preview_cache directory
    preview_dir = system_dir / "preview_cache"
    preview_dir.mkdir()
    (preview_dir / "preview_1.jpg").write_bytes(b"preview" * 20)

    return system_dir, portable_dir


class TestMigrationMode:
    def test_enum_values(self):
        assert MigrationMode.SYSTEM_TO_PORTABLE.value == "system_to_portable"
        assert MigrationMode.PORTABLE_TO_SYSTEM.value == "portable_to_system"


class TestPortableModeDetection:
    def test_not_portable_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove LOCALBOORU_PORTABLE_DATA if it exists
            os.environ.pop('LOCALBOORU_PORTABLE_DATA', None)
            assert get_portable_data_dir() is None
            assert is_portable_mode() is False
            assert get_current_mode() == "system"

    def test_portable_when_env_set(self):
        with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': '/test/path'}):
            assert get_portable_data_dir() == Path('/test/path')
            assert is_portable_mode() is True
            assert get_current_mode() == "portable"


class TestMigrationPaths:
    def test_system_to_portable_requires_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('LOCALBOORU_PORTABLE_DATA', None)
            with pytest.raises(ValueError, match="LOCALBOORU_PORTABLE_DATA not set"):
                get_migration_paths(MigrationMode.SYSTEM_TO_PORTABLE)

    def test_portable_to_system_requires_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('LOCALBOORU_PORTABLE_DATA', None)
            with pytest.raises(ValueError, match="LOCALBOORU_PORTABLE_DATA not set"):
                get_migration_paths(MigrationMode.PORTABLE_TO_SYSTEM)

    def test_system_to_portable_returns_correct_paths(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                source, dest = get_migration_paths(MigrationMode.SYSTEM_TO_PORTABLE)
                assert source == system_dir
                assert dest == portable_dir

    def test_portable_to_system_returns_correct_paths(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                source, dest = get_migration_paths(MigrationMode.PORTABLE_TO_SYSTEM)
                assert source == portable_dir
                assert dest == system_dir


class TestCalculateMigrationSize:
    def test_empty_directory(self, temp_dirs):
        system_dir, _ = temp_dirs
        total_files, total_bytes = calculate_migration_size(system_dir)
        assert total_files == 0
        assert total_bytes == 0

    def test_with_data(self, populated_source):
        system_dir, _ = populated_source
        total_files, total_bytes = calculate_migration_size(system_dir)
        assert total_files > 0
        assert total_bytes > 0
        # Should have: 1 db + 1 settings + 5 thumbnails + 1 preview = 8 files
        assert total_files == 8


class TestCheckDiskSpace:
    def test_enough_space(self, temp_dirs):
        system_dir, _ = temp_dirs
        has_space, available = check_disk_space(system_dir, 1000)
        assert has_space is True
        assert available > 0

    def test_creates_directory_if_needed(self, temp_dirs):
        system_dir, _ = temp_dirs
        new_dir = system_dir / "new_subdir"
        assert not new_dir.exists()
        has_space, _ = check_disk_space(new_dir, 1000)
        assert new_dir.exists()
        assert has_space is True


class TestValidateMigration:
    def test_source_not_exists(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        non_existent = Path("/non/existent/path")
        errors = validate_migration(
            MigrationMode.SYSTEM_TO_PORTABLE,
            non_existent,
            portable_dir,
            1000
        )
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_no_database(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        # Source exists but no database
        errors = validate_migration(
            MigrationMode.SYSTEM_TO_PORTABLE,
            system_dir,
            portable_dir,
            1000
        )
        assert any("No database found" in e for e in errors)

    def test_dest_already_has_data(self, populated_source):
        system_dir, portable_dir = populated_source
        # Create database in destination
        (portable_dir / "library.db").write_bytes(b"existing db")

        errors = validate_migration(
            MigrationMode.SYSTEM_TO_PORTABLE,
            system_dir,
            portable_dir,
            1000
        )
        assert any("already has a database" in e for e in errors)

    def test_valid_migration(self, populated_source):
        system_dir, portable_dir = populated_source
        errors = validate_migration(
            MigrationMode.SYSTEM_TO_PORTABLE,
            system_dir,
            portable_dir,
            1000
        )
        assert len(errors) == 0


class TestMigrateData:
    @pytest.mark.asyncio
    async def test_dry_run(self, populated_source):
        system_dir, portable_dir = populated_source

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                result = await migrate_data(MigrationMode.SYSTEM_TO_PORTABLE, dry_run=True)

        assert result.success is True
        assert result.files_copied > 0
        assert result.bytes_copied > 0
        # Dry run should NOT create files
        assert not (portable_dir / "library.db").exists()

    @pytest.mark.asyncio
    async def test_actual_migration(self, populated_source):
        system_dir, portable_dir = populated_source

        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress)

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                result = await migrate_data(
                    MigrationMode.SYSTEM_TO_PORTABLE,
                    progress_callback=progress_callback
                )

        assert result.success is True
        assert result.files_copied > 0
        assert (portable_dir / "library.db").exists()
        assert (portable_dir / "settings.json").exists()
        assert (portable_dir / "thumbnails").is_dir()
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_no_data_to_migrate(self, temp_dirs):
        system_dir, portable_dir = temp_dirs

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                result = await migrate_data(MigrationMode.SYSTEM_TO_PORTABLE)

        assert result.success is False
        assert "No data found" in result.error


class TestCleanupPartialMigration:
    def test_cleanup_empty_dir(self, temp_dirs):
        _, portable_dir = temp_dirs
        success, message = cleanup_partial_migration(portable_dir)
        assert success is True
        assert "Nothing to clean up" in message or "Cleaned up 0" in message

    def test_cleanup_with_data(self, temp_dirs):
        _, portable_dir = temp_dirs
        # Create some migration items
        (portable_dir / "library.db").write_bytes(b"test")
        thumb_dir = portable_dir / "thumbnails"
        thumb_dir.mkdir()
        (thumb_dir / "test.jpg").write_bytes(b"thumbnail")

        success, message = cleanup_partial_migration(portable_dir)
        assert success is True
        assert not (portable_dir / "library.db").exists()
        assert not (portable_dir / "thumbnails").exists()


class TestVerifyMigration:
    @pytest.mark.asyncio
    async def test_verify_missing_database(self, temp_dirs):
        system_dir, portable_dir = temp_dirs

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                success, issues = await verify_migration(MigrationMode.SYSTEM_TO_PORTABLE)

        assert success is False
        assert any("Database file" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_verify_successful_migration(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        # Create required files in destination
        (portable_dir / "library.db").write_bytes(b"database")
        (portable_dir / "thumbnails").mkdir()
        (portable_dir / "settings.json").write_text("{}")

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                success, issues = await verify_migration(MigrationMode.SYSTEM_TO_PORTABLE)

        assert success is True
        assert len(issues) == 0


class TestGetMigrationInfo:
    @pytest.mark.asyncio
    async def test_system_mode_info(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        # Create database in system dir
        (system_dir / "library.db").write_bytes(b"test db")

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('LOCALBOORU_PORTABLE_DATA', None)
                info = await get_migration_info()

        assert info["current_mode"] == "system"
        assert info["system_path"] == str(system_dir)
        assert info["portable_path"] is None
        assert info["system_has_data"] is True
        assert info["portable_has_data"] is False

    @pytest.mark.asyncio
    async def test_portable_mode_info(self, temp_dirs):
        system_dir, portable_dir = temp_dirs
        # Create database in portable dir
        (portable_dir / "library.db").write_bytes(b"test db")

        with patch('api.migration.get_system_data_dir', return_value=system_dir):
            with patch.dict(os.environ, {'LOCALBOORU_PORTABLE_DATA': str(portable_dir)}):
                info = await get_migration_info()

        assert info["current_mode"] == "portable"
        assert info["system_path"] == str(system_dir)
        assert info["portable_path"] == str(portable_dir)
        assert info["system_has_data"] is False
        assert info["portable_has_data"] is True
