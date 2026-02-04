"""
LocalBooru database layer - SQLite for single-user local storage

Architecture:
- Main database (library.db): Global state (WatchDirectory, Tag, TagAlias, TaskQueue, etc.)
- Per-directory databases (directories/{id}.db): Image data local to each directory
"""
from pathlib import Path
from typing import Dict, Optional
import os

from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
from sqlalchemy.orm import declarative_base

from .config import get_data_dir


def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set pragmas on every new connection for SQLite concurrency"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA busy_timeout=60000")  # 60 second wait for locks
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def get_database_url() -> str:
    """Get SQLite database URL for main library database"""
    data_dir = get_data_dir()
    db_path = data_dir / 'library.db'
    return f"sqlite+aiosqlite:///{db_path}"


# SQLite async engine with WAL mode for better concurrency
engine = create_async_engine(
    get_database_url(),
    echo=False,
    # Increase pool size for concurrent API requests (preview frames, etc.)
    pool_size=10,
    max_overflow=20,
    pool_timeout=60,  # Wait up to 60s for a connection
    # SQLite specific settings for better concurrency
    connect_args={
        "check_same_thread": False,
        "timeout": 60  # Wait up to 60s for locks
    }
)

# Set pragmas on every new connection
event.listen(engine.sync_engine, "connect", set_sqlite_pragma)


# =============================================================================
# Per-Directory Database Manager
# =============================================================================

class DirectoryDatabaseManager:
    """
    Manages per-directory SQLite databases for image data.

    Each WatchDirectory gets its own database file at:
        {data_dir}/directories/{directory_id}.db

    This allows instant directory deletion by simply removing the file.
    """

    def __init__(self):
        self._engines: Dict[int, AsyncEngine] = {}
        self._session_factories: Dict[int, async_sessionmaker] = {}
        self._data_dir = get_data_dir()
        self._directories_path = self._data_dir / 'directories'
        self._directories_path.mkdir(parents=True, exist_ok=True)

    def get_db_path(self, directory_id: int) -> Path:
        """Get the database file path for a directory"""
        return self._directories_path / f"{directory_id}.db"

    def _create_engine(self, directory_id: int) -> AsyncEngine:
        """Create an async engine for a directory database"""
        db_path = self.get_db_path(directory_id)
        db_url = f"sqlite+aiosqlite:///{db_path}"

        dir_engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=60,
            connect_args={
                "check_same_thread": False,
                "timeout": 60
            }
        )

        # Set pragmas on every connection
        event.listen(dir_engine.sync_engine, "connect", set_sqlite_pragma)

        return dir_engine

    def _get_engine(self, directory_id: int) -> AsyncEngine:
        """Get or create an engine for a directory"""
        if directory_id not in self._engines:
            self._engines[directory_id] = self._create_engine(directory_id)
        return self._engines[directory_id]

    def _get_session_factory(self, directory_id: int) -> async_sessionmaker:
        """Get or create a session factory for a directory"""
        if directory_id not in self._session_factories:
            dir_engine = self._get_engine(directory_id)
            self._session_factories[directory_id] = async_sessionmaker(
                dir_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
        return self._session_factories[directory_id]

    async def get_session(self, directory_id: int) -> AsyncSession:
        """Get an async session for a directory database"""
        factory = self._get_session_factory(directory_id)
        return factory()

    async def create_directory_db(self, directory_id: int):
        """Create a new directory database with schema"""
        # Import here to avoid circular imports
        from .models import DirectoryBase

        dir_engine = self._get_engine(directory_id)

        # Initialize pragmas
        async with dir_engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA cache_size=-32000"))  # 32MB cache
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA busy_timeout=60000"))

        # Create tables
        async with dir_engine.begin() as conn:
            await conn.run_sync(DirectoryBase.metadata.create_all)

        print(f"[DirectoryDB] Created database for directory {directory_id}")

    async def delete_directory_db(self, directory_id: int):
        """Delete a directory database file - instant deletion!"""
        # Close and remove engine/session factory first
        if directory_id in self._engines:
            await self._engines[directory_id].dispose()
            del self._engines[directory_id]
        if directory_id in self._session_factories:
            del self._session_factories[directory_id]

        # Delete the database file
        db_path = self.get_db_path(directory_id)
        if db_path.exists():
            os.remove(db_path)
            print(f"[DirectoryDB] Deleted database for directory {directory_id}")

        # Also delete WAL and SHM files if they exist
        wal_path = db_path.with_suffix('.db-wal')
        shm_path = db_path.with_suffix('.db-shm')
        if wal_path.exists():
            os.remove(wal_path)
        if shm_path.exists():
            os.remove(shm_path)

    def db_exists(self, directory_id: int) -> bool:
        """Check if a directory database exists"""
        return self.get_db_path(directory_id).exists()

    async def ensure_db_exists(self, directory_id: int):
        """Ensure a directory database exists, creating it if needed"""
        if not self.db_exists(directory_id):
            await self.create_directory_db(directory_id)

    async def close_all(self):
        """Close all directory database connections"""
        for directory_id, dir_engine in list(self._engines.items()):
            await dir_engine.dispose()
        self._engines.clear()
        self._session_factories.clear()

    def get_all_directory_ids(self) -> list[int]:
        """Get list of all directory IDs that have databases"""
        ids = []
        for db_file in self._directories_path.glob("*.db"):
            try:
                dir_id = int(db_file.stem)
                ids.append(dir_id)
            except ValueError:
                continue
        return ids


# Global directory database manager instance
directory_db_manager = DirectoryDatabaseManager()


async def init_db_pragmas():
    """Initialize SQLite pragmas for better performance and concurrency"""
    async with engine.begin() as conn:
        # WAL mode allows concurrent reads during writes
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        # Increase cache size (negative = KB, so -64000 = 64MB)
        await conn.execute(text("PRAGMA cache_size=-64000"))
        # Normal synchronous is faster but still safe with WAL
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        # Enable memory-mapped I/O (256MB)
        await conn.execute(text("PRAGMA mmap_size=268435456"))
        # Wait up to 60 seconds for locks to clear (60000ms)
        await conn.execute(text("PRAGMA busy_timeout=60000"))
        # Use IMMEDIATE locking to fail fast on conflicts
        await conn.execute(text("PRAGMA locking_mode=NORMAL"))

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()


async def get_db():
    """Dependency for getting database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables and run migrations."""
    # Initialize SQLite pragmas for better performance
    await init_db_pragmas()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Run migrations for new columns (ignore if already exists)
        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN auto_age_detect BOOLEAN DEFAULT 0"
            ))
        except Exception:
            pass  # Column already exists

        # ComfyUI metadata configuration columns
        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN comfyui_prompt_node_ids TEXT"
            ))
        except Exception:
            pass

        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN comfyui_negative_node_ids TEXT"
            ))
        except Exception:
            pass

        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN metadata_format VARCHAR(50) DEFAULT 'auto'"
            ))
        except Exception:
            pass

        # Network access control - public_access for directories
        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN public_access BOOLEAN DEFAULT 0"
            ))
        except Exception:
            pass

        # Media type filtering - show_images and show_videos for directories
        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN show_images BOOLEAN DEFAULT 1"
            ))
        except Exception:
            pass

        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN show_videos BOOLEAN DEFAULT 1"
            ))
        except Exception:
            pass

        # Performance indexes for tagging queries
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id ON image_tags(tag_id)"
            ))
        except Exception:
            pass

        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_tag_post_count ON tags(post_count DESC)"
            ))
        except Exception:
            pass

        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_tag_name_post_count ON tags(name, post_count DESC)"
            ))
        except Exception:
            pass


async def close_db():
    """Close all database connections."""
    await engine.dispose()
    await directory_db_manager.close_all()
