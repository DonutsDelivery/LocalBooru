"""
LocalBooru database layer - SQLite for single-user local storage
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from .config import get_data_dir


def get_database_url() -> str:
    """Get SQLite database URL"""
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
        "timeout": 30  # Wait up to 30s for locks
    }
)


async def init_db_pragmas():
    """Initialize SQLite pragmas for better performance"""
    async with engine.begin() as conn:
        # WAL mode allows concurrent reads during writes
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        # Increase cache size (negative = KB, so -64000 = 64MB)
        await conn.execute(text("PRAGMA cache_size=-64000"))
        # Normal synchronous is faster but still safe with WAL
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        # Enable memory-mapped I/O (256MB)
        await conn.execute(text("PRAGMA mmap_size=268435456"))

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
    """Close database connections."""
    await engine.dispose()
