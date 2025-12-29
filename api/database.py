"""
LocalBooru database layer - SQLite for single-user local storage
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from pathlib import Path
import os

# Get data directory - ~/.localbooru/ on Linux/Mac, AppData on Windows
def get_data_dir() -> Path:
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:  # Linux/Mac
        base = Path.home()

    data_dir = base / '.localbooru'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_database_url() -> str:
    """Get SQLite database URL"""
    data_dir = get_data_dir()
    db_path = data_dir / 'library.db'
    return f"sqlite+aiosqlite:///{db_path}"


# SQLite async engine
engine = create_async_engine(
    get_database_url(),
    echo=False,
    # SQLite specific settings
    connect_args={"check_same_thread": False}
)

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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Run migrations for new columns (ignore if already exists)
        try:
            await conn.execute(text(
                "ALTER TABLE watch_directories ADD COLUMN auto_age_detect BOOLEAN DEFAULT 0"
            ))
        except Exception:
            pass  # Column already exists


async def close_db():
    """Close database connections."""
    await engine.dispose()
