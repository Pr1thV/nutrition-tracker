"""
Database connection, session management, and table initialization.
Supports PostgreSQL (asyncpg) with automatic local SQLite fallback for seamless zero-setup development.
"""
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from db.models import Base

logger = logging.getLogger(__name__)

# Default Database URL configuration
DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{Path(__file__).resolve().parent.parent}/data/nutrition_tracker.db",
)

# Fix for standard postgres:// URI strings if provided by cloud platforms
if DEFAULT_DB_URL.startswith("postgres://"):
    DEFAULT_DB_URL = DEFAULT_DB_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DEFAULT_DB_URL.startswith("postgresql://") and "+asyncpg" not in DEFAULT_DB_URL:
    DEFAULT_DB_URL = DEFAULT_DB_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Ensure local data directory exists if using sqlite
if "sqlite" in DEFAULT_DB_URL:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

engine: AsyncEngine = create_async_engine(
    DEFAULT_DB_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async dependency yielding an active SQLAlchemy database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initializes database tables."""
    logger.info(f"Connecting to database: {engine.url.render_as_string(hide_password=True)}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schemas initialized successfully.")
