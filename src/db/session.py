"""Database session management.

Creates the SQLAlchemy engine and session factory from config.
Tables are auto-created on startup via ``create_all()``.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.models import Base
from src.util.logging_setup import get_logger

logger = get_logger(__name__)

_engine = None
_SessionLocal = None


def init_db() -> None:
    """Initialise the database engine and create tables if they don't exist."""
    global _engine, _SessionLocal
    settings = get_settings()

    _engine = create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
    )
    _SessionLocal = sessionmaker(bind=_engine)

    # Create tables (idempotent)
    Base.metadata.create_all(bind=_engine)
    logger.info("Database initialised: %s", settings.database_url.split("@")[-1])


def get_session() -> Session:
    """Get a new database session. Caller must close it."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
