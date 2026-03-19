"""SQLAlchemy ORM models for document persistence.

Stores document metadata and full text in MySQL so that
uploaded documents survive server restarts.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    """Persisted document record — survives server restarts."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False)
    text = Column(Text, nullable=False)
    num_chunks = Column(Integer, default=0)
    num_pages = Column(Integer, default=1)
    status = Column(String(20), default="processed")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
