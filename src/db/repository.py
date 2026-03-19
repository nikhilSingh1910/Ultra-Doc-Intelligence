"""Document repository — data access layer over MySQL.

Provides CRUD operations for document records, abstracting
away SQLAlchemy session management from the service layer.
"""

from typing import Optional

from src.db.models import DocumentModel
from src.db.session import get_session
from src.models.document import DocumentRecord
from src.util.logging_setup import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """MySQL-backed document storage."""

    def save(self, record: DocumentRecord) -> None:
        """Persist a document record (upsert by document_id)."""
        session = get_session()
        try:
            existing = (
                session.query(DocumentModel)
                .filter_by(document_id=record.document_id)
                .first()
            )
            if existing:
                existing.filename = record.filename
                existing.file_type = record.file_type
                existing.text = record.text
                existing.num_chunks = record.num_chunks
                existing.num_pages = record.num_pages
                existing.status = record.status
            else:
                model = DocumentModel(
                    document_id=record.document_id,
                    filename=record.filename,
                    file_type=record.file_type,
                    text=record.text,
                    num_chunks=record.num_chunks,
                    num_pages=record.num_pages,
                    status=record.status,
                )
                session.add(model)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document by ID. Returns None if not found."""
        session = get_session()
        try:
            model = (
                session.query(DocumentModel)
                .filter_by(document_id=document_id)
                .first()
            )
            if not model:
                return None
            return DocumentRecord(
                document_id=model.document_id,
                filename=model.filename,
                file_type=model.file_type,
                text=model.text,
                num_chunks=model.num_chunks,
                num_pages=model.num_pages,
                status=model.status,
            )
        finally:
            session.close()

    def delete(self, document_id: str) -> None:
        """Delete a document record."""
        session = get_session()
        try:
            session.query(DocumentModel).filter_by(document_id=document_id).delete()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
