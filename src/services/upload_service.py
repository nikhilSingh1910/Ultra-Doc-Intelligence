"""Document upload orchestrator: parse -> chunk -> embed -> store.

Manages the full ingestion pipeline and persists document records
to MySQL for durability across server restarts.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from src.core.chunker import LogisticsChunker
from src.core.document_parser import DocumentParser
from src.core.embedder import Embedder
from src.core.vector_store import VectorStore
from src.db.repository import DocumentRepository
from src.models.document import DocumentRecord
from src.models.response import UploadResponse
from src.util.logging_setup import get_logger

logger = get_logger(__name__)


class UploadService:
    """Orchestrates document upload: parsing, chunking, embedding, and vector storage.

    Document records are persisted to MySQL so uploads survive server restarts.

    Args:
        embedder: Embedding adapter (async).
        vector_store: Vector store adapter (sync — called via to_thread).
        chunker: Logistics-aware chunker (injected so config controls chunk sizes).
        upload_dir: Directory to persist raw uploaded files.
        document_repo: MySQL document repository (optional — creates default if None).
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunker: LogisticsChunker | None = None,
        upload_dir: str = "./data/uploads",
        document_repo: DocumentRepository | None = None,
    ):
        self.parser = DocumentParser()
        self.chunker = chunker or LogisticsChunker()
        self.embedder = embedder
        self.vector_store = vector_store
        self.upload_dir = upload_dir
        self.document_repo = document_repo or DocumentRepository()

        Path(upload_dir).mkdir(parents=True, exist_ok=True)

    async def upload(self, file_bytes: bytes, filename: str) -> UploadResponse:
        """Process and index an uploaded document.

        Returns:
            UploadResponse with document_id, chunk count, and status.

        Raises:
            UnsupportedFileTypeError: If the file type is not supported.
            DocumentParsingError: If text extraction fails.
            EmbeddingServiceError: If the embedding API fails.
        """
        document_id = str(uuid.uuid4())

        # 1. Parse (CPU-bound — run in threadpool)
        parsed = await asyncio.to_thread(self.parser.parse, file_bytes, filename)

        # 2. Persist raw file (disk I/O — run in threadpool)
        safe_filename = Path(filename).name
        save_path = Path(self.upload_dir) / f"{document_id}_{safe_filename}"
        await asyncio.to_thread(save_path.write_bytes, file_bytes)

        # 3. Chunk (CPU-bound — run in threadpool)
        chunks = await asyncio.to_thread(
            self.chunker.chunk, parsed.text, document_id, parsed.pages or None,
        )

        # 4. Embed (async — native non-blocking OpenAI call)
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedder.embed_texts(texts) if texts else []

        # 5. Store in vector index (sync ChromaDB — run in threadpool)
        if chunks and embeddings:
            await asyncio.to_thread(self.vector_store.upsert, document_id, chunks, embeddings)

        # 6. Persist document record to MySQL (sync SQLAlchemy — run in threadpool)
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_type=parsed.file_type,
            text=parsed.text,
            num_chunks=len(chunks),
            num_pages=parsed.num_pages,
        )
        await asyncio.to_thread(self.document_repo.save, record)

        logger.info(
            "Uploaded document %s: %s (%d pages, %d chunks)",
            document_id[:8], filename, parsed.num_pages, len(chunks),
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            num_chunks=len(chunks),
            num_pages=parsed.num_pages,
            status="processed",
        )

    async def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Look up a document by ID from MySQL."""
        return await asyncio.to_thread(self.document_repo.get, document_id)
