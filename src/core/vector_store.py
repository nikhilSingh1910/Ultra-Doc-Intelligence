"""ChromaDB vector store adapter.

Provides a clean interface over ChromaDB for storing, searching,
and managing document chunk embeddings. Each document gets its own
collection for isolation and easy deletion.
"""

import re
from typing import Any, Dict, List

import chromadb
from chromadb.errors import InvalidCollectionException

from src.exceptions import VectorStoreError
from src.models.document import Chunk
from src.util.logging_setup import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Persistent ChromaDB adapter for document chunk embeddings.

    Each document is stored in its own collection (``doc_{id}``),
    enabling per-document search isolation and clean deletion.

    Args:
        persist_dir: Directory for ChromaDB on-disk persistence.
    """

    def __init__(self, persist_dir: str = "./data/chroma"):
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _collection_name(self, document_id: str) -> str:
        """Sanitise document_id into a valid ChromaDB collection name (3-63 chars, alphanumeric)."""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", document_id)[:58]
        return f"doc_{safe}"

    def _get_collection(self, document_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(document_id),
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """Store chunk embeddings, replacing any existing data for this document."""
        collection = self._get_collection(document_id)

        # Delete existing chunks for idempotency
        try:
            existing = collection.get()
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except chromadb.errors.ChromaError as exc:
            logger.warning("Failed to clear existing chunks for %s: %s", document_id, exc)

        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading or "",
                "section_type": chunk.section_type,
                "page_number": chunk.page_number,
                "word_count": chunk.word_count,
                # Semantic signals from chunker metadata enrichment
                "has_monetary": chunk.metadata.get("has_monetary", False),
                "has_dates": chunk.metadata.get("has_dates", False),
                "has_reference_numbers": chunk.metadata.get("has_reference_numbers", False),
                "has_weight": chunk.metadata.get("has_weight", False),
                "has_table": chunk.metadata.get("has_table", False),
            }
            for chunk in chunks
        ]

        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("Upserted %d chunks for document %s", len(chunks), document_id)

    def search(
        self,
        document_id: str,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar chunks to a query vector.

        Returns a list of dicts with keys: text, chunk_id, score, heading,
        section_type, page_number, word_count.  Score is cosine similarity (0-1).
        """
        try:
            collection = self._client.get_collection(name=self._collection_name(document_id))
        except (ValueError, InvalidCollectionException):
            return []

        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        search_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance  # cosine similarity = 1 - cosine distance

            metadata = results["metadatas"][0][i]
            search_results.append({
                "text": results["documents"][0][i],
                "chunk_id": chunk_id,
                "score": similarity,
                "heading": metadata.get("heading", ""),
                "section_type": metadata.get("section_type", ""),
                "page_number": metadata.get("page_number", 1),
                "word_count": metadata.get("word_count", 0),
            })

        return search_results

    def delete_document(self, document_id: str) -> None:
        """Remove all data for a document."""
        try:
            self._client.delete_collection(name=self._collection_name(document_id))
            logger.info("Deleted collection for document %s", document_id)
        except (ValueError, InvalidCollectionException):
            pass  # Collection doesn't exist — nothing to delete

    def get_document_text(self, document_id: str) -> str:
        """Reconstruct full document text from stored chunks.

        Note: chunk ordering may not match original document order.
        Prefer using the primary ``DocumentRecord.text`` when available.
        """
        try:
            collection = self._client.get_collection(name=self._collection_name(document_id))
        except (ValueError, InvalidCollectionException):
            return ""

        results = collection.get(include=["documents"])
        if not results["documents"]:
            return ""

        return "\n\n".join(results["documents"])
