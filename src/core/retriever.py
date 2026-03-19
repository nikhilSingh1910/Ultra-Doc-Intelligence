"""RAG retrieval pipeline: embed query -> vector search -> keyword rerank.

Combines semantic similarity (cosine distance from embeddings) with
lexical matching (keyword overlap) for more robust retrieval on
semi-structured logistics documents where exact terms matter
(e.g., BOL numbers, dollar amounts, carrier names).
"""

import asyncio
from typing import Any, Dict, List

from src.core.embedder import Embedder
from src.core.vector_store import VectorStore
from src.util.logging_setup import get_logger
from src.util.text_utils import extract_keywords

logger = get_logger(__name__)

# Reranking weights — cosine captures semantic relevance, keyword overlap
# catches exact matches that embeddings may miss (reference numbers, amounts).
COSINE_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


class Retriever:
    """Retrieves and reranks document chunks relevant to a question.

    Pipeline: embed query -> vector search (top-K) -> keyword rerank -> top-N.

    Args:
        embedder: Embedding adapter for query vectorization.
        vector_store: Vector store for similarity search.
        search_top_k: Number of candidates to fetch before reranking.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        search_top_k: int = 10,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.search_top_k = search_top_k

    async def retrieve(
        self,
        question: str,
        document_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a question.

        Returns up to ``top_k`` results sorted by combined score,
        each containing: text, chunk_id, score, heading, page_number.
        """
        query_vector = await self.embedder.embed_query(question)

        # ChromaDB is synchronous — run in threadpool to avoid blocking
        candidates = await asyncio.to_thread(
            self.vector_store.search,
            document_id, query_vector, top_k=self.search_top_k,
        )

        if not candidates:
            logger.info("No candidates found for question on document %s", document_id)
            return []

        reranked = self.rerank(question, candidates, top_k=top_k)
        logger.info(
            "Retrieved %d chunks (from %d candidates) for: %s",
            len(reranked), len(candidates), question[:60],
        )
        return reranked

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cosine similarity + keyword overlap.

        The combined score replaces the original ``score`` field so
        downstream consumers (guardrails, confidence scorer) see a
        single, consistent relevance metric.
        """
        if not candidates:
            return []

        query_keywords = extract_keywords(question)

        for candidate in candidates:
            chunk_keywords = extract_keywords(candidate["text"])
            if query_keywords:
                overlap = len(query_keywords & chunk_keywords) / len(query_keywords)
            else:
                overlap = 0.0

            cosine_score = candidate["score"]
            combined = COSINE_WEIGHT * cosine_score + KEYWORD_WEIGHT * overlap
            candidate["combined_score"] = combined
            candidate["score"] = combined  # unify for downstream consumers

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]
