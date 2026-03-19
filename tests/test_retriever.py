from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.retriever import Retriever


class TestRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_ranked_results(self):
        """Retriever should return results ranked by combined score."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"text": "RATE: $2,450.00", "chunk_id": "c1", "score": 0.95,
             "heading": "RATE", "section_type": "kv", "page_number": 1, "word_count": 5},
            {"text": "SHIPPER: ACME Corp", "chunk_id": "c2", "score": 0.70,
             "heading": "SHIPPER", "section_type": "kv", "page_number": 1, "word_count": 4},
        ]

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        results = await retriever.retrieve("What is the rate?", "doc-1", top_k=5)

        assert len(results) > 0
        assert results[0]["score"] >= results[-1]["score"]

    @pytest.mark.asyncio
    async def test_retrieve_calls_embedder(self):
        """Retriever should embed the query."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        await retriever.retrieve("What is the rate?", "doc-1")

        mock_embedder.embed_query.assert_called_once_with("What is the rate?")

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        """Should return empty list if no results found."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        results = await retriever.retrieve("What is the rate?", "doc-1")
        assert results == []

    def test_rerank_boosts_keyword_matches(self):
        """Reranking should boost results with keyword overlap."""
        mock_embedder = MagicMock()
        mock_store = MagicMock()

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)

        candidates = [
            {"text": "SHIPPER: ACME Corp", "chunk_id": "c1", "score": 0.80,
             "heading": "SHIPPER", "section_type": "kv", "page_number": 1, "word_count": 4},
            {"text": "RATE: $2,450.00 carrier rate", "chunk_id": "c2", "score": 0.75,
             "heading": "RATE", "section_type": "kv", "page_number": 1, "word_count": 5},
        ]

        reranked = retriever.rerank("What is the carrier rate?", candidates, top_k=2)
        # The rate chunk should be boosted because it has keyword overlap
        assert reranked[0]["chunk_id"] == "c2"

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        """Should return at most top_k results."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"text": f"chunk {i}", "chunk_id": f"c{i}", "score": 0.9 - i * 0.1,
             "heading": "", "section_type": "general", "page_number": 1, "word_count": 2}
            for i in range(10)
        ]

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        results = await retriever.retrieve("test", "doc-1", top_k=3)
        assert len(results) == 3
