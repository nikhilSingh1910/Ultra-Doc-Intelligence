from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.embedder import Embedder


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_embed_texts_returns_vectors(self, mock_openai_client):
        """Should return list of float vectors."""
        with patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai_client):
            embedder = Embedder()
            result = await embedder.embed_texts(["Hello world"])
            assert len(result) == 1
            assert isinstance(result[0], list)
            assert all(isinstance(v, float) for v in result[0])

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self, mock_openai_client):
        """Should return a single float vector."""
        with patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai_client):
            embedder = Embedder()
            result = await embedder.embed_query("What is the rate?")
            assert isinstance(result, list)
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_texts_calls_openai_with_model(self, mock_openai_client):
        """Should call OpenAI with the correct model."""
        with patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai_client):
            embedder = Embedder(model="text-embedding-3-small")
            await embedder.embed_texts(["test"])
            mock_openai_client.embeddings.create.assert_called_once()
            call_kwargs = mock_openai_client.embeddings.create.call_args
            assert call_kwargs.kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_texts_batches_large_input(self, mock_openai_client):
        """Input > 100 texts should be batched into multiple API calls."""
        async def side_effect(**kwargs):
            mock_resp = MagicMock()
            mock_data = [MagicMock(embedding=[0.1] * 1536) for _ in kwargs["input"]]
            mock_resp.data = mock_data
            return mock_resp

        mock_openai_client.embeddings.create = AsyncMock(side_effect=side_effect)

        with patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai_client):
            embedder = Embedder()
            texts = [f"text_{i}" for i in range(250)]
            result = await embedder.embed_texts(texts)
            assert len(result) == 250
            # Should have been called 3 times (100 + 100 + 50)
            assert mock_openai_client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_empty_list_returns_empty(self, mock_openai_client):
        """Empty input should return empty list without API call."""
        with patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai_client):
            embedder = Embedder()
            result = await embedder.embed_texts([])
            assert result == []
            mock_openai_client.embeddings.create.assert_not_called()
