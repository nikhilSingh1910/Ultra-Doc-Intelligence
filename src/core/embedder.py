"""OpenAI embedding adapter with batching and retry logic.

Wraps the OpenAI async Embeddings API behind a simple interface so the
embedding provider can be swapped (e.g., to Voyage AI) by changing
only this module.
"""

from typing import List

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import get_settings
from src.exceptions import EmbeddingServiceError
from src.util.logging_setup import get_logger

logger = get_logger(__name__)

# Transient OpenAI errors worth retrying
_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)

BATCH_SIZE = 100  # OpenAI allows up to 2048, but 100 keeps memory bounded


class Embedder:
    """Creates vector embeddings via OpenAI ``text-embedding-3-small``.

    Uses the async client so embedding calls don't block the event loop.

    Args:
        model: Embedding model identifier.
        api_key: OpenAI API key (falls back to config).
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self._model = model or settings.embedding_model
        self._client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, batching to stay within API limits.

        Returns one embedding vector per input text.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = await self._call_api(batch)
            all_embeddings.extend(embeddings)

        logger.info("Embedded %d texts (%d batches)", len(texts), -(-len(texts) // BATCH_SIZE))
        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        result = await self._call_api([query])
        return result[0]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI Embeddings API with exponential backoff on transient errors."""
        try:
            response = await self._client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        except _RETRYABLE:
            raise  # let tenacity handle retry
        except openai.OpenAIError as exc:
            raise EmbeddingServiceError(str(exc)) from exc
