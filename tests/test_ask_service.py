from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.ask_service import AskService


class TestAskService:
    _DEFAULT_RESULTS = [
        {"text": "RATE: $2,450.00 USD", "chunk_id": "c1", "score": 0.9,
         "heading": "RATE", "section_type": "kv", "page_number": 1, "word_count": 5},
    ]

    def _make_service(self, retriever_results=None, llm_answer="The rate is $2,450.00"):
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=retriever_results if retriever_results is not None else self._DEFAULT_RESULTS
        )

        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value=llm_answer)

        return AskService(retriever=mock_retriever, llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_full_pipeline_returns_answer(self):
        """Ask should return answer with confidence and sources."""
        service = self._make_service()
        result = await service.ask("What is the carrier rate?", "doc-1")

        assert result.answer
        assert result.confidence.score > 0
        assert len(result.sources) > 0
        assert result.confidence.level in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_irrelevant_question_returns_refusal(self):
        """Question with no matching content should be refused."""
        service = self._make_service(retriever_results=[])
        result = await service.ask("What is the CEO's email?", "doc-1")

        assert "not found" in result.answer.lower() or "no relevant" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_out_of_scope_question_blocked(self):
        """Non-document question should be blocked by guardrail."""
        service = self._make_service()
        result = await service.ask("What do you think about AI?", "doc-1")

        assert "outside the scope" in result.answer.lower() or "scope" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_answer_includes_sources(self):
        """Answer should include source chunks."""
        service = self._make_service()
        result = await service.ask("What is the rate?", "doc-1")

        for source in result.sources:
            assert source.text
            assert source.chunk_id

    @pytest.mark.asyncio
    async def test_confidence_components_present(self):
        """Confidence should have all 4 component scores."""
        service = self._make_service()
        result = await service.ask("What is the rate?", "doc-1")

        assert "retrieval_similarity" in result.confidence.components
        assert "chunk_agreement" in result.confidence.components
        assert "answer_coverage" in result.confidence.components
        assert "heuristic" in result.confidence.components
