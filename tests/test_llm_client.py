from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm_client import LLMClient


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_ask_returns_answer(self):
        """Should return an answer string."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "The carrier rate is $2,450.00."
        choice.message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[choice])
        )

        with patch("src.core.llm_client.openai.AsyncOpenAI", return_value=mock_client):
            llm = LLMClient()
            answer = await llm.ask(
                question="What is the carrier rate?",
                context="RATE: $2,450.00 USD\nFUEL SURCHARGE: $175.00",
            )
            assert isinstance(answer, str)
            assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_ask_passes_context_in_system_prompt(self):
        """Context should be passed to the LLM."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Answer"
        choice.message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[choice])
        )

        with patch("src.core.llm_client.openai.AsyncOpenAI", return_value=mock_client):
            llm = LLMClient()
            await llm.ask("What?", "My context here")

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            # System message should contain the context
            system_msg = [m for m in messages if m["role"] == "system"][0]
            assert "My context here" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_extract_returns_dict(self):
        """Extract should return a dictionary with shipment fields."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        tool_call = MagicMock()
        tool_call.function.arguments = '{"shipment_id": "LOAD-123", "shipper": "ACME", "rate": 2450.0}'
        choice.message.tool_calls = [tool_call]
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[choice])
        )

        with patch("src.core.llm_client.openai.AsyncOpenAI", return_value=mock_client):
            llm = LLMClient()
            result = await llm.extract("Full document text here")
            assert isinstance(result, dict)
            assert "shipment_id" in result

    @pytest.mark.asyncio
    async def test_ask_includes_grounding_instruction(self):
        """System prompt must instruct LLM to answer only from context."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Answer"
        choice.message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[choice])
        )

        with patch("src.core.llm_client.openai.AsyncOpenAI", return_value=mock_client):
            llm = LLMClient()
            await llm.ask("What?", "context")

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            system_msg = [m for m in messages if m["role"] == "system"][0]
            content = system_msg["content"].lower()
            assert "only" in content or "context" in content
