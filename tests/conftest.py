from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.document import Chunk


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_rate_confirmation_text():
    return (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()


@pytest.fixture
def sample_bol_text():
    return (FIXTURES_DIR / "sample_bol.txt").read_text()


@pytest.fixture
def sample_invoice_text():
    return (FIXTURES_DIR / "sample_invoice.txt").read_text()


@pytest.fixture
def sample_chunks():
    """Pre-built chunks for testing downstream components."""
    return [
        Chunk(
            chunk_id="chunk-001",
            document_id="doc-test",
            text="RATE CONFIRMATION\nLoad Number: LOAD-2024-78543\nDate: March 14, 2024",
            section_type="key_value_block",
            heading="RATE CONFIRMATION",
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-002",
            document_id="doc-test",
            text="CARRIER INFORMATION:\nCarrier: Swift Transportation\nMC#: MC-123456\nDOT#: 1234567",
            section_type="key_value_block",
            heading="CARRIER INFORMATION",
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-003",
            document_id="doc-test",
            text="SHIPPER:\nACME Manufacturing Inc.\n123 Industrial Blvd\nChicago, IL 60601",
            section_type="key_value_block",
            heading="SHIPPER",
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-004",
            document_id="doc-test",
            text="CONSIGNEE:\nBest Buy Distribution Center\n456 Commerce Drive\nDallas, TX 75201",
            section_type="key_value_block",
            heading="CONSIGNEE",
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-005",
            document_id="doc-test",
            text="PICKUP: March 15, 2024 08:00 AM\nDELIVERY: March 17, 2024 02:00 PM\nEQUIPMENT: 53' Dry Van\nMODE: FTL (Full Truckload)",
            section_type="key_value_block",
            heading="",
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-006",
            document_id="doc-test",
            text="RATE: $2,450.00 USD\nFUEL SURCHARGE: $175.00\nTOTAL: $2,625.00",
            section_type="key_value_block",
            heading="",
            page_number=1,
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Deterministic fake embeddings for testing."""
    def _make_embedding(dim=1536):
        import random
        random.seed(42)
        return [random.random() for _ in range(dim)]
    return _make_embedding


@pytest.fixture
def mock_openai_client():
    """Mock async OpenAI client with deterministic responses."""
    mock = MagicMock()

    # Mock embeddings (async)
    embedding_response = MagicMock()
    embedding_data = MagicMock()
    embedding_data.embedding = [0.1] * 1536
    embedding_response.data = [embedding_data]
    mock.embeddings.create = AsyncMock(return_value=embedding_response)

    # Mock chat completions (async)
    chat_response = MagicMock()
    choice = MagicMock()
    choice.message.content = "The carrier rate is $2,450.00."
    choice.message.tool_calls = None
    chat_response.choices = [choice]
    mock.chat.completions.create = AsyncMock(return_value=chat_response)

    return mock
