"""API endpoint integration tests.

Uses a manually constructed FastAPI app with mocked OpenAI
and an in-memory document repository (no MySQL needed).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.core.chunker import LogisticsChunker
from src.core.vector_store import VectorStore
from src.models.document import DocumentRecord
from src.services.ask_service import AskService
from src.services.extract_service import ExtractService
from src.services.upload_service import UploadService


class InMemoryDocumentRepo:
    """Test double for DocumentRepository — stores in a dict instead of MySQL."""

    def __init__(self):
        self._docs = {}

    def save(self, record: DocumentRecord):
        self._docs[record.document_id] = record

    def get(self, document_id: str):
        return self._docs.get(document_id)

    def delete(self, document_id: str):
        self._docs.pop(document_id, None)


@pytest.fixture
def test_client(tmp_path):
    """Build a FastAPI test client with mocked OpenAI and in-memory DB."""
    mock_openai = MagicMock()

    # Mock embeddings (async)
    async def embed_side_effect(**kwargs):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in kwargs.get("input", [""])]
        return resp

    mock_openai.embeddings.create = AsyncMock(side_effect=embed_side_effect)

    # Mock chat completions (async)
    choice = MagicMock()
    choice.message.content = "The carrier rate is $2,450.00."
    choice.message.tool_calls = None
    mock_openai.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[choice])
    )

    with (
        patch("src.core.embedder.openai.AsyncOpenAI", return_value=mock_openai),
        patch("src.core.llm_client.openai.AsyncOpenAI", return_value=mock_openai),
    ):
        from src.core.embedder import Embedder
        from src.core.llm_client import LLMClient
        from src.core.retriever import Retriever
        from src.guardrails.confidence import ConfidenceScorer
        from src.guardrails.threshold import ThresholdGuardrail

        # Build real components with test paths
        chroma_dir = str(tmp_path / "chroma")
        upload_dir = str(tmp_path / "uploads")
        os.makedirs(chroma_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)

        embedder = Embedder()
        vector_store = VectorStore(persist_dir=chroma_dir)
        llm_client = LLMClient()
        chunker = LogisticsChunker()
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        confidence_scorer = ConfidenceScorer()
        threshold_guard = ThresholdGuardrail(retrieval_threshold=0.3, confidence_threshold=0.4)

        upload_service = UploadService(
            embedder=embedder, vector_store=vector_store,
            chunker=chunker, upload_dir=upload_dir,
            document_repo=InMemoryDocumentRepo(),
        )
        ask_service = AskService(
            retriever=retriever, llm_client=llm_client,
            confidence_scorer=confidence_scorer, threshold_guard=threshold_guard,
        )
        extract_service = ExtractService(
            llm_client=llm_client, upload_service=upload_service,
            vector_store=vector_store,
        )

        # Build app
        app = FastAPI()
        app.state.settings = MagicMock(max_upload_bytes=10 * 1024 * 1024)
        app.state.upload_service = upload_service
        app.state.ask_service = ask_service
        app.state.extract_service = extract_service
        app.include_router(router)

        yield TestClient(app)


class TestUploadEndpoint:
    def test_upload_txt(self, test_client):
        """POST /upload with TXT should return 200."""
        response = test_client.post(
            "/upload",
            files={"file": ("test.txt", b"RATE: $2,450.00\nSHIPPER: ACME Corp", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "processed"
        assert data["num_chunks"] > 0

    def test_upload_unsupported_type(self, test_client):
        """POST /upload with .exe should return 400."""
        response = test_client.post(
            "/upload",
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_empty_file(self, test_client):
        """POST /upload with empty file should return 400."""
        response = test_client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400


class TestAskEndpoint:
    def test_ask_valid_question(self, test_client):
        """POST /ask with valid doc_id should return answer."""
        upload_resp = test_client.post(
            "/upload",
            files={"file": ("test.txt", b"RATE: $2,450.00 USD\nCarrier: Swift", "text/plain")},
        )
        doc_id = upload_resp.json()["document_id"]

        response = test_client.post(
            "/ask",
            json={"document_id": doc_id, "question": "What is the rate?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data

    def test_ask_missing_document(self, test_client):
        """POST /ask with nonexistent doc_id should return 404."""
        response = test_client.post(
            "/ask",
            json={"document_id": "nonexistent", "question": "What?"},
        )
        assert response.status_code == 404


class TestExtractEndpoint:
    def test_extract_returns_json(self, test_client):
        """POST /extract should return structured JSON."""
        upload_resp = test_client.post(
            "/upload",
            files={"file": ("test.txt", b"RATE: $2,450.00\nCarrier: Swift\nSHIPPER: ACME", "text/plain")},
        )
        doc_id = upload_resp.json()["document_id"]

        # Patch LLM extract to return structured data
        with patch.object(
            test_client.app.state.extract_service.llm_client,
            "extract",
            new_callable=AsyncMock,
            return_value={
                "shipment_id": "LOAD-123", "shipper": "ACME", "consignee": None,
                "pickup_datetime": None, "delivery_datetime": None,
                "equipment_type": None, "mode": None,
                "rate": 2450.0, "currency": "USD",
                "weight": None, "carrier_name": "Swift",
            },
        ):
            response = test_client.post("/extract", json={"document_id": doc_id})
            assert response.status_code == 200
            data = response.json()
            assert "shipment_data" in data
            assert "missing_fields" in data

    def test_extract_missing_document(self, test_client):
        """POST /extract with nonexistent doc_id should return 404."""
        response = test_client.post("/extract", json={"document_id": "nonexistent"})
        assert response.status_code == 404


class TestHealthEndpoint:
    def test_health(self, test_client):
        """GET /health should return healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
