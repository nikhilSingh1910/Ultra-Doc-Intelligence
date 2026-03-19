from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.upload_service import UploadService


class TestUploadService:
    def _make_service(self, tmp_path, mock_embedder=None, mock_store=None):
        embedder = mock_embedder or MagicMock()
        embedder.embed_texts = AsyncMock(return_value=[[0.1] * 10])
        store = mock_store or MagicMock()
        mock_repo = MagicMock()
        # mock get to return the saved record
        mock_repo.get.side_effect = lambda doc_id: getattr(mock_repo, '_last_saved', None)

        def save_side_effect(record):
            mock_repo._last_saved = record

        mock_repo.save.side_effect = save_side_effect

        return UploadService(
            embedder=embedder,
            vector_store=store,
            upload_dir=str(tmp_path),
            document_repo=mock_repo,
        ), mock_repo

    @pytest.mark.asyncio
    async def test_upload_returns_document_id(self, tmp_path):
        """Upload should return a document_id."""
        service, _ = self._make_service(tmp_path)
        result = await service.upload(b"RATE: $2,450.00", "rate.txt")
        assert result.document_id
        assert result.filename == "rate.txt"
        assert result.status == "processed"

    @pytest.mark.asyncio
    async def test_upload_creates_chunks(self, tmp_path):
        """Upload should create chunks and call embedder."""
        mock_embedder = MagicMock()
        mock_embedder.embed_texts = AsyncMock(return_value=[[0.1] * 10 for _ in range(5)])
        mock_store = MagicMock()

        service, _ = self._make_service(tmp_path, mock_embedder, mock_store)
        content = b"SHIPPER:\nACME Corp\n\nCONSIGNEE:\nBest Buy\n\nRATE: $2,450.00"
        result = await service.upload(content, "test.txt")

        assert result.num_chunks > 0
        mock_embedder.embed_texts.assert_called_once()
        mock_store.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_saves_file(self, tmp_path):
        """Upload should save the file to disk."""
        service, _ = self._make_service(tmp_path)
        await service.upload(b"test content", "test.txt")

        saved_files = list(tmp_path.glob("*.txt"))
        assert len(saved_files) >= 1

    @pytest.mark.asyncio
    async def test_upload_persists_to_repository(self, tmp_path):
        """Upload should persist document record to repository."""
        service, mock_repo = self._make_service(tmp_path)
        result = await service.upload(b"RATE: $2,450.00 USD", "test.txt")

        mock_repo.save.assert_called_once()
        saved = mock_repo.save.call_args[0][0]
        assert saved.document_id == result.document_id
        assert "RATE" in saved.text

    @pytest.mark.asyncio
    async def test_get_document_delegates_to_repo(self, tmp_path):
        """get_document should query the repository."""
        service, mock_repo = self._make_service(tmp_path)
        result = await service.upload(b"RATE: $2,450.00 USD", "test.txt")

        doc = await service.get_document(result.document_id)
        assert doc is not None
        assert doc.document_id == result.document_id
