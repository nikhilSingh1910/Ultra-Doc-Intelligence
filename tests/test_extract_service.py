from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exceptions import DocumentNotFoundError
from src.services.extract_service import ExtractService


class TestExtractService:
    @pytest.mark.asyncio
    async def test_extract_returns_all_fields(self):
        """Extraction should return all 11 fields."""
        mock_llm = MagicMock()
        mock_llm.extract = AsyncMock(return_value={
            "shipment_id": "LOAD-2024-78543",
            "shipper": "ACME Manufacturing Inc.",
            "consignee": "Best Buy Distribution Center",
            "pickup_datetime": "March 15, 2024 08:00 AM",
            "delivery_datetime": "March 17, 2024 02:00 PM",
            "equipment_type": "53' Dry Van",
            "mode": "FTL",
            "rate": 2450.0,
            "currency": "USD",
            "weight": "42,000 lbs",
            "carrier_name": "Swift Transportation",
        })

        mock_upload = MagicMock()
        mock_upload.get_document = AsyncMock(return_value=MagicMock(text="Full document text"))

        mock_store = MagicMock()
        mock_store.get_document_text.return_value = "Full document text"

        service = ExtractService(
            llm_client=mock_llm,
            upload_service=mock_upload,
            vector_store=mock_store,
        )
        result = await service.extract("doc-1")

        assert result.shipment_data.shipment_id == "LOAD-2024-78543"
        assert result.shipment_data.shipper == "ACME Manufacturing Inc."
        assert result.shipment_data.rate == 2450.0
        assert result.shipment_data.carrier_name == "Swift Transportation"

    @pytest.mark.asyncio
    async def test_extract_missing_fields_are_null(self):
        """Missing fields should be null, not fabricated."""
        mock_llm = MagicMock()
        mock_llm.extract = AsyncMock(return_value={
            "shipment_id": None,
            "shipper": "ACME Corp",
            "consignee": None,
            "pickup_datetime": None,
            "delivery_datetime": None,
            "equipment_type": None,
            "mode": None,
            "rate": None,
            "currency": None,
            "weight": None,
            "carrier_name": None,
        })

        mock_upload = MagicMock()
        mock_upload.get_document = AsyncMock(return_value=MagicMock(text="Sparse document"))

        mock_store = MagicMock()
        mock_store.get_document_text.return_value = "Sparse document"

        service = ExtractService(
            llm_client=mock_llm,
            upload_service=mock_upload,
            vector_store=mock_store,
        )
        result = await service.extract("doc-1")

        assert result.shipment_data.shipment_id is None
        assert result.shipment_data.consignee is None
        assert result.shipment_data.shipper == "ACME Corp"
        assert "shipment_id" in result.missing_fields

    @pytest.mark.asyncio
    async def test_extract_reports_missing_fields(self):
        """Result should list which fields are missing."""
        mock_llm = MagicMock()
        mock_llm.extract = AsyncMock(return_value={
            "shipment_id": "LOAD-123",
            "shipper": "ACME",
            "consignee": None,
            "pickup_datetime": None,
            "delivery_datetime": None,
            "equipment_type": None,
            "mode": None,
            "rate": 2000.0,
            "currency": "USD",
            "weight": None,
            "carrier_name": None,
        })

        mock_upload = MagicMock()
        mock_upload.get_document = AsyncMock(return_value=MagicMock(text="text"))

        mock_store = MagicMock()
        mock_store.get_document_text.return_value = "text"

        service = ExtractService(
            llm_client=mock_llm,
            upload_service=mock_upload,
            vector_store=mock_store,
        )
        result = await service.extract("doc-1")

        assert "consignee" in result.missing_fields
        assert "weight" in result.missing_fields
        assert "shipment_id" not in result.missing_fields

    @pytest.mark.asyncio
    async def test_extract_document_not_found(self):
        """Should raise error if document doesn't exist."""
        mock_llm = MagicMock()
        mock_upload = MagicMock()
        mock_upload.get_document = AsyncMock(return_value=None)
        mock_store = MagicMock()
        mock_store.get_document_text.return_value = ""

        service = ExtractService(
            llm_client=mock_llm,
            upload_service=mock_upload,
            vector_store=mock_store,
        )

        with pytest.raises(DocumentNotFoundError):
            await service.extract("doc-nonexistent")
