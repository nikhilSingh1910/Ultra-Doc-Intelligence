"""Structured shipment data extraction service.

Uses LLM function calling to extract 11 logistics fields from
the full document text, returning JSON with nulls for missing fields.
"""

import asyncio
from typing import List

from src.core.llm_client import LLMClient
from src.core.vector_store import VectorStore
from src.exceptions import DocumentNotFoundError
from src.models.extraction import ShipmentData
from src.models.response import ExtractResponse
from src.services.upload_service import UploadService
from src.util.logging_setup import get_logger

logger = get_logger(__name__)

ALL_FIELDS = [
    "shipment_id", "shipper", "consignee", "pickup_datetime",
    "delivery_datetime", "equipment_type", "mode", "rate",
    "currency", "weight", "carrier_name",
]


class ExtractService:
    """Extracts structured shipment data from an uploaded document.

    Uses the full document text (not chunks) because extraction
    requires global context to identify all 11 fields.

    Args:
        llm_client: LLM adapter with ``extract()`` method (async).
        upload_service: For retrieving stored document text.
        vector_store: Fallback text source if document record is unavailable.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        upload_service: UploadService,
        vector_store: VectorStore,
    ):
        self.llm_client = llm_client
        self.upload_service = upload_service
        self.vector_store = vector_store

    async def extract(self, document_id: str) -> ExtractResponse:
        """Extract structured shipment data from the specified document.

        Returns:
            ExtractResponse with ShipmentData (nulls for missing fields),
            extraction confidence, and list of missing field names.

        Raises:
            DocumentNotFoundError: If the document_id doesn't exist.
            LLMServiceError: If the LLM call fails after retries.
        """
        # Get full document text
        doc = await self.upload_service.get_document(document_id)
        if doc:
            document_text = doc.text
        else:
            # Fallback: reconstruct from vector store (sync — run in threadpool)
            document_text = await asyncio.to_thread(
                self.vector_store.get_document_text, document_id,
            )

        if not document_text:
            raise DocumentNotFoundError(document_id)

        # LLM structured extraction via function calling (async)
        raw = await self.llm_client.extract(document_text)

        # Build ShipmentData with nulls for missing fields
        shipment = ShipmentData(**{field: raw.get(field) for field in ALL_FIELDS})

        # Identify missing fields
        missing = [f for f in ALL_FIELDS if getattr(shipment, f) is None]

        # Confidence = completeness (40%) + grounding (60%)
        # Completeness: proportion of fields filled
        filled_count = len(ALL_FIELDS) - len(missing)
        completeness = filled_count / len(ALL_FIELDS)

        # Grounding: verify extracted values appear in the source text
        grounded_count = 0
        source_lower = document_text.lower().replace(",", "")
        for f in ALL_FIELDS:
            val = getattr(shipment, f)
            if val is None:
                continue
            val_str = str(val).lower().replace(",", "").strip()
            if val_str and val_str in source_lower:
                grounded_count += 1

        grounding = grounded_count / max(filled_count, 1)
        extraction_confidence = round(0.4 * completeness + 0.6 * grounding, 2)

        logger.info(
            "Extracted %d/%d fields for document %s (confidence=%.2f)",
            filled_count, len(ALL_FIELDS), document_id[:8], extraction_confidence,
        )

        return ExtractResponse(
            shipment_data=shipment,
            extraction_confidence=extraction_confidence,
            missing_fields=missing,
            document_id=document_id,
        )
