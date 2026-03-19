from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.models.extraction import ShipmentData


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    num_pages: int
    status: str = "processed"


class SourceChunk(BaseModel):
    text: str
    chunk_id: str
    page_number: int
    section: str


class ConfidenceDetail(BaseModel):
    score: float
    level: str  # "high", "medium", "low"
    components: Dict[str, float]


class GuardrailStatus(BaseModel):
    grounding_check: str = "passed"
    retrieval_quality: str = "passed"
    reason: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    confidence: ConfidenceDetail
    sources: List[SourceChunk]
    guardrails: GuardrailStatus


class ExtractResponse(BaseModel):
    shipment_data: ShipmentData
    extraction_confidence: float
    missing_fields: List[str]
    document_id: str
