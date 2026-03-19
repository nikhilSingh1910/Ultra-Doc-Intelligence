"""FastAPI route definitions for document upload, Q&A, and extraction.

All three endpoints follow a consistent pattern:
1. Validate request.
2. Check document existence (for /ask and /extract).
3. Delegate to the appropriate async service.
4. Return a structured Pydantic response.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.models import AskRequest, ExtractRequest
from src.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    EmbeddingServiceError,
    LLMServiceError,
    UnsupportedFileTypeError,
)
from src.models.response import AskResponse, ExtractResponse, UploadResponse
from src.util.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and process a logistics document (PDF, DOCX, or TXT)."""
    upload_service = request.app.state.upload_service
    settings = request.app.state.settings

    filename = file.filename or "unknown.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}. Accepted: pdf, docx, txt")

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_upload_bytes // (1024*1024)}MB")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        return await upload_service.upload(file_bytes, filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DocumentParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except EmbeddingServiceError as exc:
        logger.error("Embedding service failed during upload: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding service temporarily unavailable")


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest):
    """Ask a natural language question about an uploaded document."""
    upload_service = request.app.state.upload_service
    ask_service = request.app.state.ask_service

    if not await upload_service.get_document(body.document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        return await ask_service.ask(body.question, body.document_id)
    except LLMServiceError as exc:
        logger.error("LLM service failed during ask: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service temporarily unavailable")
    except EmbeddingServiceError as exc:
        logger.error("Embedding service failed during ask: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding service temporarily unavailable")


@router.post("/extract", response_model=ExtractResponse)
async def extract_data(request: Request, body: ExtractRequest):
    """Extract structured shipment data from an uploaded document."""
    upload_service = request.app.state.upload_service
    extract_service = request.app.state.extract_service

    if not await upload_service.get_document(body.document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        return await extract_service.extract(body.document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LLMServiceError as exc:
        logger.error("LLM service failed during extraction: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service temporarily unavailable")
