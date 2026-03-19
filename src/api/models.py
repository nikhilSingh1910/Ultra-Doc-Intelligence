"""Pydantic request models with input validation."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=2000)


class ExtractRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100)
