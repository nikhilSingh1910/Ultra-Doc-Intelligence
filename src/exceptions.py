"""Custom exception hierarchy for Ultra Doc-Intelligence.

Provides specific, meaningful exceptions instead of generic ValueError/Exception
so callers can handle different failure modes appropriately.
"""


class UltraDocError(Exception):
    """Base exception for all application errors."""


class DocumentNotFoundError(UltraDocError):
    """Raised when a document_id does not exist in the system."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found")


class UnsupportedFileTypeError(UltraDocError):
    """Raised when an uploaded file has an unsupported extension."""

    SUPPORTED = {"pdf", "docx", "txt"}

    def __init__(self, file_type: str):
        self.file_type = file_type
        super().__init__(
            f"Unsupported file type: .{file_type}. "
            f"Accepted: {', '.join(sorted(self.SUPPORTED))}"
        )


class DocumentParsingError(UltraDocError):
    """Raised when text extraction from a document fails."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        super().__init__(f"Failed to parse '{filename}': {reason}")


class EmbeddingServiceError(UltraDocError):
    """Raised when the embedding API call fails after retries."""

    def __init__(self, reason: str):
        super().__init__(f"Embedding service error: {reason}")


class LLMServiceError(UltraDocError):
    """Raised when the LLM API call fails after retries."""

    def __init__(self, reason: str):
        super().__init__(f"LLM service error: {reason}")


class VectorStoreError(UltraDocError):
    """Raised when vector store operations fail."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(f"Vector store {operation} failed: {reason}")
