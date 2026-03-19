"""Application settings loaded from environment variables and .env file.

All configurable thresholds, model IDs, and directory paths are centralised here.
Components receive these values via dependency injection in ``main.py``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — every value here is consumed by at least one component."""

    # --- API keys ---
    openai_api_key: str = ""

    # --- Database ---
    database_url: str = "mysql+pymysql://root:password@localhost:3306/ultra_doc_intelligence"

    # --- Directories ---
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # --- Models ---
    llm_model: str = "gpt-4o-mini"
    extraction_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # --- Chunking ---
    chunk_min_chars: int = 50
    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 200

    # --- Retrieval ---
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    retrieval_similarity_threshold: float = 0.3

    # --- Confidence ---
    confidence_refuse_threshold: float = 0.4
    confidence_high_threshold: float = 0.7

    # --- Observability ---
    log_level: str = "INFO"

    # --- File limits ---
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self) -> None:
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
