"""FastAPI application factory.

Initialises all components with dependency injection from config,
wires them together, and mounts the API router.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import get_settings
from src.db.session import init_db
from src.core.chunker import LogisticsChunker
from src.core.embedder import Embedder
from src.core.llm_client import LLMClient
from src.core.retriever import Retriever
from src.core.vector_store import VectorStore
from src.guardrails.confidence import ConfidenceScorer
from src.guardrails.threshold import ThresholdGuardrail
from src.services.ask_service import AskService
from src.services.extract_service import ExtractService
from src.services.upload_service import UploadService
from src.util.logging_setup import configure_root_logger, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    configure_root_logger(settings.log_level)
    init_db()

    application = FastAPI(
        title="Ultra Doc-Intelligence",
        description="Logistics Document Q&A with Structured Extraction",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Component initialisation (all config wired through) ---
    embedder = Embedder()
    vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
    llm_client = LLMClient()
    chunker = LogisticsChunker(
        min_chars=settings.chunk_min_chars,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        search_top_k=settings.retrieval_top_k,
    )
    confidence_scorer = ConfidenceScorer(
        high_threshold=settings.confidence_high_threshold,
        refuse_threshold=settings.confidence_refuse_threshold,
    )
    threshold_guard = ThresholdGuardrail(
        retrieval_threshold=settings.retrieval_similarity_threshold,
        confidence_threshold=settings.confidence_refuse_threshold,
    )

    # --- Service layer ---
    upload_service = UploadService(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        upload_dir=settings.upload_dir,
    )
    ask_service = AskService(
        retriever=retriever,
        llm_client=llm_client,
        confidence_scorer=confidence_scorer,
        threshold_guard=threshold_guard,
    )
    extract_service = ExtractService(
        llm_client=llm_client,
        upload_service=upload_service,
        vector_store=vector_store,
    )

    # Store on app state for route access
    application.state.settings = settings
    application.state.upload_service = upload_service
    application.state.ask_service = ask_service
    application.state.extract_service = extract_service

    application.include_router(router)

    logger.info("Ultra Doc-Intelligence started (model=%s, embedding=%s)", settings.llm_model, settings.embedding_model)
    return application


app = create_app()
