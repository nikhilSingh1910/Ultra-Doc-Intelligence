"""RAG-powered document Q&A service with multi-layer guardrails.

Orchestrates the full pipeline: scope check -> retrieve -> LLM -> grounding
check -> confidence scoring -> threshold gate -> response assembly.
"""

from typing import Any, Dict, List

from src.core.llm_client import LLMClient
from src.core.retriever import Retriever
from src.guardrails.confidence import ConfidenceScorer
from src.guardrails.grounding import GroundingChecker
from src.guardrails.threshold import ThresholdGuardrail
from src.models.response import (
    AskResponse,
    ConfidenceDetail,
    GuardrailStatus,
    SourceChunk,
)
from src.util.logging_setup import get_logger

logger = get_logger(__name__)


class AskService:
    """Answers questions about uploaded documents using RAG with guardrails.

    The guardrail pipeline runs in four stages:
    1. **Out-of-scope detection** — blocks non-document questions before retrieval.
    2. **Retrieval quality gate** — refuses if no relevant chunks are found.
    3. **Grounding check** — verifies factual claims in the answer exist in sources.
    4. **Confidence threshold** — refuses if composite confidence is too low.

    Args:
        retriever: Retrieval pipeline (embed -> search -> rerank).
        llm_client: LLM adapter for answer generation (async).
        confidence_scorer: Multi-signal confidence scorer (optional, uses defaults).
        threshold_guard: Threshold guardrail (optional, uses config defaults).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        confidence_scorer: ConfidenceScorer | None = None,
        threshold_guard: ThresholdGuardrail | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.grounding_checker = GroundingChecker()
        self.threshold_guard = threshold_guard or ThresholdGuardrail()

    async def ask(self, question: str, document_id: str) -> AskResponse:
        """Answer a question about a document, applying all guardrails.

        Returns an AskResponse with either a grounded answer or a
        refusal message, always including confidence scoring.
        """
        logger.info("Ask: doc=%s question=%r", document_id[:8], question[:80])

        # 1. Out-of-scope check (pre-retrieval — saves API cost)
        scope_check = self.threshold_guard.check_out_of_scope(question)
        if not scope_check.passed:
            logger.info("Blocked by out-of-scope guardrail: %s", question[:60])
            return self._refusal_response(scope_check.reason, "out_of_scope")

        # 2. Retrieve relevant chunks (async — embedder + vector search)
        results = await self.retriever.retrieve(question, document_id)

        # 3. Retrieval quality check
        retrieval_check = self.threshold_guard.check_retrieval_quality(results)
        if not retrieval_check.passed:
            logger.info("Blocked by retrieval quality guardrail (top_score=%.3f)",
                        results[0]["score"] if results else 0.0)
            return self._refusal_response(retrieval_check.reason, "retrieval_failed")

        # 4. Build context from top chunks and get LLM answer (async)
        context = "\n\n---\n\n".join(r["text"] for r in results)
        answer = await self.llm_client.ask(question, context)

        # 5. Grounding check — verify factual claims against sources
        grounding = self.grounding_checker.check(answer, results)
        grounding_status = "passed" if grounding.grounded else "warning"

        # 6. Compute confidence
        confidence = self.confidence_scorer.compute(
            retrieval_results=results,
            answer=answer,
            question=question,
            context=context,
            chunks=results,
        )

        # 7. Confidence threshold check
        conf_check = self.threshold_guard.check_confidence(confidence.score)
        if not conf_check.passed:
            logger.info("Blocked by confidence guardrail (score=%.3f)", confidence.score)
            return AskResponse(
                answer=(
                    "Not found in document. The answer could not be determined "
                    "with sufficient confidence from the uploaded document."
                ),
                confidence=ConfidenceDetail(
                    score=confidence.score, level=confidence.level,
                    components=confidence.components,
                ),
                sources=self._build_sources(results),
                guardrails=GuardrailStatus(
                    grounding_check=grounding_status,
                    retrieval_quality="passed",
                    reason=conf_check.reason,
                ),
            )

        # 8. Success — return grounded answer
        logger.info("Answer generated (confidence=%.3f, level=%s)", confidence.score, confidence.level)
        return AskResponse(
            answer=answer,
            confidence=ConfidenceDetail(
                score=confidence.score, level=confidence.level,
                components=confidence.components,
            ),
            sources=self._build_sources(results),
            guardrails=GuardrailStatus(
                grounding_check=grounding_status,
                retrieval_quality="passed",
                reason=None if grounding.grounded else grounding.message,
            ),
        )

    def _refusal_response(self, reason: str, guardrail_type: str) -> AskResponse:
        """Build a standardised refusal response."""
        return AskResponse(
            answer=reason,
            confidence=ConfidenceDetail(
                score=0.0, level="low",
                components={
                    "retrieval_similarity": 0.0, "chunk_agreement": 0.0,
                    "answer_coverage": 0.0, "heuristic": 0.0,
                },
            ),
            sources=[],
            guardrails=GuardrailStatus(
                grounding_check="n/a",
                retrieval_quality="failed" if guardrail_type == "retrieval_failed" else "n/a",
                reason=reason,
            ),
        )

    def _build_sources(self, results: List[Dict[str, Any]]) -> List[SourceChunk]:
        """Convert retrieval results to SourceChunk response models."""
        return [
            SourceChunk(
                text=r["text"], chunk_id=r["chunk_id"],
                page_number=r.get("page_number", 1),
                section=r.get("heading", ""),
            )
            for r in results
        ]
