"""Pre-flight guardrails: scope checking, retrieval quality gating, confidence thresholding.

These guardrails run at specific points in the ask pipeline to reject
queries that cannot be answered reliably, before or after the LLM call.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import get_settings


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""


OUT_OF_SCOPE_PATTERNS = [
    r"(?i)\bwhat do you think\b",
    r"(?i)\byour opinion\b",
    r"(?i)\btell me about yourself\b",
    r"(?i)\bexplain\b.{0,20}\b(?:ai|machine learning|neural|deep learning)\b",
    r"(?i)\bwrite\b.{0,20}\b(?:code|poem|essay|story|email)\b",
    r"(?i)\btranslate\b",
    r"(?i)\bwho (?:are|is) (?:you|claude|chatgpt|gpt)\b",
    r"(?i)\bhow (?:are|do) you (?:work|feel)\b",
]


class ThresholdGuardrail:
    def __init__(
        self,
        retrieval_threshold: float | None = None,
        confidence_threshold: float | None = None,
    ):
        settings = get_settings()
        self.retrieval_threshold = (
            retrieval_threshold if retrieval_threshold is not None
            else settings.retrieval_similarity_threshold
        )
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else settings.confidence_refuse_threshold
        )

    def check_out_of_scope(self, question: str) -> GuardrailResult:
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, question):
                return GuardrailResult(
                    passed=False,
                    reason=(
                        "This question is outside the scope of document Q&A. "
                        "I can only answer questions about the uploaded document."
                    ),
                )
        return GuardrailResult(passed=True)

    def check_retrieval_quality(
        self, results: List[Dict[str, Any]]
    ) -> GuardrailResult:
        if not results:
            return GuardrailResult(
                passed=False,
                reason="Not found in document. No relevant content was found.",
            )
        top_score = results[0].get("score", 0.0)
        if top_score < self.retrieval_threshold:
            return GuardrailResult(
                passed=False,
                reason=(
                    "Not found in document. The question does not match "
                    "any content in the uploaded document."
                ),
            )
        return GuardrailResult(passed=True)

    def check_confidence(self, confidence_score: float) -> GuardrailResult:
        if confidence_score < self.confidence_threshold:
            return GuardrailResult(
                passed=False,
                reason=(
                    "The answer could not be determined with sufficient "
                    "confidence from the document."
                ),
            )
        return GuardrailResult(passed=True)
