"""Multi-signal confidence scoring for RAG answers.

Computes a composite confidence score from four independent signals,
each capturing a different failure mode of retrieval-augmented generation.
The weighted sum produces a score in [0.0, 1.0] that determines whether
the answer is returned, flagged, or refused.

Signal weights were calibrated against logistics document Q&A scenarios
where retrieval quality varies significantly between structured fields
(high similarity) and free-text questions (lower similarity).
"""

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List

from src.util.text_utils import extract_factual_claims, extract_keywords


@dataclass
class ConfidenceResult:
    score: float
    level: str  # "high", "medium", "low"
    components: Dict[str, float] = field(default_factory=dict)


def retrieval_similarity_score(results: List[Dict[str, Any]]) -> float:
    """Signal 1: How well do the retrieved chunks match the query?

    Blends the top result's score with the average of the top-3 to
    penalise queries where only a single chunk is marginally relevant.
    Returns 0.0 when no results are found.
    """
    if not results:
        return 0.0
    top_score = results[0]["score"]
    top3 = [r["score"] for r in results[:3]]
    avg_score = mean(top3) if top3 else top_score
    return 0.6 * top_score + 0.4 * avg_score


def chunk_agreement_score(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """Signal 2: Do multiple retrieved chunks support the same answer?

    Counts how many of the top-5 chunks share >30% keyword overlap
    with the answer.  Higher agreement = higher confidence that the
    evidence is not coming from a single noisy chunk.
    """
    if not chunks:
        return 0.0

    answer_keywords = extract_keywords(answer)
    if not answer_keywords:
        return 0.5  # No extractable keywords — neutral score

    supporting = 0
    check_count = min(len(chunks), 5)

    for chunk in chunks[:check_count]:
        chunk_keywords = extract_keywords(chunk["text"])
        overlap = len(answer_keywords & chunk_keywords) / len(answer_keywords)
        if overlap > 0.3:
            supporting += 1

    agreement_ratio = supporting / check_count
    return min(1.0, agreement_ratio * 1.5)


def answer_coverage_score(answer: str, context: str) -> float:
    """Signal 3: Are the factual claims in the answer present in the source?

    Extracts verifiable claims (dollar amounts, dates, weights, percentages)
    from the answer and checks what fraction appear verbatim in the context.
    This directly catches hallucinated numbers and dates.
    """
    if not answer:
        return 0.0

    claims = extract_factual_claims(answer)
    if not claims:
        return 0.8  # No specific claims to verify — moderate confidence

    context_normalized = context.lower().replace(",", "")
    covered = sum(
        1 for claim in claims
        if claim.lower().replace(",", "").strip() in context_normalized
    )

    return covered / len(claims)


def heuristic_score(
    question: str, answer: str, chunks: List[Dict[str, Any]]
) -> float:
    """Signal 4: Domain-specific heuristics for logistics documents.

    Applies targeted boosts when the answer contains the expected data
    type for the question category (e.g., rate questions should yield
    dollar amounts, date questions should yield dates).  Penalises
    hedging language and suspiciously short answers.
    """
    score = 0.5  # neutral baseline

    q_lower = question.lower()
    a_lower = answer.lower()

    # Boost: rate/cost questions answered with dollar amounts
    if re.search(r"rate|price|cost|charge|total|amount", q_lower):
        if re.search(r"\$[\d,]+\.?\d*", answer):
            score += 0.3

    # Boost: date/time questions answered with recognisable dates
    if re.search(r"date|when|pickup|delivery|schedule", q_lower):
        if re.search(
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2}",
            a_lower,
        ):
            score += 0.3

    # Boost: identity questions answered with proper nouns
    if re.search(r"who|shipper|consignee|carrier|name", q_lower):
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+|[A-Z]{2,}", answer):
            score += 0.2

    # Penalty: hedging / refusal language
    if any(phrase in a_lower for phrase in [
        "i cannot", "not sure", "unclear", "unable to",
        "not found", "no information",
    ]):
        score -= 0.3

    # Penalty: suspiciously short answer for a complex question
    if len(answer.split()) < 3 and len(question.split()) > 5:
        score -= 0.2

    return max(0.0, min(1.0, score))


class ConfidenceScorer:
    """Composite confidence scorer combining four independent signals.

    Args:
        w_retrieval: Weight for retrieval similarity (default 0.35).
        w_agreement: Weight for chunk agreement (default 0.25).
        w_coverage: Weight for answer coverage (default 0.25).
        w_heuristic: Weight for domain heuristics (default 0.15).
        high_threshold: Score >= this is classified as "high" (default 0.7).
        refuse_threshold: Score < this is classified as "low" (default 0.4).
    """

    def __init__(
        self,
        w_retrieval: float = 0.35,
        w_agreement: float = 0.25,
        w_coverage: float = 0.25,
        w_heuristic: float = 0.15,
        high_threshold: float = 0.7,
        refuse_threshold: float = 0.4,
    ):
        self.w_retrieval = w_retrieval
        self.w_agreement = w_agreement
        self.w_coverage = w_coverage
        self.w_heuristic = w_heuristic
        self.high_threshold = high_threshold
        self.refuse_threshold = refuse_threshold

    def compute(
        self,
        retrieval_results: List[Dict[str, Any]],
        answer: str,
        question: str,
        context: str,
        chunks: List[Dict[str, Any]],
    ) -> ConfidenceResult:
        """Compute composite confidence from all four signals."""
        s1 = retrieval_similarity_score(retrieval_results)
        s2 = chunk_agreement_score(answer, chunks)
        s3 = answer_coverage_score(answer, context)
        s4 = heuristic_score(question, answer, chunks)

        composite = (
            self.w_retrieval * s1
            + self.w_agreement * s2
            + self.w_coverage * s3
            + self.w_heuristic * s4
        )
        composite = round(max(0.0, min(1.0, composite)), 3)

        if composite >= self.high_threshold:
            level = "high"
        elif composite >= self.refuse_threshold:
            level = "medium"
        else:
            level = "low"

        return ConfidenceResult(
            score=composite,
            level=level,
            components={
                "retrieval_similarity": round(s1, 3),
                "chunk_agreement": round(s2, 3),
                "answer_coverage": round(s3, 3),
                "heuristic": round(s4, 3),
            },
        )
