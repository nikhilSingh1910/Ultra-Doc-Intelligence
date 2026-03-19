"""Post-generation grounding check.

Extracts verifiable factual claims (dollar amounts, dates, weights,
percentages) from the LLM's answer and verifies each one appears in
the source chunks.  If fewer than 50% of claims are grounded, the
answer is flagged as potentially hallucinated.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.util.text_utils import extract_factual_claims


@dataclass
class GroundingResult:
    grounded: bool
    unsupported_claims: List[str] = field(default_factory=list)
    message: str = ""


class GroundingChecker:
    def check(
        self,
        answer: str,
        source_chunks: List[Dict[str, Any]],
    ) -> GroundingResult:
        if not answer:
            return GroundingResult(grounded=True)

        claims = extract_factual_claims(answer)
        if not claims:
            return GroundingResult(grounded=True)

        all_source_text = " ".join(c["text"] for c in source_chunks).lower()
        # Normalize for comparison
        source_normalized = all_source_text.replace(",", "")

        unsupported = []
        for claim in claims:
            claim_normalized = claim.lower().replace(",", "").strip()
            if claim_normalized not in source_normalized:
                unsupported.append(claim)

        grounding_ratio = 1 - (len(unsupported) / len(claims))

        if grounding_ratio < 0.5:
            return GroundingResult(
                grounded=False,
                unsupported_claims=unsupported,
                message="Some information in the answer could not be verified against the document.",
            )

        return GroundingResult(
            grounded=True,
            unsupported_claims=unsupported,
        )
