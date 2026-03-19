import pytest

from src.guardrails.grounding import GroundingChecker
from src.guardrails.threshold import ThresholdGuardrail


class TestGroundingChecker:
    def setup_method(self):
        self.checker = GroundingChecker()

    def test_grounded_answer_passes(self):
        """Answer with all claims in source text should pass."""
        result = self.checker.check(
            answer="The rate is $2,450.00",
            source_chunks=[{"text": "RATE: $2,450.00 USD"}],
        )
        assert result.grounded is True

    def test_hallucinated_answer_fails(self):
        """Answer with claims not in source text should fail."""
        result = self.checker.check(
            answer="The rate is $5,000.00 and delivery is on 12/25/2024",
            source_chunks=[{"text": "SHIPPER: ACME Corp"}],
        )
        assert result.grounded is False

    def test_generic_answer_passes(self):
        """Answer without specific claims should pass."""
        result = self.checker.check(
            answer="The document is a rate confirmation.",
            source_chunks=[{"text": "RATE CONFIRMATION"}],
        )
        assert result.grounded is True

    def test_partial_grounding_passes(self):
        """Answer where most claims are grounded should pass."""
        result = self.checker.check(
            answer="The rate is $2,450.00 with carrier Swift Transportation",
            source_chunks=[
                {"text": "RATE: $2,450.00 USD"},
                {"text": "Carrier: Swift Transportation"},
            ],
        )
        assert result.grounded is True

    def test_empty_answer(self):
        """Empty answer should pass."""
        result = self.checker.check(answer="", source_chunks=[])
        assert result.grounded is True

    def test_result_has_unsupported_claims(self):
        """Result should list unsupported claims."""
        result = self.checker.check(
            answer="The rate is $5,000.00",
            source_chunks=[{"text": "RATE: $2,450.00 USD"}],
        )
        assert len(result.unsupported_claims) > 0


class TestThresholdGuardrail:
    def setup_method(self):
        self.guardrail = ThresholdGuardrail()

    def test_out_of_scope_blocks(self):
        """Out-of-scope questions should be blocked."""
        result = self.guardrail.check_out_of_scope("What do you think about AI?")
        assert result.passed is False

    def test_in_scope_passes(self):
        """Document questions should pass."""
        result = self.guardrail.check_out_of_scope("What is the carrier rate?")
        assert result.passed is True

    def test_retrieval_quality_low_blocks(self):
        """Low retrieval similarity should block."""
        result = self.guardrail.check_retrieval_quality(
            [{"score": 0.1}]
        )
        assert result.passed is False
        assert "Not found" in result.reason

    def test_retrieval_quality_high_passes(self):
        """High retrieval similarity should pass."""
        result = self.guardrail.check_retrieval_quality(
            [{"score": 0.8}]
        )
        assert result.passed is True

    def test_retrieval_quality_empty_blocks(self):
        """No results should block."""
        result = self.guardrail.check_retrieval_quality([])
        assert result.passed is False

    def test_confidence_low_blocks(self):
        """Low confidence should block."""
        result = self.guardrail.check_confidence(0.2)
        assert result.passed is False

    def test_confidence_high_passes(self):
        """High confidence should pass."""
        result = self.guardrail.check_confidence(0.8)
        assert result.passed is True

    def test_confidence_threshold_boundary(self):
        """Score at threshold should pass."""
        result = self.guardrail.check_confidence(0.4)
        assert result.passed is True
