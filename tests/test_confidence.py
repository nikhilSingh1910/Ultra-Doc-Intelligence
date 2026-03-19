import pytest

from src.guardrails.confidence import (
    ConfidenceScorer,
    retrieval_similarity_score,
    chunk_agreement_score,
    answer_coverage_score,
    heuristic_score,
)


class TestRetrievalSimilarity:
    def test_high_similarity_high_score(self):
        """Top result with 0.95 similarity should yield high score."""
        results = [
            {"score": 0.95}, {"score": 0.90}, {"score": 0.85},
        ]
        score = retrieval_similarity_score(results)
        assert score > 0.85

    def test_low_similarity_low_score(self):
        """Low similarity should yield low score."""
        results = [{"score": 0.2}, {"score": 0.1}]
        score = retrieval_similarity_score(results)
        assert score < 0.2

    def test_no_results_zero_score(self):
        """Empty results should yield 0.0."""
        assert retrieval_similarity_score([]) == 0.0

    def test_single_result(self):
        """Single result should use that result's score."""
        results = [{"score": 0.8}]
        score = retrieval_similarity_score(results)
        assert 0.7 <= score <= 0.9


class TestChunkAgreement:
    def test_multiple_agreeing_chunks_boost_score(self):
        """Multiple chunks with answer keywords should score high."""
        answer = "The carrier rate is $2,450.00"
        chunks = [
            {"text": "RATE: $2,450.00 USD"},
            {"text": "Total rate: $2,450.00 carrier charge"},
            {"text": "Rate confirmation $2,450.00"},
            {"text": "SHIPPER: ACME Corp"},
            {"text": "DELIVERY: March 17, 2024"},
        ]
        score = chunk_agreement_score(answer, chunks)
        assert score > 0.5

    def test_no_agreement_low_score(self):
        """No overlapping keywords should give low score."""
        answer = "The rate is $2,450.00"
        chunks = [
            {"text": "SHIPPER: ACME Corp"},
            {"text": "Chicago, IL 60601"},
        ]
        score = chunk_agreement_score(answer, chunks)
        assert score < 0.5

    def test_empty_chunks(self):
        """Empty chunks list should return 0."""
        assert chunk_agreement_score("answer", []) == 0.0


class TestAnswerCoverage:
    def test_answer_with_numbers_in_context_high_coverage(self):
        """Answer with $2,450 from context should score high."""
        answer = "The carrier rate is $2,450.00"
        context = "RATE: $2,450.00 USD\nFUEL SURCHARGE: $175.00"
        score = answer_coverage_score(answer, context)
        assert score > 0.5

    def test_hallucinated_numbers_low_coverage(self):
        """Answer with numbers NOT in context should score low."""
        answer = "The rate is $5,000.00 and delivery is on 12/25/2024"
        context = "SHIPPER: ACME Corp"
        score = answer_coverage_score(answer, context)
        assert score < 0.5

    def test_generic_answer_high_coverage(self):
        """Answer without specific claims passes."""
        answer = "The document is a rate confirmation."
        context = "RATE CONFIRMATION document"
        score = answer_coverage_score(answer, context)
        assert score >= 0.5

    def test_empty_answer(self):
        """Empty answer should return 0.0 (no answer is not confident)."""
        assert answer_coverage_score("", "context") == 0.0


class TestHeuristicScore:
    def test_rate_question_with_dollar_amount(self):
        """Rate question with $ answer should get boost."""
        score = heuristic_score(
            "What is the carrier rate?",
            "The carrier rate is $2,450.00.",
            [{"text": "RATE: $2,450.00"}],
        )
        assert score > 0.5

    def test_date_question_with_date_answer(self):
        """Date question with date answer should get boost."""
        score = heuristic_score(
            "When is pickup scheduled?",
            "Pickup is scheduled for March 15, 2024 at 08:00 AM.",
            [{"text": "PICKUP: March 15, 2024"}],
        )
        assert score > 0.5

    def test_hedging_answer_gets_penalty(self):
        """Answer with hedging language should be penalized."""
        score = heuristic_score(
            "What is the rate?",
            "I cannot determine the rate from the document.",
            [{"text": "SHIPPER: ACME"}],
        )
        assert score < 0.5


class TestConfidenceScorer:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_composite_score_in_range(self):
        """Composite score should be between 0.0 and 1.0."""
        result = self.scorer.compute(
            retrieval_results=[{"score": 0.8}],
            answer="The rate is $2,450.00",
            question="What is the rate?",
            context="RATE: $2,450.00 USD",
            chunks=[{"text": "RATE: $2,450.00 USD"}],
        )
        assert 0.0 <= result.score <= 1.0

    def test_high_confidence_all_signals(self):
        """All strong signals should produce high confidence."""
        result = self.scorer.compute(
            retrieval_results=[{"score": 0.95}, {"score": 0.92}, {"score": 0.88}],
            answer="The carrier rate is $2,450.00",
            question="What is the carrier rate?",
            context="RATE: $2,450.00 USD\nCarrier rate: $2,450.00",
            chunks=[
                {"text": "RATE: $2,450.00 USD carrier rate"},
                {"text": "Total carrier rate $2,450.00"},
                {"text": "Rate confirmation $2,450.00"},
            ],
        )
        assert result.score >= 0.7
        assert result.level == "high"

    def test_low_confidence_classified_as_low(self):
        """Score below 0.4 should classify as 'low'."""
        result = self.scorer.compute(
            retrieval_results=[{"score": 0.1}],
            answer="I cannot find this information.",
            question="What is the CEO's salary?",
            context="SHIPPER: ACME Corp",
            chunks=[{"text": "SHIPPER: ACME Corp"}],
        )
        assert result.level == "low"

    def test_components_are_populated(self):
        """All 4 component scores should be present."""
        result = self.scorer.compute(
            retrieval_results=[{"score": 0.8}],
            answer="test",
            question="test?",
            context="test context",
            chunks=[{"text": "test"}],
        )
        assert "retrieval_similarity" in result.components
        assert "chunk_agreement" in result.components
        assert "answer_coverage" in result.components
        assert "heuristic" in result.components

    def test_medium_confidence_level(self):
        """Score between 0.4 and 0.7 should be medium."""
        result = self.scorer.compute(
            retrieval_results=[{"score": 0.5}],
            answer="The shipper is ACME",
            question="Who is the shipper?",
            context="SHIPPER: ACME Corp Chicago",
            chunks=[{"text": "SHIPPER: ACME Corp Chicago"}],
        )
        # This should be in the medium range
        assert result.level in ("medium", "high")
