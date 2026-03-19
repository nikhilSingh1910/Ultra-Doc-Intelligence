import pytest

from src.util.text_utils import (
    extract_factual_claims,
    extract_keywords,
    normalize_text,
    split_sentences,
)


class TestExtractKeywords:
    def test_basic_extraction(self):
        keywords = extract_keywords("The carrier rate is $2,450.00")
        assert "carrier" in keywords
        assert "rate" in keywords
        assert "2,450.00" in keywords  # $ stripped by word boundary regex

    def test_removes_stopwords(self):
        keywords = extract_keywords("The shipper is in the warehouse")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "in" not in keywords
        assert "shipper" in keywords

    def test_empty_string(self):
        assert extract_keywords("") == set()

    def test_case_insensitive(self):
        keywords = extract_keywords("ACME Corp")
        assert "acme" in keywords
        assert "corp" in keywords


class TestSplitSentences:
    def test_basic_split(self):
        sentences = split_sentences("First sentence. Second sentence. Third.")
        assert len(sentences) == 3

    def test_single_sentence(self):
        sentences = split_sentences("Just one sentence.")
        assert len(sentences) == 1

    def test_empty_string(self):
        assert split_sentences("") == []


class TestExtractFactualClaims:
    def test_extracts_dollar_amounts(self):
        claims = extract_factual_claims("The rate is $2,450.00")
        assert "$2,450.00" in claims

    def test_extracts_dates(self):
        claims = extract_factual_claims("Pickup on March 15, 2024")
        assert any("March 15, 2024" in c for c in claims)

    def test_extracts_weights(self):
        claims = extract_factual_claims("Weight: 42,000 lbs")
        assert any("42,000 lbs" in c for c in claims)

    def test_extracts_percentages(self):
        claims = extract_factual_claims("Discount: 10.5%")
        assert any("10.5%" in c for c in claims)

    def test_no_claims(self):
        claims = extract_factual_claims("The document is a rate confirmation.")
        assert len(claims) == 0

    def test_empty_string(self):
        assert extract_factual_claims("") == []


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_newlines(self):
        assert normalize_text("line1\n\n\nline2") == "line1 line2"
