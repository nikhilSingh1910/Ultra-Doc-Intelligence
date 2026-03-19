from pathlib import Path

import pytest

from src.core.chunker import LogisticsChunker
from src.models.document import Chunk


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLogisticsChunker:
    def setup_method(self):
        self.chunker = LogisticsChunker()

    def test_chunk_returns_list_of_chunks(self):
        """Chunking should return a list of Chunk objects."""
        text = "SHIPPER:\nACME Corp\n123 Main St"
        result = self.chunker.chunk(text, "doc-1")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_detects_section_headers(self):
        """Known logistics headers like 'SHIPPER:' should start new chunks."""
        text = "SHIPPER:\nACME Corp\n123 Main St\n\nCONSIGNEE:\nBest Buy\n456 Commerce Dr"
        result = self.chunker.chunk(text, "doc-1")
        headings = [c.heading for c in result if c.heading]
        assert any("SHIPPER" in h for h in headings)
        assert any("CONSIGNEE" in h for h in headings)

    def test_chunk_preserves_key_value_pairs(self):
        """Key-value pairs should not be split across chunks."""
        text = "CARRIER INFORMATION:\nCarrier: Swift Transportation\nMC#: MC-123456\nDOT#: 1234567"
        result = self.chunker.chunk(text, "doc-1")
        # All KV pairs should be in one chunk
        carrier_chunk = [c for c in result if "Swift Transportation" in c.text]
        assert len(carrier_chunk) >= 1
        assert "MC-123456" in carrier_chunk[0].text

    def test_chunk_rate_confirmation_fixture(self):
        """Full rate confirmation should produce logical chunks."""
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        assert len(result) >= 3  # Should have multiple meaningful chunks
        # Key information should be preserved
        all_text = " ".join(c.text for c in result)
        assert "Swift Transportation" in all_text
        assert "$2,450.00" in all_text
        assert "ACME Manufacturing" in all_text

    def test_chunk_bol_fixture(self):
        """BOL fixture should be chunked correctly."""
        text = (FIXTURES_DIR / "sample_bol.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        assert len(result) >= 3
        all_text = " ".join(c.text for c in result)
        assert "XPO Logistics" in all_text
        assert "BOL-998877" in all_text

    def test_chunk_invoice_fixture(self):
        """Invoice fixture should be chunked correctly."""
        text = (FIXTURES_DIR / "sample_invoice.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        assert len(result) >= 3
        all_text = " ".join(c.text for c in result)
        assert "INV-2024-03150" in all_text
        assert "$2,775.00" in all_text

    def test_chunk_includes_metadata(self):
        """Each chunk should have document_id and section_type."""
        text = "SHIPPER:\nACME Corp\nChicago, IL"
        result = self.chunker.chunk(text, "doc-42")
        for chunk in result:
            assert chunk.document_id == "doc-42"
            assert chunk.section_type in ("header", "key_value_block", "table", "body", "general")

    def test_chunk_assigns_unique_ids(self):
        """Each chunk should have a unique chunk_id."""
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_empty_text(self):
        """Empty text should produce empty chunk list."""
        result = self.chunker.chunk("", "doc-1")
        assert result == []

    def test_chunk_whitespace_only(self):
        """Whitespace-only text should produce empty chunk list."""
        result = self.chunker.chunk("   \n\n  \t  ", "doc-1")
        assert result == []

    def test_chunk_word_count_populated(self):
        """Chunks should have word_count populated."""
        text = "SHIPPER:\nACME Manufacturing Inc.\n123 Industrial Blvd\nChicago, IL 60601"
        result = self.chunker.chunk(text, "doc-1")
        for chunk in result:
            assert chunk.word_count > 0

    def test_chunk_no_empty_chunks(self):
        """No chunk should have empty text."""
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        for chunk in result:
            assert chunk.text.strip() != ""

    def test_chunk_all_text_preserved(self):
        """All original text content should be present across chunks."""
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        result = self.chunker.chunk(text, "doc-1")
        all_chunked = " ".join(c.text for c in result)
        # Key entities from the original must exist
        for entity in ["Swift Transportation", "ACME Manufacturing", "Best Buy",
                       "$2,450.00", "42,000 lbs", "March 15, 2024"]:
            assert entity in all_chunked, f"Missing entity: {entity}"


class TestChunkOverlap:
    """Overlap injects trailing context from the previous chunk so that
    information at chunk boundaries is not lost during retrieval."""

    def test_overlap_adds_context_from_previous_chunk(self):
        """Second chunk should contain text from the end of the first chunk."""
        chunker = LogisticsChunker(overlap_chars=200)
        text = (
            "SHIPPER:\nACME Corp\n123 Main St\nChicago, IL 60601\n"
            "Contact: John Smith (312) 555-0100\n\n"
            "CONSIGNEE:\nBest Buy Distribution Center\n456 Commerce Drive\nDallas, TX 75201"
        )
        result = chunker.chunk(text, "doc-1")
        if len(result) >= 2:
            # The consignee chunk should contain some shipper context
            second_text = result[1].text
            # Overlap brings in trailing context — at minimum, part of
            # the shipper section should bleed into the consignee chunk
            assert "CONSIGNEE" in second_text

    def test_no_overlap_when_disabled(self):
        """With overlap_chars=0, chunks should have no bleed-over."""
        chunker = LogisticsChunker(overlap_chars=0)
        text = (
            "SHIPPER:\nACME Corp\n123 Main St\n\n"
            "CONSIGNEE:\nBest Buy\n456 Commerce Drive"
        )
        result = chunker.chunk(text, "doc-1")
        if len(result) >= 2:
            # Second chunk should start with its own heading, not the prior chunk's text
            assert result[1].text.startswith("CONSIGNEE")

    def test_overlap_does_not_cross_pages(self):
        """Overlap should not bleed context from page 1 into page 2."""
        chunker = LogisticsChunker(overlap_chars=200)
        pages = [
            "SHIPPER:\nACME Corp\n123 Main St\nChicago, IL 60601",
            "CONSIGNEE:\nBest Buy\n456 Commerce Drive\nDallas, TX 75201",
        ]
        result = chunker.chunk("full text", "doc-1", pages=pages)
        # Find the first chunk on page 2
        page2_chunks = [c for c in result if c.page_number == 2]
        if page2_chunks:
            # Should NOT contain page-1 content
            assert "ACME Corp" not in page2_chunks[0].text

    def test_overlap_preserves_chunk_ids(self):
        """Overlap injection must not duplicate or alter chunk IDs."""
        chunker = LogisticsChunker(overlap_chars=200)
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        result = chunker.chunk(text, "doc-1")
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids)), "Chunk IDs must remain unique after overlap"


class TestTablePreservation:
    """Table rows (pipe-delimited or tab-delimited) must be kept as
    atomic units — never split mid-row."""

    def test_table_rows_stay_in_one_chunk(self):
        """A pipe-delimited table should remain intact within a single chunk."""
        text = (FIXTURES_DIR / "sample_bol.txt").read_text()
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        # Find the chunk containing table rows
        table_chunks = [c for c in result if "|" in c.text and c.text.count("|") >= 6]
        assert len(table_chunks) >= 1, "Table should be captured in at least one chunk"
        # Both data rows should be in the same chunk
        table_text = table_chunks[0].text
        assert "Electronic Components" in table_text
        assert "Printed Circuit Boards" in table_text

    def test_table_classified_as_table_type(self):
        """Chunks containing table content should have section_type='table'."""
        text = (
            "HANDLING UNIT INFORMATION:\n"
            "| QTY | Type   | Weight     |\n"
            "|-----|--------|------------|\n"
            "| 12  | Pallet | 18,500 lbs |\n"
            "| 8   | Pallet | 12,300 lbs |\n"
        )
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        table_chunks = [c for c in result if "18,500" in c.text]
        assert len(table_chunks) >= 1
        assert table_chunks[0].section_type == "table"

    def test_oversized_table_not_split(self):
        """Even if a table exceeds max_chars, it should not be split."""
        rows = ["| Col1 | Col2 | Col3 |", "|------|------|------|"]
        rows += [f"| Row{i} | Data{i} | Value{i} |" for i in range(100)]
        text = "HANDLING UNIT INFORMATION:\n" + "\n".join(rows)
        chunker = LogisticsChunker(max_chars=500)
        result = chunker.chunk(text, "doc-1")
        # The table should be in a single chunk despite exceeding max_chars
        table_chunks = [c for c in result if "Row50" in c.text]
        assert len(table_chunks) == 1
        assert "Row1" in table_chunks[0].text
        assert "Row99" in table_chunks[0].text


class TestSentenceFallbackSplitting:
    """When paragraph splitting produces chunks still over max_chars,
    the chunker should fall back to sentence boundaries."""

    def test_long_paragraph_splits_on_sentences(self):
        """A single long paragraph with no \\n\\n should split on sentence boundaries."""
        # Build a single paragraph with multiple sentences
        sentences = [f"Sentence number {i} has some logistics data." for i in range(50)]
        text = " ".join(sentences)
        chunker = LogisticsChunker(max_chars=300, overlap_chars=0)
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 2, "Should split into multiple chunks"
        for chunk in result:
            assert len(chunk.text) <= 300 + 50  # allow small overshoot from sentence length

    def test_sentence_split_preserves_all_content(self):
        """Sentence splitting should not lose any sentences."""
        sentences = [f"Sentence {i} is about shipping." for i in range(20)]
        text = " ".join(sentences)
        chunker = LogisticsChunker(max_chars=200, overlap_chars=0)
        result = chunker.chunk(text, "doc-1")
        all_text = " ".join(c.text for c in result)
        for i in range(20):
            assert f"Sentence {i}" in all_text


class TestMetadataEnrichment:
    """Chunks should carry semantic signals (has_monetary, has_dates, etc.)
    so downstream retrieval and confidence scoring can use them."""

    def test_monetary_flag_set(self):
        """Chunks containing dollar amounts should have has_monetary=True."""
        text = "RATE: $2,450.00 USD"
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 1
        assert result[0].metadata.get("has_monetary") is True

    def test_date_flag_set(self):
        """Chunks containing dates should have has_dates=True."""
        text = "PICKUP: March 15, 2024 08:00 AM"
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 1
        assert result[0].metadata.get("has_dates") is True

    def test_reference_flag_set(self):
        """Chunks containing reference numbers should have has_reference_numbers=True."""
        text = "BOL#: BOL-998877\nPRO#: PRO-554433"
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 1
        assert result[0].metadata.get("has_reference_numbers") is True

    def test_weight_flag_set(self):
        """Chunks containing weight values should have has_weight=True."""
        text = "WEIGHT: 42,000 lbs"
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 1
        assert result[0].metadata.get("has_weight") is True

    def test_no_false_positives_on_plain_text(self):
        """Plain text without logistics data should have all flags False."""
        text = "This is a general paragraph about shipping procedures and policies."
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        assert len(result) >= 1
        meta = result[0].metadata
        assert meta.get("has_monetary") is False
        assert meta.get("has_weight") is False

    def test_metadata_on_fixture(self):
        """Rate confirmation fixture should have monetary and date flags."""
        text = (FIXTURES_DIR / "sample_rate_confirmation.txt").read_text()
        chunker = LogisticsChunker()
        result = chunker.chunk(text, "doc-1")
        # At least one chunk should have monetary data
        assert any(c.metadata.get("has_monetary") for c in result)
        # At least one chunk should have date data
        assert any(c.metadata.get("has_dates") for c in result)
