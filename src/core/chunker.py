"""Logistics-aware document chunker.

Splits document text into semantically meaningful chunks by detecting
section headers, key-value pair blocks, and tables — structures common
in Rate Confirmations, BOLs, Freight Invoices, and Shipment Instructions.

Unlike naive fixed-size chunking, this preserves logical boundaries so
that related information (e.g., shipper name + address) stays together,
improving retrieval accuracy for downstream RAG queries.

Chunking pipeline (6 stages):
1. Section header detection (logistics-specific + ALL-CAPS).
2. Table detection — pipe/tab-delimited rows kept as atomic units.
3. Key-value pair grouping (Label: Value blocks stay together).
4. Small chunk merging (fragments < min_chars join neighbours).
5. Oversized chunk splitting (paragraph → sentence → word fallback).
6. Overlap injection — trailing context from prior chunk prepended
   so boundary information is not lost during retrieval.
"""

import re
import uuid
from typing import List, Optional

from src.models.document import Chunk
from src.util.logging_setup import get_logger

logger = get_logger(__name__)


# Logistics-specific section headers
LOGISTICS_HEADERS = {
    "RATE CONFIRMATION", "BILL OF LADING", "FREIGHT INVOICE",
    "SHIPPER", "CONSIGNEE", "SHIP FROM", "SHIP TO",
    "CARRIER INFORMATION", "CARRIER NAME", "CARRIER",
    "PICKUP", "DELIVERY", "ORIGIN", "DESTINATION",
    "EQUIPMENT", "COMMODITY", "LOAD DETAILS",
    "SPECIAL INSTRUCTIONS", "PAYMENT TERMS", "CHARGES",
    "HANDLING UNIT INFORMATION", "SHIPMENT DETAILS",
    "BILL TO", "REMIT PAYMENT TO",
    "THIRD PARTY FREIGHT CHARGES BILL TO",
}

# Patterns that indicate a section header line
HEADER_PATTERNS = [
    # Known logistics headers (with optional colon)
    re.compile(
        r"^(" + "|".join(re.escape(h) for h in LOGISTICS_HEADERS) + r"):?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ALL-CAPS lines under 80 chars (likely headers)
    re.compile(r"^[A-Z][A-Z\s\-/&]{2,78}:?\s*$", re.MULTILINE),
    # Numbered sections: "1.", "2.", "Section 1", "Article I"
    re.compile(r"^\s*(?:Section|Article|Part)?\s*\d+[.)]\s+.{3,}$", re.IGNORECASE | re.MULTILINE),
]

# Key-value pair pattern
KV_PATTERN = re.compile(
    r"^([A-Za-z][A-Za-z\s#./()]+?):\s+(.+)$",
    re.MULTILINE,
)

# Table row: line containing 2+ pipe characters or tab-delimited columns
_TABLE_ROW_RE = re.compile(r"^[|].*[|]|^.*\t.*\t", re.MULTILINE)
# Separator row in markdown-style tables: |---|---|
_TABLE_SEP_RE = re.compile(r"^[|\s\-:]+$")

# Sentence boundary — split after ". ", "! ", "? " but not abbreviations
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Metadata detection patterns
_MONETARY_RE = re.compile(r"\$[\d,]+\.?\d*|\d+\.\d{2}\s*(?:USD|CAD|MXN)", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b"
    r"|\b\d{4}[-/]\d{2}[-/]\d{2}\b",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"\b(?:BOL|PRO|PO|MC|DOT|LOAD|LD|INV|SID|CID|SEAL)[#\-]?\s*[:\-#]?\s*[\w\-]+",
    re.IGNORECASE,
)
_WEIGHT_RE = re.compile(r"\b[\d,]+\.?\d*\s*(?:lbs?|kg|tons?)\b", re.IGNORECASE)


class LogisticsChunker:
    """Splits logistics document text into semantic chunks.

    Strategies applied in order:
    1. Section header detection (logistics-specific + ALL-CAPS).
    2. Table detection — pipe/tab rows are kept as atomic blocks.
    3. Key-value pair grouping (Label: Value blocks stay together).
    4. Small chunk merging (fragments < min_chars join neighbours).
    5. Oversized chunk splitting (paragraph → sentence → word fallback).
    6. Overlap injection — trailing context from the prior chunk is
       prepended so retrieval doesn't miss boundary information.

    Args:
        min_chars: Minimum chunk size — smaller fragments are merged.
        max_chars: Maximum chunk size — larger chunks are split.
        overlap_chars: Characters of trailing context to prepend from the
                       previous chunk.  Set to 0 to disable overlap.
    """

    def __init__(self, min_chars: int = 50, max_chars: int = 1500, overlap_chars: int = 200):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(
        self,
        text: str,
        document_id: str,
        pages: List[str] | None = None,
    ) -> List[Chunk]:
        """Chunk document text into semantically meaningful pieces.

        Args:
            text: Full document text (used when pages is not provided).
            document_id: Parent document ID for chunk association.
            pages: Optional per-page text list for accurate page_number assignment.
                   When provided, chunking is done per-page to preserve page boundaries.
        """
        if not text or not text.strip():
            return []

        # If per-page text is available, chunk each page separately
        # to preserve accurate page numbers for source attribution.
        if pages and len(pages) > 1:
            return self._chunk_with_pages(pages, document_id)

        return self._chunk_text(text, document_id, page_number=1)

    def _chunk_with_pages(self, pages: List[str], document_id: str) -> List[Chunk]:
        """Chunk each page independently, add overlap within each page."""
        all_chunks: List[Chunk] = []

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text or not page_text.strip():
                continue
            page_chunks = self._chunk_text(page_text, document_id, page_number=page_num)
            all_chunks.extend(page_chunks)

        logger.info(
            "Chunked document %s: %d pages -> %d chunks",
            document_id[:8], len(pages), len(all_chunks),
        )
        return all_chunks

    def _chunk_text(self, text: str, document_id: str, page_number: int = 1) -> List[Chunk]:
        """Core chunking logic for a single block of text."""
        if not text or not text.strip():
            return []

        # Split text into sections based on header detection
        sections = self._split_into_sections(text)

        # Build chunks from sections
        chunks = []
        for section in sections:
            heading = section["heading"]
            body = section["body"].strip()
            if not body:
                continue

            section_type = self._classify_section(heading, body)

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=body if not heading else f"{heading}\n{body}",
                section_type=section_type,
                heading=heading,
                page_number=page_number,
                char_offset_start=section.get("start", 0),
                char_offset_end=section.get("end", 0),
            ))

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Split oversized chunks (preserves tables as atomic units)
        chunks = self._split_oversized_chunks(chunks, document_id, page_number)

        # Filter empty
        chunks = [c for c in chunks if c.text.strip()]

        # Inject overlap from previous chunk so boundary context is preserved
        chunks = self._add_overlap(chunks)

        # Enrich with metadata signals for downstream retrieval
        for chunk in chunks:
            chunk.metadata = self._extract_metadata(chunk.text)

        return chunks

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> List[dict]:
        lines = text.split("\n")
        sections: List[dict] = []
        current_heading = ""
        current_body_lines: List[str] = []
        current_start = 0
        in_table = False

        for i, line in enumerate(lines):
            # Table continuity: once we detect a table row, keep
            # accumulating rows until a non-table line appears.
            if self._is_table_row(line):
                in_table = True
                current_body_lines.append(line)
                continue

            if in_table and not line.strip():
                # Blank line right after table — end the table block
                in_table = False
                current_body_lines.append(line)
                continue

            in_table = False

            is_header = self._is_header_line(line)

            if is_header and current_body_lines:
                # Save previous section
                body = "\n".join(current_body_lines)
                sections.append({
                    "heading": current_heading,
                    "body": body,
                    "start": current_start,
                    "end": current_start + len(body),
                })
                current_heading = line.strip().rstrip(":")
                current_body_lines = []
                current_start = sum(len(l) + 1 for l in lines[:i])
            elif is_header and not current_body_lines:
                current_heading = line.strip().rstrip(":")
                current_start = sum(len(l) + 1 for l in lines[:i])
            else:
                current_body_lines.append(line)

        # Don't forget the last section
        if current_body_lines:
            body = "\n".join(current_body_lines)
            sections.append({
                "heading": current_heading,
                "body": body,
                "start": current_start,
                "end": current_start + len(body),
            })
        elif current_heading:
            # Header with no body
            sections.append({
                "heading": current_heading,
                "body": current_heading,
                "start": current_start,
                "end": current_start + len(current_heading),
            })

        return sections

    # ------------------------------------------------------------------
    # Line-level classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Detect pipe-delimited or tab-delimited table rows."""
        stripped = line.strip()
        if not stripped:
            return False
        # Markdown-style separator row (|---|---|)
        if _TABLE_SEP_RE.match(stripped):
            return True
        # Pipe-delimited row with 2+ cells
        if stripped.startswith("|") and stripped.count("|") >= 3:
            return True
        # Tab-delimited row with 2+ tabs
        if stripped.count("\t") >= 2:
            return True
        return False

    def _is_header_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        # Table rows are never headers
        if self._is_table_row(line):
            return False

        # Check against known logistics headers
        normalized = stripped.rstrip(":").upper()
        if normalized in LOGISTICS_HEADERS:
            return True

        # Check ALL-CAPS lines (short, no digits dominant)
        # But exclude lines that are key-value pairs (e.g., "BOL#: BOL-998877")
        if (stripped == stripped.upper()
                and len(stripped) < 80
                and len(stripped) > 2
                and re.search(r"[A-Z]{3,}", stripped)
                and not re.match(r"^[\d$.,\-\s]+$", stripped)):
            # Exclude lines that are clearly data or key-value pairs
            if re.search(r"\$\d|^\d+\s*\||\|\s*\d", stripped):
                return False
            # Exclude KV pairs: "LABEL: some_value" where value has digits/mixed
            if re.match(r"^[A-Z][A-Z\s#./()]+:\s+\S", stripped):
                return False
            return True

        return False

    def _classify_section(self, heading: str, body: str) -> str:
        # Table detection takes priority — a section whose body contains
        # pipe/tab-delimited rows is a table regardless of its heading.
        if self._has_table_content(body):
            return "table"

        if not heading:
            kv_matches = KV_PATTERN.findall(body)
            if len(kv_matches) >= 2:
                return "key_value_block"
            return "general"

        heading_upper = heading.upper()
        if heading_upper in LOGISTICS_HEADERS:
            return "key_value_block"

        return "body"

    @staticmethod
    def _has_table_content(text: str) -> bool:
        """Check whether text contains pipe-delimited or tab-delimited table rows."""
        lines = text.split("\n")
        table_rows = sum(1 for line in lines if LogisticsChunker._is_table_row(line))
        return table_rows >= 2

    # ------------------------------------------------------------------
    # Merge, split, overlap
    # ------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            # If chunk is too small, has no meaningful heading, and there's a next chunk
            if (len(current.text) < self.min_chars
                    and not current.heading
                    and i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                merged_text = current.text + "\n\n" + next_chunk.text
                merged.append(Chunk(
                    chunk_id=next_chunk.chunk_id,
                    document_id=current.document_id,
                    text=merged_text,
                    section_type=next_chunk.section_type,
                    heading=next_chunk.heading or current.heading,
                    page_number=current.page_number,
                    char_offset_start=current.char_offset_start,
                    char_offset_end=next_chunk.char_offset_end,
                ))
                i += 2
            else:
                merged.append(current)
                i += 1

        return merged

    def _split_oversized_chunks(
        self, chunks: List[Chunk], document_id: str, page_number: int
    ) -> List[Chunk]:
        result = []
        for chunk in chunks:
            if len(chunk.text) <= self.max_chars:
                result.append(chunk)
                continue

            # Never split table chunks — they are atomic units.
            # If a table exceeds max_chars, keep it whole so downstream
            # consumers see complete rows rather than broken fragments.
            if chunk.section_type == "table":
                result.append(chunk)
                continue

            # Stage 1: split on paragraph boundaries (\n\n)
            sub_chunks = self._split_on_paragraphs(
                chunk.text, chunk, document_id, page_number,
            )

            # Stage 2: any still-oversized parts get split on sentence boundaries
            final = []
            for sc in sub_chunks:
                if len(sc.text) <= self.max_chars:
                    final.append(sc)
                else:
                    final.extend(
                        self._split_on_sentences(sc.text, sc, document_id, page_number)
                    )
            result.extend(final)

        return result

    def _split_on_paragraphs(
        self, text: str, source: Chunk, document_id: str, page_number: int
    ) -> List[Chunk]:
        """Split text on double-newline paragraph boundaries."""
        parts = re.split(r"\n\n+", text)
        return self._accumulate_parts(parts, "\n\n", source, document_id, page_number)

    def _split_on_sentences(
        self, text: str, source: Chunk, document_id: str, page_number: int
    ) -> List[Chunk]:
        """Split text on sentence boundaries (fallback when paragraphs are too large)."""
        parts = _SENTENCE_RE.split(text)
        if len(parts) <= 1:
            # No sentence boundaries found — hard split on word boundary
            return self._split_on_words(text, source, document_id, page_number)
        return self._accumulate_parts(parts, " ", source, document_id, page_number)

    def _split_on_words(
        self, text: str, source: Chunk, document_id: str, page_number: int
    ) -> List[Chunk]:
        """Last-resort split: break on word boundaries."""
        words = text.split()
        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > self.max_chars and current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += len(word) + 1
        if current:
            parts.append(" ".join(current))

        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=p,
                section_type=source.section_type,
                heading=source.heading,
                page_number=page_number,
            )
            for p in parts if p.strip()
        ]

    def _accumulate_parts(
        self,
        parts: List[str],
        joiner: str,
        source: Chunk,
        document_id: str,
        page_number: int,
    ) -> List[Chunk]:
        """Greedily accumulate text parts into chunks up to max_chars."""
        chunks: List[Chunk] = []
        current_part = ""
        for part in parts:
            candidate = current_part + joiner + part if current_part else part
            if len(candidate) <= self.max_chars:
                current_part = candidate
            else:
                if current_part.strip():
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=current_part.strip(),
                        section_type=source.section_type,
                        heading=source.heading,
                        page_number=page_number,
                    ))
                current_part = part

        if current_part.strip():
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=current_part.strip(),
                section_type=source.section_type,
                heading=source.heading,
                page_number=page_number,
            ))

        return chunks

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Prepend trailing context from the previous chunk.

        Overlap gives each chunk awareness of what came immediately
        before it, preventing information loss at chunk boundaries.
        Only applied between chunks on the **same page** to avoid
        cross-page context pollution that would harm source attribution.
        """
        if self.overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Only overlap within the same page
            if prev.page_number != curr.page_number:
                result.append(curr)
                continue

            tail = prev.text[-self.overlap_chars:]

            # Find a clean break (newline or sentence end) so we don't
            # start mid-word
            for sep in ("\n", ". ", "? ", "! ", ", ", " "):
                pos = tail.find(sep)
                if pos >= 0:
                    tail = tail[pos + len(sep):]
                    break

            if not tail.strip():
                result.append(curr)
                continue

            result.append(Chunk(
                chunk_id=curr.chunk_id,
                document_id=curr.document_id,
                text=tail.strip() + "\n\n" + curr.text,
                section_type=curr.section_type,
                heading=curr.heading,
                page_number=curr.page_number,
                char_offset_start=curr.char_offset_start,
                char_offset_end=curr.char_offset_end,
                metadata=curr.metadata,
            ))

        return result

    # ------------------------------------------------------------------
    # Metadata enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(text: str) -> dict:
        """Extract semantic signals from chunk text for downstream use.

        These flags let the retriever or confidence scorer know what
        *kind* of information a chunk contains without re-parsing.
        """
        return {
            "has_monetary": bool(_MONETARY_RE.search(text)),
            "has_dates": bool(_DATE_RE.search(text)),
            "has_reference_numbers": bool(_REFERENCE_RE.search(text)),
            "has_weight": bool(_WEIGHT_RE.search(text)),
            "has_table": LogisticsChunker._has_table_content(text),
        }
