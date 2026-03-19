# Ultra Doc-Intelligence

A POC AI system that allows users to upload logistics documents (Rate Confirmations, BOLs, Invoices, Shipment Instructions) and interact with them using natural language questions. Built as an AI assistant for Transportation Management Systems (TMS).

## Evaluation Results (3 Real Logistics PDFs)

```
Q&A Accuracy:        16/16 (100%)
Extraction Accuracy: 12/12 (100%)
Guardrail Pass Rate:  2/2  (100%)
```

Tested against: Carrier Rate Confirmation (2 pages), Bill of Lading (2 pages), Shipper Rate Confirmation (1 page). Run `python scripts/demo.py` to reproduce.

## Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key
- MySQL 8.0+ (for document persistence)

### Local Setup

```bash
# Clone and install
git clone <repo-url>
cd ultra-doc-intelligence
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and DATABASE_URL

# Create MySQL database
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS ultra_doc_intelligence;"

# Run backend (auto-creates tables on startup)
uvicorn src.main:app --reload --port 8000

# Run UI (in a separate terminal)
streamlit run ui/app.py

# Run evaluation (tests against real PDFs)
python scripts/demo.py
```

### Docker

```bash
export OPENAI_API_KEY=sk-your-key
docker-compose up --build
```

Starts three services:
- **MySQL 8.0**: `localhost:3307` — document persistence (auto-creates database)
- **Backend API**: http://localhost:8000 — FastAPI (waits for MySQL healthy)
- **Streamlit UI**: http://localhost:8501 — upload, ask, extract (waits for backend healthy)
- **API Docs**: http://localhost:8000/docs

### Run Tests

```bash
pytest tests/ -v   # 133 tests
```

---

## Architecture

```
                          +------------------+
                          |   Streamlit UI   |
                          | (Upload/Ask/     |
                          |  Extract tabs)   |
                          +--------+---------+
                                   |
                            HTTP (REST API)
                                   |
                          +--------v---------+
                          |   FastAPI         |
                          | POST /upload     |
                          | POST /ask        |
                          | POST /extract    |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
    +---------v--------+  +-------v--------+  +---------v--------+
    |  Upload Service  |  |  Ask Service   |  | Extract Service  |
    | parse->chunk->   |  | retrieve->LLM  |  | full-text->LLM   |
    | embed->store     |  | ->guardrails   |  | ->structured JSON |
    +--+----+----+-----+  +--+----+----+---+  +--------+---------+
       |    |    |           |    |    |                |
       v    v    v           v    v    v                v
   +------+ +-------+ +--------+ +----------+ +----------+
   |Parser| |Chunker| |Embedder| |Retriever | |LLM Client|
   +------+ +-------+ +---+----+ +----+-----+ +----------+
                           |           |
                    +------v-----------v------+
                    |       ChromaDB           |
                    |   (Vector Store)         |
                    +------+------------------+
                           |
                    +------v------------------+
                    |       MySQL              |
                    |   (Document Records)     |
                    +-------------------------+
```

### Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| **DocumentParser** | `src/core/document_parser.py` | Extracts text from PDF (pdfplumber), DOCX (python-docx), TXT. Per-page extraction for page attribution. Fallback from pdfplumber to PyPDF2 for encrypted PDFs. |
| **LogisticsChunker** | `src/core/chunker.py` | 6-stage pipeline: section detection, table preservation, KV grouping, merge, split (paragraph -> sentence -> word), overlap injection, metadata enrichment |
| **Embedder** | `src/core/embedder.py` | OpenAI `text-embedding-3-small` (1536 dims), batched (100/batch), tenacity retry on transient errors |
| **VectorStore** | `src/core/vector_store.py` | ChromaDB adapter — per-document collections, cosine similarity, idempotent upsert |
| **Retriever** | `src/core/retriever.py` | Query embedding -> vector search (top-10) -> keyword reranking (top-5). 60% cosine + 40% keyword overlap. |
| **LLMClient** | `src/core/llm_client.py` | GPT-4o-mini (Q&A) / GPT-4o (extraction). Domain-expert system prompts. Forced function calling for extraction. |
| **ConfidenceScorer** | `src/guardrails/confidence.py` | 4-signal composite: retrieval similarity, chunk agreement, answer coverage, heuristic scoring |
| **GroundingChecker** | `src/guardrails/grounding.py` | Extracts factual claims ($, dates, weights, %) from answer, verifies they exist in source text |
| **ThresholdGuardrail** | `src/guardrails/threshold.py` | Out-of-scope detection (8 regex patterns) + similarity/confidence thresholds |

---

## Chunking Strategy

Logistics documents are **semi-structured** — they contain labeled fields, tables, key-value pairs, and short sections. Naive character-based chunking destroys this structure (splitting an address mid-line, breaking a rate table across chunks).

### 6-Stage Pipeline

```
Raw text -> Per-Page Split -> Section Detection -> Classify -> Merge Small
         -> Split Oversized (paragraph -> sentence -> word) -> Overlap Injection
         -> Metadata Enrichment
```

**Stage 1: Per-Page Isolation**
Multi-page documents are chunked page-by-page independently. This preserves accurate page numbers in Q&A responses ("Source: page 2").

**Stage 2: Section Header Detection**
Recognizes 25 logistics-specific headers (`SHIPPER`, `CONSIGNEE`, `CARRIER INFORMATION`, `RATE CONFIRMATION`, `BILL OF LADING`, `PICKUP`, `DELIVERY`, `EQUIPMENT`, `HANDLING UNIT INFORMATION`, etc.). Also detects ALL-CAPS lines as headers, with safeguards to exclude data lines (amounts, table rows, KV pairs). Each header starts a new chunk so semantically related content stays together.

**Stage 3: Table Preservation**
Pipe-delimited (`| Col1 | Col2 |`) and tab-delimited table rows are detected and kept as atomic blocks. Table rows never trigger section breaks, and table-type chunks are **never split** even if oversized — splitting a table mid-row makes it unreadable and hurts retrieval.

**Stage 4: Small Chunk Merging**
Orphan chunks under 50 characters (without meaningful headings) are merged into the next chunk to avoid low-information fragments.

**Stage 5: Oversized Chunk Splitting (3-tier fallback)**
Chunks exceeding 1,500 characters are split with graceful degradation:
1. **Paragraph boundaries** (`\n\n`) — try first
2. **Sentence boundaries** (`. ` before capital letter) — fallback if paragraphs are too large
3. **Word boundaries** — last resort, guarantees all chunks fit within max_chars

**Stage 6: Overlap Injection**
After all chunks are finalized, 200 characters of trailing context from each chunk are prepended to the next chunk **on the same page**. This prevents information loss at section boundaries — a question spanning two sections (e.g., "What carrier ships from ACME?") can find both facts in one chunk. Overlap never crosses page boundaries to preserve source attribution accuracy. Clean break points (newline, sentence end, word boundary) prevent mid-word starts.

**Stage 7: Metadata Enrichment**
Each chunk is scanned for semantic signals stored alongside the embeddings in ChromaDB:
- `has_monetary` — contains dollar amounts ($2,450.00, etc.)
- `has_dates` — contains date patterns (March 15, 2024 / 02-08-2026)
- `has_reference_numbers` — contains BOL#, LOAD-, MC#, PRO#, etc.
- `has_weight` — contains weight values (42,000 lbs)
- `has_table` — contains pipe/tab-delimited table rows

### Why Not Character-Based Chunking?

A `RecursiveCharacterTextSplitter(chunk_size=500)` on a rate confirmation would produce:
```
Chunk 1: "...CARRIER INFORMATION:\nCarrier: Swift Transp"
Chunk 2: "ortation\nMC#: MC-123456\nDOT#: 1234567\nContact..."
```

"Swift Transportation" is split across two chunks. The carrier name and MC# end up in different chunks. Retrieval for "What is the carrier's MC number?" would need both chunks and hope the LLM can stitch them together.

The logistics-aware chunker keeps `Carrier: Swift Transportation / MC#: MC-123456 / DOT#: 1234567` in a single chunk.

---

## Retrieval Method

### Pipeline: Embed -> Search -> Rerank -> Context Assembly

1. **Query Embedding**: The user's question is embedded using OpenAI `text-embedding-3-small` (1536 dimensions).

2. **Vector Search**: ChromaDB cosine similarity search retrieves top-10 candidate chunks from the document's collection.

3. **Keyword-Based Reranking**: Candidates are reranked using a combined score:
   - **60% cosine similarity** (semantic relevance)
   - **40% keyword overlap** (exact term matching between question and chunk)
   - Top-5 results are kept after reranking.

4. **Context Assembly**: Top chunks are concatenated with `---` separators and passed to the LLM as document context.

### Why Keyword Reranking?

Pure semantic search can miss exact matches. For "What is the BOL number?", a chunk containing "BOL#: BOL-998877" has high keyword overlap but might not be the top semantic match. Logistics documents are dense with reference numbers, amounts, and identifiers — keyword overlap catches these exact matches that embeddings may miss.

---

## Guardrails Approach

Four layers of guardrails prevent hallucination and ensure answer quality:

### 1. Out-of-Scope Detection (Pre-Retrieval)
Blocks questions unrelated to document Q&A using 8 regex patterns: "What do you think about...", "Write me a poem", "Tell me about yourself", etc. Runs before retrieval to save API cost.

### 2. Retrieval Similarity Threshold (Post-Retrieval)
If the highest cosine similarity score is below 0.3, the system refuses to answer: *"Not found in document. The question does not match any content in the uploaded document."*

### 3. Grounding Check (Post-LLM)
Extracts factual claims from the LLM's answer (dollar amounts, dates, percentages, weights) and verifies each claim exists in the source chunks. If fewer than 50% of claims are grounded, a warning is issued. This catches hallucinated numbers/dates.

### 4. Confidence Threshold (Final Gate)
If the composite confidence score is below 0.4, the system refuses to answer: *"The answer could not be determined with sufficient confidence from the document."*

---

## Confidence Scoring Method

A **4-signal composite score** on [0.0, 1.0]:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| **Retrieval Similarity** | 35% | How well the retrieved chunks match the query (0.6 * top score + 0.4 * avg top-3) |
| **Chunk Agreement** | 25% | How many of the top-5 chunks share keywords with the answer (consensus) |
| **Answer Coverage** | 25% | What fraction of factual claims in the answer (amounts, dates, names) appear in the source text |
| **Heuristic Scoring** | 15% | Domain-specific checks: rate questions should contain `$`, date questions should contain dates; penalties for hedging language |

### Formula
```
confidence = 0.35 * retrieval_similarity + 0.25 * chunk_agreement + 0.25 * answer_coverage + 0.15 * heuristic
```

### Thresholds
- **>= 0.7**: HIGH confidence — answer returned normally
- **0.4 - 0.7**: MEDIUM confidence — answer returned with advisory
- **< 0.4**: LOW confidence — answer refused

### Why 4 Signals?
Each signal catches a different failure mode:
- **Retrieval Similarity**: Catches irrelevant questions (no matching content)
- **Chunk Agreement**: Catches thin evidence (answer supported by only 1 of 5 chunks)
- **Answer Coverage**: Catches hallucination (LLM invents numbers not in the document)
- **Heuristic**: Catches wrong-type answers (rate question answered with a name)

---

## Structured Extraction

Uses GPT-4o with **forced function calling** (`tool_choice`) to extract 11 fields from the full document text:

| Field | Description |
|-------|-------------|
| `shipment_id` | Primary reference: Load Number, Shipment ID (not BOL# or PRO#) |
| `shipper` | SHIP FROM party (not broker or freight payer) |
| `consignee` | SHIP TO / DELIVER TO party |
| `pickup_datetime` | Pickup date and time as written |
| `delivery_datetime` | Delivery date and time as written |
| `equipment_type` | Trailer spec (53' Dry Van, Reefer, Flatbed) |
| `mode` | FTL, LTL, Intermodal, Partial |
| `rate` | Line haul / base rate as a number |
| `currency` | ISO 4217 code (USD, CAD, MXN) |
| `weight` | Total weight with unit (42,000 lbs) |
| `carrier_name` | Trucking company (not broker or 3PL) |

Missing fields are returned as `null`. Extraction confidence = 40% completeness + 60% grounding (extracted values verified against source text).

---

## API Endpoints

### POST /upload
Upload a logistics document for processing.

**Request**: `multipart/form-data` with `file` field (PDF, DOCX, or TXT, max 10MB)

**Response**:
```json
{
  "document_id": "abc123-uuid",
  "filename": "rate_confirmation.pdf",
  "num_chunks": 12,
  "num_pages": 2,
  "status": "processed"
}
```

### POST /ask
Ask a question about an uploaded document.

**Request**:
```json
{"document_id": "abc123-uuid", "question": "What is the carrier rate?"}
```

**Response**:
```json
{
  "answer": "The carrier rate is $2,450.00 USD.",
  "confidence": {
    "score": 0.87,
    "level": "high",
    "components": {
      "retrieval_similarity": 0.92,
      "chunk_agreement": 0.80,
      "answer_coverage": 0.85,
      "heuristic": 0.90
    }
  },
  "sources": [{"text": "RATE: $2,450.00 USD", "chunk_id": "...", "page_number": 1, "section": "RATE"}],
  "guardrails": {"grounding_check": "passed", "retrieval_quality": "passed"}
}
```

### POST /extract
Extract structured shipment data from an uploaded document.

**Request**:
```json
{"document_id": "abc123-uuid"}
```

**Response**:
```json
{
  "shipment_data": {
    "shipment_id": "LOAD-2024-78543",
    "shipper": "ACME Manufacturing Inc.",
    "consignee": "Best Buy Distribution Center",
    "pickup_datetime": "March 15, 2024 08:00 AM",
    "delivery_datetime": "March 17, 2024 02:00 PM",
    "equipment_type": "53' Dry Van",
    "mode": "FTL",
    "rate": 2450.0,
    "currency": "USD",
    "weight": "42,000 lbs",
    "carrier_name": "Swift Transportation"
  },
  "extraction_confidence": 1.0,
  "missing_fields": [],
  "document_id": "abc123-uuid"
}
```

---

## Known Failure Cases

1. **Scanned PDFs / Image-Only PDFs**: pdfplumber cannot extract text from scanned documents. PyPDF2 fallback also fails. OCR (Tesseract) integration would be needed.

2. **Ambiguous Questions**: "What is the rate?" on a document with multiple rates (line haul, fuel surcharge, total) — the system returns the most prominent value and notes related figures, but disambiguation could be improved.

3. **Non-English Documents**: The system is tuned for English logistics documents. Non-English documents will have degraded chunking and retrieval quality.

4. **Very Short Documents**: Documents with fewer than 3 lines produce few or no chunks, reducing retrieval effectiveness.

5. **Complex Nested Structures**: Documents with deeply nested sections (sub-sub-clauses, appendices within appendices) may not chunk optimally.

6. **Handwritten Annotations**: Any handwritten notes on printed documents are invisible to text extraction.

---

## Project Structure

```
ultra-doc-intelligence/
├── src/
│   ├── api/           # FastAPI routes and request/response models
│   ├── core/          # Document parser, chunker, embedder, vector store, retriever, LLM client
│   ├── db/            # SQLAlchemy ORM models, session management, repository
│   ├── services/      # Upload, ask, and extract orchestration services
│   ├── guardrails/    # Confidence scoring, grounding check, threshold guardrails
│   ├── models/        # Pydantic/dataclass models (Chunk, ShipmentData, responses)
│   └── util/          # Text utilities, logging setup
├── ui/                # Streamlit application (3 tabs: Upload, Ask, Extract)
├── scripts/           # E2E demo/evaluation script (16 Q&A + extraction tests)
├── tests/             # 133 unit and integration tests (TDD)
│   └── fixtures/      # Sample logistics documents (3 PDFs + 3 TXT samples)
├── Dockerfile
├── docker-compose.yml # Backend + UI + MySQL (health checks, dependency ordering)
├── render.yaml        # Render.com deployment config
└── requirements.txt
```

## Tech Stack

- **Backend**: FastAPI (Python 3.12) — fully async route handlers
- **LLM**: OpenAI GPT-4o-mini (Q&A) + GPT-4o (extraction) with tenacity exponential backoff
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions), batched (100/batch)
- **Vector Store**: ChromaDB (persistent, cosine similarity, per-document collections)
- **Database**: MySQL 8.0 (SQLAlchemy ORM) for document record persistence
- **Document Parsing**: pdfplumber (primary) + PyPDF2 (fallback) + python-docx
- **UI**: Streamlit (3 tabs: Upload, Ask Questions, Structured Extraction)
- **Testing**: pytest — 133 tests across 13 test files, TDD
- **Deployment**: Docker Compose (MySQL + Backend + UI with health checks), Render.com
