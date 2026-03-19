import pytest

from src.core.vector_store import VectorStore
from src.models.document import Chunk


class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Temporary ChromaDB instance for testing."""
        return VectorStore(persist_dir=str(tmp_path / "test_chroma"))

    @pytest.fixture
    def sample_data(self):
        """Sample chunks and embeddings for testing."""
        chunks = [
            Chunk(chunk_id="c1", document_id="doc-1",
                  text="Carrier: Swift Transportation MC#: MC-123456",
                  heading="CARRIER", section_type="key_value_block"),
            Chunk(chunk_id="c2", document_id="doc-1",
                  text="RATE: $2,450.00 USD FUEL SURCHARGE: $175.00",
                  heading="RATE", section_type="key_value_block"),
            Chunk(chunk_id="c3", document_id="doc-1",
                  text="SHIPPER: ACME Manufacturing Inc. Chicago, IL",
                  heading="SHIPPER", section_type="key_value_block"),
        ]
        # Simple fake embeddings (3 chunks x 10 dims for speed)
        embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        return chunks, embeddings

    def test_upsert_and_search(self, store, sample_data):
        """Upserted chunks should be searchable."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        # Search with a query similar to the rate chunk
        query_vec = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        results = store.search("doc-1", query_vec, top_k=3)
        assert len(results) > 0
        assert any("$2,450.00" in r["text"] for r in results)

    def test_search_returns_scores(self, store, sample_data):
        """Search results should include similarity scores."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        query_vec = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        results = store.search("doc-1", query_vec, top_k=3)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_search_returns_chunk_metadata(self, store, sample_data):
        """Search results should include chunk text and metadata."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        query_vec = [0.5] * 10
        results = store.search("doc-1", query_vec, top_k=1)
        assert len(results) == 1
        assert "text" in results[0]
        assert "chunk_id" in results[0]
        assert "heading" in results[0]
        assert "page_number" in results[0]

    def test_search_filters_by_document(self, store, sample_data):
        """Search should only return chunks from specified document."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        # Search for a non-existent document
        query_vec = [0.5] * 10
        results = store.search("doc-999", query_vec, top_k=3)
        assert len(results) == 0

    def test_delete_document(self, store, sample_data):
        """Deleted document chunks should not appear in search."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        store.delete_document("doc-1")

        query_vec = [0.5] * 10
        results = store.search("doc-1", query_vec, top_k=3)
        assert len(results) == 0

    def test_upsert_is_idempotent(self, store, sample_data):
        """Upserting same document twice should not duplicate chunks."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)
        store.upsert("doc-1", chunks, embeddings)

        query_vec = [0.5] * 10
        results = store.search("doc-1", query_vec, top_k=10)
        assert len(results) == 3  # Still only 3 chunks

    def test_get_document_text(self, store, sample_data):
        """Should be able to retrieve full document text."""
        chunks, embeddings = sample_data
        store.upsert("doc-1", chunks, embeddings)

        text = store.get_document_text("doc-1")
        assert "Swift Transportation" in text
        assert "$2,450.00" in text
        assert "ACME Manufacturing" in text

    def test_get_document_text_missing(self, store):
        """Getting text for non-existent document returns empty string."""
        text = store.get_document_text("doc-999")
        assert text == ""
