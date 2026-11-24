import logging
from typing import List, Tuple
from backend.ingestion.text_splitter import Chunk
from backend.storage.bm25_store import BM25Store
from backend.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

class KeywordSearchEngine:
    """Executes BM25 keyword search using rank-bm25 and SQLite metadata store."""

    def __init__(self, bm25_store: BM25Store, metadata_store: MetadataStore):
        self.bm25_store = bm25_store
        self.metadata_store = metadata_store

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Computes BM25 scores and maps matched chunk_ids to Chunk objects.
        Returns list of (Chunk, bm25_score).
        """
        ranked_chunk_ids = self.bm25_store.search(query, top_k=top_k)
        results: List[Tuple[Chunk, float]] = []

        for chunk_id, score in ranked_chunk_ids:
            chunk = self.metadata_store.get_chunk_by_id(chunk_id)
            if chunk:
                results.append((chunk, score))

        return results
