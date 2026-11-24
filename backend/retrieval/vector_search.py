import logging
from typing import List, Tuple
from backend.ingestion.text_splitter import Chunk
from backend.embeddings.embedder import EmbeddingModel
from backend.storage.faiss_index import FaissIndex
from backend.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Executes dense semantic vector search using Sentence Transformers and FAISS."""

    def __init__(self, embedder: EmbeddingModel, faiss_index: FaissIndex, metadata_store: MetadataStore):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.metadata_store = metadata_store

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Embeds the query, searches FAISS, and retrieves corresponding Chunks with cosine scores.
        Returns list of (Chunk, similarity_score).
        """
        if self.faiss_index.total_vectors == 0:
            return []

        query_vector = self.embedder.embed_query(query)
        scores, faiss_ids = self.faiss_index.search(query_vector, top_k=top_k)

        results: List[Tuple[Chunk, float]] = []
        for score, fid in zip(scores, faiss_ids):
            if fid == -1:
                continue
            chunk = self.metadata_store.get_chunk_by_faiss_id(int(fid))
            if chunk:
                results.append((chunk, float(score)))

        return results
