from typing import List, Dict, Any, Tuple
from backend.ingestion.text_splitter import Chunk

class ReciprocalRankFusion:
    """
    Combines ranked lists from multiple search algorithms (e.g. Vector Search & BM25)
    using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in methods} (1 / (k + rank_m(d)))
    where k is a constant (typically 60) that dampens high-ranking outliers.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        vector_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuses vector and BM25 results into a combined ranked list.
        Returns list of dictionaries containing:
            - chunk: Chunk object
            - rrf_score: Combined RRF score
            - vector_rank: 1-based rank in vector results (or None)
            - vector_score: Original cosine similarity score (or None)
            - bm25_rank: 1-based rank in BM25 results (or None)
            - bm25_score: Original BM25 score (or None)
        """
        candidates: Dict[str, Dict[str, Any]] = {}

        # Process vector search results
        for rank, (chunk, score) in enumerate(vector_results, start=1):
            cid = chunk.chunk_id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "vector_rank": rank,
                    "vector_score": score,
                    "bm25_rank": None,
                    "bm25_score": None
                }
            candidates[cid]["rrf_score"] += 1.0 / (self.k + rank)

        # Process BM25 keyword search results
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": rank,
                    "bm25_score": score
                }
            else:
                candidates[cid]["bm25_rank"] = rank
                candidates[cid]["bm25_score"] = score
            candidates[cid]["rrf_score"] += 1.0 / (self.k + rank)

        # Sort descending by rrf_score
        fused_list = list(candidates.values())
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused_list[:top_k]
