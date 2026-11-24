import logging
from typing import List, Dict, Any
from .vector_search import VectorSearchEngine
from .keyword_search import KeywordSearchEngine
from .rrf import ReciprocalRankFusion

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Coordinates semantic vector search and BM25 keyword search,
    combining them via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        rrf_k: int = 60
    ):
        self.vector_engine = vector_engine
        self.keyword_engine = keyword_engine
        self.rrf = ReciprocalRankFusion(k=rrf_k)

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Executes both search streams and fuses their outputs with RRF.
        Returns top_k fused candidate items.
        """
        logger.info(f"Executing hybrid search for query: '{query}'")

        vector_results = self.vector_engine.search(query, top_k=top_k)
        keyword_results = self.keyword_engine.search(query, top_k=top_k)

        logger.debug(f"Retrieved {len(vector_results)} vector results and {len(keyword_results)} BM25 results.")

        fused = self.rrf.fuse(vector_results, keyword_results, top_k=top_k)
        logger.info(f"RRF fusion yielded {len(fused)} candidates.")
        return fused
