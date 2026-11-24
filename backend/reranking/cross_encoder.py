import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate chunks using a lightweight Cross-Encoder model.
    Runs joint full self-attention over (query, document) pairs to score deep semantic relevance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}'...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("CrossEncoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate dictionary items (output of RRF).
        Each candidate contains at least 'chunk' (Chunk object).
        Appends 'rerank_score' to each candidate and returns the top_k sorted descending.
        """
        if not candidates:
            return []

        self._load_model()

        # Build query-chunk text pairs
        pairs = [(query, item["chunk"].text) for item in candidates]
        logger.debug(f"Predicting CrossEncoder scores for {len(pairs)} pairs...")

        # Compute raw logits / scores
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Handle scalar output for single pair
        if isinstance(scores, (float, np.floating)):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        # Attach scores to items
        reranked_items = []
        for item, score in zip(candidates, scores):
            # Create a shallow copy with rerank_score
            item_copy = dict(item)
            item_copy["rerank_score"] = round(score, 4)
            reranked_items.append(item_copy)

        # Sort descending by rerank_score
        reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_items[:top_k]
