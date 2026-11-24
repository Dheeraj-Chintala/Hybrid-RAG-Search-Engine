import re
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from rank_bm25 import BM25Okapi
from backend.ingestion.text_splitter import Chunk

logger = logging.getLogger(__name__)

class BM25Store:
    """
    In-memory BM25 index with persistence.
    Tokenizes chunks, computes BM25Okapi keyword scores, and saves tokenized corpus to disk.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load_or_create()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, fast tokenization: lowercases and extracts alphanumeric words."""
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _load_or_create(self) -> None:
        """Loads serialized tokenized corpus if available, and initializes BM25Okapi."""
        if self.persistence_path and self.persistence_path.exists():
            try:
                logger.info(f"Loading BM25 index from {self.persistence_path}")
                with open(self.persistence_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunk_ids = data.get("chunk_ids", [])
                    self.corpus_tokens = data.get("corpus_tokens", [])
                if self.corpus_tokens:
                    self._bm25 = BM25Okapi(self.corpus_tokens)
                    logger.info(f"BM25 index initialized with {len(self.chunk_ids)} documents.")
                return
            except Exception as e:
                logger.error(f"Error loading BM25 from {self.persistence_path}: {e}. Starting fresh.")

        self.chunk_ids = []
        self.corpus_tokens = []
        self._bm25 = None

    @property
    def total_documents(self) -> int:
        return len(self.chunk_ids)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Tokenizes new chunks, appends to the corpus, and rebuilds BM25."""
        if not chunks:
            return

        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            self.chunk_ids.append(chunk.chunk_id)
            self.corpus_tokens.append(tokens)

        # Re-initialize BM25Okapi with the updated full corpus
        self._bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Updated BM25 corpus to {len(self.chunk_ids)} chunks.")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Calculates BM25 scores for query across corpus.
        Returns top_k tuples of (chunk_id, bm25_score) ordered by descending score.
        """
        if not self._bm25 or not self.chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        
        # Rank descending by score
        indexed_scores = [(self.chunk_ids[i], float(scores[i])) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Only return items with score > 0 to avoid irrelevant noise
        filtered = [item for item in indexed_scores if item[1] > 0.0]
        return filtered[:top_k]

    def save(self, path: Optional[Path] = None) -> None:
        """Persists the BM25 tokenized corpus to disk."""
        target_path = Path(path) if path else self.persistence_path
        if not target_path:
            raise ValueError("No path specified to save BM25 corpus.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens
            }, f)
        logger.info(f"Saved BM25 corpus ({len(self.chunk_ids)} docs) to {target_path}")

    def reset(self) -> None:
        """Clears all stored BM25 documents and removes persistence file if present."""
        self.chunk_ids = []
        self.corpus_tokens = []
        self._bm25 = None
        if self.persistence_path and self.persistence_path.exists():
            try:
                self.persistence_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {self.persistence_path}: {e}")
        logger.info("BM25 store has been reset.")
