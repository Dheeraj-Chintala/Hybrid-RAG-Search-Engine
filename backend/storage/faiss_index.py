import logging
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# On Windows Python 3.8+, register faiss_cpu.libs directory for OpenMP DLL resolution
if sys.platform == "win32":
    for _p in sys.path:
        _libs_dir = Path(_p) / "faiss_cpu.libs"
        if _libs_dir.is_dir():
            try:
                os.add_dll_directory(str(_libs_dir.resolve()))
            except Exception:
                pass

try:
    import torch  # noqa: F401
except Exception:
    pass

import faiss

logger = logging.getLogger(__name__)

class FaissIndex:
    """
    Manages a persistent FAISS index using cosine similarity (IndexFlatIP on normalized vectors)
    with an IndexIDMap to assign explicit integer chunk IDs.
    """

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self._index: Optional[faiss.IndexIDMap] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Loads index from disk if it exists, otherwise creates a new IndexIDMap."""
        if self.index_path and self.index_path.exists():
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                raw_index = faiss.read_index(str(self.index_path))
                if isinstance(raw_index, faiss.IndexIDMap):
                    self._index = raw_index
                else:
                    self._index = faiss.IndexIDMap(raw_index)
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors.")
                return
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}. Creating new index.")

        # Create IndexFlatIP (Inner Product) wrapped in IndexIDMap
        flat_index = faiss.IndexFlatIP(self.dimension)
        self._index = faiss.IndexIDMap(flat_index)
        logger.info(f"Initialized new FAISS IndexIDMap(IndexFlatIP({self.dimension})).")

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def add(self, embeddings: np.ndarray, ids: List[int]) -> None:
        """
        Adds vectors with corresponding integer IDs.
        Vectors must be float32 and normalized.
        """
        if len(ids) == 0:
            return

        if embeddings.shape[0] != len(ids):
            raise ValueError(f"Mismatch between embedding count ({embeddings.shape[0]}) and ids ({len(ids)})")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        id_array = np.ascontiguousarray(np.array(ids, dtype=np.int64))

        self._index.add_with_ids(embeddings, id_array)
        logger.info(f"Added {len(ids)} vectors to FAISS index. Total vectors: {self.total_vectors}")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs inner product (cosine similarity) search for a single query vector.
        Returns:
            scores: float array of cosine similarity scores
            ids: integer array of retrieved chunk IDs
        """
        if self.total_vectors == 0:
            return np.array([]), np.array([], dtype=np.int64)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(top_k, self.total_vectors)
        scores, ids = self._index.search(query_vector, k)
        return scores[0], ids[0]

    def save(self, path: Optional[Path] = None) -> None:
        """Persists the FAISS index to disk."""
        target_path = Path(path) if path else self.index_path
        if not target_path:
            raise ValueError("No path specified to save FAISS index.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target_path))
        logger.info(f"Saved FAISS index ({self.total_vectors} vectors) to {target_path}")

    def reset(self) -> None:
        """Resets the index to empty."""
        flat_index = faiss.IndexFlatIP(self.dimension)
        self._index = faiss.IndexIDMap(flat_index)
        if self.index_path and self.index_path.exists():
            self.save(self.index_path)
        logger.info("FAISS index has been reset.")
