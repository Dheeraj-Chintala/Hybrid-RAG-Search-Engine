import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Generates dense embeddings using Sentence Transformers on CPU.
    Normalizes embeddings to unit length so Inner Product computes Cosine Similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer model loaded successfully.")

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        self._load_model()
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of strings, returning a float32 numpy array of shape (N, dimension).
        Embeddings are L2 normalized for cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning float32 numpy array of shape (1, dimension)."""
        self._load_model()
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)
