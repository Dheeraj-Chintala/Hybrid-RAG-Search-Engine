import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # API & Server
    backend_host: str = Field(default=os.getenv("BACKEND_HOST", "127.0.0.1"))
    backend_port: int = Field(default=int(os.getenv("BACKEND_PORT", "8000")))

    # LLM Settings (OpenRouter & Gemini)
    openrouter_api_key: str = Field(default=os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = Field(default=os.getenv("OPENROUTER_MODEL", "openrouter/free"))
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    llm_provider: str = Field(default=os.getenv("LLM_PROVIDER", "openrouter"))

    # Embeddings & Reranking
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    reranker_model: str = Field(default=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Chunking Defaults
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "100")))

    # Retrieval Defaults
    top_k_retrieval: int = Field(default=int(os.getenv("TOP_K_RETRIEVAL", "10")))
    top_k_rerank: int = Field(default=int(os.getenv("TOP_K_RERANK", "5")))
    rrf_k: int = Field(default=int(os.getenv("RRF_K", "60")))

    # Storage Paths
    storage_dir: Path = Field(default=BASE_DIR / os.getenv("STORAGE_DIR", "data"))
    uploads_dir: Path = Field(default=BASE_DIR / os.getenv("STORAGE_DIR", "data") / "uploads")
    faiss_index_path: Path = Field(default=BASE_DIR / os.getenv("STORAGE_DIR", "data") / "faiss.index")
    metadata_db_path: Path = Field(default=BASE_DIR / os.getenv("STORAGE_DIR", "data") / "metadata.db")
    bm25_corpus_path: Path = Field(default=BASE_DIR / os.getenv("STORAGE_DIR", "data") / "bm25_corpus.pkl")

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
