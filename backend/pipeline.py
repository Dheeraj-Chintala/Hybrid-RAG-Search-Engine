import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.ingestion.pdf_extractor import PDFExtractor
from backend.ingestion.text_splitter import TextSplitter, Chunk
from backend.embeddings.embedder import EmbeddingModel
from backend.storage.faiss_index import FaissIndex
from backend.storage.metadata_store import MetadataStore
from backend.storage.bm25_store import BM25Store
from backend.retrieval.vector_search import VectorSearchEngine
from backend.retrieval.keyword_search import KeywordSearchEngine
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.reranking.cross_encoder import CrossEncoderReranker
from backend.generation.generator import GeminiGenerator, GenerationResult

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Central Coordinator for the Hybrid RAG Search Engine.
    Handles document ingestion, dual-index persistence, hybrid retrieval,
    Cross-Encoder reranking, and Gemini answer generation with citations.
    """

    def __init__(
        self,
        config=settings,
        embedder: Optional[EmbeddingModel] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.config = config

        # Ingestion
        self.pdf_extractor = PDFExtractor()
        self.text_splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Embeddings & Reranking (lazy loaded on demand or injected)
        self.embedder = embedder or EmbeddingModel(model_name=self.config.embedding_model)
        self.reranker = reranker or CrossEncoderReranker(model_name=self.config.reranker_model)

        # Persistent Storage
        self.metadata_store = MetadataStore(self.config.metadata_db_path)
        self.faiss_index = FaissIndex(
            dimension=self.embedder.dimension,
            index_path=self.config.faiss_index_path
        )
        self.bm25_store = BM25Store(persistence_path=self.config.bm25_corpus_path)

        # Retrieval engines
        self.vector_engine = VectorSearchEngine(
            embedder=self.embedder,
            faiss_index=self.faiss_index,
            metadata_store=self.metadata_store
        )
        self.keyword_engine = KeywordSearchEngine(
            bm25_store=self.bm25_store,
            metadata_store=self.metadata_store
        )
        self.hybrid_engine = HybridSearchEngine(
            vector_engine=self.vector_engine,
            keyword_engine=self.keyword_engine,
            rrf_k=self.config.rrf_k
        )

    def _compute_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def ingest_pdf_file(self, file_path: Path, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """Ingests a PDF file from disk, processes pages, chunks, embeds, and persists."""
        path = Path(file_path)
        with open(path, "rb") as f:
            content = f.read()
        doc_name = original_filename or path.name
        return self.ingest_pdf_bytes(content, doc_name)

    def ingest_pdf_bytes(self, content: bytes, doc_name: str) -> Dict[str, Any]:
        """Ingests raw PDF bytes with complete document tracking and index updates."""
        start_time = time.time()
        file_hash = self._compute_hash(content)

        logger.info(f"Starting ingestion for '{doc_name}' ({len(content)} bytes)")

        # 1. Extract text page-by-page
        pages = self.pdf_extractor.extract_from_bytes(content, doc_name)
        if not pages:
            raise ValueError(f"No readable text pages could be extracted from '{doc_name}'.")

        # 2. Split into chunks preserving document name and page numbers
        chunks = self.text_splitter.split_pages(pages)
        if not chunks:
            raise ValueError(f"No chunks generated from '{doc_name}'.")

        # 3. Compute vector embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # 4. Determine next sequential FAISS ID
        current_max_id = self.metadata_store.get_max_faiss_id()
        start_faiss_id = current_max_id + 1

        # 5. Save to Metadata SQLite Store
        self.metadata_store.add_document(
            doc_name=doc_name,
            file_hash=file_hash,
            page_count=len(pages),
            chunk_count=len(chunks)
        )
        assigned_ids = self.metadata_store.add_chunks(chunks, start_faiss_id=start_faiss_id)

        # 6. Add to FAISS and persist to disk
        self.faiss_index.add(embeddings, assigned_ids)
        self.faiss_index.save()

        # 7. Add to BM25 and persist to disk
        self.bm25_store.add_chunks(chunks)
        self.bm25_store.save()

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Ingested '{doc_name}' successfully: {len(pages)} pages, {len(chunks)} chunks in {elapsed}s.")

        return {
            "doc_name": doc_name,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "duration_seconds": elapsed,
            "status": "success"
        }

    def search_passages(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid Retrieval (Vector + BM25 + RRF) followed by Cross-Encoder Reranking."""
        k_retrieve = top_k_retrieval or self.config.top_k_retrieval
        k_rerank = top_k_rerank or self.config.top_k_rerank

        # Stage 1: Hybrid Search (Vector + BM25 via RRF)
        candidates = self.hybrid_engine.search(query, top_k=k_retrieve)
        if not candidates:
            return []

        # Stage 2: Cross-Encoder Reranker
        reranked = self.reranker.rerank(query, candidates, top_k=k_rerank)
        return reranked

    def answer_query(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end question answering:
        1. Hybrid Search + RRF
        2. Cross-Encoder Rerank
        3. Grounded Generation with Citations (OpenRouter or Gemini)
        """
        reranked_passages = self.search_passages(query, top_k_retrieval, top_k_rerank)

        generator = GeminiGenerator(api_key=api_key, model_name=model_name, provider=provider)
        gen_result = generator.generate(query, reranked_passages)

        # Format retrieved passages for API response
        formatted_passages = []
        for p in reranked_passages:
            chunk = p["chunk"]
            formatted_passages.append({
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "rrf_score": round(p.get("rrf_score", 0.0), 5),
                "vector_rank": p.get("vector_rank"),
                "vector_score": round(p.get("vector_score", 0.0), 4) if p.get("vector_score") is not None else None,
                "bm25_rank": p.get("bm25_rank"),
                "bm25_score": round(p.get("bm25_score", 0.0), 4) if p.get("bm25_score") is not None else None,
                "rerank_score": p.get("rerank_score")
            })

        return {
            "query": query,
            "answer": gen_result.answer,
            "citations": [c.model_dump() for c in gen_result.citations],
            "retrieved_passages": formatted_passages,
            "has_sufficient_context": gen_result.has_sufficient_context,
            "model_used": gen_result.model_used,
            "error": gen_result.error
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns indexed documents with metadata."""
        return self.metadata_store.list_documents()

    def delete_document(self, doc_name: str) -> None:
        """Deletes a document from metadata and re-synchronizes the FAISS and BM25 indexes."""
        logger.info(f"Deleting document '{doc_name}'...")
        self.metadata_store.delete_document(doc_name)

        # Re-build indexes from remaining chunks to guarantee complete index integrity
        remaining_chunks = self.metadata_store.get_all_chunks()
        self.faiss_index.reset()
        self.bm25_store.reset()

        if remaining_chunks:
            chunk_texts = [c.text for c in remaining_chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)
            faiss_ids = [c.metadata.get("faiss_id", idx) for idx, c in enumerate(remaining_chunks)]
            self.faiss_index.add(embeddings, faiss_ids)
            self.faiss_index.save()

            self.bm25_store.add_chunks(remaining_chunks)
            self.bm25_store.save()

        logger.info(f"Document '{doc_name}' removed. Index synchronized with {len(remaining_chunks)} chunks.")

    def reset_all(self) -> None:
        """Completely purges all indexed documents, vectors, BM25 corpus, and metadata."""
        self.metadata_store.clear_all()
        self.faiss_index.reset()
        self.bm25_store.reset()
        logger.info("All documents and indexes have been reset.")
