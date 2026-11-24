import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import settings
from backend.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HybridRAG.Backend")

# Pipeline singleton
pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing Hybrid RAG Pipeline...")
    pipeline = RAGPipeline(config=settings)
    doc_count = len(pipeline.list_documents())
    logger.info(f"Pipeline ready. Currently holding {doc_count} indexed documents and {pipeline.faiss_index.total_vectors} vectors.")
    yield
    logger.info("Shutting down Hybrid RAG Backend.")

app = FastAPI(
    title="Hybrid RAG Search Engine API",
    description="Locally runnable Hybrid RAG with FAISS, BM25, Reciprocal Rank Fusion, Cross-Encoder Reranking, and Gemini Citations.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class HealthResponse(BaseModel):
    status: str
    documents_count: int
    total_chunks: int
    gemini_key_configured: bool
    embedding_model: str
    reranker_model: str


class SearchRequest(BaseModel):
    query: str = Field(description="Search question or keyword query")
    top_k_retrieval: Optional[int] = Field(default=None, ge=1, le=50)
    top_k_rerank: Optional[int] = Field(default=None, ge=1, le=20)


class QueryRequest(BaseModel):
    query: str = Field(description="User question to answer")
    top_k_retrieval: Optional[int] = Field(default=None, ge=1, le=50)
    top_k_rerank: Optional[int] = Field(default=None, ge=1, le=20)
    provider: Optional[str] = Field(default=None, description="LLM provider: 'openrouter' or 'gemini'")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")
    model_name: Optional[str] = Field(default=None, description="Optional model name override")
    # Backwards compatibility fields
    gemini_api_key: Optional[str] = Field(default=None, description="Legacy Gemini key field")
    gemini_model: Optional[str] = Field(default=None, description="Legacy Gemini model field")


class IngestResponse(BaseModel):
    filename: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns backend system status and index telemetry."""
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")

    docs = pipeline.list_documents()
    total_chunks = pipeline.faiss_index.total_vectors
    has_gemini_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))

    return HealthResponse(
        status="healthy",
        documents_count=len(docs),
        total_chunks=total_chunks,
        gemini_key_configured=has_gemini_key,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model
    )


@app.post("/api/documents/upload", response_model=List[IngestResponse], tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads and indexes one or more PDF documents."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results: List[IngestResponse] = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            results.append(IngestResponse(
                filename=filename,
                status="rejected",
                error="Only PDF documents are supported."
            ))
            continue

        try:
            content = await file.read()
            if not content:
                results.append(IngestResponse(
                    filename=filename,
                    status="error",
                    error="Empty file provided."
                ))
                continue

            # Ingest bytes
            ingest_stat = pipeline.ingest_pdf_bytes(content, filename)
            results.append(IngestResponse(
                filename=filename,
                status="success",
                page_count=ingest_stat["page_count"],
                chunk_count=ingest_stat["chunk_count"],
                duration_seconds=ingest_stat["duration_seconds"]
            ))
        except Exception as e:
            logger.error(f"Failed to ingest file '{filename}': {e}", exc_info=True)
            results.append(IngestResponse(
                filename=filename,
                status="error",
                error=str(e)
            ))

    return results


@app.get("/api/documents", tags=["Documents"])
async def get_documents():
    """Lists all currently indexed documents."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.list_documents()


@app.delete("/api/documents/{doc_name}", tags=["Documents"])
async def delete_document(doc_name: str):
    """Deletes an indexed document and re-syncs FAISS and BM25 indexes."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.delete_document(doc_name)
    return {"message": f"Document '{doc_name}' deleted successfully."}


@app.post("/api/search", tags=["Search"])
async def search_passages(req: SearchRequest):
    """
    Retrieves and reranks passages without generating an LLM response.
    Useful for inspecting hybrid search candidates, RRF scores, and Cross-Encoder ranks.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    passages = pipeline.search_passages(
        query=req.query,
        top_k_retrieval=req.top_k_retrieval,
        top_k_rerank=req.top_k_rerank
    )

    formatted = []
    for p in passages:
        chunk = p["chunk"]
        formatted.append({
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

    return {"query": req.query, "passages": formatted}


@app.post("/api/query", tags=["RAG"])
async def query_rag(req: QueryRequest):
    """
    End-to-End RAG Q&A:
    Performs Hybrid Search (FAISS + BM25 with RRF), Cross-Encoder Reranking,
    and Gemini Answer Generation with In-Text Citations and Grounding.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    key_to_use = req.api_key or req.gemini_api_key
    model_to_use = req.model_name or req.gemini_model

    result = pipeline.answer_query(
        query=req.query,
        top_k_retrieval=req.top_k_retrieval,
        top_k_rerank=req.top_k_rerank,
        api_key=key_to_use,
        model_name=model_to_use,
        provider=req.provider
    )
    return result


@app.post("/api/reset", tags=["System"])
async def reset_index():
    """Clears all indexed documents, FAISS index, and BM25 corpus."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.reset_all()
    return {"message": "All documents and indexes have been reset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
