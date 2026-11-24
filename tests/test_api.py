import io
import pytest
from fastapi.testclient import TestClient
import pymupdf
from backend.main import app

def create_in_memory_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_api_health():
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "embedding_model" in data
        assert "reranker_model" in data

def test_api_upload_search_and_query():
    with TestClient(app) as client:
        # 1. Reset first to have clean slate
        client.post("/api/reset")
        
        # 2. Upload sample PDF
        pdf_bytes = create_in_memory_pdf(
            "Reciprocal Rank Fusion (RRF) is a method to combine rankings from multiple information retrieval systems. "
            "It computes the sum of reciprocal ranks for each document."
        )
        
        files = [
            ("files", ("test_rrf_paper.pdf", pdf_bytes, "application/pdf"))
        ]
        upload_resp = client.post("/api/documents/upload", files=files)
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert len(upload_data) == 1
        assert upload_data[0]["filename"] == "test_rrf_paper.pdf"
        assert upload_data[0]["status"] == "success"
        
        # 3. List documents
        docs_resp = client.get("/api/documents")
        assert docs_resp.status_code == 200
        docs = docs_resp.json()
        assert len(docs) >= 1
        assert any(d["doc_name"] == "test_rrf_paper.pdf" for d in docs)
        
        # 4. Search passages (retrieval + rerank without LLM generation)
        search_resp = client.post("/api/search", json={
            "query": "reciprocal rank fusion rankings",
            "top_k_retrieval": 5,
            "top_k_rerank": 3
        })
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert len(search_data["passages"]) > 0
        top_passage = search_data["passages"][0]
        assert top_passage["doc_name"] == "test_rrf_paper.pdf"
        assert top_passage["page_number"] == 1
        assert "rrf_score" in top_passage
        assert "rerank_score" in top_passage
