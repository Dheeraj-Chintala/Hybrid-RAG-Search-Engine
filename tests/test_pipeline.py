import pytest
import pymupdf
from backend.config import Settings
from backend.pipeline import RAGPipeline
from backend.generation.prompt_templates import build_grounded_rag_prompt
from backend.generation.generator import GeminiGenerator, GenerationResult

def create_synthetic_pdf(path, title: str, pages_content: list) -> str:
    doc = pymupdf.open()
    for text in pages_content:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()
    return str(path)

def test_full_pipeline_ingestion_and_search(tmp_path):
    # Setup isolated test settings
    test_settings = Settings(
        storage_dir=tmp_path / "data",
        uploads_dir=tmp_path / "data" / "uploads",
        faiss_index_path=tmp_path / "data" / "faiss.index",
        metadata_db_path=tmp_path / "data" / "metadata.db",
        bm25_corpus_path=tmp_path / "data" / "bm25.pkl",
        chunk_size=150,
        chunk_overlap=30,
        top_k_retrieval=4,
        top_k_rerank=2,
        rrf_k=60
    )
    test_settings.ensure_directories()
    
    pipeline = RAGPipeline(config=test_settings)
    
    # Create two PDFs
    pdf1 = create_synthetic_pdf(
        tmp_path / "attention.pdf",
        "Attention Paper",
        [
            "The Transformer architecture relies entirely on self-attention mechanisms without recurrent connections.",
            "Multi-head attention allows the model to jointly attend to information from different representation subspaces."
        ]
    )
    pdf2 = create_synthetic_pdf(
        tmp_path / "retrieval.pdf",
        "Retrieval Paper",
        [
            "Dense passage retrieval uses bi-encoder embeddings to index passages in a FAISS vector database.",
            "Reciprocal Rank Fusion merges rankings from BM25 and vector search without requiring score normalization."
        ]
    )
    
    # 1. Ingest both PDFs
    res1 = pipeline.ingest_pdf_file(pdf1, original_filename="attention.pdf")
    res2 = pipeline.ingest_pdf_file(pdf2, original_filename="retrieval.pdf")
    
    assert res1["status"] == "success"
    assert res1["page_count"] == 2
    assert res2["status"] == "success"
    assert res2["page_count"] == 2
    
    # Check document listing
    docs = pipeline.list_documents()
    assert len(docs) == 2
    doc_names = [d["doc_name"] for d in docs]
    assert "attention.pdf" in doc_names
    assert "retrieval.pdf" in doc_names
    
    # 2. Search for attention query
    passages = pipeline.search_passages("self-attention without recurrent connections", top_k_retrieval=4, top_k_rerank=2)
    assert len(passages) > 0
    top_hit = passages[0]
    assert top_hit["chunk"].doc_name == "attention.pdf"
    assert top_hit["chunk"].page_number == 1
    assert "rerank_score" in top_hit
    assert "rrf_score" in top_hit
    
    # 3. Test citation formatting
    prompt = build_grounded_rag_prompt("Explain self attention", passages)
    assert "[attention.pdf, Page 1]" in prompt
    assert "CRITICAL GROUNDING RULES" in prompt
    
    # 4. Test delete document and index re-synchronization
    pipeline.delete_document("attention.pdf")
    remaining_docs = pipeline.list_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["doc_name"] == "retrieval.pdf"
    
    # Search again - attention.pdf passages should no longer be returned
    new_passages = pipeline.search_passages("recurrent connections", top_k_retrieval=4, top_k_rerank=2)
    for p in new_passages:
        assert p["chunk"].doc_name != "attention.pdf"

def test_insufficient_context_guardrail():
    """Verify that when no relevant passages exist, the generator asserts insufficient information."""
    generator = GeminiGenerator(api_key="mock_key", model_name="gemini-3.8-flash")
    result = generator.generate(query="What is quantum gravity?", context_passages=[])
    
    assert result.has_sufficient_context is False
    assert "do not have sufficient information" in result.answer
