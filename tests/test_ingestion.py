import pytest
import pymupdf
from backend.ingestion.pdf_extractor import PDFExtractor, ExtractedPage
from backend.ingestion.text_splitter import TextSplitter

def create_sample_pdf(tmp_path) -> str:
    """Creates a sample multi-page PDF for testing."""
    pdf_path = tmp_path / "test_sample.pdf"
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Hybrid RAG Architecture.\n"
        "This is page one discussing Reciprocal Rank Fusion (RRF) and Vector embeddings.\n"
        "FAISS provides efficient dense vector search for high dimensional similarity."
    )
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Keyword Search and BM25.\n"
        "This is page two explaining BM25 token frequencies and exact keyword matching.\n"
        "Cross-Encoder reranking evaluates query and document pairs jointly to improve NDCG."
    )
    
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)

def test_pdf_extraction(tmp_path):
    pdf_path = create_sample_pdf(tmp_path)
    extractor = PDFExtractor()
    pages = extractor.extract_from_path(pdf_path)
    
    assert len(pages) == 2
    assert pages[0].doc_name == "test_sample.pdf"
    assert pages[0].page_number == 1
    assert "Hybrid RAG Architecture" in pages[0].text
    
    assert pages[1].doc_name == "test_sample.pdf"
    assert pages[1].page_number == 2
    assert "Keyword Search and BM25" in pages[1].text

def test_text_splitter_and_citations():
    pages = [
        ExtractedPage(
            doc_name="paper.pdf",
            page_number=1,
            text="First sentence about attention mechanisms. Second sentence about transformer layers. Third sentence about feedforward networks.",
            char_count=130
        ),
        ExtractedPage(
            doc_name="paper.pdf",
            page_number=2,
            text="Second page content covering cross entropy loss and optimization algorithms.",
            char_count=78
        )
    ]
    
    splitter = TextSplitter(chunk_size=70, chunk_overlap=20)
    chunks = splitter.split_pages(pages)
    
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.doc_name == "paper.pdf"
        assert chunk.page_number in [1, 2]
        assert chunk.citation.startswith("[paper.pdf, Page ")
        assert len(chunk.text) > 0
