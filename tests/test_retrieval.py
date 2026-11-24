import numpy as np
import pytest
from backend.ingestion.text_splitter import Chunk
from backend.storage.metadata_store import MetadataStore
from backend.storage.faiss_index import FaissIndex
from backend.storage.bm25_store import BM25Store
from backend.retrieval.rrf import ReciprocalRankFusion

def sample_chunks():
    return [
        Chunk(
            chunk_id="c1",
            doc_name="doc1.pdf",
            page_number=1,
            chunk_index=0,
            text="Attention is all you need for sequence to sequence models and machine translation.",
            token_count=15
        ),
        Chunk(
            chunk_id="c2",
            doc_name="doc1.pdf",
            page_number=2,
            chunk_index=1,
            text="BM25 is a probabilistic term-matching keyword retrieval model based on TF-IDF.",
            token_count=14
        ),
        Chunk(
            chunk_id="c3",
            doc_name="doc2.pdf",
            page_number=1,
            chunk_index=0,
            text="Cross-Encoders achieve higher precision by evaluating query and passage jointly.",
            token_count=13
        )
    ]

def test_metadata_store(tmp_path):
    db_path = tmp_path / "test_meta.db"
    store = MetadataStore(db_path)
    
    chunks = sample_chunks()
    store.add_document("doc1.pdf", "hash123", page_count=2, chunk_count=2)
    ids = store.add_chunks(chunks[:2], start_faiss_id=0)
    
    assert ids == [0, 1]
    
    fetched = store.get_chunk_by_faiss_id(0)
    assert fetched is not None
    assert fetched.chunk_id == "c1"
    assert fetched.doc_name == "doc1.pdf"
    assert fetched.page_number == 1
    
    all_chunks = store.get_all_chunks()
    assert len(all_chunks) == 2
    
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_name"] == "doc1.pdf"

def test_faiss_index_lifecycle(tmp_path):
    index_path = tmp_path / "test.faiss"
    dim = 4
    index = FaissIndex(dimension=dim, index_path=index_path)
    
    # 2 normalized vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    index.add(vecs, [100, 200])
    assert index.total_vectors == 2
    
    # Search for vector close to first vector
    q = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    scores, ids = index.search(q, top_k=2)
    
    assert ids[0] == 100
    assert scores[0] > 0.8
    
    # Save & reload
    index.save(index_path)
    reloaded = FaissIndex(dimension=dim, index_path=index_path)
    assert reloaded.total_vectors == 2

def test_bm25_store_and_persistence(tmp_path):
    bm25_path = tmp_path / "test_bm25.pkl"
    store = BM25Store(persistence_path=bm25_path)
    chunks = sample_chunks()
    store.add_chunks(chunks)
    
    results = store.search("BM25 TF-IDF keyword", top_k=2)
    assert len(results) > 0
    # Top match should be c2
    assert results[0][0] == "c2"
    
    # Save & reload
    store.save(bm25_path)
    reloaded = BM25Store(persistence_path=bm25_path)
    assert reloaded.total_documents == 3
    reloaded_results = reloaded.search("Cross-Encoder joint", top_k=1)
    assert reloaded_results[0][0] == "c3"

def test_reciprocal_rank_fusion():
    chunks = sample_chunks()
    c1, c2, c3 = chunks[0], chunks[1], chunks[2]
    
    # Vector results rank: c1 (rank 1), c2 (rank 2)
    vector_results = [(c1, 0.95), (c2, 0.80)]
    # BM25 results rank: c2 (rank 1), c3 (rank 2)
    bm25_results = [(c2, 8.5), (c3, 4.2)]
    
    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse(vector_results, bm25_results, top_k=3)
    
    assert len(fused) == 3
    # c2 was present in both (rank 2 in vector + rank 1 in BM25):
    # RRF(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.016129 + 0.016393 = 0.032522
    # RRF(c1) = 1/(60+1) = 1/61 ~ 0.016393
    # RRF(c3) = 1/(60+2) = 1/62 ~ 0.016129
    # Therefore c2 MUST be ranked 1st!
    assert fused[0]["chunk"].chunk_id == "c2"
    assert fused[0]["vector_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert fused[1]["chunk"].chunk_id == "c1"
    assert fused[2]["chunk"].chunk_id == "c3"
