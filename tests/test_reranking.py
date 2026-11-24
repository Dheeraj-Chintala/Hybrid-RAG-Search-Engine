import pytest
from backend.ingestion.text_splitter import Chunk
from backend.reranking.cross_encoder import CrossEncoderReranker

def test_cross_encoder_reranking():
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    candidates = [
        {
            "chunk": Chunk(
                chunk_id="c1",
                doc_name="physics.pdf",
                page_number=1,
                chunk_index=0,
                text="The speed of light in vacuum is approximately 299,792 kilometers per second.",
                token_count=15
            ),
            "rrf_score": 0.03
        },
        {
            "chunk": Chunk(
                chunk_id="c2",
                doc_name="recipes.pdf",
                page_number=4,
                chunk_index=3,
                text="To make delicious pancakes, mix flour, eggs, milk, and melted butter.",
                token_count=14
            ),
            "rrf_score": 0.02
        }
    ]
    
    query = "What is the physical velocity and speed of light?"
    reranked = reranker.rerank(query, candidates, top_k=2)
    
    assert len(reranked) == 2
    # Physics chunk must have higher rerank score for speed of light query
    assert reranked[0]["chunk"].chunk_id == "c1"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
    assert "rerank_score" in reranked[0]
