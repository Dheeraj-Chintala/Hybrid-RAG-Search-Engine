from typing import List, Dict, Any

STRICT_SYSTEM_INSTRUCTION = """You are a precise, honest, and strictly grounded AI research assistant.
Your task is to answer the user's question using ONLY the provided context passages below.

CRITICAL GROUNDING RULES:
1. ONLY state facts that are directly mentioned in the provided Context Passages. Do NOT use outside knowledge, speculate, extrapolate, or hallucinate.
2. If the provided context does not contain enough information to fully or partially answer the question, you MUST explicitly state:
   "I do not have sufficient information in the provided documents to answer this question."
   Do NOT attempt to guess or invent details.
3. IN-TEXT CITATIONS: Whenever you state a fact or reference information from a passage, immediately append an in-text bracketed citation referencing the document and page number, for example: [document_name.pdf, Page X].
4. Be concise, coherent, and professional in your response.
5. After your answer, list the unique source citations used in your answer under a 'Sources' header."""

def format_context_passages(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved & reranked chunks into clean, numbered passage blocks with source headers."""
    if not passages:
        return "No relevant context passages found in indexed documents."

    formatted_blocks = []
    for i, item in enumerate(passages, start=1):
        chunk = item["chunk"]
        doc = chunk.doc_name
        page = chunk.page_number
        text = chunk.text.strip()
        rerank_score = item.get("rerank_score", "N/A")
        
        block = f"[Passage {i}]\nSource: [{doc}, Page {page}] (Relevance Score: {rerank_score})\nContent:\n{text}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)

def build_grounded_rag_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    """Builds the full prompt combining instructions, formatted context passages, and the user query."""
    context_str = format_context_passages(passages)
    
    prompt = f"""{STRICT_SYSTEM_INSTRUCTION}

==================== CONTEXT PASSAGES ====================
{context_str}
==========================================================

USER QUESTION:
{query}

GROUNDED ANSWER (with [Document, Page] citations):"""
    return prompt
