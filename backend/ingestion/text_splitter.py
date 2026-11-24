import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .pdf_extractor import ExtractedPage

logger = logging.getLogger(__name__)

class Chunk(BaseModel):
    """Represents a text chunk with complete source provenance and citation metadata."""
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_name: str = Field(description="Original document name")
    page_number: int = Field(description="1-based page number")
    chunk_index: int = Field(description="Sequential chunk index within document")
    text: str = Field(description="Text content of the chunk")
    token_count: int = Field(default=0, description="Approximate token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def citation(self) -> str:
        """Formatted source citation string."""
        return f"[{self.doc_name}, Page {self.page_number}]"


class TextSplitter:
    """
    Splits document pages into semantically cohesive chunks with configurable size and overlap.
    Preserves document name and page number on every chunk.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _approx_token_count(self, text: str) -> int:
        """Estimates token count (avg ~4 characters per token)."""
        return max(1, len(text) // 4)

    def _split_page_text(self, text: str) -> List[str]:
        """
        Splits page text into chunks using recursive separator fallback:
        Paragraphs (\\n\\n) -> Sentences (. ) -> Whitespace (' ') -> Characters.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Break text into paragraphs first
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph fits within chunk_size
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                # If current_chunk has content, push it
                if current_chunk:
                    chunks.append(current_chunk)
                    # Start next chunk with overlap from the tail of current_chunk
                    overlap_seed = current_chunk[-self.chunk_overlap:].strip()
                    # Find a clean boundary in overlap if possible
                    space_idx = overlap_seed.find(" ")
                    if space_idx != -1:
                        overlap_seed = overlap_seed[space_idx + 1:]
                    current_chunk = f"{overlap_seed}\n\n{para}".strip() if overlap_seed else para
                else:
                    # Paragraph itself is larger than chunk_size, split by sentences
                    sentences = para.split(". ")
                    for s in sentences:
                        s = s.strip()
                        if not s:
                            continue
                        if not s.endswith("."):
                            s = s + "."
                        if len(current_chunk) + len(s) + 1 <= self.chunk_size:
                            current_chunk = f"{current_chunk} {s}".strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                                overlap_seed = current_chunk[-self.chunk_overlap:].strip()
                                current_chunk = f"{overlap_seed} {s}".strip()
                            else:
                                # Single sentence is huge, hard split by words
                                words = s.split(" ")
                                for word in words:
                                    if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                                        current_chunk = f"{current_chunk} {word}".strip()
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                            overlap_seed = current_chunk[-self.chunk_overlap:].strip()
                                            current_chunk = f"{overlap_seed} {word}".strip()
                                        else:
                                            chunks.append(word[:self.chunk_size])
                                            current_chunk = word[self.chunk_size:]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_pages(self, pages: List[ExtractedPage]) -> List[Chunk]:
        """Splits extracted pages into chunks, assigning unique IDs and exact page citations."""
        all_chunks: List[Chunk] = []
        global_chunk_idx = 0

        for page in pages:
            raw_chunks = self._split_page_text(page.text)
            for raw_chunk in raw_chunks:
                clean_chunk_text = raw_chunk.strip()
                if not clean_chunk_text:
                    continue

                chunk_obj = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=clean_chunk_text,
                    token_count=self._approx_token_count(clean_chunk_text),
                    metadata={
                        "char_length": len(clean_chunk_text),
                        "doc_name": page.doc_name,
                        "page_number": page.page_number
                    }
                )
                all_chunks.append(chunk_obj)
                global_chunk_idx += 1

        logger.info(f"Split {len(pages)} pages into {len(all_chunks)} chunks.")
        return all_chunks
