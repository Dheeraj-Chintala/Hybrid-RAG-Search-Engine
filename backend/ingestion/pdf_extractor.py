import io
import logging
import re
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ExtractedPage(BaseModel):
    """Represents text extracted from a single page of a document."""
    doc_name: str = Field(description="Name or filename of the document")
    page_number: int = Field(description="1-based page number")
    text: str = Field(description="Extracted clean text content")
    char_count: int = Field(default=0, description="Character count of extracted text")


class PDFExtractor:
    """Extracts text from PDF files using PyMuPDF (fitz) preserving document names and page numbers."""

    def __init__(self):
        try:
            import pymupdf
            self._fitz = pymupdf
        except ImportError:
            import fitz
            self._fitz = fitz

    def clean_text(self, text: str) -> str:
        """Cleans extracted text: normalizes whitespace, fixes linebreaks, removes null bytes."""
        if not text:
            return ""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize whitespace (preserve paragraphs)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_from_path(self, file_path: Union[str, Path]) -> List[ExtractedPage]:
        """Extracts text page-by-page from a PDF file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        doc_name = path.name
        logger.info(f"Extracting PDF pages from: {doc_name}")

        try:
            doc = self._fitz.open(str(path))
            return self._process_doc(doc, doc_name)
        except Exception as e:
            logger.error(f"Failed to extract text from {doc_name}: {e}")
            raise

    def extract_from_bytes(self, content: bytes, doc_name: str) -> List[ExtractedPage]:
        """Extracts text page-by-page from raw PDF bytes."""
        logger.info(f"Extracting PDF pages from memory: {doc_name}")
        try:
            stream = io.BytesIO(content)
            doc = self._fitz.open(stream=stream, filetype="pdf")
            return self._process_doc(doc, doc_name)
        except Exception as e:
            logger.error(f"Failed to extract text from memory bytes of {doc_name}: {e}")
            raise

    def _process_doc(self, doc, doc_name: str) -> List[ExtractedPage]:
        pages: List[ExtractedPage] = []
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                raw_text = page.get_text("text")
                cleaned = self.clean_text(raw_text)
                
                # We record pages even if text is minimal, but store character count
                # 1-based page numbering for natural user citation: Page 1, Page 2...
                page_num = page_idx + 1
                if cleaned:
                    pages.append(
                        ExtractedPage(
                            doc_name=doc_name,
                            page_number=page_num,
                            text=cleaned,
                            char_count=len(cleaned)
                        )
                    )
                else:
                    logger.warning(f"Page {page_num} in '{doc_name}' is empty or contains non-extractable raster text.")
        finally:
            doc.close()

        logger.info(f"Extracted {len(pages)} non-empty pages from '{doc_name}'")
        return pages
