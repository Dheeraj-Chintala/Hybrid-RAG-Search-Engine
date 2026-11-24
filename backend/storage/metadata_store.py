import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from backend.ingestion.text_splitter import Chunk

logger = logging.getLogger(__name__)

class MetadataStore:
    """
    SQLite-backed store for persisting indexed document records and chunk metadata.
    Enables instant reload upon app restarts without re-embedding.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables for documents and chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    faiss_id INTEGER UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_name) REFERENCES documents (doc_name) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks (doc_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks (faiss_id)")
            conn.commit()

    def add_document(self, doc_name: str, file_hash: str, page_count: int, chunk_count: int) -> int:
        """Adds or updates a document entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (doc_name, file_hash, page_count, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_name) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    created_at = CURRENT_TIMESTAMP
            """, (doc_name, file_hash, page_count, chunk_count))
            conn.commit()
            return cursor.lastrowid

    def add_chunks(self, chunks: List[Chunk], start_faiss_id: int) -> List[int]:
        """
        Inserts chunks into SQLite, associating each with a sequential integer faiss_id.
        Returns the assigned faiss_ids.
        """
        faiss_ids = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                assigned_id = start_faiss_id + i
                faiss_ids.append(assigned_id)
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks (id, doc_name, page_number, chunk_index, text, token_count, faiss_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.doc_name,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.token_count,
                    assigned_id
                ))
            conn.commit()
        return faiss_ids

    def get_chunk_by_faiss_id(self, faiss_id: int) -> Optional[Chunk]:
        """Fetches a Chunk by its assigned FAISS vector integer ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE faiss_id = ?", (faiss_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Chunk(
                chunk_id=row["id"],
                doc_name=row["doc_name"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                token_count=row["token_count"],
                metadata={"faiss_id": row["faiss_id"]}
            )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Fetches a Chunk by its string chunk_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Chunk(
                chunk_id=row["id"],
                doc_name=row["doc_name"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                token_count=row["token_count"],
                metadata={"faiss_id": row["faiss_id"]}
            )

    def get_all_chunks(self) -> List[Chunk]:
        """Returns all chunks ordered by faiss_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks ORDER BY faiss_id ASC")
            rows = cursor.fetchall()
            return [
                Chunk(
                    chunk_id=r["id"],
                    doc_name=r["doc_name"],
                    page_number=r["page_number"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    token_count=r["token_count"],
                    metadata={"faiss_id": r["faiss_id"]}
                )
                for r in rows
            ]

    def get_max_faiss_id(self) -> int:
        """Returns the highest faiss_id currently stored, or -1 if empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(faiss_id) as max_id FROM chunks")
            row = cursor.fetchone()
            if row and row["max_id"] is not None:
                return int(row["max_id"])
            return -1

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of all indexed documents with statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(r) for r in cursor.fetchall()]

    def delete_document(self, doc_name: str) -> None:
        """Removes a document and its chunks from metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
            cursor.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))
            conn.commit()

    def clear_all(self) -> None:
        """Clears all stored documents and chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            conn.commit()
