from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from app.config import FINAL_K, TOP_K_KEYWORD, TOP_K_SEMANTIC
from app.embeddings import Embedder
from app.storage import connect


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    score: float


def fts_query(text: str) -> str:
    """Convert plain text into a safe FTS5 OR query."""
    terms = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


class HybridRetriever:
    def __init__(self, db_path: Path, index_path: Path, embedder: Embedder) -> None:
        if not db_path.is_file() or not index_path.is_file():
            raise FileNotFoundError("Knowledge-base files are missing; build the index first.")
        self.db_path = db_path
        self.embedder = embedder
        self.index = faiss.read_index(str(index_path))
        with connect(db_path) as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        if count == 0 or self.index.ntotal != count:
            raise ValueError("FAISS and SQLite indexes do not match; rebuild the knowledge base.")
        if metadata.get("embedding_model") != embedder.model_name:
            raise ValueError("Configured embedding model differs from the index; rebuild the knowledge base.")
        if self.index.d != embedder.dimension:
            raise ValueError("Embedding dimension differs from the index; rebuild the knowledge base.")
        self.chunk_count = count

    def search(self, question: str, final_k: int = FINAL_K) -> list[SearchHit]:
        candidates: dict[int, dict[str, Any]] = {}
        vector = self.embedder.query(question)
        semantic_k = min(max(1, TOP_K_SEMANTIC), self.chunk_count)
        scores, ids = self.index.search(vector, semantic_k)
        for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vector_id >= 0:
                candidates[int(vector_id) + 1] = {"semantic_rank": rank, "semantic_score": float(score)}

        expression = fts_query(question)
        if expression:
            with connect(self.db_path) as connection:
                rows = connection.execute(
                    """SELECT c.id, c.chunk_id, d.filename, c.page_start, c.page_end, c.text,
                              bm25(chunks_fts) AS rank
                       FROM chunks_fts
                       JOIN chunks c ON c.id = chunks_fts.rowid
                       JOIN documents d ON d.doc_id = c.doc_id
                       WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                    (expression, TOP_K_KEYWORD),
                ).fetchall()
            for rank, row in enumerate(rows, start=1):
                item = candidates.setdefault(int(row["id"]), {})
                item["keyword_rank"] = rank
                item["row"] = row

        if not candidates:
            return []
        missing_ids = [chunk_id for chunk_id, item in candidates.items() if "row" not in item]
        if missing_ids:
            placeholders = ",".join("?" for _ in missing_ids)
            with connect(self.db_path) as connection:
                rows = connection.execute(
                    f"""SELECT c.id, c.chunk_id, d.filename, c.page_start, c.page_end, c.text
                        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                        WHERE c.id IN ({placeholders})""",
                    missing_ids,
                ).fetchall()
            for row in rows:
                candidates[int(row["id"])]["row"] = row

        hits: list[SearchHit] = []
        for item in candidates.values():
            row = item.get("row")
            if row is None:
                continue
            score = 0.0
            if "semantic_rank" in item:
                score += 0.7 / (60 + item["semantic_rank"])
            if "keyword_rank" in item:
                score += 0.3 / (60 + item["keyword_rank"])
            hits.append(SearchHit(
                chunk_id=str(row["chunk_id"]), filename=str(row["filename"]),
                page_start=int(row["page_start"]), page_end=int(row["page_end"]),
                text=str(row["text"]), score=score,
            ))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:max(1, min(final_k, 12))]
