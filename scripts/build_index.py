#!/usr/bin/env python3
"""Build a local SQLite FTS5 + FAISS index from user-provided PDFs."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, EMBEDDING_MODEL  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.storage import initialize_database  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_index")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[str], target_words: int, overlap_words: int, doc_id: str) -> list[Chunk]:
    if target_words < 50:
        raise ValueError("target_words must be at least 50")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be >= 0 and smaller than target_words")

    words: list[tuple[str, int]] = []
    for page_number, page_text in enumerate(pages, start=1):
        words.extend((word, page_number) for word in re.findall(r"\S+", page_text))
    if not words:
        return []

    step = target_words - overlap_words
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        part = words[start:start + target_words]
        if not part:
            break
        text = " ".join(word for word, _ in part).strip()
        if not text:
            continue
        index = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}:{index}",
            doc_id=doc_id,
            page_start=part[0][1],
            page_end=part[-1][1],
            text=text,
        ))
        if start + target_words >= len(words):
            break
    return chunks


def read_pdf(path: Path) -> tuple[str, str, int, list[str]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = digest[:24]
    with fitz.open(path) as pdf:
        pages = [clean_text(page.get_text("text")) for page in pdf]
        page_count = pdf.page_count
    return doc_id, digest, page_count, pages


def build(input_dir: Path, db_path: Path, index_path: Path, model_name: str) -> int:
    pdf_paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {input_dir}. Add documents you are allowed to use.")

    documents: list[tuple[str, str, str, int]] = []
    chunks: list[Chunk] = []
    seen_digests: set[str] = set()
    for pdf_path in pdf_paths:
        doc_id, digest, page_count, pages = read_pdf(pdf_path)
        if digest in seen_digests:
            logger.info("Skipping duplicate PDF: %s", pdf_path.name)
            continue
        seen_digests.add(digest)
        doc_chunks = chunk_pages(pages, CHUNK_WORDS, CHUNK_OVERLAP_WORDS, doc_id)
        if not doc_chunks:
            logger.warning("No extractable text in %s; skipping", pdf_path.name)
            continue
        documents.append((doc_id, pdf_path.name, digest, page_count))
        chunks.extend(doc_chunks)
        logger.info("Read %s (%d pages, %d chunks)", pdf_path.name, page_count, len(doc_chunks))
    if not chunks:
        raise RuntimeError("No text chunks were extracted. The PDFs may be scanned images requiring OCR.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_db = db_path.with_name(db_path.stem + ".building" + db_path.suffix)
    temp_index = index_path.with_name(index_path.stem + ".building" + index_path.suffix)
    for temp_path in (temp_db, temp_index):
        if temp_path.exists():
            temp_path.unlink()

    connection = initialize_database(temp_db)
    try:
        connection.executemany(
            "INSERT INTO documents(doc_id, filename, sha256, page_count) VALUES (?, ?, ?, ?)",
            documents,
        )
        connection.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, page_start, page_end, text) VALUES (?, ?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.page_start, c.page_end, c.text) for c in chunks],
        )
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            [("embedding_model", model_name), ("chunk_count", str(len(chunks)))],
        )
        connection.commit()
    finally:
        connection.close()

    embedder = Embedder(model_name)
    index = faiss.IndexFlatIP(embedder.dimension)
    batch_size = 32
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        vectors = embedder.passages([chunk.text for chunk in batch], batch_size=batch_size)
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite embedding values in batch starting at {offset}")
        index.add(vectors)
        logger.info("Embedded %d/%d chunks", min(offset + len(batch), len(chunks)), len(chunks))
    if index.ntotal != len(chunks):
        raise RuntimeError("FAISS vector count did not match the stored chunk count")
    faiss.write_index(index, str(temp_index))
    # Keep the current pair intact until both replacement artifacts are ready.
    os.replace(temp_index, index_path)
    os.replace(temp_db, db_path)
    logger.info("Built %d vectors. SQLite: %s | FAISS: %s", index.ntotal, db_path, index_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "pdfs")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "generated" / "kb.db")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "generated" / "faiss.index")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    args = parser.parse_args()
    build(args.input_dir.resolve(), args.db.resolve(), args.index.resolve(), args.embedding_model)


if __name__ == "__main__":
    main()
