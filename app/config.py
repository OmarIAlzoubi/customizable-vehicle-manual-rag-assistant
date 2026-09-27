from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _path_setting(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


PDF_DIR = _path_setting("PDF_DIR", "data/pdfs")
DB_PATH = _path_setting("KB_DB_PATH", "data/generated/kb.db")
FAISS_PATH = _path_setting("KB_FAISS_PATH", "data/generated/faiss.index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
VEHICLE_NAME = os.getenv("VEHICLE_NAME", "2023 Hyundai Elantra N").strip() or "Your vehicle"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.7")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_TIMEOUT_SECONDS = float(os.getenv("XAI_TIMEOUT_SECONDS", "120"))

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "380"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "6"))
TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "6"))
FINAL_K = int(os.getenv("FINAL_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
