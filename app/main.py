from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import DB_PATH, EMBEDDING_MODEL, FAISS_PATH, GROK_MODEL, MAX_QUESTION_CHARS, VEHICLE_NAME
from app.embeddings import Embedder
from app.retrieval import HybridRetriever
from app.service import AssistantService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("elantra_rag")
STATE: dict[str, object] = {"assistant": None, "startup_error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        if not DB_PATH.is_file() or not FAISS_PATH.is_file():
            raise FileNotFoundError("Knowledge base files are missing. Add PDFs to data/pdfs and run scripts/build_index.py.")
        embedder = Embedder(EMBEDDING_MODEL)
        retriever = HybridRetriever(DB_PATH, FAISS_PATH, embedder)
        STATE["assistant"] = AssistantService(retriever)
        logger.info("Knowledge base ready: %s chunks", retriever.chunk_count)
    except Exception as exc:
        STATE["startup_error"] = str(exc)
        logger.warning("Knowledge base is not ready: %s", exc)
    yield
    STATE.clear()


app = FastAPI(title=f"{VEHICLE_NAME} Manual Assistant", version="2.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    timings_ms: dict[str, float]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    assistant = STATE.get("assistant")
    retriever = getattr(assistant, "retriever", None)
    return {
        "ok": assistant is not None,
        "knowledge_base_ready": retriever is not None,
        "indexed_chunks": getattr(retriever, "chunk_count", 0),
        "embedding_model": EMBEDDING_MODEL,
        "grok_model": GROK_MODEL,
        "vehicle_name": VEHICLE_NAME,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    assistant = STATE.get("assistant")
    if assistant is None:
        detail = str(STATE.get("startup_error") or "Knowledge base is not ready.")
        raise HTTPException(status_code=503, detail=detail)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    started = time.perf_counter()
    try:
        result = assistant.ask(question)
    except RuntimeError as exc:
        logger.warning("Assistant configuration or generation error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Request failed")
        raise HTTPException(status_code=502, detail="The assistant could not complete this request. Check the server log and try again.") from exc
    return AskResponse(**result, timings_ms={"total": round((time.perf_counter() - started) * 1000, 1)})
