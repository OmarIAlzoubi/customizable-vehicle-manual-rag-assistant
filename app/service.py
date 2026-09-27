from __future__ import annotations

from app.grok import answer_with_grok
from app.retrieval import HybridRetriever


class AssistantService:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def ask(self, question: str) -> dict:
        hits = self.retriever.search(question)
        if not hits:
            return {
                "answer": "I couldn't find a relevant passage in the indexed documents. Try different wording or add the relevant document to your local knowledge base.",
                "sources": [],
            }
        answer = answer_with_grok(question, hits)
        sources = []
        for i, hit in enumerate(hits, start=1):
            sources.append({
                "label": f"S{i}",
                "filename": hit.filename,
                "page_start": hit.page_start,
                "page_end": hit.page_end,
                "excerpt": hit.text[:700],
                "score": round(hit.score, 5),
            })
        return {"answer": answer, "sources": sources}
