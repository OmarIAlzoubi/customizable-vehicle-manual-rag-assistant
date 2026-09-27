import numpy as np
import fitz

from scripts import build_index
from scripts.build_index import chunk_pages
from app import grok
from app.grok import make_user_input
from app.retrieval import HybridRetriever, SearchHit, fts_query


def test_chunk_pages_tracks_page_ranges_and_overlap() -> None:
    pages = [" ".join(f"p1w{i}" for i in range(1, 46)), " ".join(f"p2w{i}" for i in range(1, 46))]
    chunks = chunk_pages(pages, target_words=60, overlap_words=10, doc_id="doc")
    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[1].page_start == 2
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_fts_query_quotes_terms_and_ignores_operators() -> None:
    result = fts_query('NGS OR "tire-pressure"')
    assert '"NGS"' in result
    assert '"OR"' in result
    assert '"tire"' in result
    assert '"pressure"' in result


def test_grok_context_has_only_assigned_source_labels() -> None:
    hit = SearchHit("c1", "Manual.pdf", 12, 13, "Set tire pressure when cold.", 0.01)
    text = make_user_input("How do I check tire pressure?", [hit])
    assert "[S1] Manual.pdf, page 12-13" in text
    assert "Set tire pressure when cold." in text
    assert "How do I check tire pressure?" in text


def test_build_and_hybrid_search_with_a_temporary_pdf(tmp_path, monkeypatch) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "sample-manual.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Tire pressure warning reset procedure. Check tire pressure when tires are cold.")
    pdf.save(pdf_path)
    pdf.close()

    class FakeEmbedder:
        def __init__(self, model_name):
            self.model_name = model_name
            self.dimension = 3

        def passages(self, texts, batch_size=32):
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

        def query(self, text):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(build_index, "Embedder", FakeEmbedder)
    db_path = tmp_path / "generated" / "kb.db"
    faiss_path = tmp_path / "generated" / "faiss.index"
    build_index.build(pdf_dir, db_path, faiss_path, "test-embedder")

    retriever = HybridRetriever(db_path, faiss_path, FakeEmbedder("test-embedder"))
    results = retriever.search("How do I reset the tire pressure warning?")
    assert results
    assert results[0].filename == "sample-manual.pdf"
    assert results[0].page_start == 1
    assert "Tire pressure" in results[0].text


def test_grok_request_uses_configured_model_and_disables_response_storage(monkeypatch) -> None:
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_text": "Set the pressure when the tires are cold. [S1]"})()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(grok, "OpenAI", FakeClient)
    hit = SearchHit("c1", "Manual.pdf", 12, 12, "Check pressure when cold.", 0.01)
    answer = grok.answer_with_grok("How do I check pressure?", [hit])

    assert answer.endswith("[S1]")
    assert captured["model"] == "grok-4.7"
    assert captured["store"] is False
    assert "2023 Hyundai Elantra N" in captured["input"][0]["content"]
    assert captured["client_kwargs"]["base_url"] == "https://api.x.ai/v1"
