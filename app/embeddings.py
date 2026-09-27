from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Multilingual E5 encoder using query/passage prefixes."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def passages(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts], batch_size)

    def query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"], 1)

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
