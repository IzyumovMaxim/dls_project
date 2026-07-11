"""SentenceTransformer wrapper driven by ModelConfig"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from fever_search.config import ModelConfig


class Encoder:
    def __init__(self, config: ModelConfig, model_path: str | None = None) -> None:
        self.config = config
        self.model = SentenceTransformer(model_path or config.name)

    def encode_documents(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, self.config.doc_prompt, show_progress_bar)

    def encode_queries(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, self.config.query_prompt, show_progress_bar)

    def _encode(self, texts: list[str], prompt: str | None, show_progress_bar: bool) -> np.ndarray:
        kwargs: dict = {
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize,
            "show_progress_bar": show_progress_bar,
            "convert_to_numpy": True,
        }
        if prompt:
            kwargs["prompt_name"] = prompt
        vectors = self.model.encode(texts, **kwargs)
        return np.asarray(vectors, dtype=np.float32)
