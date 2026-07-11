"""SentenceTransformer wrapper driven by ModelConfig"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from fever_search.config import ModelConfig


class Encoder:
    def __init__(self, config: ModelConfig, model_path: str | None = None) -> None:
        self.config = config
        if config.device and config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"model.device={config.device!r} is set in the config, but torch.cuda.is_available() "
                "is False here (no GPU visible, or a CPU-only torch build). Fix the CUDA setup, or set "
                "device to null/'cpu' in the config to run on CPU on purpose."
            )
        self.model = SentenceTransformer(model_path or config.name, device=config.device)
        if config.fp16 and config.device and config.device.startswith("cuda"):
            self.model = self.model.half()
        dtype = next(self.model.parameters()).dtype
        print(f"[Encoder] {model_path or config.name} -> device: {self.model.device}, dtype: {dtype}")

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
            kwargs["prompt"] = prompt   # literal instruction text prepended to each input
        vectors = self.model.encode(texts, **kwargs)
        return np.asarray(vectors, dtype=np.float32)
    