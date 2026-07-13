"""Search engine: encode a query and retrieve top-k corpus documents from a FAISS index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fever_search import index, paths
from fever_search.config import ExperimentConfig
from fever_search.data_io import load_corpus
from fever_search.encoder import Encoder


@dataclass(frozen=True)
class SearchHit:
    rank: int
    doc_id: str
    score: float
    title: str
    text: str


class SearchEngine:
    def __init__(self, config: ExperimentConfig, model_path: str | None = None) -> None:
        self.config = config
        self._index, self._doc_ids, self.manifest = index.load(config.name, index_cfg=config.index)
        self._encoder = Encoder(config.model, model_path=model_path)
        self._corpus = load_corpus(paths.CORPUS_PATH)

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        vector = self._encoder.encode_queries([query])
        scores, indices = self._index.search(np.asarray(vector, dtype=np.float32), top_k)

        hits: list[SearchHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            doc_id = self._doc_ids[idx]
            doc = self._corpus.get(doc_id, {})
            hits.append(SearchHit(
                rank=rank,
                doc_id=doc_id,
                score=float(score),
                title=str(doc.get("title") or ""),
                text=str(doc.get("text") or ""),
            ))
        return hits


def format_hit(hit: SearchHit, text_preview: int = 280) -> str:
    text = hit.text.strip().replace("\n", " ")
    if len(text) > text_preview:
        text = text[: text_preview - 3] + "..."
    header = f"{hit.rank}. score={hit.score:.4f}  id={hit.doc_id}"
    if hit.title:
        header += f"  title={hit.title}"
    return f"{header}\n   {text}"
