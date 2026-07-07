"""
Vector search API over fever_500k (bge-small-en-v1.5 + FAISS Flat).

Usage:
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path

    spec = spec_from_file_location(
        "query_search_bge",
        Path("scripts/query_search_bge_small_en_v1.5.py"),
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    hits = mod.vector_search("Einstein developed E=mc2", top_k=10)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSONL_PATH = PROJECT_ROOT / "data" / "corpus" / "fever_500k.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "bge-small-en-v1.5"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class SearchHit:
    rank: int
    doc_id: str
    score: float
    title: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "score": self.score,
            "title": self.title,
            "text": self.text,
        }


class BgeSearchEngine:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._doc_ids: list[str] | None = None
        self._corpus: dict[str, dict] | None = None
        self._model: SentenceTransformer | None = None
        self._manifest: dict[str, Any] = {}

    @property
    def manifest(self) -> dict[str, Any]:
        self._ensure_loaded()
        return self._manifest

    @property
    def document_count(self) -> int:
        self._ensure_loaded()
        return len(self._doc_ids or [])

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return

        if not CORPUS_JSONL_PATH.exists():
            raise FileNotFoundError(
                f"Corpus not found: {CORPUS_JSONL_PATH}. Run scripts/create_corpus.py"
            )

        faiss_path = INDEX_DIR / "faiss.index"
        doc_ids_path = INDEX_DIR / "doc_ids.json"
        manifest_path = INDEX_DIR / "manifest.json"

        for path in (faiss_path, doc_ids_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Index file not found: {path}. "
                    "Run scripts/corpus_vector_bge_small_en_v1.5.py"
                )

        self._index = faiss.read_index(str(faiss_path))
        self._doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        model_name = self._manifest.get("model_name", MODEL_NAME)
        self._model = SentenceTransformer(model_name)

        corpus: dict[str, dict] = {}
        with CORPUS_JSONL_PATH.open(encoding="utf-8") as file:
            for line in file:
                doc = json.loads(line)
                corpus[str(doc["_id"])] = doc
        self._corpus = corpus

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        self._ensure_loaded()
        assert self._index is not None
        assert self._doc_ids is not None
        assert self._corpus is not None
        assert self._model is not None

        query = query.strip()
        if not query:
            return []

        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            prompt_name="query",
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        hits: list[SearchHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            doc_id = self._doc_ids[idx]
            doc = self._corpus.get(doc_id, {})
            hits.append(
                SearchHit(
                    rank=rank,
                    doc_id=doc_id,
                    score=float(score),
                    title=str(doc.get("title") or ""),
                    text=str(doc.get("text") or ""),
                )
            )
        return hits


_ENGINE: BgeSearchEngine | None = None


def get_search_engine() -> BgeSearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = BgeSearchEngine()
    return _ENGINE


def vector_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
    """Encode query and return top-k corpus documents with scores."""
    return get_search_engine().search(query, top_k=top_k)


def format_hit(hit: SearchHit, text_preview: int = 280) -> str:
    text = hit.text.strip().replace("\n", " ")
    if len(text) > text_preview:
        text = text[: text_preview - 3] + "..."
    header = f"{hit.rank}. score={hit.score:.4f}  id={hit.doc_id}"
    if hit.title:
        header += f"  title={hit.title}"
    return f"{header}\n   {text}"
