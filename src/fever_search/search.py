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


@dataclass(frozen=True)
class Evidence:
    """The sentence of a hit that best supports the query, as an index into split_sentences(text)."""
    sentence_index: int | None
    score: float


class SearchEngine:
    def __init__(self, config: ExperimentConfig, model_path: str | None = None) -> None:
        self.config = config
        self._index, self._doc_ids, self.manifest = index.load(config.name, index_cfg=config.index)
        self._encoder = Encoder(config.model, model_path=model_path)
        self._corpus = load_corpus(paths.CORPUS_PATH)
        self._doc_position = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}

        # binary_rerank is built from the sign bits of another index's vectors and re-scores against
        # them, so its fp32 and sentence vectors live in that index's directory.
        self._vectors_dir = paths.index_dir(config.index.vectors_from or config.name)
        self._doc_vectors = self._load_doc_vectors()
        self._sentence_vectors, self._sentence_offsets = self._load_sentence_index()
        self._last_query: str | None = None
        self._last_query_vector: np.ndarray | None = None

    def _load_doc_vectors(self) -> np.ndarray | None:
        """fp32 doc vectors for re-scoring, memmapped so only the shortlisted rows are read."""
        if not index.is_binary(self.config.index.type):
            return None
        path = self._vectors_dir / "doc_embeddings.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.config.index.type} re-scores against fp32 vectors, but {path} is missing. "
                f"Point index.vectors_from at the index that has them (e.g. e5_base_flat)."
            )
        vectors = np.load(path, mmap_mode="r")
        if vectors.shape[0] != len(self._doc_ids):
            raise ValueError(
                f"{path} holds {vectors.shape[0]:,} vectors but the index has {len(self._doc_ids):,} "
                "docs; they are addressed by row, so a mismatch reranks the wrong documents."
            )
        print(f"[rerank] {vectors.shape[0]:,} fp32 doc vectors (memmap)")
        return vectors

    def _encode_query(self, query: str) -> np.ndarray:
        """search() and locate_evidence() need the same vector; the forward pass dominates both."""
        if query != self._last_query:
            self._last_query = query
            self._last_query_vector = self._encoder.encode_queries([query])[0]
        assert self._last_query_vector is not None
        return self._last_query_vector

    def _load_sentence_index(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Precomputed sentence vectors, memmapped. Absent is fine: we fall back to encoding."""
        vectors_path = self._vectors_dir / "sentence_vectors.npy"
        offsets_path = self._vectors_dir / "sentence_offsets.npy"
        if not (vectors_path.exists() and offsets_path.exists()):
            return None, None
        offsets = np.load(offsets_path)
        if len(offsets) != len(self._doc_ids) + 1:
            # Built against a different corpus: it would address the wrong sentences.
            print(f"[sentences] {offsets_path.name} covers {len(offsets) - 1:,} docs, index has "
                  f"{len(self._doc_ids):,} — ignoring; rebuild with scripts/index/build_sentence_index.py")
            return None, None
        vectors = np.load(vectors_path, mmap_mode="r")
        print(f"[sentences] {vectors.shape[0]:,} precomputed vectors (memmap)")
        return vectors, offsets

    @property
    def has_sentence_index(self) -> bool:
        return self._sentence_vectors is not None

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    def locate_evidence(self, query: str, hits: list[SearchHit], candidates: list[list[int]]) -> list[Evidence]:
        """For each hit, the sentence best supporting the query.

        `candidates[i]` holds the eligible positions within split_sentences(hits[i].text).
        Without a precomputed sentence index this re-encodes the sentences per query: correct, but
        ~1.5 s on CPU. Build one with scripts/index/build_sentence_index.py.
        """
        query_vector = self._encode_query(query.strip())

        if self._sentence_vectors is None:
            return self._locate_by_encoding(query_vector, hits, candidates)
        return self._locate_by_lookup(query_vector, hits, candidates)

    def _locate_by_lookup(
        self, query_vector: np.ndarray, hits: list[SearchHit], candidates: list[list[int]]
    ) -> list[Evidence]:
        assert self._sentence_vectors is not None and self._sentence_offsets is not None
        out: list[Evidence] = []
        for hit, eligible in zip(hits, candidates):
            position = self._doc_position.get(hit.doc_id)
            if position is None or not eligible:
                out.append(Evidence(None, 0.0))
                continue
            start = int(self._sentence_offsets[position])
            rows = [start + i for i in eligible]
            vectors = np.asarray(self._sentence_vectors[rows], dtype=np.float32)
            scores = vectors @ query_vector
            best = int(np.argmax(scores))
            out.append(Evidence(eligible[best], float(scores[best])))
        return out

    def _locate_by_encoding(
        self, query_vector: np.ndarray, hits: list[SearchHit], candidates: list[list[int]]
    ) -> list[Evidence]:
        from fever_search.text import split_sentences

        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for hit, eligible in zip(hits, candidates):
            sentences = split_sentences(hit.text)
            start = len(flat)
            flat.extend(sentences[i] for i in eligible)
            spans.append((start, len(flat)))

        if not flat:
            return [Evidence(None, 0.0) for _ in hits]

        scores = np.asarray(self._encoder.encode_documents(flat) @ query_vector, dtype=np.float32)
        out: list[Evidence] = []
        for eligible, (start, end) in zip(candidates, spans):
            if start == end:
                out.append(Evidence(None, 0.0))
                continue
            best = int(np.argmax(scores[start:end]))
            out.append(Evidence(eligible[best], float(scores[start + best])))
        return out

    def _search_binary_rerank(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Hamming shortlist over sign bits, then exact fp32 re-scoring of just that shortlist.

        Hamming alone costs ~19 nDCG points; re-scoring `rerank_depth` candidates recovers all but
        one of them. Those rows come from a memmap, so search time depends on whether the OS page
        cache already holds them — see the cold/warm split in data/analysis/RESULTS.md.
        """
        assert self._doc_vectors is not None
        depth = min(self.config.index.rerank_depth, len(self._doc_ids))
        query_bits = index.pack_bits(query_vector[None, :])
        _, shortlist = self._index.search(query_bits, depth)

        candidates = shortlist[0][shortlist[0] >= 0]
        if candidates.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        exact = np.asarray(self._doc_vectors[candidates], dtype=np.float32) @ query_vector
        best = np.argsort(-exact)[:top_k]
        return exact[best][None, :], candidates[best][None, :]

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        query_vector = self._encode_query(query)

        if index.is_binary(self.config.index.type):
            scores, indices = self._search_binary_rerank(query_vector, top_k)
        else:
            scores, indices = self._index.search(
                np.asarray(query_vector[None, :], dtype=np.float32), top_k
            )

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
