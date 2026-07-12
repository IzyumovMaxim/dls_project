"""Benchmark helpers: latency percentiles, index memory, query vectors, quality metrics.

Metrics come from fever_search.eval, so numbers match scripts/evaluate.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from fever_search import eval as evallib
from fever_search import paths
from fever_search.data_io import load_qrels, load_queries
from fever_search.encoder import Encoder


def stats_ms(times_s: list[float]) -> dict[str, float]:
    """Percentile summary (ms) of durations given in seconds."""
    arr = np.asarray(times_s, dtype=np.float64) * 1000.0
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "mean": round(float(arr.mean()), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "n": int(arr.size),
    }


def time_calls(fn: Callable[[], object], n_warmup: int, n_repeat: int) -> dict[str, float]:
    """Time fn(): n_warmup discarded runs, then n_repeat measured."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return stats_ms(times)


def index_num_bytes(index: faiss.Index, *, binary: bool = False) -> int:
    """Serialized index size in bytes (~ its in-RAM vector storage)."""
    serialized = faiss.serialize_index_binary(index) if binary else faiss.serialize_index(index)
    return int(serialized.nbytes)


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def load_query_vectors(config, cache_dir, benchmark: str, split: str, use_cache: bool = True):
    """Return (qids, vectors, qrels), encoding once and caching to cache_dir.

    qid order is deterministic (sorted qrels ∩ non-empty queries) so a cached
    query_emb_<benchmark>_<split>.npy stays aligned across runs and scripts.
    """
    queries_path, qrels_path = paths.benchmark_files(benchmark, split)
    qrels = load_qrels(qrels_path)
    all_queries = load_queries(queries_path)
    qids = [qid for qid in sorted(qrels) if qid in all_queries and all_queries[qid]]

    cache = Path(cache_dir) / f"query_emb_{benchmark}_{split}.npy"
    if use_cache and cache.exists():
        vectors = np.load(cache)
        if vectors.shape[0] == len(qids):
            print(f"[cache] query vectors <- {cache}")
            return qids, vectors.astype(np.float32), {q: qrels[q] for q in qids}

    texts = [all_queries[qid] for qid in qids]
    print(f"Encoding {len(texts):,} queries ...")
    vectors = np.asarray(Encoder(config.model).encode_queries(texts, show_progress_bar=True), dtype=np.float32)
    if use_cache:
        np.save(cache, vectors)
        print(f"[cache] query vectors -> {cache}")
    return qids, vectors, {q: qrels[q] for q in qids}


def retrieved_from_ids(indices: np.ndarray, qids: list[str], doc_ids: np.ndarray) -> dict[str, list[str]]:
    """Map a (num_queries, k) FAISS index matrix to {qid: [doc_id, ...]}, dropping -1 padding."""
    return {qid: doc_ids[row[row >= 0]].tolist() for row, qid in zip(indices, qids)}


def aggregate_metrics(
    retrieved_by_qid: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    """Mean P@k / R@k / MRR / nDCG@10 across queries."""
    acc: dict[str, list[float]] = {}
    for k in k_values:
        acc[f"precision@{k}"] = []
        acc[f"recall@{k}"] = []
    acc["mrr"] = []
    acc["ndcg@10"] = []

    for qid, retrieved in retrieved_by_qid.items():
        relevant = qrels[qid]
        for k in k_values:
            acc[f"precision@{k}"].append(evallib.precision_at_k(retrieved, relevant, k))
            acc[f"recall@{k}"].append(evallib.recall_at_k(retrieved, relevant, k))
        acc["mrr"].append(evallib.mrr(retrieved, relevant))
        acc["ndcg@10"].append(evallib.ndcg_at_k(retrieved, relevant, 10))

    return {key: round(float(np.mean(vals)), 4) for key, vals in acc.items()}
