"""Retrieval metrics (Precision/Recall@k, MRR, nDCG@10) and evaluation runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from fever_search.data_io import load_qrels, load_queries
from fever_search.search import SearchEngine


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 1)
              for rank, doc_id in enumerate(retrieved[:k], start=1) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def run_eval(
    engine: SearchEngine,
    queries_path: Path,
    qrels_path: Path,
    out_dir: Path,
    label: str,
    top_k: int = 100,
    k_values: tuple[int, ...] = (1, 5, 10, 100),
) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    all_queries = load_queries(queries_path)
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    if len(queries) != len(qrels):
        print(f"WARNING: {len(qrels) - len(queries)} query ids missing from {queries_path.name}")

    per_query: list[dict[str, Any]] = []
    acc: dict[str, list[float]] = {f"precision@{k}": [] for k in k_values}
    acc.update({f"recall@{k}": [] for k in k_values})
    acc["mrr"] = []
    acc["ndcg@10"] = []

    print(f"{label}: {len(queries):,} queries against {engine.document_count:,} docs (top-{top_k})")
    for qid in tqdm(sorted(queries), desc="Evaluating", unit="q"):
        relevant = qrels[qid]
        retrieved = [hit.doc_id for hit in engine.search(queries[qid], top_k=top_k)]
        row: dict[str, Any] = {"query_id": qid, "num_relevant": len(relevant)}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row["mrr"] = mrr(retrieved, relevant)
        row["ndcg@10"] = ndcg_at_k(retrieved, relevant, 10)
        for key in acc:
            acc[key].append(row[key])
        per_query.append(row)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": label,
        "model": engine.manifest.get("model_name"),
        "index_type": engine.manifest.get("index_type"),
        "index_docs": engine.document_count,
        "num_queries": len(per_query),
        "metrics": {key: _mean(vals) for key, vals in acc.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as file:
        for row in per_query:
            file.write(json.dumps(row) + "\n")

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Report -> {out_dir / 'report.json'}")
    return summary
