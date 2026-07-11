"""Mine hard negatives: top retrieved non-gold docs per FEVER training query."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from fever_search import paths
from fever_search.data_io import load_qrels, load_queries
from fever_search.search import SearchEngine


def mine_hard_negatives(
    engine: SearchEngine,
    num_negatives: int = 4,
    split: str = "train",
    out_path: Path | None = None,
    max_queries: int | None = None,
) -> Path:
    queries_path, qrels_path = paths.benchmark_files("fever", split)
    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    qids = sorted(qid for qid in qrels if qid in queries)
    if max_queries:
        qids = qids[:max_queries]

    out_path = out_path or (paths.TRAIN_DIR / f"hard_negatives_{split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    search_k = num_negatives + 20

    with out_path.open("w", encoding="utf-8") as file:
        for qid in tqdm(qids, desc="Mining", unit="q"):
            gold = qrels[qid]
            retrieved = [hit.doc_id for hit in engine.search(queries[qid], top_k=search_k)]
            negatives = [doc_id for doc_id in retrieved if doc_id not in gold][:num_negatives]
            file.write(json.dumps({
                "query_id": qid,
                "positive_ids": sorted(gold),
                "negative_ids": negatives,
            }) + "\n")

    print(f"Mined hard negatives for {len(qids):,} queries -> {out_path}")
    return out_path
