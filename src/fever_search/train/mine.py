"""Mine hard negatives: top retrieved non-gold docs per FEVER training query."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fever_search import bench, index, paths
from fever_search.config import ExperimentConfig


def mine_hard_negatives(
    config: ExperimentConfig,
    num_negatives: int = 4,
    split: str = "train",
    out_path: Path | None = None,
    max_queries: int | None = None,
    chunk_size: int = 4096,
) -> Path:
    # Pass the config: on an ANN index faiss would otherwise default to nprobe=1 and
    # the "hard" negatives would be near-random.
    faiss_index, doc_id_list, _ = index.load(config.name, index_cfg=config.index)
    doc_ids = np.asarray(doc_id_list)

    qids, qvecs, qrels = bench.load_query_vectors(
        config, paths.index_dir(config.name), "fever", split
    )
    if max_queries:
        qids, qvecs = qids[:max_queries], qvecs[:max_queries]

    # Gold docs are dropped after retrieval, so leave headroom for the query with the most of them.
    max_gold = max(len(qrels[qid]) for qid in qids)
    search_k = num_negatives + max_gold + 5

    out_path = out_path or (paths.TRAIN_DIR / f"hard_negatives_{split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    short = 0
    with out_path.open("w", encoding="utf-8") as file:
        for start in tqdm(range(0, len(qids), chunk_size), desc="Mining", unit="chunk"):
            batch_qids = qids[start : start + chunk_size]
            _, indices = faiss_index.search(qvecs[start : start + chunk_size], search_k)
            for qid, row in zip(batch_qids, indices):
                gold = qrels[qid]
                retrieved = doc_ids[row[row >= 0]].tolist()
                negatives = [doc_id for doc_id in retrieved if doc_id not in gold][:num_negatives]
                if len(negatives) < num_negatives:
                    short += 1
                file.write(json.dumps({
                    "query_id": qid,
                    "positive_ids": sorted(gold),
                    "negative_ids": negatives,
                }) + "\n")
                written += 1

    print(f"Mined hard negatives for {written:,} queries (top-{search_k}) -> {out_path}")
    if short:
        print(f"NOTE: {short:,} queries yielded fewer than {num_negatives} negatives")
    return out_path
