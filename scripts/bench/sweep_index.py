"""Sweep the search-time knob of already-built ANN indexes to trace recall<->latency.

nprobe (IVF/IVFPQ) and efSearch (HNSW) are search-time params: no rebuild. Queries load from cache.

Defaults to validation: picking an operating point off a curve drawn on test is tuning on test.
Use scripts/index/tune_ann.py to select a value and write it into the configs; this script only
draws the curve. Pass --split test only to report the chosen point, never to choose it.

    python scripts/bench/sweep_index.py --benchmark fever

Output: data/analysis/sweep_<benchmark>_<split>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import bench, index, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100

SWEEP_VALUES = {
    "e5_base_ivf": [8, 16, 32, 64, 128, 256],
    "e5_base_ivfpq": [8, 16, 32, 64, 128, 256],
    "e5_base_ivfpq192": [8, 16, 32, 64, 128, 256, 512],
    "e5_base_hnsw": [16, 32, 64, 128, 256],
}


def flat_recall_100(qvecs: np.ndarray, qids: list[str], qrels: dict) -> float:
    """Exact-search reference for the Pareto plot — measured, not assumed: it moves with the model."""
    idx, doc_ids, _ = index.load("e5_base_flat")
    _, indices = idx.search(qvecs, TOP_K)
    metrics = bench.aggregate_metrics(
        bench.retrieved_from_ids(indices, qids, np.array(doc_ids)), qrels, K_VALUES)
    return metrics["recall@100"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache-dir", default=str(paths.index_dir("e5_base_flat")),
                        help="dir holding the cached query_emb_*.npy")
    parser.add_argument("--configs", default=",".join(SWEEP_VALUES))
    parser.add_argument("--lat-warmup", type=int, default=20)
    parser.add_argument("--lat-repeat", type=int, default=100)
    args = parser.parse_args()

    # e2e = encode + search. Without a measured encode cost we report search only rather than
    # fold in a guess, which would silently skew every e2e number on the plot.
    flat_lat = paths.DATA_DIR / "latency" / "e5_base_flat.json"
    if flat_lat.exists():
        encode_p50 = json.loads(flat_lat.read_text())["stages_ms"]["encode"]["p50"]
    else:
        encode_p50 = None
        print(f"NOTE: {flat_lat} missing — run scripts/bench/latency.py for e2e numbers; reporting search only")

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    qids, qvecs, qrels = bench.load_query_vectors(
        load_config(f"configs/{names[0]}.yaml"), args.cache_dir, args.benchmark, args.split)

    flat_recall = flat_recall_100(qvecs, qids, qrels)
    print(f"flat baseline recall@100 = {flat_recall:.4f} ({len(qids):,} queries)")

    curves = {}
    for name in names:
        cfg = load_config(f"configs/{name}.yaml").index
        knob = index.SEARCH_KNOB[cfg.type]
        idx, doc_ids, _ = index.load(name)
        doc_ids = np.array(doc_ids)
        print(f"\n=== {name} ({knob}) ===")
        points = []
        for v in SWEEP_VALUES[name]:
            index.set_search_knob(idx, knob, v)
            _, indices = idx.search(qvecs, TOP_K)
            metrics = bench.aggregate_metrics(bench.retrieved_from_ids(indices, qids, doc_ids), qrels, K_VALUES)
            p50 = bench.time_calls(lambda: idx.search(qvecs[:1], TOP_K), args.lat_warmup, args.lat_repeat)["p50"]
            point = {knob: v, "search_p50_ms": p50, **metrics}
            if encode_p50 is not None:
                point["e2e_p50_ms"] = round(encode_p50 + p50, 2)
            points.append(point)
            print(f"  {knob}={v:<4} recall@100={metrics['recall@100']:.4f} "
                  f"nDCG@10={metrics['ndcg@10']:.4f} search_p50={p50}ms")
        curves[name] = {"knob": knob, "points": points}

    out = paths.DATA_DIR / "analysis" / f"sweep_{args.benchmark}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": f"{args.benchmark}/{args.split}", "num_queries": len(qids),
        "encode_p50_ms": encode_p50, "flat_recall@100": flat_recall, "curves": curves,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
