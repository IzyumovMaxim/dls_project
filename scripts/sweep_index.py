"""Axis B - sweep the search-time knob of already-built ANN indexes to trace recall<->latency.

nprobe (IVF/IVFPQ) and efSearch (HNSW) are search-time params: no rebuild. Queries load from cache.

    python scripts/sweep_index.py --benchmark fever

Output: data/analysis/sweep_<benchmark>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import bench, index, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100
FLAT_RECALL_100 = 0.9786  # exact reference for the plot

SWEEPS = {
    "e5_base_ivf": ("nprobe", [8, 16, 32, 64, 128, 256]),
    "e5_base_ivfpq": ("nprobe", [8, 16, 32, 64, 128, 256]),
    "e5_base_hnsw": ("efSearch", [16, 32, 64, 128, 256]),
}


def set_knob(idx, knob: str, value: int) -> None:
    if knob == "nprobe":
        idx.nprobe = value
    else:
        idx.hnsw.efSearch = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", default=str(paths.index_dir("e5_base_flat")),
                        help="dir holding the cached query_emb_*.npy")
    parser.add_argument("--configs", default=",".join(SWEEPS))
    parser.add_argument("--lat-warmup", type=int, default=20)
    parser.add_argument("--lat-repeat", type=int, default=100)
    args = parser.parse_args()

    flat_lat = paths.DATA_DIR / "latency" / "e5_base_flat.json"
    encode_p50 = json.loads(flat_lat.read_text())["stages_ms"]["encode"]["p50"] if flat_lat.exists() else 20.4

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    qids, qvecs, qrels = bench.load_query_vectors(
        load_config(f"configs/{names[0]}.yaml"), args.cache_dir, args.benchmark, args.split)

    curves = {}
    for name in names:
        knob, values = SWEEPS[name]
        idx, doc_ids, _ = index.load(name)
        doc_ids = np.array(doc_ids)
        print(f"\n=== {name} ({knob}) ===")
        points = []
        for v in values:
            set_knob(idx, knob, v)
            _, indices = idx.search(qvecs, TOP_K)
            metrics = bench.aggregate_metrics(bench.retrieved_from_ids(indices, qids, doc_ids), qrels, K_VALUES)
            p50 = bench.time_calls(lambda: idx.search(qvecs[:1], TOP_K), args.lat_warmup, args.lat_repeat)["p50"]
            points.append({knob: v, "search_p50_ms": p50, "e2e_p50_ms": round(encode_p50 + p50, 2), **metrics})
            print(f"  {knob}={v:<4} recall@100={metrics['recall@100']:.4f} "
                  f"nDCG@10={metrics['ndcg@10']:.4f} search_p50={p50}ms")
        curves[name] = {"knob": knob, "points": points}

    out = paths.DATA_DIR / "analysis" / f"sweep_{args.benchmark}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": f"{args.benchmark}/{args.split}", "num_queries": len(qids),
        "encode_p50_ms": encode_p50, "flat_recall@100": FLAT_RECALL_100, "curves": curves,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
