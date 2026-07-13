"""Tune ANN search-time hyperparameters on a train/val split, then apply to configs.

Sweeps nprobe (IVF/IVFPQ) and efSearch (HNSW) without rebuilding indexes.
Picks the fastest setting that keeps recall@100 within a margin of flat on the tune split.
Updates configs/*.yaml and optionally re-evaluates on test.

    python scripts/tune_ann.py --split train
    python scripts/tune_ann.py --split train --eval-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import bench, index, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100

SWEEPS = {
    "e5_base_ivf": ("nprobe", [8, 16, 32, 64, 128, 256]),
    "e5_base_ivfpq": ("nprobe", [8, 16, 32, 64, 128, 256]),
    "e5_base_hnsw": ("ef_search", [16, 32, 64, 128, 256]),
}

KNOB_YAML_KEY = {"nprobe": "nprobe", "ef_search": "ef_search"}


def set_knob(faiss_index, knob: str, value: int) -> None:
    if knob == "nprobe":
        faiss_index.nprobe = value
    else:
        faiss_index.hnsw.efSearch = value


def sweep_one(
    name: str,
    knob: str,
    values: list[int],
    qvecs: np.ndarray,
    qids: list[str],
    qrels: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> list[dict]:
    faiss_index, ids, _ = index.load(name)
    doc_ids_arr = np.array(ids)
    points = []
    print(f"\n=== {name} ({knob}) ===")
    for value in values:
        set_knob(faiss_index, knob, value)
        _, indices = faiss_index.search(qvecs, TOP_K)
        metrics = bench.aggregate_metrics(
            bench.retrieved_from_ids(indices, qids, doc_ids_arr), qrels, K_VALUES
        )
        p50 = bench.time_calls(lambda: faiss_index.search(qvecs[:1], TOP_K), lat_warmup, lat_repeat)["p50"]
        row = {knob: value, "search_p50_ms": p50, **metrics}
        points.append(row)
        print(
            f"  {knob}={value:<4} recall@100={metrics['recall@100']:.4f} "
            f"nDCG@10={metrics['ndcg@10']:.4f} P@1={metrics['precision@1']:.4f} search_p50={p50}ms"
        )
    return points


def flat_baseline(qvecs: np.ndarray, qids: list[str], qrels: dict) -> dict:
    faiss_index, ids, _ = index.load("e5_base_flat")
    doc_ids_arr = np.array(ids)
    _, indices = faiss_index.search(qvecs, TOP_K)
    metrics = bench.aggregate_metrics(
        bench.retrieved_from_ids(indices, qids, doc_ids_arr), qrels, K_VALUES
    )
    print(
        f"\n=== flat baseline ===\n"
        f"  recall@100={metrics['recall@100']:.4f} nDCG@10={metrics['ndcg@10']:.4f} "
        f"P@1={metrics['precision@1']:.4f}"
    )
    return metrics


def pick_best(points: list[dict], flat_recall: float, recall_margin: float, knob: str) -> dict:
    """Fastest point with recall@100 >= flat - margin; else highest recall."""
    threshold = flat_recall - recall_margin
    candidates = [p for p in points if p["recall@100"] >= threshold]
    pool = candidates if candidates else points
    return min(
        pool,
        key=lambda p: (p["search_p50_ms"], -p["ndcg@10"], -p["recall@100"]),
    )


def update_config_yaml(config_path: Path, knob: str, value: int) -> None:
    yaml_key = KNOB_YAML_KEY[knob]
    text = config_path.read_text(encoding="utf-8")
    pattern = rf"^(\s*{re.escape(yaml_key)}:\s*)\d+"
    replacement = rf"\g<1>{value}"
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"Could not update {yaml_key} in {config_path}")
    config_path.write_text(new_text, encoding="utf-8")
    print(f"  updated {config_path.name}: {yaml_key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="train", help="split for tuning (train or validation)")
    parser.add_argument("--recall-margin", type=float, default=0.01,
                        help="max allowed recall@100 drop vs flat on tune split")
    parser.add_argument("--eval-test", action="store_true",
                        help="after tuning, run evaluate.py on test for each ANN config")
    parser.add_argument("--lat-warmup", type=int, default=10)
    parser.add_argument("--lat-repeat", type=int, default=50)
    args = parser.parse_args()

    base_config = load_config("configs/e5_base_flat.yaml")
    cache_dir = paths.index_dir("e5_base_flat")
    qids, qvecs, qrels = bench.load_query_vectors(
        base_config, cache_dir, args.benchmark, args.split, use_cache=True
    )
    print(f"Tuning on {args.benchmark}/{args.split}: {len(qids):,} queries")

    flat_metrics = flat_baseline(qvecs, qids, qrels)
    flat_recall = flat_metrics["recall@100"]

    curves: dict[str, dict] = {}
    chosen: dict[str, dict] = {}
    for name, (knob, values) in SWEEPS.items():
        points = sweep_one(
            name, knob, values, qvecs, qids, qrels,
            lat_warmup=args.lat_warmup, lat_repeat=args.lat_repeat,
        )
        best = pick_best(points, flat_recall, args.recall_margin, knob)
        curves[name] = {"knob": knob, "points": points, "chosen": best}
        chosen[name] = best
        print(
            f"  -> chosen {knob}={best[knob]} "
            f"(recall@100={best['recall@100']:.4f}, nDCG@10={best['ndcg@10']:.4f}, "
            f"search_p50={best['search_p50_ms']}ms)"
        )
        update_config_yaml(paths.PROJECT_ROOT / "configs" / f"{name}.yaml", knob, best[knob])

    out = paths.DATA_DIR / "analysis" / f"tune_ann_{args.benchmark}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": f"{args.benchmark}/{args.split}",
        "num_queries": len(qids),
        "flat_baseline": flat_metrics,
        "recall_margin": args.recall_margin,
        "chosen": {name: {k: v for k, v in row.items()} for name, row in chosen.items()},
        "curves": curves,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    if args.eval_test:
        import subprocess
        for name in SWEEPS:
            cmd = [
                sys.executable,
                str(paths.PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--config", f"configs/{name}.yaml",
                "--benchmark", args.benchmark,
                "--split", "test",
            ]
            print(f"\n>> {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
