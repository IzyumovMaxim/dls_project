"""RAM / search latency / quality on one split: compression variants + ANN (default & tuned).

Reuses cached query embeddings when available. Compression builds variants from
doc_embeddings.npy; ANN loads pre-built indexes and only changes search-time knobs.

    python scripts/benchmark_split.py --split validation
    python scripts/benchmark_split.py --split train
    python scripts/benchmark_split.py --split test --ann-only

Output: data/analysis/benchmark_<benchmark>_<split>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import bench, index, paths  # noqa: E402
from fever_search.config import IndexConfig, load_config  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100

COMPRESSION_VARIANTS = ("flat", "sq8", "pq", "binary", "binary_rerank")

ANN_INDEXES = {
    "e5_base_ivf": {
        "knob": "nprobe",
        "default": 32,
        "tuned": 64,
    },
    "e5_base_ivfpq": {
        "knob": "nprobe",
        "default": 32,
        "tuned": 128,
    },
    "e5_base_hnsw": {
        "knob": "ef_search",
        "default": 64,
        "tuned": 256,
    },
}


def _load_compression_runner():
    path = Path(__file__).resolve().parent / "benchmark_compression.py"
    spec = importlib.util.spec_from_file_location("benchmark_compression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_variant


def set_knob(faiss_index, knob: str, value: int) -> None:
    if knob == "nprobe":
        faiss_index.nprobe = value
    else:
        faiss_index.hnsw.efSearch = value


def run_ann(
    name: str,
    knob: str,
    value: int,
    label: str,
    qvecs: np.ndarray,
    qids: list[str],
    qrels: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> dict:
    faiss_index, doc_ids, manifest = index.load(name)
    set_knob(faiss_index, knob, value)
    doc_ids_arr = np.array(doc_ids)
    mem = bench.index_num_bytes(faiss_index, binary=False)

    _, indices = faiss_index.search(qvecs, TOP_K)
    metrics = bench.aggregate_metrics(
        bench.retrieved_from_ids(indices, qids, doc_ids_arr), qrels, K_VALUES
    )
    p50 = bench.time_calls(
        lambda: faiss_index.search(qvecs[:1], TOP_K), lat_warmup, lat_repeat
    )["p50"]

    row = {
        "group": "ann",
        "name": name,
        "index_type": manifest.get("index_type", name),
        "setting": label,
        knob: value,
        "memory": bench.human_bytes(mem),
        "memory_bytes": mem,
        "search_p50_ms": p50,
        **metrics,
    }
    print(
        f"  {name} {label} {knob}={value}: "
        f"RAM={row['memory']} search_p50={p50}ms "
        f"P@1={metrics['precision@1']:.4f} nDCG@10={metrics['ndcg@10']:.4f} "
        f"R@100={metrics['recall@100']:.4f}"
    )
    return row


def run_compression(
    run_variant,
    emb: np.ndarray,
    doc_ids: np.ndarray,
    qvecs: np.ndarray,
    qids: list[str],
    qrels: dict,
    pq_m: int,
    pq_nbits: int,
    rerank_depth: int,
    lat_warmup: int,
    lat_repeat: int,
) -> list[dict]:
    rows = []
    flat_recall = None
    for variant in COMPRESSION_VARIANTS:
        print(f"\n=== compression / {variant} ===")
        v = run_variant(
            variant, emb, qvecs, pq_m, pq_nbits, rerank_depth, lat_warmup, lat_repeat
        )
        metrics = bench.aggregate_metrics(
            bench.retrieved_from_ids(v["indices"], qids, doc_ids), qrels, K_VALUES
        )
        if variant == "flat":
            flat_recall = metrics["recall@100"]
            d_recall = 0.0
        else:
            d_recall = round(metrics["recall@100"] - flat_recall, 4) if flat_recall is not None else None
        rows.append({
            "group": "compression",
            "variant": variant,
            "setting": None,
            "memory": v["memory_human"],
            "memory_bytes": v["memory_bytes"],
            "search_p50_ms": v["search_p50_ms"],
            "build_s": v["build_s"],
            "d_recall": d_recall,
            **metrics,
        })
        print(
            f"  RAM={v['memory_human']} search_p50={v['search_p50_ms']}ms "
            f"P@1={metrics['precision@1']:.4f} nDCG@10={metrics['ndcg@10']:.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5_base_flat.yaml")
    parser.add_argument("--index-dir", default=str(paths.index_dir("e5_base_flat")))
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument(
        "--split",
        default="validation",
        help="train (~110k queries, slow) | validation (~6.5k) | test",
    )
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--rerank-depth", type=int, default=1000)
    parser.add_argument("--lat-warmup", type=int, default=20)
    parser.add_argument("--lat-repeat", type=int, default=100)
    parser.add_argument("--compression-only", action="store_true")
    parser.add_argument("--ann-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON (default: data/analysis/benchmark_<benchmark>_<split>.json)",
    )
    args = parser.parse_args()

    if args.compression_only and args.ann_only:
        raise SystemExit("Use at most one of --compression-only / --ann-only")

    config = load_config(args.config)
    index_dir = Path(args.index_dir)
    qids, qvecs, qrels = bench.load_query_vectors(
        config, index_dir, args.benchmark, args.split, use_cache=not args.no_cache
    )
    print(f"Benchmark: {args.benchmark}/{args.split} — {len(qids):,} queries\n")

    results: list[dict] = []
    run_compression_fn = _load_compression_runner()

    if not args.ann_only:
        emb = np.load(index_dir / "doc_embeddings.npy").astype(np.float32)
        doc_ids = np.array(json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8")))
        print(f"Embeddings: {emb.shape[0]:,} x {emb.shape[1]} ({bench.human_bytes(emb.nbytes)})")
        results.extend(run_compression(
            run_compression_fn, emb, doc_ids, qvecs, qids, qrels,
            args.pq_m, args.pq_nbits, args.rerank_depth,
            args.lat_warmup, args.lat_repeat,
        ))

    if not args.compression_only:
        print("\n=== ann / e5_base_flat (baseline) ===")
        flat_cfg = IndexConfig(type="flat")
        faiss_index, doc_ids, manifest = index.load("e5_base_flat", flat_cfg)
        doc_ids_arr = np.array(doc_ids)
        mem = bench.index_num_bytes(faiss_index, binary=False)
        _, indices = faiss_index.search(qvecs, TOP_K)
        flat_metrics = bench.aggregate_metrics(
            bench.retrieved_from_ids(indices, qids, doc_ids_arr), qrels, K_VALUES
        )
        flat_p50 = bench.time_calls(
            lambda: faiss_index.search(qvecs[:1], TOP_K), args.lat_warmup, args.lat_repeat
        )["p50"]
        flat_row = {
            "group": "ann",
            "name": "e5_base_flat",
            "index_type": "flat",
            "setting": "baseline",
            "memory": bench.human_bytes(mem),
            "memory_bytes": mem,
            "search_p50_ms": flat_p50,
            **flat_metrics,
        }
        results.append(flat_row)
        print(
            f"  flat baseline: RAM={flat_row['memory']} search_p50={flat_p50}ms "
            f"P@1={flat_metrics['precision@1']:.4f} nDCG@10={flat_metrics['ndcg@10']:.4f}"
        )

        for name, spec in ANN_INDEXES.items():
            print(f"\n=== ann / {name} ===")
            for label, value in (("default", spec["default"]), ("tuned", spec["tuned"])):
                results.append(run_ann(
                    name, spec["knob"], value, label,
                    qvecs, qids, qrels, args.lat_warmup, args.lat_repeat,
                ))

    out_path = (
        Path(args.out)
        if args.out
        else paths.DATA_DIR / "analysis" / f"benchmark_{args.benchmark}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.name,
        "benchmark": f"{args.benchmark}/{args.split}",
        "num_queries": len(qids),
        "pq": {"m": args.pq_m, "nbits": args.pq_nbits},
        "rerank_depth": args.rerank_depth,
        "ann_params": ANN_INDEXES,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
