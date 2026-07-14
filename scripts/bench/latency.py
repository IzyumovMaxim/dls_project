"""Latency hammer: single-query encode / search / end-to-end p50/p95 for any float index.

Warmup runs are discarded; p50/p95 are the headline, max is a noisy tail. Measure alone.

    python scripts/bench/latency.py --config configs/e5_base_flat.yaml --index-dir data/index/e5_base_flat
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import bench, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.data_io import load_qrels, load_queries  # noqa: E402
from fever_search.encoder import Encoder  # noqa: E402


def _sample_queries(benchmark: str, split: str, n: int) -> list[str]:
    queries_path, qrels_path = paths.benchmark_files(benchmark, split)
    qrels = load_qrels(qrels_path)
    all_queries = load_queries(queries_path)
    texts = [all_queries[qid] for qid in sorted(qrels) if qid in all_queries and all_queries[qid]]
    return texts[:n] if n > 0 else texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-dir", required=True, help="dir with faiss.index (built over the corpus)")
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", default=None, help="write JSON summary here (default: data/latency/<config>.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    index = faiss.read_index(str(Path(args.index_dir) / "faiss.index"))
    encoder = Encoder(config.model)

    # A rotating pool of real queries so encode timing is not one cached string.
    pool = _sample_queries(args.benchmark, args.split, args.warmup + args.repeat + 1)
    if not pool:
        raise SystemExit("No queries found for latency sampling.")
    counter = {"i": 0}

    def next_query() -> str:
        text = pool[counter["i"] % len(pool)]
        counter["i"] += 1
        return text

    # Pre-encode one vector for the isolated search stage.
    warm_vec = np.asarray(encoder.encode_queries([pool[0]]), dtype=np.float32)

    def encode_once() -> None:
        encoder.encode_queries([next_query()])

    def search_once() -> None:
        index.search(warm_vec, args.top_k)

    def e2e_once() -> None:
        vec = np.asarray(encoder.encode_queries([next_query()]), dtype=np.float32)
        index.search(vec, args.top_k)

    print(f"Latency: {index.ntotal:,} docs, top-{args.top_k}, warmup={args.warmup}, repeat={args.repeat}")
    stages = {
        "encode": bench.time_calls(encode_once, args.warmup, args.repeat),
        "search": bench.time_calls(search_once, args.warmup, args.repeat),
        "end_to_end": bench.time_calls(e2e_once, args.warmup, args.repeat),
    }
    throughput = round(1000.0 / stages["end_to_end"]["mean"], 2)

    header = f"{'stage':<16}{'p50':>8}{'p95':>8}{'mean':>8}{'min':>8}{'max':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, s in stages.items():
        print(f"{name:<16}{s['p50']:>8}{s['p95']:>8}{s['mean']:>8}{s['min']:>8}{s['max']:>8}")
    print(f"\nthroughput (single-query) = {throughput} q/s")

    summary = {
        "config": config.name,
        "index_dir": str(args.index_dir),
        "index_docs": int(index.ntotal),
        "top_k": args.top_k,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "stages_ms": stages,
        "throughput_qps": throughput,
    }
    out_path = Path(args.out) if args.out else paths.DATA_DIR / "latency" / f"{config.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
