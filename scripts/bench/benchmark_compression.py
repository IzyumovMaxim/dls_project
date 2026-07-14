"""Axis A - compression: build SQ8 / PQ / binary variants from the saved fp32 embeddings and
measure index memory, single-query search p50 and full quality (P@k, R@k, MRR, nDCG@10) vs flat.

Variants are built from doc_embeddings.npy (no re-encoding). Queries are encoded once and cached.

    python scripts/bench/benchmark_compression.py --index-dir data/index/e5_base_flat --benchmark fever

Output: <index-dir>/compression_<benchmark>.json + a markdown table on stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import bench, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100


def binary_rerank_search(bindex, qbits, qvecs, emb, top_k, rerank_depth):
    """Hamming shortlist of `rerank_depth`, then exact fp32 re-scoring to top_k."""
    nq = qbits.shape[0]          # qbits may be a leading slice of qvecs (single-query latency call)
    qvecs = qvecs[:nq]
    depth = min(rerank_depth, emb.shape[0])
    _, cand = bindex.search(qbits, depth)
    out = np.full((nq, top_k), -1, dtype=np.int64)
    for qi in range(nq):
        row = cand[qi][cand[qi] >= 0]
        top = row[np.argsort(-(emb[row] @ qvecs[qi]))[:top_k]]
        out[qi, : len(top)] = top
    return out


def run_variant(name, emb, qvecs, pq_m, pq_nbits, rerank_depth, lat_warmup, lat_repeat):
    """Build one variant, then measure memory + search p50 and return its top-k retrieval."""
    dim = emb.shape[1]
    binary = name in ("binary", "binary_rerank")
    t0 = time.perf_counter()

    if name == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        search = lambda q, k: index.search(q, k)[1]
        q_all, q_one = qvecs, qvecs[:1]
    elif name == "sq8":
        index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        search = lambda q, k: index.search(q, k)[1]
        q_all, q_one = qvecs, qvecs[:1]
    elif name == "pq":
        if dim % pq_m != 0:
            raise SystemExit(f"pq: dim {dim} not divisible by m={pq_m}")
        index = faiss.IndexPQ(dim, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        search = lambda q, k: index.search(q, k)[1]
        q_all, q_one = qvecs, qvecs[:1]
    elif name == "opq":
        # Same code size as pq; the rotation matrix costs an extra dim x dim floats (2.4 MB).
        if dim % pq_m != 0:
            raise SystemExit(f"opq: dim {dim} not divisible by m={pq_m}")
        index = faiss.index_factory(dim, f"OPQ{pq_m},PQ{pq_m}x{pq_nbits}", faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        search = lambda q, k: index.search(q, k)[1]
        q_all, q_one = qvecs, qvecs[:1]
    else:  # binary / binary_rerank (sign bits, Hamming search)
        index = faiss.IndexBinaryFlat(dim)
        index.add(np.packbits(emb > 0, axis=1))
        qbits = np.packbits(qvecs > 0, axis=1)
        if name == "binary":
            search = lambda q, k: index.search(q, k)[1]
        else:
            search = lambda q, k: binary_rerank_search(index, q, qvecs, emb, k, rerank_depth)
        q_all, q_one = qbits, qbits[:1]

    build_s = time.perf_counter() - t0
    index_bytes = bench.index_num_bytes(index, binary=binary)

    # binary_rerank re-scores its Hamming shortlist against the fp32 vectors, so those vectors are
    # part of its footprint — counting only the 48 MB binary index would understate it by ~30x.
    # They need not be resident: memmapping doc_embeddings.npy touches ~3 MB per query and measures
    # the same p50, which is what `rerank_bytes_resident` reports.
    rerank_bytes = emb.nbytes if name == "binary_rerank" else 0
    total_bytes = index_bytes + rerank_bytes

    indices = search(q_all, TOP_K)
    latency = bench.time_calls(lambda: search(q_one, TOP_K), lat_warmup, lat_repeat)
    return {
        "indices": indices,
        "index_bytes": index_bytes,
        "rerank_bytes": rerank_bytes,
        "memory_bytes": total_bytes,
        "memory_human": bench.human_bytes(total_bytes),
        "memory_resident_human": bench.human_bytes(index_bytes),  # fp32 side memmapped off-RAM
        "search_p50_ms": latency["p50"],
        "build_s": round(build_s, 1),
    }


def render_table(rows, columns, header_map):
    head = [header_map.get(c, c) for c in columns]
    widths = [max(len(h), *(len(str(r.get(c, ""))) for r in rows)) for h, c in zip(head, columns)]
    line = lambda vals: "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(head), sep, *(line([r.get(c, "") for c in columns]) for r in rows)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5_base_flat.yaml")
    parser.add_argument("--index-dir", default=str(paths.index_dir("e5_base_flat")),
                        help="dir holding doc_embeddings.npy + doc_ids.json")
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", default="flat,sq8,pq,binary,binary_rerank")
    parser.add_argument("--pq-m", type=int, default=96, help="PQ sub-quantizers (must divide dim)")
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--rerank-depth", type=int, default=1000, help="binary_rerank shortlist size")
    parser.add_argument("--lat-warmup", type=int, default=20)
    parser.add_argument("--lat-repeat", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    index_dir = Path(args.index_dir)
    emb = np.load(index_dir / "doc_embeddings.npy").astype(np.float32)
    doc_ids = np.array(json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8")))
    print(f"Embeddings: {emb.shape[0]:,} x {emb.shape[1]} fp32  ({bench.human_bytes(emb.nbytes)} in RAM)")

    qids, qvecs, qrels = bench.load_query_vectors(config, index_dir, args.benchmark, args.split, not args.no_cache)

    results, flat_recall = [], None
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"\n=== {name} ===")
        v = run_variant(name, emb, qvecs, args.pq_m, args.pq_nbits,
                        args.rerank_depth, args.lat_warmup, args.lat_repeat)
        metrics = bench.aggregate_metrics(bench.retrieved_from_ids(v["indices"], qids, doc_ids), qrels, K_VALUES)
        if name == "flat":
            flat_recall = metrics["recall@100"]
        results.append({
            "variant": name,
            "memory": v["memory_human"],
            "memory_resident": v["memory_resident_human"],
            "memory_bytes": v["memory_bytes"],
            "index_bytes": v["index_bytes"],
            "rerank_bytes": v["rerank_bytes"],
            "d_recall": "" if flat_recall is None else round(metrics["recall@100"] - flat_recall, 4),
            "search_p50_ms": v["search_p50_ms"],
            "build_s": v["build_s"],
            **metrics,
        })
        note = "" if not v["rerank_bytes"] else (
            f" (index {bench.human_bytes(v['index_bytes'])} + fp32 rerank vectors "
            f"{bench.human_bytes(v['rerank_bytes'])}; memmap the latter to keep only the index resident)"
        )
        print(f"  memory={v['memory_human']}{note}  recall@100={metrics['recall@100']}  "
              f"nDCG@10={metrics['ndcg@10']}  search_p50={v['search_p50_ms']}ms")

    compact = ["variant", "memory", "memory_resident", "recall@100", "d_recall", "search_p50_ms", "ndcg@10", "build_s"]
    full = ["variant", "memory", "precision@1", "recall@10", "recall@100", "mrr", "ndcg@10", "search_p50_ms"]
    hdr = {
        "d_recall": "Δrecall vs flat",
        "search_p50_ms": "search p50, ms",
        "build_s": "build, s",
        "memory": "RAM (all in-memory)",
        "memory_resident": "RAM (fp32 memmapped)",
    }
    print(f"\n\n## Axis A - compression ({args.benchmark}/{args.split}, {len(qids):,} queries)\n")
    print(render_table(results, compact, hdr))
    print("\n## Full metrics\n")
    print(render_table(results, full, hdr))

    out = index_dir / f"compression_{args.benchmark}_{args.split}_m{args.pq_m}.json"
    out.write_text(json.dumps({
        "config": config.name,
        "benchmark": f"{args.benchmark}/{args.split}",
        "num_queries": len(qids),
        "pq": {"m": args.pq_m, "nbits": args.pq_nbits},
        "rerank_depth": args.rerank_depth,
        "results": [{k: val for k, val in r.items() if k != "indices"} for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
