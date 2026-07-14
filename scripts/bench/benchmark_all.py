"""Every number in the report, from one command.

Writes data/analysis/RESULTS.md (tables) and benchmark_all.json (raw numbers), so any figure can be
traced back to the run that produced it. Query vectors come from the cache the flat build wrote.

    python scripts/bench/benchmark_all.py                 # index + compression + serving
    python scripts/bench/benchmark_all.py --with-bm25     # adds the lexical baseline (slow to index)
    python scripts/bench/benchmark_all.py --report-only   # rebuild the tables from the last run

Axes:
    index        quality / RAM / search latency of each built index
    compression  the same vectors stored as fp32 / SQ8 / PQ / binary, built on the fly
    serving      the query path end to end: encode -> search -> locate evidence, plus process RSS
    bm25         lexical baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import bench, index, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.search import SearchEngine  # noqa: E402
from fever_search.text import evidence_candidates, split_sentences  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100

# Every index we built, in the order the report compares them.
INDEXES = ["e5_base_flat", "e5_base_ivf", "e5_base_ivfpq", "e5_base_hnsw", "e5_base_binary_rerank"]
COMPRESSION_VARIANTS = ("flat", "sq8", "pq", "binary", "binary_rerank")
SERVING_CONFIGS = ["e5_base_flat", "e5_base_binary_rerank"]

BASELINE = "e5_base_flat"  # exact search: the quality ceiling everything else is measured against


def rss_bytes() -> int:
    """Peak resident set of this process. On macOS getrusage reports bytes, on Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def load_compression_runner():
    path = Path(__file__).resolve().parent / "benchmark_compression.py"
    spec = importlib.util.spec_from_file_location("benchmark_compression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_variant


# ------------------------------------------------------------------------------- axis: index

def search_index(name: str, qvecs: np.ndarray, qids: list[str], qrels: dict, args) -> dict:
    """Quality, RAM and search latency of one built index, through the code the app actually runs."""
    config = load_config(paths.PROJECT_ROOT / "configs" / f"{name}.yaml")
    faiss_index, doc_ids, _ = index.load(name, index_cfg=config.index)
    doc_ids_arr = np.array(doc_ids)
    binary = index.is_binary(config.index.type)
    index_bytes = bench.index_num_bytes(faiss_index, binary=binary)

    if binary:
        # Same path as SearchEngine._search_binary_rerank: Hamming shortlist, exact fp32 re-scoring.
        vectors = np.load(paths.index_dir(config.index.vectors_from) / "doc_embeddings.npy", mmap_mode="r")
        depth = min(config.index.rerank_depth, len(doc_ids))

        def run(queries: np.ndarray) -> np.ndarray:
            _, shortlist = faiss_index.search(index.pack_bits(queries), depth)
            out = np.full((queries.shape[0], TOP_K), -1, dtype=np.int64)
            for row in range(queries.shape[0]):
                candidates = shortlist[row][shortlist[row] >= 0]
                exact = np.asarray(vectors[candidates], dtype=np.float32) @ queries[row]
                best = candidates[np.argsort(-exact)[:TOP_K]]
                out[row, : len(best)] = best
            return out

        # The fp32 vectors are memmapped, so they are not resident; the binary codes are.
        rerank_bytes = int(vectors.nbytes)
    else:
        def run(queries: np.ndarray) -> np.ndarray:
            return faiss_index.search(queries, TOP_K)[1]

        rerank_bytes = 0

    indices = run(qvecs)
    metrics = bench.aggregate_metrics(bench.retrieved_from_ids(indices, qids, doc_ids_arr), qrels, K_VALUES)
    latency = bench.time_calls(lambda: run(qvecs[:1]), args.lat_warmup, args.lat_repeat)

    return {
        "name": name,
        "index_type": config.index.type,
        "resident": bench.human_bytes(index_bytes),
        "resident_bytes": index_bytes,
        "on_disk_bytes": index_bytes + rerank_bytes,
        "on_disk": bench.human_bytes(index_bytes + rerank_bytes),
        "search_p50_ms": latency["p50"],
        "search_p95_ms": latency["p95"],
        **metrics,
    }


# ------------------------------------------------------------------------- axis: compression

def run_compression(emb, doc_ids, qvecs, qids, qrels, args) -> list[dict]:
    run_variant = load_compression_runner()
    rows = []
    for variant in COMPRESSION_VARIANTS:
        print(f"\n=== compression / {variant} ===")
        result = run_variant(variant, emb, qvecs, args.pq_m, args.pq_nbits,
                             args.rerank_depth, args.lat_warmup, args.lat_repeat)
        metrics = bench.aggregate_metrics(
            bench.retrieved_from_ids(result["indices"], qids, doc_ids), qrels, K_VALUES)
        rows.append({
            "variant": variant,
            "in_memory": result["memory_human"],
            "resident": result["memory_resident_human"],
            "search_p50_ms": result["search_p50_ms"],
            "build_s": result["build_s"],
            **metrics,
        })
        print(f"  RAM={result['memory_human']} (resident {result['memory_resident_human']}) "
              f"nDCG@10={metrics['ndcg@10']:.4f} search_p50={result['search_p50_ms']}ms")
    return rows


# ---------------------------------------------------------------------------- axis: serving

def serving_in_subprocess(name: str, args) -> dict:
    """Measure one config in a fresh process.

    Peak RSS is monotonic within a process, so measuring two configs in one would charge the second
    with the first one's footprint — and RAM is the number this whole comparison turns on.
    """
    import subprocess

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--serving-worker", name,
        "--benchmark", args.benchmark, "--split", args.split,
        "--device", args.device, "--serving-queries", str(args.serving_queries),
        "--serving-warmup", str(args.serving_warmup),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    for line in completed.stdout.splitlines():
        if line.startswith("SERVING_JSON "):
            return json.loads(line[len("SERVING_JSON "):])
    raise RuntimeError(f"serving worker for {name} produced no result:\n{completed.stdout}\n{completed.stderr}")


def time_queries(engine, queries: list[str]) -> dict[str, list[float]]:
    stages: dict[str, list[float]] = {"encode": [], "search": [], "evidence": []}
    for query in queries:
        engine._last_query = None  # every query pays the encoder, as it would in production

        t0 = time.perf_counter()
        engine._encode_query(query)
        t1 = time.perf_counter()
        hits = engine.search(query, top_k=10)
        t2 = time.perf_counter()
        candidates = [evidence_candidates(split_sentences(hit.text)) for hit in hits]
        engine.locate_evidence(query, hits, candidates)
        t3 = time.perf_counter()

        stages["encode"].append(t1 - t0)
        stages["search"].append(t2 - t1)  # the encode inside search() hits the one-query cache
        stages["evidence"].append(t3 - t2)
    return stages


def run_serving(name: str, queries: list[str], args) -> dict:
    """What a user actually waits for: encode + search + evidence, and what the process holds in RAM.

    Measured twice, because for the memmapped indexes the two answers differ by an order of magnitude:

      cold  a fresh process, every query touching pages of doc_embeddings.npy for the first time
      warm  after `--serving-warmup` queries, once the OS page cache holds the rows being re-scored

    Reporting only the warm number would be a lie by omission: it is fast precisely because the fp32
    vectors are effectively in RAM, which is the memory the small index was supposed to save.
    """
    config = load_config(paths.PROJECT_ROOT / "configs" / f"{name}.yaml")
    config.model.device = args.device

    start = time.perf_counter()
    engine = SearchEngine(config)
    load_s = time.perf_counter() - start

    n = args.serving_queries
    cold = time_queries(engine, queries[:n])
    rss_cold = rss_bytes()

    time_queries(engine, queries[n : n + args.serving_warmup])  # warm the page cache
    warm = time_queries(engine, queries[n + args.serving_warmup : 2 * n + args.serving_warmup])
    rss_warm = rss_bytes()

    def summarise(stages: dict[str, list[float]]) -> dict:
        percentiles = {stage: bench.stats_ms(times) for stage, times in stages.items()}
        return {
            "stages_ms": percentiles,
            "total_p50_ms": round(sum(p["p50"] for p in percentiles.values()), 2),
        }

    return {
        "name": name,
        "index_type": config.index.type,
        "device": args.device,
        "queries": n,
        "warmup_queries": args.serving_warmup,
        "load_s": round(load_s, 1),
        "sentence_index": engine.has_sentence_index,
        "cold": {**summarise(cold), "rss": bench.human_bytes(rss_cold), "rss_bytes": rss_cold},
        "warm": {**summarise(warm), "rss": bench.human_bytes(rss_warm), "rss_bytes": rss_warm},
    }


# ------------------------------------------------------------------------------- axis: bm25

def run_bm25(qids: list[str], qrels: dict, args) -> dict:
    import bm25s
    from Stemmer import Stemmer

    from fever_search.data_io import iter_jsonl, load_queries

    docs = list(iter_jsonl(paths.CORPUS_PATH))
    doc_ids = np.array([str(d["_id"]) for d in docs])
    corpus = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in docs]

    stemmer = Stemmer("english")
    start = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer))
    build_s = time.perf_counter() - start

    queries_path, _ = paths.benchmark_files(args.benchmark, args.split)
    all_queries = load_queries(queries_path)
    texts = [all_queries[qid] for qid in qids]
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    indices, _ = retriever.retrieve(tokens, k=TOP_K)

    metrics = bench.aggregate_metrics(
        bench.retrieved_from_ids(np.asarray(indices), qids, doc_ids), qrels, K_VALUES)
    return {"variant": "bm25", "build_s": round(build_s, 1), **metrics}


# ----------------------------------------------------------------------------------- report

def format_cell(key: str, value) -> str:
    if not isinstance(value, float):
        return str(value if value is not None else "")
    # Timings read as durations, metrics as scores; four decimals on a millisecond is noise.
    timing = key.endswith(("_ms", "_s")) or key in {"encode", "search", "evidence", "total"}
    return f"{value:.2f}" if timing else f"{value:.4f}"


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Markdown table: columns is [(key, header)]."""
    headers = [header for _, header in columns]
    cells = [[format_cell(key, row.get(key)) for key, _ in columns] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    line = lambda vals: "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep, *(line(c) for c in cells)])


def write_report(payload: dict, out_path: Path) -> None:
    lines = [
        "# Results",
        "",
        f"Corpus: {payload['corpus_docs']:,} passages · benchmark: {payload['benchmark']} "
        f"· {payload['num_queries']:,} queries · model: {payload['model']}",
        "",
        f"Generated by `scripts/bench/benchmark_all.py` on {payload['host']}.",
        "",
    ]

    if payload.get("index"):
        lines += [
            "## Index: quality vs cost",
            "",
            "Search latency is a single query against the 500k index, p50 over repeated calls. "
            "*Resident* is what the process must hold in RAM; *on disk* also counts the fp32 vectors "
            "that binary+rerank re-scores against, which are memmapped rather than loaded.",
            "",
            table(payload["index"], [
                ("index_type", "index"), ("resident", "resident RAM"), ("on_disk", "on disk"),
                ("search_p50_ms", "search p50, ms"), ("precision@1", "P@1"),
                ("ndcg@10", "nDCG@10"), ("recall@100", "R@100"),
            ]),
            "",
        ]

    if payload.get("compression"):
        lines += [
            "## Compression: how the same vectors can be stored",
            "",
            "Built from the same fp32 embeddings, no re-encoding. binary+rerank keeps a 48 MB Hamming "
            "index and re-scores its top-1000 shortlist against the fp32 vectors: those can stay on "
            "disk, which is the difference between the two RAM columns.",
            "",
            table(payload["compression"], [
                ("variant", "variant"), ("in_memory", "RAM, all in memory"),
                ("resident", "RAM, fp32 memmapped"), ("search_p50_ms", "search p50, ms"),
                ("precision@1", "P@1"), ("ndcg@10", "nDCG@10"), ("recall@100", "R@100"),
                ("build_s", "build, s"),
            ]),
            "",
        ]

    if payload.get("bm25"):
        rows = [payload["bm25"]] + [
            {**row, "variant": row["index_type"]} for row in payload.get("index", [])
            if row["name"] == BASELINE
        ]
        lines += [
            "## Lexical baseline",
            "",
            table(rows, [
                ("variant", "retriever"), ("precision@1", "P@1"), ("ndcg@10", "nDCG@10"),
                ("recall@100", "R@100"), ("mrr", "MRR"),
            ]),
            "",
        ]

    if payload.get("serving"):
        rows = []
        for entry in payload["serving"]:
            for cache in ("cold", "warm"):
                stages = entry[cache]["stages_ms"]
                rows.append({
                    "index_type": entry["index_type"],
                    "cache": cache,
                    "encode": stages["encode"]["p50"],
                    "search": stages["search"]["p50"],
                    "evidence": stages["evidence"]["p50"],
                    "total": entry[cache]["total_p50_ms"],
                    "rss": entry[cache]["rss"],
                })
        first = payload["serving"][0]
        lines += [
            "## Serving: the full query path",
            "",
            f"Device: {first['device']}. Each stage is p50 over {first['queries']} distinct real queries; "
            f"*warm* is the same measurement repeated after {first['warmup_queries']} further queries. "
            "*evidence* picks the sentence supporting the claim from precomputed vectors, so it is a "
            "dot product.",
            "",
            "The cold/warm split matters for binary+rerank: its shortlist is re-scored against fp32 "
            "vectors read from disk, so its search time is dominated by page faults until the OS cache "
            "holds them — and once it does, those vectors are in RAM after all. A 48 MB index and a "
            "3 ms search are not simultaneously true on a memory-constrained box.",
            "",
            table(rows, [
                ("index_type", "index"), ("cache", "cache"), ("encode", "encode, ms"),
                ("search", "search, ms"), ("evidence", "evidence, ms"), ("total", "total, ms"),
                ("rss", "process RSS"),
            ]),
            "",
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--axes", default="index,compression,serving",
                        help="comma-separated: index,compression,serving,bm25")
    parser.add_argument("--with-bm25", action="store_true", help="shorthand for adding the bm25 axis")
    parser.add_argument("--device", default="cpu", help="device for the serving axis")
    parser.add_argument("--serving-queries", type=int, default=25)
    parser.add_argument("--serving-warmup", type=int, default=200,
                        help="queries run between the cold and warm serving measurements")
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--rerank-depth", type=int, default=1000)
    parser.add_argument("--lat-warmup", type=int, default=20)
    parser.add_argument("--lat-repeat", type=int, default=100)
    parser.add_argument("--serving-worker", default=None,
                        help="internal: measure one config in this process and print its JSON")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild RESULTS.md from the last benchmark_all.json without re-measuring")
    args = parser.parse_args()

    out_dir = paths.DATA_DIR / "analysis"
    if args.report_only:
        previous = json.loads((out_dir / "benchmark_all.json").read_text(encoding="utf-8"))
        write_report(previous, out_dir / "RESULTS.md")
        return

    if args.serving_worker:
        from fever_search.data_io import load_queries

        queries_path, _ = paths.benchmark_files(args.benchmark, args.split)
        all_queries = load_queries(queries_path)
        qrels_qids = sorted(bench.load_qrels(paths.benchmark_files(args.benchmark, args.split)[1]))
        texts = [all_queries[qid] for qid in qrels_qids if qid in all_queries and all_queries[qid]]
        print("SERVING_JSON " + json.dumps(run_serving(args.serving_worker, texts, args)))
        return

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    if args.with_bm25 and "bm25" not in axes:
        axes.append("bm25")

    base_config = load_config(paths.PROJECT_ROOT / "configs" / f"{BASELINE}.yaml")
    base_dir = paths.index_dir(BASELINE)
    qids, qvecs, qrels = bench.load_query_vectors(base_config, base_dir, args.benchmark, args.split)
    doc_ids = np.array(json.loads((base_dir / "doc_ids.json").read_text(encoding="utf-8")))
    print(f"{args.benchmark}/{args.split}: {len(qids):,} queries, {len(doc_ids):,} docs\n")

    payload: dict = {
        "benchmark": f"{args.benchmark}/{args.split}",
        "num_queries": len(qids),
        "corpus_docs": len(doc_ids),
        "model": base_config.model.name,
        "host": f"{sys.platform}",
    }

    if "index" in axes:
        rows = []
        for name in INDEXES:
            if not (paths.index_dir(name) / "faiss.index").exists():
                print(f"skipping {name}: not built")
                continue
            print(f"=== index / {name} ===")
            row = search_index(name, qvecs, qids, qrels, args)
            rows.append(row)
            print(f"  resident={row['resident']} search_p50={row['search_p50_ms']}ms "
                  f"nDCG@10={row['ndcg@10']:.4f} R@100={row['recall@100']:.4f}")
        payload["index"] = rows

    if "compression" in axes:
        emb = np.load(base_dir / "doc_embeddings.npy").astype(np.float32)
        payload["compression"] = run_compression(emb, doc_ids, qvecs, qids, qrels, args)
        del emb

    if "bm25" in axes:
        print("\n=== bm25 ===")
        payload["bm25"] = run_bm25(qids, qrels, args)
        print(f"  nDCG@10={payload['bm25']['ndcg@10']:.4f} R@100={payload['bm25']['recall@100']:.4f}")

    if "serving" in axes:
        rows = []
        for name in SERVING_CONFIGS:
            if not (paths.index_dir(name) / "faiss.index").exists():
                print(f"skipping serving/{name}: not built")
                continue
            print(f"\n=== serving / {name} (fresh process) ===")
            row = serving_in_subprocess(name, args)
            rows.append(row)
            for cache in ("cold", "warm"):
                stages = row[cache]["stages_ms"]
                print(f"  {cache:4} encode={stages['encode']['p50']}ms search={stages['search']['p50']}ms "
                      f"evidence={stages['evidence']['p50']}ms total={row[cache]['total_p50_ms']}ms "
                      f"RSS={row[cache]['rss']}")
        payload["serving"] = rows

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_all.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRaw numbers -> {out_dir / 'benchmark_all.json'}")
    write_report(payload, out_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
