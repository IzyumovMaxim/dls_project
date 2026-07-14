"""Shrink the encoder and its vectors, driven by an analysis of the vectors themselves.

Three stages, each answering a different part of "make inference cheaper without losing quality":

  spectrum  Analyse the generated vectors: PCA eigenspectrum of doc_embeddings.npy. How many of
            the 768 dimensions actually carry variance? This is the evidence for stage `pca`.
  pca       Act on it: project docs+queries to d' dims, renormalise, rebuild flat IP, and measure
            quality / RAM / search latency at each d'. Vector storage shrinks linearly with d'.
  quantize  Speed up encoder inference: dynamic int8 on the Linear layers. Queries are re-encoded
            with the quantized model and searched against the *unchanged* fp32 doc index, which is
            how it would be served — so the quality delta reported here is the real one.
  prune     Drop the top transformer layers. This changes the embedding space, so the corpus has to
            be re-encoded to evaluate it honestly; needs --reencode and a GPU to be practical.

    python scripts/shrink_model.py                       # spectrum + pca + quantize
    python scripts/shrink_model.py --stages spectrum
    python scripts/shrink_model.py --stages prune --reencode --keep-layers 6,8,10

Output: data/analysis/shrink_model_<benchmark>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import bench, paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.data_io import doc_to_passage, iter_jsonl  # noqa: E402
from fever_search.encoder import Encoder  # noqa: E402

K_VALUES = (1, 5, 10, 100)
TOP_K = 100


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def evaluate_vectors(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    qids: list[str],
    doc_ids: np.ndarray,
    qrels: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> dict:
    """Flat IP index over the given vectors -> quality + RAM + search p50."""
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(np.ascontiguousarray(doc_vectors, dtype=np.float32))
    memory = bench.index_num_bytes(index)

    _, indices = index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), TOP_K)
    metrics = bench.aggregate_metrics(bench.retrieved_from_ids(indices, qids, doc_ids), qrels, K_VALUES)
    p50 = bench.time_calls(
        lambda: index.search(query_vectors[:1].astype(np.float32), TOP_K), lat_warmup, lat_repeat
    )["p50"]
    return {
        "dim": int(doc_vectors.shape[1]),
        "memory": bench.human_bytes(memory),
        "memory_bytes": memory,
        "search_p50_ms": p50,
        **metrics,
    }


# --------------------------------------------------------------------------------------- spectrum

def run_spectrum(emb: np.ndarray, sample_size: int, seed: int) -> dict:
    """PCA eigenspectrum of the document vectors: where does the variance actually live?"""
    rng = np.random.default_rng(seed)
    n = min(sample_size, emb.shape[0])
    sample = emb[rng.choice(emb.shape[0], n, replace=False)].astype(np.float32)
    sample -= sample.mean(axis=0, keepdims=True)

    singular = np.linalg.svd(sample, compute_uv=False)
    variance = singular**2
    explained = variance / variance.sum()
    cumulative = np.cumsum(explained)

    # Dimensions needed to retain X% of the variance -- the whole argument for cutting dims.
    thresholds = {f"dims_for_{int(p * 100)}pct": int(np.searchsorted(cumulative, p) + 1)
                  for p in (0.80, 0.90, 0.95, 0.99)}
    # Participation ratio: a basis-free "effective dimensionality" of the cloud.
    effective_dim = float(variance.sum() ** 2 / (variance**2).sum())

    print(f"\n=== spectrum ({n:,} sampled vectors, {emb.shape[1]} dims) ===")
    for name, value in thresholds.items():
        print(f"  {name.replace('_', ' ')}: {value}")
    print(f"  effective dimensionality (participation ratio): {effective_dim:.1f}")
    for d in (64, 128, 256, 384, 512):
        if d < emb.shape[1]:
            print(f"  variance retained by top {d:>3} dims: {cumulative[d - 1]:.4f}")

    return {
        "sampled_vectors": n,
        "input_dim": int(emb.shape[1]),
        "effective_dim_participation_ratio": round(effective_dim, 1),
        **thresholds,
        "cumulative_explained_variance": [round(float(x), 6) for x in cumulative],
    }


# -------------------------------------------------------------------------------------------- pca

def run_pca(
    emb: np.ndarray,
    qvecs: np.ndarray,
    qids: list[str],
    doc_ids: np.ndarray,
    qrels: dict,
    dims: list[int],
    baseline: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> list[dict]:
    """Project to each target dim with faiss PCA, renormalise, evaluate."""
    rows = []
    for dim in dims:
        if dim >= emb.shape[1]:
            print(f"  skipping d'={dim}: not smaller than {emb.shape[1]}")
            continue
        print(f"\n=== pca / d'={dim} ===")
        t0 = time.perf_counter()
        pca = faiss.PCAMatrix(emb.shape[1], dim)
        pca.train(emb)
        # Renormalise: the index scores inner products, which only equal cosine on unit vectors.
        doc_reduced = normalize(pca.apply(emb))
        query_reduced = normalize(pca.apply(np.ascontiguousarray(qvecs, dtype=np.float32)))
        fit_s = time.perf_counter() - t0

        row = evaluate_vectors(doc_reduced, query_reduced, qids, doc_ids, qrels, lat_warmup, lat_repeat)
        row["stage"] = "pca"
        row["fit_s"] = round(fit_s, 1)
        row["d_ndcg@10"] = round(row["ndcg@10"] - baseline["ndcg@10"], 4)
        row["d_recall@100"] = round(row["recall@100"] - baseline["recall@100"], 4)
        row["compression"] = f"{emb.shape[1] / dim:.1f}x"
        rows.append(row)
        print(
            f"  RAM={row['memory']} search_p50={row['search_p50_ms']}ms "
            f"P@1={row['precision@1']:.4f} nDCG@10={row['ndcg@10']:.4f} "
            f"(Δ{row['d_ndcg@10']:+.4f}) R@100={row['recall@100']:.4f}"
        )
    return rows


# --------------------------------------------------------------------------------- encoder surgery

def encode_latency(encoder: Encoder, texts: list[str], warmup: int, repeat: int) -> dict:
    """Single-query encode latency -- the encoder cost actually paid per search."""
    one = texts[:1]
    return bench.time_calls(lambda: encoder.encode_queries(one), warmup, repeat)


def quantize_encoder(encoder: Encoder) -> Encoder:
    """Dynamic int8 on Linear layers: weights int8, activations quantized on the fly."""
    import torch

    shrunk = copy.copy(encoder)
    shrunk.model = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(encoder.model), {torch.nn.Linear}, dtype=torch.qint8
    )
    return shrunk


def model_num_bytes(encoder: Encoder) -> int:
    import torch

    total = 0
    for param in encoder.model.state_dict().values():
        if isinstance(param, torch.Tensor):
            total += param.numel() * param.element_size()
    return total


def run_quantize(
    encoder: Encoder,
    emb: np.ndarray,
    query_texts: list[str],
    qids: list[str],
    doc_ids: np.ndarray,
    qrels: dict,
    baseline: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> list[dict]:
    """int8 query encoder against the unchanged fp32 doc index -- the serving configuration."""
    rows = []
    fp32_bytes = model_num_bytes(encoder)
    fp32_latency = encode_latency(encoder, query_texts, lat_warmup, lat_repeat)
    print(f"\n=== quantize / fp32 baseline ===")
    print(f"  weights={bench.human_bytes(fp32_bytes)}  encode_p50={fp32_latency['p50']}ms")

    print(f"\n=== quantize / int8 dynamic ===")
    shrunk = quantize_encoder(encoder)
    int8_bytes = model_num_bytes(shrunk)
    int8_latency = encode_latency(shrunk, query_texts, lat_warmup, lat_repeat)

    print(f"  re-encoding {len(query_texts):,} queries with the int8 encoder ...")
    qvecs_int8 = shrunk.encode_queries(query_texts, show_progress_bar=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _, indices = index.search(np.ascontiguousarray(qvecs_int8, dtype=np.float32), TOP_K)
    metrics = bench.aggregate_metrics(bench.retrieved_from_ids(indices, qids, doc_ids), qrels, K_VALUES)

    for label, weights, latency, quality in (
        ("fp32", fp32_bytes, fp32_latency, baseline),
        ("int8_dynamic", int8_bytes, int8_latency, metrics),
    ):
        rows.append({
            "stage": "quantize",
            "variant": label,
            "weights": bench.human_bytes(weights),
            "weights_bytes": weights,
            "encode_p50_ms": latency["p50"],
            "encode_p95_ms": latency["p95"],
            "speedup": round(fp32_latency["p50"] / latency["p50"], 2),
            "d_ndcg@10": round(quality["ndcg@10"] - baseline["ndcg@10"], 4),
            **{k: quality[k] for k in ("precision@1", "ndcg@10", "recall@100")},
        })

    row = rows[-1]
    print(
        f"  weights={row['weights']} ({fp32_bytes / int8_bytes:.1f}x smaller)  "
        f"encode_p50={row['encode_p50_ms']}ms ({row['speedup']}x faster)  "
        f"nDCG@10={row['ndcg@10']:.4f} (Δ{row['d_ndcg@10']:+.4f})"
    )
    return rows


def run_prune(
    config,
    keep_layers: list[int],
    query_texts: list[str],
    qids: list[str],
    doc_ids: np.ndarray,
    qrels: dict,
    baseline: dict,
    lat_warmup: int,
    lat_repeat: int,
) -> list[dict]:
    """Truncate top transformer layers. Changes the vector space -> the corpus is re-encoded."""
    passages = [doc_to_passage(doc) for doc in iter_jsonl(paths.CORPUS_PATH)]
    print(f"\nRe-encoding {len(passages):,} passages per pruned variant — this is the expensive path.")

    rows = []
    for keep in keep_layers:
        print(f"\n=== prune / keep {keep} layers ===")
        encoder = Encoder(config.model)
        layers = encoder.model[0].auto_model.encoder.layer
        if keep >= len(layers):
            print(f"  skipping: model has {len(layers)} layers")
            continue
        encoder.model[0].auto_model.encoder.layer = layers[:keep]

        t0 = time.perf_counter()
        doc_vectors = encoder.encode_documents(passages, show_progress_bar=True)
        encode_s = time.perf_counter() - t0
        query_vectors = encoder.encode_queries(query_texts, show_progress_bar=True)

        row = evaluate_vectors(doc_vectors, query_vectors, qids, doc_ids, qrels, lat_warmup, lat_repeat)
        row["stage"] = "prune"
        row["kept_layers"] = keep
        row["weights"] = bench.human_bytes(model_num_bytes(encoder))
        row["encode_p50_ms"] = encode_latency(encoder, query_texts, lat_warmup, lat_repeat)["p50"]
        row["corpus_encode_s"] = round(encode_s, 1)
        row["d_ndcg@10"] = round(row["ndcg@10"] - baseline["ndcg@10"], 4)
        rows.append(row)
        print(
            f"  weights={row['weights']}  encode_p50={row['encode_p50_ms']}ms  "
            f"nDCG@10={row['ndcg@10']:.4f} (Δ{row['d_ndcg@10']:+.4f})"
        )
    return rows


# ------------------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5_base_flat.yaml")
    parser.add_argument("--index-dir", default=str(paths.index_dir("e5_base_flat")))
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--stages", default="spectrum,pca,quantize",
                        help="comma-separated: spectrum,pca,quantize,prune")
    parser.add_argument("--dims", default="64,128,256,384,512", help="pca target dims")
    parser.add_argument("--keep-layers", default="6,8,10", help="prune: transformer layers to keep")
    parser.add_argument("--reencode", action="store_true",
                        help="required for --stages prune: re-encodes the whole corpus")
    parser.add_argument("--encoder-device", default="cpu",
                        help="device for the quantize/prune stages; dynamic int8 is CPU-only in torch, "
                             "and CPU is also the honest baseline for single-query serving latency")
    parser.add_argument("--sample-size", type=int, default=50000, help="vectors sampled for the spectrum")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="subsample queries (quantize/prune re-encode them on CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat-warmup", type=int, default=10)
    parser.add_argument("--lat-repeat", type=int, default=50)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "prune" in stages and not args.reencode:
        raise SystemExit("--stages prune re-encodes the 500k corpus; pass --reencode to confirm.")

    config = load_config(args.config)
    index_dir = Path(args.index_dir)
    emb = np.load(index_dir / "doc_embeddings.npy").astype(np.float32)
    doc_ids = np.array(json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8")))
    print(f"Embeddings: {emb.shape[0]:,} x {emb.shape[1]} ({bench.human_bytes(emb.nbytes)})")

    qids, qvecs, qrels = bench.load_query_vectors(config, index_dir, args.benchmark, args.split)
    if args.max_queries:
        qids, qvecs = qids[: args.max_queries], qvecs[: args.max_queries]
        qrels = {qid: qrels[qid] for qid in qids}
    print(f"Benchmark: {args.benchmark}/{args.split} — {len(qids):,} queries")

    print("\n=== baseline / fp32, 768 dims ===")
    baseline = evaluate_vectors(emb, qvecs, qids, doc_ids, qrels, args.lat_warmup, args.lat_repeat)
    baseline["stage"] = "baseline"
    print(
        f"  RAM={baseline['memory']} search_p50={baseline['search_p50_ms']}ms "
        f"P@1={baseline['precision@1']:.4f} nDCG@10={baseline['ndcg@10']:.4f} "
        f"R@100={baseline['recall@100']:.4f}"
    )

    payload: dict = {
        "config": config.name,
        "model": config.model.name,
        "benchmark": f"{args.benchmark}/{args.split}",
        "num_queries": len(qids),
        "baseline": baseline,
    }

    if "spectrum" in stages:
        payload["spectrum"] = run_spectrum(emb, args.sample_size, args.seed)

    if "pca" in stages:
        dims = [int(d) for d in args.dims.split(",") if d.strip()]
        payload["pca"] = run_pca(emb, qvecs, qids, doc_ids, qrels, dims, baseline,
                                 args.lat_warmup, args.lat_repeat)

    if "quantize" in stages or "prune" in stages:
        from fever_search.data_io import load_queries

        queries_path, _ = paths.benchmark_files(args.benchmark, args.split)
        all_queries = load_queries(queries_path)
        query_texts = [all_queries[qid] for qid in qids]

        # The config's device targets the training box; these stages measure serving-time encode
        # cost, and torch's dynamic int8 kernels are CPU-only regardless.
        config.model.device = args.encoder_device
        print(f"\nEncoder stages on device: {config.model.device}")

        if "quantize" in stages:
            encoder = Encoder(config.model)
            payload["quantize"] = run_quantize(
                encoder, emb, query_texts, qids, doc_ids, qrels, baseline,
                args.lat_warmup, args.lat_repeat,
            )
        if "prune" in stages:
            keep = [int(k) for k in args.keep_layers.split(",") if k.strip()]
            payload["prune"] = run_prune(
                config, keep, query_texts, qids, doc_ids, qrels, baseline,
                args.lat_warmup, args.lat_repeat,
            )

    out = paths.DATA_DIR / "analysis" / f"shrink_model_{args.benchmark}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
