"""
Evaluate bge-small-en-v1.5 retrieval on FEVER test split.

Run from dls_project root:
    python scripts/test_bge_small_en_v1.5.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SEARCH_MODULE_PATH = SCRIPT_DIR / "query_search_bge_small_en_v1.5.py"

QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "queries.jsonl"
QRELS_TEST_PATH = PROJECT_ROOT / "data" / "qrels" / "qrels_test.tsv"
QUALITY_DIR = PROJECT_ROOT / "data" / "quality"
FIGURES_DIR = QUALITY_DIR / "figures"

EVAL_TOP_K = 100
K_VALUES = (1, 5, 10, 100)
METRIC_NAMES = ("precision", "recall", "mrr", "ndcg")


def load_search_module():
    spec = importlib.util.spec_from_file_location("query_search_bge", SEARCH_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {SEARCH_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_test_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with QRELS_TEST_PATH.open(encoding="utf-8") as file:
        next(file)
        for line in file:
            query_id, corpus_id, _ = line.rstrip("\n").split("\t")
            qrels[query_id].add(corpus_id)
    return dict(qrels)


def load_test_queries(qrels: dict[str, set[str]]) -> dict[str, str]:
    test_ids = set(qrels)
    queries: dict[str, str] = {}
    with QUERIES_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in test_ids:
                queries[query_id] = str(row.get("text") or "").strip()
    return queries


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "std": round(float(arr.std()), 4),
    }


def plot_metric_histogram(
    values: list[float],
    title: str,
    xlabel: str,
    filename: str,
    summary: dict[str, float],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(summary["mean"], color="orange", linestyle="--", label=f"mean={summary['mean']:.3f}")
    ax.axvline(summary["median"], color="red", linestyle="--", label=f"median={summary['median']:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Queries")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mean_metrics_bar(summary: dict[str, Any]) -> Path:
    labels: list[str] = []
    values: list[float] = []

    for k in K_VALUES:
        labels.append(f"P@{k}")
        values.append(summary["precision"][f"@{k}"]["mean"])
    for k in K_VALUES:
        labels.append(f"R@{k}")
        values.append(summary["recall"][f"@{k}"]["mean"])
    labels.extend(["MRR", "nDCG@10"])
    values.extend([summary["mrr"]["mean"], summary["ndcg"]["@10"]["mean"]])

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="seagreen", edgecolor="white")
    ax.set_title("Mean retrieval metrics on FEVER test")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "mean_metrics_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_by_k(metric_name: str, per_k_summaries: dict[str, dict[str, float]]) -> Path:
    labels = [f"@{k}" for k in K_VALUES]
    means = [per_k_summaries[f"@{k}"]["mean"] for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, means, marker="o", linewidth=2, color="darkorange")
    ax.set_title(f"Mean {metric_name} by K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    path = FIGURES_DIR / f"{metric_name}_by_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    for path in (QUERIES_PATH, QRELS_TEST_PATH):
        if not path.exists():
            print(f"Missing file: {path}. Run scripts/create_test.py first.")
            sys.exit(1)

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    search_mod = load_search_module()
    engine = search_mod.get_search_engine()

    qrels = load_test_qrels()
    queries = load_test_queries(qrels)
    if len(queries) != len(qrels):
        missing = set(qrels) - set(queries)
        print(f"WARNING: {len(missing)} test query ids missing from queries.jsonl")

    per_query: list[dict[str, Any]] = []
    metric_values: dict[str, dict[str, list[float]]] = {
        "precision": {f"@{k}": [] for k in K_VALUES},
        "recall": {f"@{k}": [] for k in K_VALUES},
        "mrr": {"@100": []},
        "ndcg": {"@10": []},
    }

    print("=" * 60)
    print("FEVER test evaluation — bge-small-en-v1.5")
    print("=" * 60)
    print(f"Test queries : {len(queries):,}")
    print(f"Search top-K : {EVAL_TOP_K}")
    print()

    for query_id in tqdm(sorted(queries, key=int), desc="Evaluating", unit="query"):
        query_text = queries[query_id]
        relevant = qrels[query_id]
        hits = search_mod.vector_search(query_text, top_k=EVAL_TOP_K)
        retrieved = [hit.doc_id for hit in hits]

        row: dict[str, Any] = {
            "query_id": query_id,
            "query_text": query_text,
            "num_relevant": len(relevant),
            "retrieved_ids": retrieved,
        }

        for k in K_VALUES:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            metric_values["precision"][f"@{k}"].append(p)
            metric_values["recall"][f"@{k}"].append(r)
            row[f"precision@{k}"] = round(p, 4)
            row[f"recall@{k}"] = round(r, 4)

        mrr_score = mrr(retrieved, relevant)
        ndcg_score = ndcg_at_k(retrieved, relevant, 10)
        metric_values["mrr"]["@100"].append(mrr_score)
        metric_values["ndcg"]["@10"].append(ndcg_score)
        row["mrr"] = round(mrr_score, 4)
        row["ndcg@10"] = round(ndcg_score, 4)
        per_query.append(row)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": engine.manifest.get("model_name", "BAAI/bge-small-en-v1.5"),
        "index_type": engine.manifest.get("index_type", "IndexFlatIP"),
        "eval_top_k": EVAL_TOP_K,
        "num_queries": len(per_query),
        "precision": {key: summarize(vals) for key, vals in metric_values["precision"].items()},
        "recall": {key: summarize(vals) for key, vals in metric_values["recall"].items()},
        "mrr": summarize(metric_values["mrr"]["@100"]),
        "ndcg": {"@10": summarize(metric_values["ndcg"]["@10"])},
    }

    figures = [
        plot_mean_metrics_bar(summary),
        plot_metric_by_k("recall", summary["recall"]),
        plot_metric_by_k("precision", summary["precision"]),
        plot_metric_histogram(
            metric_values["mrr"]["@100"],
            "MRR distribution (per query)",
            "MRR",
            "mrr_histogram.png",
            summary["mrr"],
        ),
        plot_metric_histogram(
            metric_values["recall"]["@10"],
            "Recall@10 distribution (per query)",
            "Recall@10",
            "recall_at_10_histogram.png",
            summary["recall"]["@10"],
        ),
        plot_metric_histogram(
            metric_values["precision"]["@10"],
            "Precision@10 distribution (per query)",
            "Precision@10",
            "precision_at_10_histogram.png",
            summary["precision"]["@10"],
        ),
        plot_metric_histogram(
            metric_values["ndcg"]["@10"],
            "nDCG@10 distribution (per query)",
            "nDCG@10",
            "ndcg_at_10_histogram.png",
            summary["ndcg"]["@10"],
        ),
    ]

    report = {
        "summary": summary,
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
    }

    report_path = QUALITY_DIR / "report.json"
    per_query_path = QUALITY_DIR / "per_query_metrics.jsonl"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with per_query_path.open("w", encoding="utf-8") as file:
        for row in per_query:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("Mean metrics:")
    for k in K_VALUES:
        print(f"  Precision@{k}: {summary['precision'][f'@{k}']['mean']:.4f}")
        print(f"  Recall@{k}   : {summary['recall'][f'@{k}']['mean']:.4f}")
    print(f"  MRR          : {summary['mrr']['mean']:.4f}")
    print(f"  nDCG@10      : {summary['ndcg']['@10']['mean']:.4f}")
    print()
    print(f"Report : {report_path}")
    print(f"Per-q  : {per_query_path}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
