"""
FEVER dataset analytics: corpus, queries, qrels stats, figures, README, hash.

Run from dls_project root:
    python scripts/data/analyze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CORPUS_PATH = DATA_DIR / "corpus" / "fever_500k.jsonl"
QUERIES_PATH = DATA_DIR / "queries" / "queries.jsonl"
QRELS_DIR = DATA_DIR / "qrels"
QRELS_SPLITS = ("train", "validation", "test")

REPORT_PATH = ANALYSIS_DIR / "report.json"
README_PATH = ANALYSIS_DIR / "README.md"
HASH_PATH = ANALYSIS_DIR / "hash.md"

TEXT_LENGTH_BINS = 50
TEXT_LENGTH_PLOT_MAX = 5000
REL_DOCS_PLOT_MAX = 15


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    arr = np.array(lengths, dtype=np.int64)
    return {
        "count": len(lengths),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(arr.std()), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def load_qrels_split(split: str) -> list[dict[str, Any]]:
    path = QRELS_DIR / f"qrels_{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Qrels file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        header = file.readline().strip().split("\t")
        for line in file:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(header, values))
            row["query-id"] = int(row["query-id"])
            row["score"] = int(row["score"])
            rows.append(row)
    return rows


def analyze_qrels() -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    all_pairs = 0
    all_query_ids: set[int] = set()
    per_query_all: Counter[int] = Counter()

    for split in QRELS_SPLITS:
        rows = load_qrels_split(split)
        per_query = Counter(row["query-id"] for row in rows)
        corpus_ids = {row["corpus-id"] for row in rows}

        rel_counts = list(per_query.values())
        split_stats[split] = {
            "pairs": len(rows),
            "unique_queries": len(per_query),
            "unique_corpus_ids": len(corpus_ids),
            "rel_docs_per_query": {
                **length_summary(rel_counts),
                "distribution": dict(sorted(Counter(rel_counts).items())),
            },
        }

        all_pairs += len(rows)
        all_query_ids.update(per_query)
        per_query_all.update(per_query)

    overall_rel_counts = list(per_query_all.values())
    return {
        "splits": split_stats,
        "overall": {
            "pairs": all_pairs,
            "unique_queries": len(all_query_ids),
            "unique_corpus_ids": sum(
                split_stats[split]["unique_corpus_ids"] for split in QRELS_SPLITS
            ),
            "rel_docs_per_query": length_summary(overall_rel_counts),
        },
    }


def analyze_queries() -> dict[str, Any]:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Queries file not found: {QUERIES_PATH}")

    count = 0
    lengths: list[int] = []
    query_ids: set[int] = set()

    with QUERIES_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            count += 1
            query_ids.add(int(row["_id"]))
            lengths.append(len(str(row.get("text") or "")))

    split_query_ids: dict[str, set[int]] = {}
    for split in QRELS_SPLITS:
        rows = load_qrels_split(split)
        split_query_ids[split] = {row["query-id"] for row in rows}

    unmatched = query_ids.copy()
    for ids in split_query_ids.values():
        unmatched -= ids

    return {
        "total": count,
        "unique_ids": len(query_ids),
        "text_length": length_summary(lengths),
        "queries_per_split": {
            split: len(ids) for split, ids in split_query_ids.items()
        },
        "queries_not_in_any_qrels_split": len(unmatched),
    }


def analyze_corpus(test_qrels_doc_ids: set[str]) -> dict[str, Any] | None:
    if not CORPUS_PATH.exists():
        return None

    lengths: list[int] = []
    qrels_lengths: list[int] = []
    filler_lengths: list[int] = []
    doc_ids: set[str] = set()

    with CORPUS_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            doc_id = str(row["_id"])
            doc_ids.add(doc_id)
            text_len = len(str(row.get("text") or ""))
            lengths.append(text_len)
            if doc_id in test_qrels_doc_ids:
                qrels_lengths.append(text_len)
            else:
                filler_lengths.append(text_len)

    test_qrels_in_corpus = test_qrels_doc_ids & doc_ids
    return {
        "path": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "document_count": len(lengths),
        "unique_ids": len(doc_ids),
        "file_size_mb": round(CORPUS_PATH.stat().st_size / (1024 * 1024), 2),
        "text_length": {
            "all": length_summary(lengths),
            "test_qrels_docs": length_summary(qrels_lengths),
            "filler_docs": length_summary(filler_lengths),
        },
        "test_qrels_coverage": {
            "expected_test_qrels_docs": len(test_qrels_doc_ids),
            "present_in_corpus": len(test_qrels_in_corpus),
            "missing_in_corpus": sorted(test_qrels_doc_ids - doc_ids),
        },
    }


def plot_corpus_text_length(lengths: list[int], stats: dict[str, Any]) -> Path:
    capped = [min(length, TEXT_LENGTH_PLOT_MAX) for length in lengths]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(capped, bins=TEXT_LENGTH_BINS, color="steelblue", edgecolor="white")
    ax.set_title("Corpus: text length distribution")
    ax.set_xlabel(f"Characters (capped at {TEXT_LENGTH_PLOT_MAX})")
    ax.set_ylabel("Documents")
    ax.axvline(stats["median"], color="red", linestyle="--", label=f"median={stats['median']}")
    ax.axvline(stats["mean"], color="orange", linestyle="--", label=f"mean={stats['mean']}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "corpus_text_length_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_qrels_vs_filler_lengths(qrels_lengths: list[int], filler_lengths: list[int]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, TEXT_LENGTH_PLOT_MAX, TEXT_LENGTH_BINS + 1)
    ax.hist(
        [min(length, TEXT_LENGTH_PLOT_MAX) for length in qrels_lengths],
        bins=bins,
        alpha=0.7,
        label=f"test qrels docs ({len(qrels_lengths)})",
        color="seagreen",
    )
    ax.hist(
        [min(length, TEXT_LENGTH_PLOT_MAX) for length in filler_lengths],
        bins=bins,
        alpha=0.5,
        label=f"filler docs ({len(filler_lengths)})",
        color="gray",
    )
    ax.set_title("Corpus: test qrels docs vs filler")
    ax.set_xlabel(f"Characters (capped at {TEXT_LENGTH_PLOT_MAX})")
    ax.set_ylabel("Documents")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "corpus_qrels_vs_filler_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rel_docs_per_query(qrels_stats: dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(1, REL_DOCS_PLOT_MAX + 1)

    for idx, split in enumerate(QRELS_SPLITS):
        dist = qrels_stats["splits"][split]["rel_docs_per_query"]["distribution"]
        counts = [int(dist.get(str(n), dist.get(n, 0))) for n in x]
        ax.bar(x + (idx - 1) * width, counts, width=width, label=split)

    ax.set_title("Qrels: relevant documents per query")
    ax.set_xlabel("Relevant docs per query")
    ax.set_ylabel("Number of queries")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "qrels_rel_docs_per_query.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_query_text_length(lengths: list[int], stats: dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=40, color="mediumpurple", edgecolor="white")
    ax.set_title("Queries: claim text length distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Queries")
    ax.axvline(stats["median"], color="red", linestyle="--", label=f"median={stats['median']}")
    ax.axvline(stats["mean"], color="orange", linestyle="--", label=f"mean={stats['mean']}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "query_text_length_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_readme(report: dict[str, Any]) -> str:
    corpus = report.get("corpus")
    corpus_line = (
        f"- Local 500k slice: `{corpus['path']}` — **{corpus['document_count']:,}** documents"
        if corpus
        else "- Local 500k slice: not built yet (`data/corpus/fever_500k.jsonl`)"
    )

    return f"""# FEVER / BEIR — аналитика датасета

Сгенерировано: {report['generated_at_utc']}

## Что это за датасет

**FEVER** (Fact Extraction and VERification) в упаковке **BEIR** — задача retrieval:
по claim (утверждению) найти в Wikipedia пассажи-доказательства.

Проект использует три компонента:

| Компонент | Файл | Описание |
|-----------|------|----------|
| Corpus | `data/corpus/fever_500k.jsonl` | Пассажи Wikipedia для поиска |
| Queries | `data/queries/queries.jsonl` | Все claim'ы ({report['queries']['total']:,}) |
| Qrels | `data/qrels/qrels_*.tsv` | Разметка: какой query → какой doc релевантен |

Источник на HuggingFace: `BeIR/fever`, `BeIR/fever-qrels`.

## Формат данных

### Corpus (JSONL)

Одна строка = один документ:

```json
{{"_id": "Albert_Einstein", "title": "Albert Einstein", "text": "..."}}
```

- `_id` — id пассажа/статьи
- `title` — заголовок Wikipedia-статьи
- `text` — текст пассажа

{corpus_line}

### Queries (JSONL)

```json
{{"_id": "75397", "title": "", "text": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."}}
```

- `_id` — id claim'а
- `text` — сам claim
- **Сплитов в файле нет** — train/validation/test задаются через qrels

### Qrels (TSV)

```tsv
query-id\tcorpus-id\tscore
163803\tUkrainian_Soviet_Socialist_Republic\t1
```

- `query-id` — id из `queries._id`
- `corpus-id` — `_id` из corpus
- `score` — всегда `1` (релевантен)

Сплиты не пересекаются по query-id:

| Split | Queries | Qrels pairs | Avg rel docs / query |
|-------|---------|-------------|----------------------|
| train | {report['qrels']['splits']['train']['unique_queries']:,} | {report['qrels']['splits']['train']['pairs']:,} | {report['qrels']['splits']['train']['rel_docs_per_query']['mean']} |
| validation | {report['qrels']['splits']['validation']['unique_queries']:,} | {report['qrels']['splits']['validation']['pairs']:,} | {report['qrels']['splits']['validation']['rel_docs_per_query']['mean']} |
| test | {report['qrels']['splits']['test']['unique_queries']:,} | {report['qrels']['splits']['test']['pairs']:,} | {report['qrels']['splits']['test']['rel_docs_per_query']['mean']} |

## Как с этим работать

1. **Собрать данные**
   - `python scripts/data/export_fever.py` — выгрузить queries и qrels
   - `python scripts/data/build_corpus.py` — собрать корпус 500k

2. **Взять test queries для eval**

```python
import json

test_qids = set()
with open("data/qrels/qrels_test.tsv") as f:
    next(f)
    for line in f:
        qid, _, _ = line.strip().split("\\t")
        test_qids.add(int(qid))

test_queries = []
with open("data/queries/queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        if int(q["_id"]) in test_qids:
            test_queries.append(q)
```

3. **Retrieval eval** — для каждого test query найти top-k документов в корпусе,
   метрики считать по `qrels_test.tsv` (nDCG@10, Recall@k, MRR).

4. **Важно для корпуса 500k** — в срез должны попасть все документы из `qrels_test`,
   иначе метрики будут занижены.

## Файлы в этой папке

- `report.json` — числовая сводка
- `hash.md` — SHA256 корпуса
- `figures/` — графики распределений
"""


def build_hash_md(corpus_path: Path) -> str:
    if not corpus_path.exists():
        return (
            "# Corpus hash\n\n"
            "Corpus file not found. Run `python scripts/data/build_corpus.py` first.\n"
        )

    digest = sha256_of_file(corpus_path)
    size_mb = corpus_path.stat().st_size / (1024 * 1024)
    return (
        "# Corpus hash\n\n"
        f"File: `{corpus_path.relative_to(PROJECT_ROOT)}`\n\n"
        f"SHA256: `{digest}`\n\n"
        f"Size: {size_mb:.2f} MB\n"
    )


def print_summary(report: dict[str, Any]) -> None:
    print("=" * 60)
    print("FEVER analytics")
    print("=" * 60)

    queries = report["queries"]
    print(f"\nQueries: {queries['total']:,}")
    print(f"  mean claim length: {queries['text_length']['mean']} chars")

    qrels = report["qrels"]
    print(f"\nQrels overall:")
    print(f"  pairs: {qrels['overall']['pairs']:,}")
    print(f"  unique queries: {qrels['overall']['unique_queries']:,}")
    print(f"  avg rel docs/query: {qrels['overall']['rel_docs_per_query']['mean']}")

    for split in QRELS_SPLITS:
        s = qrels["splits"][split]
        print(f"\nQrels/{split}:")
        print(f"  pairs: {s['pairs']:,}")
        print(f"  unique queries: {s['unique_queries']:,}")
        print(f"  unique corpus ids: {s['unique_corpus_ids']:,}")
        print(f"  avg rel docs/query: {s['rel_docs_per_query']['mean']}")

    corpus = report.get("corpus")
    if corpus:
        print(f"\nCorpus: {corpus['document_count']:,} docs ({corpus['file_size_mb']} MB)")
        print(f"  mean text length: {corpus['text_length']['all']['mean']} chars")
        print(f"  test qrels docs in corpus: {corpus['test_qrels_coverage']['present_in_corpus']:,}"
              f" / {corpus['test_qrels_coverage']['expected_test_qrels_docs']:,}")
    else:
        print("\nCorpus: not found (skipped corpus stats)")

    print(f"\nOutput: {ANALYSIS_DIR}")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not QUERIES_PATH.exists() or not (QRELS_DIR / "qrels_test.tsv").exists():
        print("Run scripts/data/export_fever.py first to export queries and qrels.")
        sys.exit(1)

    test_qrels_rows = load_qrels_split("test")
    test_qrels_doc_ids = {row["corpus-id"] for row in test_qrels_rows}

    qrels_stats = analyze_qrels()
    queries_stats = analyze_queries()
    corpus_stats = analyze_corpus(test_qrels_doc_ids)

    figures: list[str] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    query_lengths: list[int] = []
    with QUERIES_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            query_lengths.append(len(str(row.get("text") or "")))
    figures.append(
        str(plot_query_text_length(query_lengths, queries_stats["text_length"]).relative_to(PROJECT_ROOT))
    )
    figures.append(
        str(plot_rel_docs_per_query(qrels_stats).relative_to(PROJECT_ROOT))
    )

    if corpus_stats and CORPUS_PATH.exists():
        lengths: list[int] = []
        qrels_lengths: list[int] = []
        filler_lengths: list[int] = []
        with CORPUS_PATH.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                text_len = len(str(row.get("text") or ""))
                lengths.append(text_len)
                if str(row["_id"]) in test_qrels_doc_ids:
                    qrels_lengths.append(text_len)
                else:
                    filler_lengths.append(text_len)

        figures.append(
            str(plot_corpus_text_length(lengths, corpus_stats["text_length"]["all"]).relative_to(PROJECT_ROOT))
        )
        if qrels_lengths and filler_lengths:
            figures.append(
                str(plot_qrels_vs_filler_lengths(qrels_lengths, filler_lengths).relative_to(PROJECT_ROOT))
            )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": queries_stats,
        "qrels": qrels_stats,
        "corpus": corpus_stats,
        "figures": figures,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    README_PATH.write_text(build_readme(report), encoding="utf-8")
    HASH_PATH.write_text(build_hash_md(CORPUS_PATH), encoding="utf-8")

    print_summary(report)


if __name__ == "__main__":
    main()
