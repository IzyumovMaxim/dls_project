import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_JSONL_PATH = PROJECT_ROOT / "data" / "queries" / "queries.jsonl"
QRELS_DIR = PROJECT_ROOT / "data" / "qrels"


queries = load_dataset("BeIR/fever", "queries", split="queries")
QUERIES_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

with QUERIES_JSONL_PATH.open("w", encoding="utf-8") as f:
    for row in queries:
        f.write(json.dumps({
            "_id": str(row["_id"]),
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or "").strip(),
        }, ensure_ascii=False) + "\n")

print(f"queries: {len(queries):,} -> {QUERIES_JSONL_PATH}")


QRELS_DIR.mkdir(parents=True, exist_ok=True)

QRELS_OUTPUTS = {
    "train": QRELS_DIR / "qrels_train.tsv",
    "validation": QRELS_DIR / "qrels_validation.tsv",
    "test": QRELS_DIR / "qrels_test.tsv",
}

for split, path in QRELS_OUTPUTS.items():
    qrels = load_dataset("BeIR/fever-qrels", split=split)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    print(f"qrels/{split}: {len(qrels):,} -> {path}")
