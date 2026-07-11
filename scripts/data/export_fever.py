"""Export FEVER queries + qrels (train/validation/test) from HuggingFace."""

import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

for split in ("train", "validation", "test"):
    path = QRELS_DIR / f"qrels_{split}.tsv"
    qrels = load_dataset("BeIR/fever-qrels", split=split)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    print(f"qrels/{split}: {len(qrels):,} -> {path}")
