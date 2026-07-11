"""Export Climate-FEVER queries + qrels (test) — evaluated on the shared 500k index."""

import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIMATE_DIR = PROJECT_ROOT / "data" / "climate-fever"

CLIMATE_DIR.mkdir(parents=True, exist_ok=True)

queries = load_dataset("BeIR/climate-fever", "queries", split="queries")
with (CLIMATE_DIR / "queries.jsonl").open("w", encoding="utf-8") as f:
    for row in queries:
        f.write(json.dumps({
            "_id": str(row["_id"]),
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or "").strip(),
        }, ensure_ascii=False) + "\n")
print(f"queries: {len(queries):,}")

qrels = load_dataset("BeIR/climate-fever-qrels", split="test")
with (CLIMATE_DIR / "qrels_test.tsv").open("w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for row in qrels:
        f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
print(f"qrels/test: {len(qrels):,} -> {CLIMATE_DIR}")
