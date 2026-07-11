"""Build the 500k corpus: all FEVER + Climate-FEVER gold docs plus random filler."""

import json
import random
from pathlib import Path

from datasets import load_dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_JSONL_PATH = PROJECT_ROOT / "data" / "corpus" / "fever_500k.jsonl"
MISSING_GOLD_PATH = PROJECT_ROOT / "data" / "corpus" / "missing_gold_ids.txt"
QRELS_DIRS = (
    PROJECT_ROOT / "data" / "qrels",
    PROJECT_ROOT / "data" / "climate-fever",
)

TARGET_SIZE = 500000
RANDOM_SEED = 42


def load_gold_doc_ids() -> set[str]:
    tsv_paths = [p for d in QRELS_DIRS for p in sorted(d.glob("qrels_*.tsv"))]
    if not tsv_paths:
        raise SystemExit("No qrels found. Run export_fever.py and export_climate.py first.")
    doc_ids: set[str] = set()
    for path in tsv_paths:
        with path.open(encoding="utf-8") as file:
            next(file)
            for line in file:
                _, corpus_id, _ = line.rstrip("\n").split("\t")
                doc_ids.add(corpus_id)
    return doc_ids


def row_to_document(row: dict) -> dict[str, str]:
    return {
        "_id": str(row["_id"]),
        "title": str(row.get("title") or ""),
        "text": str(row.get("text") or "").strip(),
    }


def open_corpus_stream():
    return load_dataset("BeIR/fever", "corpus", split="corpus", streaming=True)


def collect_gold_documents(gold_doc_ids: set[str]) -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    stream = open_corpus_stream()
    iterator = tqdm(stream, desc="Phase 1: gold docs", unit="docs") if tqdm else stream
    for row in iterator:
        doc = row_to_document(row)
        if doc["_id"] not in gold_doc_ids:
            continue
        documents[doc["_id"]] = doc
        if len(documents) == len(gold_doc_ids):
            break
    return documents


def sample_random_filler(used_ids: set[str], filler_target: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen_available = 0
    stream = open_corpus_stream()
    iterator = tqdm(stream, desc="Phase 2: random filler", unit="docs") if tqdm else stream
    for row in iterator:
        doc = row_to_document(row)
        if doc["_id"] in used_ids:
            continue
        seen_available += 1
        if len(reservoir) < filler_target:
            reservoir.append(doc)
        elif (pick := rng.randint(1, seen_available)) <= filler_target:
            reservoir[pick - 1] = doc
    return reservoir


CORPUS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

gold_doc_ids = load_gold_doc_ids()
print(f"Gold docs referenced by qrels: {len(gold_doc_ids):,}")

if len(gold_doc_ids) > TARGET_SIZE:
    raise SystemExit(f"ERROR: {len(gold_doc_ids):,} gold docs exceed target size {TARGET_SIZE:,}")

gold_documents = collect_gold_documents(gold_doc_ids)

missing = gold_doc_ids - set(gold_documents)
if missing:
    # a few Climate-FEVER evidence ids are absent from the FEVER dump; log, don't fail
    MISSING_GOLD_PATH.write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")
    print(f"WARNING: {len(missing):,} gold docs not found in corpus (logged to {MISSING_GOLD_PATH})")

filler_target = TARGET_SIZE - len(gold_documents)
print(f"Gold docs collected: {len(gold_documents):,}")
print(f"Filler target: {filler_target:,}")

filler_documents: list[dict[str, str]] = []
if filler_target:
    filler_documents = sample_random_filler(set(gold_documents), filler_target, RANDOM_SEED)

if len(filler_documents) != filler_target:
    raise SystemExit(f"ERROR: expected {filler_target:,} filler docs, got {len(filler_documents):,}")

corpus = list(gold_documents.values()) + filler_documents
corpus.sort(key=lambda doc: doc["_id"])

if len(corpus) != TARGET_SIZE:
    raise SystemExit(f"ERROR: expected {TARGET_SIZE:,} docs, got {len(corpus):,}")

with CORPUS_JSONL_PATH.open("w", encoding="utf-8") as file:
    for doc in corpus:
        file.write(json.dumps(doc, ensure_ascii=False) + "\n")

size_mb = CORPUS_JSONL_PATH.stat().st_size / (1024 * 1024)
print(f"Done: {len(corpus):,} docs -> {CORPUS_JSONL_PATH} ({size_mb:.1f} MB)")
print(f"  gold  : {len(gold_documents):,}")
print(f"  filler: {len(filler_documents):,}")
