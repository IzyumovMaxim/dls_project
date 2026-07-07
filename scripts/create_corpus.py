import json
import random
from pathlib import Path

from datasets import load_dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSONL_PATH = PROJECT_ROOT / "data" / "corpus" / "fever_500k.jsonl"
QRELS_TEST_PATH = PROJECT_ROOT / "data" / "qrels" / "qrels_test.tsv"

TARGET_SIZE = 500000
RANDOM_SEED = 42


def load_test_qrels_doc_ids() -> set[str]:
    if QRELS_TEST_PATH.exists():
        doc_ids: set[str] = set()
        with QRELS_TEST_PATH.open(encoding="utf-8") as file:
            next(file)  # header
            for line in file:
                _, corpus_id, _ = line.rstrip("\n").split("\t")
                doc_ids.add(corpus_id)
        return doc_ids

    qrels = load_dataset("BeIR/fever-qrels", split="test")
    return {str(doc_id) for doc_id in qrels["corpus-id"]}


def row_to_document(row: dict) -> dict[str, str]:
    return {
        "_id": str(row["_id"]),
        "title": str(row.get("title") or ""),
        "text": str(row.get("text") or "").strip(),
    }


def open_corpus_stream():
    return load_dataset("BeIR/fever", "corpus", split="corpus", streaming=True)


def collect_qrels_documents(qrels_doc_ids: set[str]) -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    stream = open_corpus_stream()
    iterator = tqdm(stream, desc="Phase 1: qrels docs", unit="docs") if tqdm else stream

    for row in iterator:
        doc = row_to_document(row)
        if doc["_id"] not in qrels_doc_ids:
            continue
        documents[doc["_id"]] = doc
        if len(documents) == len(qrels_doc_ids):
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
        else:
            pick = rng.randint(1, seen_available)
            if pick <= filler_target:
                reservoir[pick - 1] = doc

    return reservoir


qrels_doc_ids = load_test_qrels_doc_ids()
print(f"Test qrels unique docs: {len(qrels_doc_ids):,}")

if len(qrels_doc_ids) > TARGET_SIZE:
    raise SystemExit(
        f"ERROR: {len(qrels_doc_ids):,} qrels docs exceed target size {TARGET_SIZE:,}"
    )

qrels_documents = collect_qrels_documents(qrels_doc_ids)

missing = qrels_doc_ids - set(qrels_documents)
if missing:
    raise SystemExit(f"ERROR: {len(missing)} qrels docs not found in corpus")

filler_target = TARGET_SIZE - len(qrels_documents)
print(f"Qrels docs collected: {len(qrels_documents):,}")
print(f"Filler target: {filler_target:,}")

used_ids = set(qrels_documents)
filler_documents = sample_random_filler(used_ids, filler_target, RANDOM_SEED) if filler_target else []

if len(filler_documents) != filler_target:
    raise SystemExit(
        f"ERROR: expected {filler_target:,} filler docs, got {len(filler_documents):,}"
    )

corpus = list(qrels_documents.values()) + filler_documents
corpus.sort(key=lambda doc: doc["_id"])

if len(corpus) != TARGET_SIZE:
    raise SystemExit(f"ERROR: expected {TARGET_SIZE:,} docs, got {len(corpus):,}")

CORPUS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
with CORPUS_JSONL_PATH.open("w", encoding="utf-8") as file:
    for doc in corpus:
        file.write(json.dumps(doc, ensure_ascii=False) + "\n")

size_mb = CORPUS_JSONL_PATH.stat().st_size / (1024 * 1024)
print(f"Done: {len(corpus):,} docs -> {CORPUS_JSONL_PATH} ({size_mb:.1f} MB)")
print(f"  qrels: {len(qrels_documents):,}")
print(f"  filler: {len(filler_documents):,}")
