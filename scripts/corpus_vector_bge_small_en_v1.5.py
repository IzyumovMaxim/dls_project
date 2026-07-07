"""
Build FAISS index (Option A: IndexFlatIP) for fever_500k with bge-small-en-v1.5.

Run from dls_project root:
    python scripts/corpus_vector_bge_small_en_v1.5.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSONL_PATH = PROJECT_ROOT / "data" / "corpus" / "fever_500k.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "bge-small-en-v1.5"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
ENCODE_BATCH_SIZE = 64
JSONL_READ_BATCH = 256


def doc_to_passage(doc: dict) -> str:
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def iter_corpus_batches(path: Path, batch_size: int):
    batch_docs: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            batch_docs.append(json.loads(line))
            if len(batch_docs) >= batch_size:
                yield batch_docs
                batch_docs = []
    if batch_docs:
        yield batch_docs


def main() -> None:
    if not CORPUS_JSONL_PATH.exists():
        print(f"Corpus not found: {CORPUS_JSONL_PATH}")
        print("Run: python scripts/create_corpus.py")
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = INDEX_DIR / "doc_embeddings.npy"
    doc_ids_path = INDEX_DIR / "doc_ids.json"
    faiss_path = INDEX_DIR / "faiss.index"
    manifest_path = INDEX_DIR / "manifest.json"

    print("=" * 60)
    print("Corpus vectorization — bge-small-en-v1.5")
    print("=" * 60)
    print(f"Corpus : {CORPUS_JSONL_PATH}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Output : {INDEX_DIR}")
    print()

    model = SentenceTransformer(MODEL_NAME)

    doc_ids: list[str] = []
    embedding_blocks: list[np.ndarray] = []

    batches = list(iter_corpus_batches(CORPUS_JSONL_PATH, JSONL_READ_BATCH))
    for batch_docs in tqdm(batches, desc="Encoding corpus", unit="batch"):
        passages = [doc_to_passage(doc) for doc in batch_docs]
        vectors = model.encode(
            passages,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            prompt_name="document",
            show_progress_bar=False,
        )
        embedding_blocks.append(np.asarray(vectors, dtype=np.float32))
        doc_ids.extend(str(doc["_id"]) for doc in batch_docs)

    embeddings = np.vstack(embedding_blocks)
    if embeddings.shape[0] != len(doc_ids):
        raise RuntimeError("Embedding count does not match doc_ids count")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_path))

    np.save(embeddings_path, embeddings)
    doc_ids_path.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "corpus_path": str(CORPUS_JSONL_PATH.relative_to(PROJECT_ROOT)),
        "document_count": len(doc_ids),
        "embedding_dim": int(dimension),
        "index_type": "IndexFlatIP",
        "normalize_embeddings": True,
        "files": {
            "doc_embeddings": embeddings_path.name,
            "doc_ids": doc_ids_path.name,
            "faiss_index": faiss_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = embeddings.nbytes / (1024 * 1024)
    print()
    print("Done.")
    print(f"  Documents : {len(doc_ids):,}")
    print(f"  Dimension : {dimension}")
    print(f"  Embeddings: {embeddings_path} ({size_mb:.1f} MB)")
    print(f"  FAISS     : {faiss_path}")
    print(f"  Manifest  : {manifest_path}")


if __name__ == "__main__":
    main()
