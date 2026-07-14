"""Encode every corpus sentence once, offline, so highlighting evidence is free at query time.

Sentences are document-side data; encoding them per query costs ~1.5 s on CPU. Precomputed, picking
the evidence sentence is a dot product against the query vector retrieval already computed.

Stored fp16 and memmapped at serving time, so only the sentences of the retrieved documents are read.

    python scripts/index/build_sentence_index.py --config configs/e5_base_flat.yaml

Writes into the config's index dir:
    sentence_vectors.npy  (num_sentences x dim, fp16)
    sentence_offsets.npy  (num_docs + 1, int64) — doc i owns rows [offsets[i], offsets[i+1])

Row order matches doc_ids.json, so a FAISS hit maps straight to its sentence block.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.data_io import iter_jsonl  # noqa: E402
from fever_search.encoder import Encoder  # noqa: E402
from fever_search.text import split_sentences  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5_base_flat.yaml")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="sentences per encoder batch; they are short, so this can exceed the doc batch")
    parser.add_argument("--device", default=None, help="override config device (e.g. cuda, cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config.model.device = args.device
    out_dir = paths.index_dir(config.name)
    if not (out_dir / "doc_ids.json").exists():
        raise SystemExit(f"No index in {out_dir}. Run scripts/index/build_index.py --config {args.config} first.")

    doc_ids = json.loads((out_dir / "doc_ids.json").read_text(encoding="utf-8"))
    position = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Walk the corpus once, keeping sentences grouped per document in doc_ids order.
    per_doc: list[list[str]] = [[] for _ in doc_ids]
    for doc in iter_jsonl(paths.CORPUS_PATH):
        index = position.get(str(doc["_id"]))
        if index is not None:
            per_doc[index] = split_sentences(str(doc.get("text") or ""))

    counts = np.array([len(s) for s in per_doc], dtype=np.int64)
    offsets = np.zeros(len(doc_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    print(f"{len(doc_ids):,} docs -> {total:,} sentences ({total / max(len(doc_ids), 1):.1f} per doc)")

    flat = [sentence for sentences in per_doc for sentence in sentences]
    encoder = Encoder(config.model)

    start = time.perf_counter()
    # Sentences are encoded document-side: that is the side of the space the passages live in.
    vectors = encoder.encode_documents(flat, show_progress_bar=True)
    encode_s = time.perf_counter() - start

    vectors = np.asarray(vectors, dtype=np.float16)
    np.save(out_dir / "sentence_vectors.npy", vectors)
    np.save(out_dir / "sentence_offsets.npy", offsets)

    size_gb = vectors.nbytes / 1e9
    print(f"\nEncoded {total:,} sentences in {encode_s / 60:.1f} min")
    print(f"  sentence_vectors.npy: {vectors.shape} fp16 ({size_gb:.1f} GB)")
    print(f"  sentence_offsets.npy: {offsets.shape} int64")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
