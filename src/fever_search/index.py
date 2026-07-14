"""Build, persist and load a FAISS index (flat / ivf / hnsw) over the corpus."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from fever_search import paths
from fever_search.config import ExperimentConfig, IndexConfig
from fever_search.data_io import doc_to_passage, iter_jsonl
from fever_search.encoder import Encoder


def is_binary(index_type: str) -> bool:
    """Binary indexes are a separate faiss hierarchy with their own read/write calls."""
    return index_type == "binary_rerank"


def pack_bits(vectors: np.ndarray) -> np.ndarray:
    """Sign bits of each vector, packed 8 to a byte: 768 floats -> 96 bytes."""
    return np.packbits(vectors > 0, axis=1)


def build_binary(embeddings: np.ndarray) -> faiss.IndexBinary:
    """Hamming over sign bits: a coarse proxy for cosine, re-scored by SearchEngine."""
    index = faiss.IndexBinaryFlat(embeddings.shape[1])
    index.add(pack_bits(embeddings))
    return index


def build_faiss(embeddings: np.ndarray, cfg: IndexConfig):
    dim = embeddings.shape[1]
    if cfg.type == "binary_rerank":
        return build_binary(embeddings)
    if cfg.type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif cfg.type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, cfg.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = cfg.nprobe
    elif cfg.type == "pq":
        # Exhaustive scan of PQ codes: no pruning, so the only loss is the coding error.
        if dim % cfg.pq_m != 0:
            raise ValueError(f"pq: dim {dim} not divisible by pq_m={cfg.pq_m}")
        if cfg.opq:
            index = faiss.index_factory(
                dim, f"OPQ{cfg.pq_m},PQ{cfg.pq_m}x{cfg.pq_nbits}", faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexPQ(dim, cfg.pq_m, cfg.pq_nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    elif cfg.type == "ivfpq":
        if dim % cfg.pq_m != 0:
            raise ValueError(f"ivfpq: dim {dim} not divisible by pq_m={cfg.pq_m}")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, cfg.nlist, cfg.pq_m, cfg.pq_nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = cfg.nprobe
    elif cfg.type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.ef_construction
        index.hnsw.efSearch = cfg.ef_search
    else:
        raise ValueError(f"Unknown index type: {cfg.type!r} (flat | pq | ivf | ivfpq | hnsw | binary_rerank)")
    index.add(embeddings)
    return index


def build_and_save(
    config: ExperimentConfig,
    encoder: Encoder | None = None,
    reuse_embeddings_dir: str | Path | None = None,
) -> None:
    if reuse_embeddings_dir is not None:
        # Rebuild a different index type from already-encoded vectors (no re-encoding).
        src = Path(reuse_embeddings_dir)
        embeddings = np.load(src / "doc_embeddings.npy").astype(np.float32)
        doc_ids = json.loads((src / "doc_ids.json").read_text(encoding="utf-8"))
        print(f"Reusing embeddings from {src}: {len(doc_ids):,} x {embeddings.shape[1]} (no re-encode)")
    else:
        if not paths.CORPUS_PATH.exists():
            raise FileNotFoundError(f"Corpus not found: {paths.CORPUS_PATH}. Run scripts/data/build_corpus.py")
        encoder = encoder or Encoder(config.model)
        docs = list(iter_jsonl(paths.CORPUS_PATH))
        doc_ids = [str(doc["_id"]) for doc in docs]
        passages = [doc_to_passage(doc) for doc in docs]
        embeddings = encoder.encode_documents(passages, show_progress_bar=True)

    index = build_faiss(embeddings, config.index)

    out_dir = paths.index_dir(config.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_binary(config.index.type):
        faiss.write_index_binary(index, str(out_dir / "faiss.index"))
    else:
        faiss.write_index(index, str(out_dir / "faiss.index"))
    if reuse_embeddings_dir is None:
        np.save(out_dir / "doc_embeddings.npy", embeddings)  # skip 1.5 GB re-save when reusing
    (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config.name,
        "model_name": config.model.name,
        "index_type": config.index.type,
        "document_count": len(doc_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": config.model.normalize,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Index [{config.index.type}]: {len(doc_ids):,} docs, dim {embeddings.shape[1]} -> {out_dir}")


# The one search-time knob per index type, tunable without a rebuild.
# Keys match the IndexConfig field names, so getattr(cfg, SEARCH_KNOB[cfg.type]) reads the value.
SEARCH_KNOB = {"ivf": "nprobe", "ivfpq": "nprobe", "hnsw": "ef_search"}


def set_search_knob(faiss_index, knob: str, value: int) -> None:
    if knob == "nprobe":
        faiss_index.nprobe = value
    elif knob == "ef_search":
        faiss_index.hnsw.efSearch = value
    else:
        raise ValueError(f"Unknown search knob: {knob!r} (nprobe | ef_search)")


def apply_search_params(faiss_index, cfg: IndexConfig) -> None:
    """Apply search-time knobs from config. Flat and binary_rerank have none."""
    knob = SEARCH_KNOB.get(cfg.type)
    if knob is not None:
        set_search_knob(faiss_index, knob, getattr(cfg, knob))


def load(name: str, index_cfg: IndexConfig | None = None) -> tuple[object, list[str], dict]:
    out_dir = paths.index_dir(name)
    faiss_path = out_dir / "faiss.index"
    ids_path = out_dir / "doc_ids.json"
    manifest_path = out_dir / "manifest.json"
    if not faiss_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Index not found in {out_dir}. Run scripts/index/build_index.py --config ...")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    binary = is_binary(index_cfg.type if index_cfg else manifest.get("index_type", ""))

    faiss_index = faiss.read_index_binary(str(faiss_path)) if binary else faiss.read_index(str(faiss_path))
    if index_cfg is not None:
        apply_search_params(faiss_index, index_cfg)
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return faiss_index, doc_ids, manifest
