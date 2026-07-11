"""Filesystem layout: data inputs and per-experiment artifact directories."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CORPUS_PATH = DATA_DIR / "corpus" / "fever_500k.jsonl"
QUERIES_PATH = DATA_DIR / "queries" / "queries.jsonl"
QRELS_DIR = DATA_DIR / "qrels"
CLIMATE_DIR = DATA_DIR / "climate-fever"
TRAIN_DIR = DATA_DIR / "train"


def index_dir(name: str) -> Path:
    return DATA_DIR / "index" / name


def quality_dir(name: str) -> Path:
    return DATA_DIR / "quality" / name


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


def benchmark_files(benchmark: str, split: str) -> tuple[Path, Path]:
    """Return (queries_path, qrels_path) for a benchmark + split."""
    if benchmark == "fever":
        return QUERIES_PATH, QRELS_DIR / f"qrels_{split}.tsv"
    if benchmark == "climate":
        return CLIMATE_DIR / "queries.jsonl", CLIMATE_DIR / "qrels_test.tsv"
    raise ValueError(f"Unknown benchmark: {benchmark!r} (expected 'fever' or 'climate')")
