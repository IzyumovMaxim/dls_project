"""Experiment configuration loaded from a YAML file"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    name: str = "BAAI/bge-small-en-v1.5"
    normalize: bool = True
    query_prompt: str | None = None   # literal instruction prepended to queries (model-specific), e.g. "query: "
    doc_prompt: str | None = None     # literal instruction prepended to documents, e.g. "passage: "
    batch_size: int = 64
    fp16: bool = False          # load weights in fp16 on cuda (big speed-up on A100, negligible retrieval impact)
    device: str | None = None   # null = auto (sentence-transformers picks cuda > mps > cpu); or "cuda", "cuda:0", "cpu"


@dataclass
class IndexConfig:
    type: str = "flat"          # flat | pq | ivf | ivfpq | hnsw | binary_rerank
    nlist: int = 4096           # ivf/ivfpq: number of Voronoi cells
    nprobe: int = 32            # ivf/ivfpq: cells probed at search time
    pq_m: int = 96              # pq/ivfpq: PQ sub-quantizers (must divide dim)
    pq_nbits: int = 8           # pq/ivfpq: bits per PQ code
    opq: bool = False           # pq: rotate the space before quantizing (same code size)
    hnsw_m: int = 32            # hnsw: neighbours per node
    ef_construction: int = 200  # hnsw: build-time candidate list
    ef_search: int = 64         # hnsw: search-time candidate list
    rerank_depth: int = 1000    # binary_rerank: Hamming shortlist re-scored against the fp32 vectors
    vectors_from: str | None = None  # index holding the fp32 + sentence vectors this one reuses;
                                     # required by any index that stores codes rather than vectors


@dataclass
class EvalConfig:
    top_k: int = 100
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10, 100])


@dataclass
class TrainConfig:
    base_config: str | None = None
    epochs: int = 1
    lr: float = 2e-5
    batch_size: int = 64
    warmup_ratio: float = 0.1
    hard_negatives: int = 4
    max_train_pairs: int | None = None


@dataclass
class ExperimentConfig:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "name" not in data:
        raise ValueError(f"Config {path} is missing required field 'name'")
    return ExperimentConfig(
        name=data["name"],
        model=ModelConfig(**data.get("model", {})),
        index=IndexConfig(**data.get("index", {})),
        eval=EvalConfig(**data.get("eval", {})),
        train=TrainConfig(**data.get("train", {})),
    )
