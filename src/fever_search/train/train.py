"""Fine-tune the bi-encoder on FEVER train with MultipleNegativesRankingLoss."""

from __future__ import annotations

import json
import random
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from fever_search import paths
from fever_search.config import ExperimentConfig
from fever_search.data_io import doc_to_passage, load_corpus, load_qrels, load_queries

SHUFFLE_SEED = 42


def build_examples(config: ExperimentConfig, corpus: dict, queries: dict) -> list[InputExample]:
    """Build (query, positive, *negatives) examples with the encoder's search-time prefixes.

    Queries yielding fewer than num_negatives usable negatives are dropped: the
    sentence-transformers collate infers the column count from the first example
    and assumes every example matches it.
    """
    query_prompt = config.model.query_prompt or ""
    doc_prompt = config.model.doc_prompt or ""

    def passage(corpus_id: str) -> str:
        text = doc_to_passage(corpus.get(corpus_id, {}))
        return f"{doc_prompt}{text}" if text else ""

    num_negatives = config.train.hard_negatives
    examples: list[InputExample] = []

    if num_negatives > 0:
        hard_path = paths.TRAIN_DIR / "hard_negatives_train.jsonl"
        if not hard_path.exists():
            raise FileNotFoundError(
                f"{hard_path} not found, but train.hard_negatives={num_negatives}. "
                "Run scripts/mine_negatives.py first, or set hard_negatives: 0."
            )
        dropped = 0
        for line in hard_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            query = queries.get(record["query_id"])
            if not query:
                continue
            negatives = [p for p in (passage(i) for i in record["negative_ids"]) if p][:num_negatives]
            if len(negatives) < num_negatives:
                dropped += 1
                continue
            for positive_id in record["positive_ids"]:
                positive = passage(positive_id)
                if positive:
                    examples.append(
                        InputExample(texts=[f"{query_prompt}{query}", positive, *negatives])
                    )
        if dropped:
            print(f"Dropped {dropped:,} queries with fewer than {num_negatives} usable negatives")
    else:
        qrels = load_qrels(paths.benchmark_files("fever", "train")[1])
        for qid, gold in qrels.items():
            query = queries.get(qid)
            if not query:
                continue
            for positive_id in gold:
                positive = passage(positive_id)
                if positive:
                    examples.append(InputExample(texts=[f"{query_prompt}{query}", positive]))

    random.Random(SHUFFLE_SEED).shuffle(examples)  # so max_train_pairs slices a representative subset
    if config.train.max_train_pairs:
        examples = examples[: config.train.max_train_pairs]
    return examples


def train(config: ExperimentConfig) -> Path:
    corpus = load_corpus(paths.CORPUS_PATH)
    queries = load_queries(paths.QUERIES_PATH)
    examples = build_examples(config, corpus, queries)
    if not examples:
        raise RuntimeError("No training examples were built - check the corpus, queries and hard negatives.")
    print(f"Training pairs: {len(examples):,} ({len(examples[0].texts)} texts each)")

    model = SentenceTransformer(config.model.name, device=config.model.device)
    loader = DataLoader(examples, shuffle=True, batch_size=config.train.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(loader) * config.train.epochs * config.train.warmup_ratio)

    out_dir = paths.model_dir(config.name)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=config.train.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.train.lr},
        output_path=str(out_dir),
        use_amp=config.model.fp16,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned model -> {out_dir}")
    print(f"Index/eval it with: --model-path {out_dir}")
    return out_dir
