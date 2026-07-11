"""Fine-tune the bi-encoder on FEVER train with MultipleNegativesRankingLoss."""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from fever_search import paths
from fever_search.config import ExperimentConfig
from fever_search.data_io import doc_to_passage, load_corpus, load_qrels, load_queries


def _passage(corpus: dict, corpus_id: str) -> str:
    return doc_to_passage(corpus.get(corpus_id, {}))


def build_examples(config: ExperimentConfig, corpus: dict, queries: dict) -> list[InputExample]:
    hard_path = paths.TRAIN_DIR / "hard_negatives_train.jsonl"
    examples: list[InputExample] = []

    if config.train.hard_negatives > 0 and hard_path.exists():
        for line in hard_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            query = queries.get(rec["query_id"])
            if not query:
                continue
            negatives = rec["negative_ids"]
            for positive_id in rec["positive_ids"]:
                positive = _passage(corpus, positive_id)
                if not positive:
                    continue
                if negatives:
                    examples.append(InputExample(texts=[query, positive, _passage(corpus, negatives[0])]))
                else:
                    examples.append(InputExample(texts=[query, positive]))
    else:
        qrels = load_qrels(paths.benchmark_files("fever", "train")[1])
        for qid, gold in qrels.items():
            query = queries.get(qid)
            if not query:
                continue
            for positive_id in gold:
                positive = _passage(corpus, positive_id)
                if positive:
                    examples.append(InputExample(texts=[query, positive]))

    if config.train.max_train_pairs:
        examples = examples[: config.train.max_train_pairs]
    return examples


def train(config: ExperimentConfig) -> Path:
    corpus = load_corpus(paths.CORPUS_PATH)
    queries = load_queries(paths.QUERIES_PATH)
    examples = build_examples(config, corpus, queries)
    print(f"Training pairs: {len(examples):,}")

    model = SentenceTransformer(config.model.name)
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
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned model -> {out_dir}")
    print(f"Index/eval it with: --model-path {out_dir}")
    return out_dir
