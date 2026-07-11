"""Readers for the JSONL corpus / queries and TSV qrels"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator


def doc_to_passage(doc: dict) -> str:
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def load_corpus(path: Path) -> dict[str, dict]:
    return {str(doc["_id"]): doc for doc in iter_jsonl(path)}


def load_queries(path: Path) -> dict[str, str]:
    return {str(row["_id"]): str(row.get("text") or "").strip() for row in iter_jsonl(path)}


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as file:
        next(file)  # header
        for line in file:
            query_id, corpus_id, _ = line.rstrip("\n").split("\t")
            qrels[query_id].add(corpus_id)
    return dict(qrels)
