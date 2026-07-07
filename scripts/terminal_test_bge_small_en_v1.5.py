"""
Interactive terminal demo for bge-small-en-v1.5 vector search.

Run from dls_project root:
    python scripts/terminal_test_bge_small_en_v1.5.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_MODULE_PATH = SCRIPT_DIR / "query_search_bge_small_en_v1.5.py"

DEFAULT_TOP_K = 10


def load_search_module():
    spec = importlib.util.spec_from_file_location("query_search_bge", SEARCH_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {SEARCH_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_hits(query: str, hits) -> None:
    print()
    print(f"Query: {query}")
    print("-" * 60)
    if not hits:
        print("No results.")
        return
    search_mod = load_search_module()
    for hit in hits:
        print(search_mod.format_hit(hit))
        print()


def main() -> None:
    search_mod = load_search_module()

    print("Loading search engine...")
    engine = search_mod.get_search_engine()
    print(f"Model   : {engine.manifest.get('model_name', 'BAAI/bge-small-en-v1.5')}")
    print(f"Indexed : {engine.document_count:,} documents")
    print(f"Top-K   : {DEFAULT_TOP_K}")
    print()
    print("Enter a claim to search (empty line to exit).")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        hits = search_mod.vector_search(query, top_k=DEFAULT_TOP_K)
        print_hits(query, hits)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
