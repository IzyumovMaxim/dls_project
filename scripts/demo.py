"""Interactive search demo: python scripts/demo.py --config configs/bge_small_flat.yaml"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search.config import load_config  # noqa: E402
from fever_search.search import SearchEngine, format_hit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = SearchEngine(config, model_path=args.model_path)
    print(f"Model: {engine.manifest.get('model_name')}  Docs: {engine.document_count:,}")
    print("Enter a claim (empty line to exit).")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        for hit in engine.search(query, top_k=args.top_k):
            print(format_hit(hit))
            print()


if __name__ == "__main__":
    main()
