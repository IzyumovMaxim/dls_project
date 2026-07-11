"""Evaluate a config on a benchmark: python scripts/evaluate.py --config ... --benchmark fever --split test"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import paths  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.eval import run_eval  # noqa: E402
from fever_search.search import SearchEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", default="fever", choices=["fever", "climate"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", default=None, help="override model, e.g. a fine-tuned checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = SearchEngine(config, model_path=args.model_path)
    queries_path, qrels_path = paths.benchmark_files(args.benchmark, args.split)
    out_dir = paths.quality_dir(config.name) / args.benchmark
    run_eval(
        engine, queries_path, qrels_path, out_dir,
        label=f"{config.name}/{args.benchmark}",
        top_k=config.eval.top_k,
        k_values=tuple(config.eval.k_values),
    )


if __name__ == "__main__":
    main()
