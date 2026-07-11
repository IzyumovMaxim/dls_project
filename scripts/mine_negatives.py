"""Mine hard negatives using a config's index: python scripts/mine_negatives.py --config configs/bge_small_flat.yaml"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search.config import load_config  # noqa: E402
from fever_search.search import SearchEngine  # noqa: E402
from fever_search.train.mine import mine_hard_negatives  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config whose index is used to retrieve candidates")
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = SearchEngine(config)
    num_negatives = args.num_negatives if args.num_negatives is not None else config.train.hard_negatives
    mine_hard_negatives(engine, num_negatives=num_negatives, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
