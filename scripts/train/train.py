"""Fine-tune a config's model on FEVER train: python scripts/train/train.py --config configs/bge_small_ft.yaml"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search.config import load_config  # noqa: E402
from fever_search.train.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
