"""Build the FAISS index for a config: python scripts/build_index.py --config configs/bge_small_flat.yaml"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fever_search import index  # noqa: E402
from fever_search.config import load_config  # noqa: E402
from fever_search.encoder import Encoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default=None, help="override model, e.g. a fine-tuned checkpoint")
    parser.add_argument("--from-embeddings", default=None,
                        help="reuse doc_embeddings.npy/doc_ids.json from this dir instead of re-encoding the corpus")
    args = parser.parse_args()

    config = load_config(args.config)
    encoder = Encoder(config.model, model_path=args.model_path) if args.model_path else None
    index.build_and_save(config, encoder=encoder, reuse_embeddings_dir=args.from_embeddings)


if __name__ == "__main__":
    main()
