#!/usr/bin/env bash
# Reproduce the full index study on the 500k corpus (fever/test).
# Run from the repo root:  bash scripts/run_benchmarks.sh
# Prereq: data/corpus/fever_500k.jsonl present. All outputs land under data/ (gitignored).
set -euo pipefail

FLAT=data/index/e5_base_flat
BENCH=fever

# 0. Base flat index (encodes the corpus once; skipped if already built).
if [ ! -f "$FLAT/faiss.index" ]; then
  echo "== build flat (encodes 500k) =="
  python scripts/build_index.py --config configs/e5_base_flat.yaml
fi

# A. Compression variants (SQ8 / PQ / binary / binary+rerank), built from the saved embeddings.
echo "== axis A: compression =="
python scripts/benchmark_compression.py --index-dir "$FLAT" --benchmark "$BENCH"

# B. ANN indexes: build from the same embeddings (no re-encode), evaluate quality, measure latency.
for cfg in ivf ivfpq hnsw; do
  echo "== axis B: build+eval $cfg =="
  python scripts/build_index.py --config "configs/e5_base_$cfg.yaml" --from-embeddings "$FLAT"
  python scripts/evaluate.py    --config "configs/e5_base_$cfg.yaml" --benchmark "$BENCH"
  python scripts/latency.py     --config "configs/e5_base_$cfg.yaml" --index-dir "data/index/e5_base_$cfg"
done

# Flat latency + nprobe/efSearch sweep (recall<->latency curves).
python scripts/latency.py    --config configs/e5_base_flat.yaml --index-dir "$FLAT"
echo "== axis B: sweep =="
python scripts/sweep_index.py --benchmark "$BENCH"

# Quality comparison table across every config.
echo "== comparison =="
python scripts/compare.py --benchmark "$BENCH"
echo "== done: see data/quality, data/latency, data/analysis, $FLAT/compression_$BENCH.json =="
