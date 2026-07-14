import argparse
import json
from pathlib import Path
import bm25s
from Stemmer import Stemmer

from fever_search import paths
from fever_search.data_io import iter_jsonl, load_qrels, load_queries
from fever_search.eval import precision_at_k, recall_at_k, mrr, ndcg_at_k, _mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, choices=["fever", "climate"])
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    print(f"--- Running BM25 Baseline for {args.benchmark} ({args.split}) ---")

    print("Loading corpus...")
    docs = list(iter_jsonl(paths.CORPUS_PATH))
    doc_ids = [str(d["_id"]) for d in docs]
    corpus = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in docs]

    print("Indexing corpus...")
    stemmer = Stemmer("english")
    corpus_tok = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tok)

    print("Loading queries...")
    q_path, qr_path = paths.benchmark_files(args.benchmark, args.split)
    qrels = load_qrels(qr_path)
    queries = load_queries(q_path)
    qids = [q for q in qrels if q in queries]

    print("Searching...")
    queries_list = [queries[q] for q in qids]
    q_tok = bm25s.tokenize(queries_list, stopwords="en", stemmer=stemmer)
    idx, _ = retriever.retrieve(q_tok, k=100)

    print("Calculating metrics...")
    k_values = [1, 5, 10, 100]

    acc = {f"precision@{k}": [] for k in k_values}
    acc.update({f"recall@{k}": [] for k in k_values})
    acc["mrr"] = []
    acc["ndcg@10"] = []

    for i, qid in enumerate(qids):
        relevant = qrels[qid]
        retrieved = [doc_ids[j] for j in idx[i]]

        for k in k_values:
            acc[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            acc[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
        acc["mrr"].append(mrr(retrieved, relevant))
        acc["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))

    metrics = {key: _mean(vals) for key, vals in acc.items()}

    report = {
        "model": "bm25",
        "index_type": "bm25",
        "num_queries": len(qids),
        "metrics": metrics
    }

    output_dir = Path("data/quality/bm25") / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Report saved to {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()