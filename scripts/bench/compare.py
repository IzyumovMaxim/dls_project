"""Compare metrics across experiment configs.

Scans data/quality/<config>/<benchmark>/report.json (written by scripts/evaluate.py)
and prints/saves a comparison table, grouped by benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fever_search import paths  # noqa: E402

DEFAULT_METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr", "precision@1"]
BASE_COLUMNS = ["config", "benchmark", "model", "index_type", "num_queries"]


def find_reports(benchmark: str | None = None) -> list[Path]:
    quality_dir = paths.DATA_DIR / "quality"
    if not quality_dir.exists():
        return []
    reports = sorted(quality_dir.glob("*/*/report.json"))
    if benchmark:
        reports = [path for path in reports if path.parent.name == benchmark]
    return reports


def load_rows(report_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "config": path.parent.parent.name,      # data/quality/<config>/<benchmark>/report.json
            "benchmark": path.parent.name,
            "model": data.get("model"),
            "index_type": data.get("index_type"),
            "num_queries": data.get("num_queries"),
            **data.get("metrics", {}),
        })
    return rows


def sort_rows(rows: list[dict[str, Any]], sort_by: str | None) -> list[dict[str, Any]]:
    if not sort_by:
        return rows
    return sorted(rows, key=lambda row: row.get(sort_by) if row.get(sort_by) is not None else -1.0, reverse=True)


def format_table(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    headers = [*BASE_COLUMNS, *metrics]
    widths = [max(len(h), *(len(_cell(row, h)) for row in rows)) if rows else len(h) for h in headers]

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    out = [line(headers), line(["-" * width for width in widths])]
    for row in rows:
        out.append(line([_cell(row, h) for h in headers]))
    return "\n".join(out)


def _cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def write_markdown(path: Path, grouped: list[tuple[str, list[dict[str, Any]]]], metrics: list[str]) -> None:
    headers = [*BASE_COLUMNS, *metrics]
    sections = []
    for benchmark, rows in grouped:
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *("| " + " | ".join(_cell(row, h) for h in headers) + " |" for row in rows),
        ]
        sections.append(f"## {benchmark}\n\n" + "\n".join(md_lines))
    path.write_text("# Experiment comparison\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    headers = [*BASE_COLUMNS, *metrics]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: _cell(row, h) for h in headers})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=None, choices=["fever", "climate"],
                         help="show only one benchmark (default: all found, grouped)")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                         help=f"comma-separated metric keys (default: {','.join(DEFAULT_METRICS)})")
    parser.add_argument("--sort-by", default="ndcg@10",
                         help="metric to sort each benchmark's rows by, descending; '' to keep filesystem order")
    parser.add_argument("--out", default=None, help="also save the table to this file (.md or .csv)")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    report_paths = find_reports(args.benchmark)
    if not report_paths:
        print("No report.json found under data/quality/. Run scripts/evaluate.py for at least one config first.")
        sys.exit(1)

    rows = load_rows(report_paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    grouped = [(benchmark, sort_rows([row for row in rows if row["benchmark"] == benchmark], args.sort_by))
               for benchmark in benchmarks]

    for benchmark, benchmark_rows in grouped:
        print(f"\n=== {benchmark} ===")
        print(format_table(benchmark_rows, metrics))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            write_csv(out_path, sort_rows(rows, args.sort_by), metrics)
        else:
            write_markdown(out_path, grouped, metrics)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()