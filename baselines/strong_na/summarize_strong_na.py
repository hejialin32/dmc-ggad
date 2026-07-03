"""Summarize strict strong-NA baseline CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = [
    "dataset",
    "method",
    "n",
    "auc_mean",
    "auc_std",
    "ap_mean",
    "ap_std",
    "precision_at_k_mean",
    "precision_at_k_std",
    "failed",
    "protocol",
    "label_visibility",
]


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.8f}"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    failures: dict[tuple[str, str], int] = defaultdict(int)
    meta: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        meta[key] = (row.get("protocol", ""), row.get("label_visibility", ""))
        if row.get("status") == "done":
            groups[key].append(row)
        else:
            failures[key] += 1

    output = []
    for key in sorted(set(groups) | set(failures)):
        done_rows = groups.get(key, [])
        protocol, label_visibility = meta.get(key, ("", ""))
        vals = {}
        for metric in ("auc", "ap", "precision_at_k"):
            numbers = np.asarray([float(row[metric]) for row in done_rows if row.get(metric)], dtype=float)
            vals[f"{metric}_mean"] = float(np.mean(numbers)) if numbers.size else float("nan")
            vals[f"{metric}_std"] = float(np.std(numbers, ddof=1)) if numbers.size > 1 else 0.0 if numbers.size else float("nan")
        output.append(
            {
                "dataset": key[0],
                "method": key[1],
                "n": len(done_rows),
                "auc_mean": fmt(vals["auc_mean"]),
                "auc_std": fmt(vals["auc_std"]),
                "ap_mean": fmt(vals["ap_mean"]),
                "ap_std": fmt(vals["ap_std"]),
                "precision_at_k_mean": fmt(vals["precision_at_k_mean"]),
                "precision_at_k_std": fmt(vals["precision_at_k_std"]),
                "failed": failures.get(key, 0),
                "protocol": protocol,
                "label_visibility": label_visibility,
            }
        )
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default="baselines/strong_na/strong_na_results.csv")
    parser.add_argument("--output_csv", default="baselines/strong_na/strong_na_summary.csv")
    args = parser.parse_args()

    rows = summarize(read_rows(Path(args.input_csv)))
    write_rows(Path(args.output_csv), rows)
    print(f"[saved] {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
