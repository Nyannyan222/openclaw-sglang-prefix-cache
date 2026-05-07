#!/usr/bin/env python3
"""Summarize sub-context metadata CSV by model and request."""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


def as_float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    groups: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["model"], row["request_name"])].append(row)

    print("Model,Request,Rows,AvgTokenLen,AvgCached,AvgPrompt,AvgRatio")
    for (model, request), group in sorted(groups.items()):
        count = len(group)
        avg_token_len = sum(as_float(row["token_len"]) for row in group) / count
        avg_cached = sum(as_float(row["cached_tokens"]) for row in group) / count
        avg_prompt = sum(as_float(row["prompt_tokens"]) for row in group) / count
        avg_ratio = sum(as_float(row["cache_ratio"]) for row in group) / count
        print(
            f"{model},{request},{count},{avg_token_len:.1f},"
            f"{avg_cached:.1f},{avg_prompt:.1f},{avg_ratio * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
