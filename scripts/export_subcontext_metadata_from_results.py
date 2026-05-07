#!/usr/bin/env python3
"""Export sub-context metadata from existing SGLang benchmark CSV/JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bench_sglang_prefix_cache as bench


def model_from_result_dir(path: pathlib.Path) -> str:
    name = path.name
    if name.startswith("neno5_") and "_" in name:
        label = name.removeprefix("neno5_").rsplit("_", 1)[0]
        return label.replace("__", "/")
    return ""


def read_csv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def cache_fields_from_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = as_number(row.get("log_prompt_tokens")) or as_number(
        row.get("usage_prompt_tokens")
    )
    cached_tokens = as_number(row.get("log_cached_tokens"))
    if cached_tokens is None:
        cached_tokens = as_number(row.get("metric_delta_cached_tokens_total"))
    cache_ratio = None
    if prompt_tokens and cached_tokens is not None:
        cache_ratio = float(cached_tokens) / float(prompt_tokens)
    return {
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "cache_ratio": cache_ratio,
    }


def result_pairs(results_root: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    pairs = []
    for csv_path in sorted(results_root.rglob("sglang_prefix_cache_*.csv")):
        if csv_path.name.endswith("_subcontexts.csv") or csv_path.name.startswith("summary_"):
            continue
        if not csv_path.parent.name.startswith("neno5_"):
            continue
        json_path = csv_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((csv_path, json_path))
    return pairs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sub-context metadata CSV from existing benchmark outputs."
    )
    parser.add_argument(
        "--results-root",
        default="benchmark_results_qwen_20260507_162727/benchmark_results",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results_qwen_20260507_162727/benchmark_results/local_subcontext_metadata.csv",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional Hugging Face tokenizer path. If omitted, token spans are left empty.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    results_root = pathlib.Path(args.results_root)
    output_path = pathlib.Path(args.output)
    tokenizer = bench.load_tokenizer(args.tokenizer_path) if args.tokenizer_path else None

    output_rows: list[dict[str, Any]] = []
    for csv_path, json_path in result_pairs(results_root):
        result_json = read_json(json_path)
        experiment_id = result_json["experiment_id"]
        model = result_json.get("model") or model_from_result_dir(csv_path.parent)
        if model == "Qwen/Qwen2.5-0.5B-Instruct":
            folder_model = model_from_result_dir(csv_path.parent)
            if folder_model:
                model = folder_model

        req_specs = {req["name"]: req for req in bench.build_requests(experiment_id)}
        csv_rows = read_csv_rows(csv_path)
        for csv_row in csv_rows:
            request_name = csv_row["request_name"]
            req_spec = req_specs[request_name]
            base_metadata = bench.build_subcontext_metadata_for_request(
                req_spec,
                tokenizer=tokenizer,
            )
            cache_fields = cache_fields_from_csv_row(csv_row)
            for meta_row in base_metadata:
                output_rows.append(
                    {
                        "model": model,
                        "job_dir": csv_path.parent.name,
                        "source_csv": str(csv_path),
                        **meta_row,
                        **cache_fields,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bench.write_csv(output_path, output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
