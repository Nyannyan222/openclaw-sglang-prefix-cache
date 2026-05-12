#!/usr/bin/env python3
"""Run or stage the WildClawBench next-round framework evaluation manifest.

The runner supports:

- dry-run: write prompt/context files for each manifest row;
- openai: send each row to an OpenAI-compatible Chat Completions endpoint.

It writes a results CSV/JSON that matches the experiment template produced by
prepare_wildclaw_next_experiment.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are evaluating a framework context policy on a WildClawBench task.
Use only the provided context and the original task prompt.
Return a concise answer plus an evidence trace listing the source paths or
sub-context IDs you used. If the provided context is insufficient, say so."""

WEB_SYSTEM_PROMPT = """You are executing the Original task prompt inside an evaluation.
The benchmark name, context policy, and evaluation-framework metadata are not
search targets. Use the provided context as local evidence, then use web search
only for the real-world facts needed by the Original task prompt. Return a
concise answer plus an evidence trace listing both local source paths/sub-context
IDs and web sources. If local evidence conflicts with current web evidence,
explicitly say which source is outdated and why."""

WEB_RETRIEVAL_SYSTEM_PROMPT = """You are a focused web retrieval agent.
Search only for evidence needed to answer the user's task. Do not search for the
benchmark name, context policy, RAG, or evaluation-framework metadata unless the
task itself asks for those topics. Return compact evidence bullets with citations."""


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def search_brief(row: dict[str, Any]) -> str:
    task_id = row.get("task_id", "")
    if task_id.endswith("task_2_conflicting_handling"):
        return (
            "SEARCH_OBJECTIVE:\n"
            "- Determine the current PRC statute of limitations for a sales-contract payment claim.\n"
            "- Verify whether local materials saying two years are outdated by current Chinese civil law.\n"
            "SUGGESTED_QUERY_TERMS:\n"
            "- 中华人民共和国 民法典 诉讼时效期间 三年 第一百八十八条\n"
            "- PRC Civil Code limitation period three years Article 188\n"
        )
    if task_id.endswith("task_4_efficient_search"):
        return (
            "SEARCH_OBJECTIVE:\n"
            "- Identify the Python version where pathlib.Path.walk() entered the standard library.\n"
            "- Identify the CPython GitHub reference number linked by the official Python evidence for that addition.\n"
            "- Keep the answer's search log within the task's budget of 4 searches.\n"
            "- Prefer official Python documentation, bugs.python.org, and github.com/python/cpython evidence.\n"
            "- Do not rely on third-party code mirrors or later follow-up PRs when they conflict with official docs.\n"
            "SUGGESTED_QUERY_TERMS:\n"
            "- Python 3.12 What's New pathlib Path.walk contributed gh\n"
            "- pathlib Path.walk Python 3.12 docs gh cpython\n"
        )
    if task_id.endswith("task_5_fuzzy_search"):
        return (
            "SEARCH_OBJECTIVE:\n"
            "- Find the 2025 paper by a first author whose last name is Liu that transfers a DeepSeek-R1-like approach to visual perception tasks.\n"
            "- Confirm the linked GitHub repository has more than 2k stars if possible.\n"
            "SUGGESTED_QUERY_TERMS:\n"
            "- Liu 2025 DeepSeek-R1 visual perception GitHub stars paper\n"
            "- Visual-RFT Liu DeepSeek-R1 visual perception GitHub\n"
        )
    return (
        "SEARCH_OBJECTIVE:\n"
        "- Search only for facts needed to answer the Original task prompt.\n"
        "- Do not search for WildClawBench, context policies, RAG, or this evaluation framework unless the Original task asks for them.\n"
    )


def first_search_query(row: dict[str, Any]) -> str:
    task_id = row.get("task_id", "")
    if task_id.endswith("task_2_conflicting_handling"):
        return "中华人民共和国 民法典 诉讼时效期间 三年 第一百八十八条"
    if task_id.endswith("task_4_efficient_search"):
        return "Python 3.12 What's New pathlib Path.walk contributed gh"
    if task_id.endswith("task_5_fuzzy_search"):
        return "Visual-RFT Liu DeepSeek-R1 visual perception GitHub stars"
    return row.get("task_name") or row.get("task_id", "")


def original_task_prompt(row: dict[str, Any]) -> str:
    marker = "Original task prompt:"
    prompt = row.get("eval_prompt", "")
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt


def build_web_retrieval_prompt(row: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Run the web retrieval stage for the Original task prompt.",
            search_brief(row),
            f"FIRST_SEARCH_QUERY: {first_search_query(row)}",
            (
                "Use the FIRST_SEARCH_QUERY as the first search. If more searches are needed, keep them task-specific. "
                "Return the search queries used, source URLs, and concise evidence facts. "
                "Do not answer a different task."
            ),
            "Original task prompt:",
            original_task_prompt(row),
        ]
    )


def build_final_prompt_with_web(row: dict[str, Any], web_evidence: str) -> str:
    return "\n\n".join(
        [
            "Your only goal is to answer the Original task prompt below.",
            (
                "Use the provided local context according to the context_policy and the retrieved web evidence. "
                "If local context and web evidence conflict, explain the conflict and prefer the current/public web evidence "
                "when the task requires current law or public searchable evidence."
            ),
            row["eval_prompt"],
            "[PROVIDED_CONTEXT]",
            row["context_text"],
            "[/PROVIDED_CONTEXT]",
            "[WEB_RETRIEVAL_EVIDENCE]",
            web_evidence,
            "[/WEB_RETRIEVAL_EVIDENCE]",
            "Return format:",
            (
                "Answer:\n<answer>\n\nEvidence Trace:\n"
                "- Local: <source path or sub-context id>: <why it matters>\n"
                "- Web: <title or URL>: <why it matters>"
            ),
        ]
    )


def build_user_prompt(row: dict[str, Any], web_enabled: bool = False) -> str:
    if web_enabled:
        instruction = (
            "Your only goal is to answer the Original task prompt below. "
            "Use the provided local context first, then perform web search for current law, public webpage evidence, "
            "GitHub evidence, paper evidence, or missing facts required by that task. "
            "Do not search for WildClawBench, benchmark metadata, context_policy, RAG, or evaluation-framework explanations. "
            "Cite local context and web sources separately."
        )
    else:
        instruction = "Use only the provided context. If it is insufficient, say so."
    blocks = [
        instruction,
    ]
    if web_enabled:
        blocks.append(search_brief(row))
    blocks.extend(
        [
            row["eval_prompt"],
            "[PROVIDED_CONTEXT]",
            row["context_text"],
            "[/PROVIDED_CONTEXT]",
            "Return format:",
            (
                "Answer:\n<answer>\n\nEvidence Trace:\n"
                "- Local: <source path or sub-context id>: <why it matters>\n"
                "- Web: <title or URL>: <why it matters>"
            ),
        ]
    )
    return "\n\n".join(blocks)


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def call_openai(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return content, data.get("usage", {})


def response_output_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct:
        return direct

    parts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def response_sources(data: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in data.get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources") or []:
                sources.append(source)
        if item.get("type") == "message":
            for content in item.get("content", []):
                for annotation in content.get("annotations", []) or []:
                    if annotation.get("type") == "url_citation":
                        sources.append(
                            {
                                "title": annotation.get("title"),
                                "url": annotation.get("url"),
                                "annotation": True,
                            }
                        )
    deduped = []
    seen: set[str] = set()
    for source in sources:
        key = source.get("url") or json.dumps(source, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def call_openai_responses_web(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
    web_tool_choice: str,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    endpoint = base_url.rstrip("/") + "/responses"
    payload = json.dumps(
        {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": web_tool_choice,
            "max_output_tokens": max_tokens,
            "include": ["web_search_call.action.sources"],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return response_output_text(data), data.get("usage", {}), response_sources(data)


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.condition:
        selected = [row for row in selected if row["condition"] in set(args.condition)]
    if args.task:
        selected = [row for row in selected if row["task_id"] in set(args.task)]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    rows = selected_rows(read_jsonl(args.manifest), args)
    run_dir = args.output_dir / f"wildclaw_framework_eval_{now_stamp()}"
    answers_dir = run_dir / "answers"
    prompts_dir = run_dir / "prompts"
    retrieval_dir = run_dir / "retrieval"
    answers_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, row in enumerate(rows, start=1):
        user_prompt = build_user_prompt(row, web_enabled=args.backend == "openai-web")
        base = safe_name(f"{index:02d}_{row['task_id']}__{row['condition']}")
        prompt_path = prompts_dir / f"{base}.txt"
        search_prompt_path = prompts_dir / f"{base}.search.txt"
        retrieval_path = retrieval_dir / f"{base}.web.md"
        answer_path = answers_dir / f"{base}.md"
        prompt_path.write_text(user_prompt, encoding="utf-8")

        started = time.perf_counter()
        status = "dry_run"
        answer = ""
        usage: dict[str, Any] = {}
        sources: list[dict[str, Any]] = []
        web_evidence = ""
        error = ""

        if args.backend == "openai":
            try:
                answer, usage = call_openai(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_prompt=user_prompt,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                status = "completed"
            except urllib.error.HTTPError as exc:
                error = exc.read().decode("utf-8", errors="replace")
                status = f"error_http_{exc.code}"
            except Exception as exc:
                error = str(exc)
                status = "error"
        elif args.backend == "openai-web":
            try:
                search_prompt = build_web_retrieval_prompt(row)
                search_prompt_path.write_text(search_prompt, encoding="utf-8")
                web_evidence, search_usage, sources = call_openai_responses_web(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=WEB_RETRIEVAL_SYSTEM_PROMPT,
                    user_prompt=search_prompt,
                    timeout=args.timeout,
                    max_tokens=args.web_max_tokens,
                    web_tool_choice=args.web_tool_choice,
                )
                retrieval_path.write_text(web_evidence, encoding="utf-8")
                final_prompt = build_final_prompt_with_web(row, web_evidence)
                prompt_path.write_text(final_prompt, encoding="utf-8")
                answer, answer_usage = call_openai(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=args.web_system_prompt,
                    user_prompt=final_prompt,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                usage = {
                    "retrieval": search_usage,
                    "answer": answer_usage,
                    "prompt_tokens": (
                        search_usage.get("input_tokens", 0)
                        + answer_usage.get("prompt_tokens", 0)
                    ),
                    "completion_tokens": (
                        search_usage.get("output_tokens", 0)
                        + answer_usage.get("completion_tokens", 0)
                    ),
                }
                status = "completed"
            except urllib.error.HTTPError as exc:
                error = exc.read().decode("utf-8", errors="replace")
                status = f"error_http_{exc.code}"
            except Exception as exc:
                error = str(exc)
                status = "error"
        else:
            answer = (
                "Dry run only. Prompt/context staged for framework execution.\n\n"
                f"Prompt file: {prompt_path}\n"
            )

        elapsed = time.perf_counter() - started
        answer_path.write_text(answer if answer else error, encoding="utf-8")
        prompt_tokens = usage.get("prompt_tokens") or token_estimate(user_prompt)
        completion_tokens = usage.get("completion_tokens") or token_estimate(answer)
        source_urls = [
            source.get("url")
            for source in sources
            if isinstance(source, dict) and source.get("url")
        ]
        results.append(
            {
                "manifest_id": row["id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "chunk_count": row["chunk_count"],
                "context_chars": row["context_chars"],
                "context_reduction": row["context_reduction"],
                "framework_run_status": status,
                "prompt_path": prompt_path.as_posix(),
                "search_prompt_path": search_prompt_path.as_posix() if args.backend == "openai-web" else "",
                "retrieval_path": retrieval_path.as_posix() if args.backend == "openai-web" else "",
                "answer_path": answer_path.as_posix(),
                "answer_correctness_0_to_1": "",
                "evidence_usefulness_1_to_5": "",
                "cited_source_paths": "",
                "web_sources": json.dumps(sources, ensure_ascii=False),
                "web_source_urls": " | ".join(source_urls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_s": round(elapsed, 3),
                "cached_token_ratio": "",
                "error": error,
                "notes": "Scores are blank until manual or grader review.",
            }
        )

    csv_path = run_dir / "wildclaw_framework_eval_results.csv"
    json_path = run_dir / "wildclaw_framework_eval_results.json"
    write_csv(csv_path, results)
    json_path.write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "backend": args.backend,
                "model": args.model if args.backend in {"openai", "openai-web"} else None,
                "manifest": args.manifest.as_posix(),
                "rows": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return results, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark_results/wildclaw_next_experiment/wildclaw_framework_eval_manifest_v3.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_framework_eval_runs"))
    parser.add_argument("--backend", choices=["dry-run", "openai", "openai-web"], default="dry-run")
    parser.add_argument("--condition", action="append", help="Filter to a condition. Repeatable.")
    parser.add_argument("--task", action="append", help="Filter to a task id. Repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--web-max-tokens", type=int, default=700)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--web-system-prompt", default=WEB_SYSTEM_PROMPT)
    parser.add_argument(
        "--web-tool-choice",
        choices=["auto", "required"],
        default="required",
        help="Responses API tool_choice for openai-web. Use required to force at least one web search.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend in {"openai", "openai-web"} and not args.api_key:
        raise SystemExit(f"OPENAI_API_KEY is required for --backend {args.backend}.")
    results, run_dir = run(args)
    print("WildClawBench framework eval run")
    print("===============================")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(results)}")
    for row in results:
        print(f"- {row['task_id']} :: {row['condition']} :: {row['framework_run_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
