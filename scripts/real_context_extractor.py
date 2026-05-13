#!/usr/bin/env python3
"""Extract real WildClawBench task contexts for semantic sub-context pilots.

The extractor reads WildClawBench task markdown files plus their corresponding
workspace files and writes a JSONL file where each row is one long-context task.
It intentionally excludes grading/ground-truth material so the resulting pilot
data can be used as an evaluation source without answer leakage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


DEFAULT_PILOT_TASKS = [
    "04_Search_Retrieval_task_2_conflicting_handling",
    "04_Search_Retrieval_task_4_efficient_search",
    "04_Search_Retrieval_task_5_fuzzy_search",
]

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".csv",
    ".css",
    ".go",
    ".h",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "gt",
    "node_modules",
    "output",
    "outputs",
    "result",
    "results",
}

EXCLUDED_FILE_PATTERNS = (
    "answer",
    "ground_truth",
    "gt.",
    "grading",
    "score",
)


@dataclass(frozen=True)
class SourceFile:
    path: str
    role: str
    bytes: int
    sha256: str
    line_count: int | None
    included_in_context: bool
    skipped_reason: str | None
    content: str | None


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_bytes(data: bytes) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_docx_text(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError):
        return None

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return None

    paragraphs: list[str] = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def read_pdf_text(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    except Exception:
        return None
    return "\n\n".join(pages) if pages else None


def normalize_text(text: str, max_chars: int) -> tuple[str, bool]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[TRUNCATED_BY_EXTRACTOR]", True


TASK_SECTION_HEADINGS = (
    "Prompt",
    "Expected Behavior",
    "Grading Criteria",
    "Automated Checks",
    "Workspace Path",
    "Skills",
    "Env",
    "Warmup",
)


def normalize_task_markdown(markdown: str) -> str:
    """Make compact WildClawBench task markdown easier to section-parse."""
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    for heading in TASK_SECTION_HEADINGS:
        normalized = re.sub(rf"\s+##\s+{re.escape(heading)}\b", f"\n## {heading}", normalized)
    return normalized


def parse_sections(markdown: str) -> dict[str, str]:
    markdown = normalize_task_markdown(markdown)
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[heading] = markdown[start:end].strip()
    return sections


def parse_front_matter(markdown: str) -> dict[str, str]:
    markdown = normalize_task_markdown(markdown)
    lines = markdown.splitlines()
    first_line = lines[0] if lines else ""
    if not first_line.startswith("---"):
        return {}

    if first_line.strip() == "---":
        block_lines: list[str] = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            block_lines.append(line)
        block = "\n".join(block_lines)
        fields: dict[str, str] = {}
        for line in block_lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        if fields:
            return fields
    else:
        block = first_line

    # Some WildClawBench task files are compact; fields may appear on one line.
    fields: dict[str, str] = {}
    pattern = re.compile(r"\b(id|name|category|timeout_seconds):\s*(.*?)(?=\s+\b(?:id|name|category|timeout_seconds):| ---|$)")
    for key, value in pattern.findall(block):
        fields[key] = value.strip()
    return fields


def extract_code_block(section: str) -> str | None:
    match = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\s*(.*?)```", section, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def task_short_name(task_id: str) -> str:
    match = re.match(r"\d+_[A-Za-z_]+_task_(\d+_.+)", task_id)
    if match:
        return f"task_{match.group(1)}"
    return task_id


def task_file_for(task_root: Path, category: str, task_id: str) -> Path:
    direct = task_root / category / f"{task_id}.md"
    if direct.exists():
        return direct
    candidates = sorted((task_root / category).glob(f"*{task_short_name(task_id)}.md"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Task markdown not found for {task_id} under {task_root / category}")


def workspace_path_for(workspace_root: Path, task: dict[str, Any]) -> Path:
    if task.get("workspace_path"):
        rel = str(task["workspace_path"]).replace("/", "\\")
        candidate = workspace_root.parent / rel
        if candidate.exists():
            return candidate
    return workspace_root / task["category"] / task_short_name(task["task_id"])


def should_skip_path(path: Path, root: Path) -> str | None:
    rel_parts = path.relative_to(root).parts
    for part in rel_parts[:-1]:
        if part.lower() in EXCLUDED_DIR_NAMES:
            return f"excluded directory: {part}"
    lowered = path.name.lower()
    for pattern in EXCLUDED_FILE_PATTERNS:
        if pattern in lowered:
            return f"excluded leakage-like filename pattern: {pattern}"
    return None


def read_source_file(path: Path, root: Path, max_file_bytes: int, max_chars: int) -> SourceFile:
    rel = path.relative_to(root).as_posix()
    data = path.read_bytes()
    digest = sha256_bytes(data)
    skip_reason = should_skip_path(path, root)
    if skip_reason:
        return SourceFile(rel, "workspace", len(data), digest, None, False, skip_reason, None)
    if len(data) > max_file_bytes:
        return SourceFile(rel, "workspace", len(data), digest, None, False, "file exceeds max_file_bytes", None)

    text: str | None
    if path.suffix.lower() == ".docx":
        text = read_docx_text(path)
    elif path.suffix.lower() == ".pdf":
        text = read_pdf_text(path)
    elif path.suffix.lower() in TEXT_SUFFIXES or path.suffix == "":
        text = read_text_bytes(data)
    else:
        text = None

    if text is None:
        return SourceFile(rel, "workspace", len(data), digest, None, False, "binary or unsupported text extraction", None)

    normalized, truncated = normalize_text(text, max_chars=max_chars)
    reason = "content truncated by max_chars" if truncated else None
    return SourceFile(
        rel,
        "workspace",
        len(data),
        digest,
        normalized.count("\n") + 1 if normalized else 0,
        True,
        reason,
        normalized,
    )


def parse_task_markdown(path: Path) -> dict[str, Any]:
    markdown = path.read_text(encoding="utf-8")
    sections = parse_sections(markdown)
    front_matter = parse_front_matter(markdown)
    task_id = front_matter.get("id") or path.stem
    category = front_matter.get("category") or path.parent.name
    workspace_path = extract_code_block(sections.get("Workspace Path", ""))
    skills = extract_code_block(sections.get("Skills", "")) or ""
    env = extract_code_block(sections.get("Env", "")) or ""
    return {
        "task_id": task_id,
        "name": front_matter.get("name", ""),
        "category": category,
        "timeout_seconds": int(front_matter.get("timeout_seconds", "0") or 0),
        "prompt": sections.get("Prompt", "").strip(),
        "expected_behavior": sections.get("Expected Behavior", "").strip(),
        "workspace_path": workspace_path,
        "skills": [line.strip() for line in skills.splitlines() if line.strip()],
        "env": [line.strip() for line in env.splitlines() if line.strip()],
        "task_markdown_path": path.as_posix(),
    }


def build_long_context(task: dict[str, Any], files: list[SourceFile]) -> str:
    parts = [
        f"[TASK_ID]\n{task['task_id']}",
        f"[CATEGORY]\n{task['category']}",
        f"[TASK_NAME]\n{task.get('name') or ''}",
        f"[PROMPT]\n{task.get('prompt') or ''}",
    ]
    for source in files:
        if not source.included_in_context or source.content is None:
            parts.append(
                "\n".join(
                    [
                        f"[WORKSPACE_FILE_METADATA path={source.path} sha256={source.sha256[:12]}]",
                        f"bytes={source.bytes}",
                        f"skipped_reason={source.skipped_reason}",
                        "[/WORKSPACE_FILE_METADATA]",
                    ]
                )
            )
            continue
        parts.append(
            "\n".join(
                [
                    f"[WORKSPACE_FILE path={source.path} sha256={source.sha256[:12]}]",
                    source.content,
                    "[/WORKSPACE_FILE]",
                ]
            )
        )
    return "\n\n".join(parts)


def extract_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wildclaw_root = args.wildclaw_root.resolve()
    task_root = args.tasks_root.resolve() if args.tasks_root else wildclaw_root / "tasks"
    workspace_root = args.workspace_root.resolve() if args.workspace_root else wildclaw_root / "workspace"
    tasks: list[dict[str, Any]] = []
    skipped_files = 0
    included_files = 0

    for task_id in args.task:
        category = task_id.split("_task_", 1)[0]
        task_md = task_file_for(task_root, category, task_id)
        task = parse_task_markdown(task_md)
        workspace_dir = workspace_path_for(workspace_root, task)
        workspace_missing = not workspace_dir.exists()
        if workspace_missing and not args.allow_missing_workspace:
            raise FileNotFoundError(f"Workspace directory not found: {workspace_dir}")

        files = (
            [
                read_source_file(path, workspace_dir, args.max_file_bytes, args.max_chars_per_file)
                for path in sorted(workspace_dir.rglob("*"))
                if path.is_file()
            ]
            if not workspace_missing
            else []
        )
        included_files += sum(1 for item in files if item.included_in_context)
        skipped_files += sum(1 for item in files if not item.included_in_context)
        long_context = build_long_context(task, files)
        tasks.append(
            {
                "schema_version": "wildclaw-real-context-v1",
                "source": "WildClawBench",
                "created_at": now_iso(),
                "task": task,
                "workspace_dir": workspace_dir.as_posix(),
                "workspace_missing": workspace_missing,
                "source_files": [asdict(item) for item in files],
                "long_context": long_context,
                "long_context_chars": len(long_context),
                "leakage_policy": {
                    "excluded_dirs": sorted(EXCLUDED_DIR_NAMES),
                    "excluded_file_patterns": list(EXCLUDED_FILE_PATTERNS),
                    "notes": "Ground-truth, grader, result, and answer-like files are excluded from the long context.",
                },
            }
        )

    inspection = {
        "created_at": now_iso(),
        "wildclaw_root": wildclaw_root.as_posix(),
        "tasks_root": task_root.as_posix(),
        "workspace_root": workspace_root.as_posix(),
        "selected_tasks": args.task,
        "task_count": len(tasks),
        "included_workspace_files": included_files,
        "skipped_workspace_files": skipped_files,
        "total_long_context_chars": sum(item["long_context_chars"] for item in tasks),
        "tasks": [
            {
                "task_id": item["task"]["task_id"],
                "name": item["task"].get("name", ""),
                "category": item["task"]["category"],
                "workspace_dir": item["workspace_dir"],
                "workspace_missing": item["workspace_missing"],
                "included_files": sum(1 for src in item["source_files"] if src["included_in_context"]),
                "skipped_files": sum(1 for src in item["source_files"] if not src["included_in_context"]),
                "long_context_chars": item["long_context_chars"],
            }
            for item in tasks
        ],
    }
    return tasks, inspection


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wildclaw-root",
        type=Path,
        default=Path("external/WildClawBench"),
        help="Root directory containing WildClawBench tasks/ and workspace/.",
    )
    parser.add_argument("--tasks-root", type=Path, default=None, help="Override WildClawBench tasks directory.")
    parser.add_argument("--workspace-root", type=Path, default=None, help="Override WildClawBench workspace directory.")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task id to extract. Repeat for multiple tasks. Defaults to the 3-task Search Retrieval pilot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/wildclaw_semantic_subcontext_pilot"),
    )
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-chars-per-file", type=int, default=80_000)
    parser.add_argument(
        "--allow-missing-workspace",
        action="store_true",
        help="Use prompt-only context rows when a task workspace is absent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.task:
        args.task = list(DEFAULT_PILOT_TASKS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, inspection = extract_tasks(args)
    raw_context_path = args.output_dir / "wildclaw_real_contexts_pilot.jsonl"
    inspection_path = args.output_dir / "wildclaw_data_inspection.json"
    write_jsonl(raw_context_path, rows)
    inspection_path.write_text(json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WildClawBench real-context extraction")
    print("=====================================")
    print(f"Tasks: {len(rows)}")
    print(f"Raw contexts: {raw_context_path}")
    print(f"Inspection: {inspection_path}")
    for task in inspection["tasks"]:
        print(
            f"- {task['task_id']}: {task['included_files']} included files, "
            f"{task['skipped_files']} skipped, {task['long_context_chars']} chars"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
