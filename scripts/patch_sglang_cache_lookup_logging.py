#!/usr/bin/env python3
"""Patch installed SGLang to log Radix prefix-cache lookup details.

Run this with the same Python environment that will launch SGLang:

    python scripts/patch_sglang_cache_lookup_logging.py

The patch is intentionally small and guarded by:

    SGLANG_PREFIX_CACHE_DEBUG_LOG=1

When enabled, SGLang logs JSON lines with event="cache_lookup" from
schedule_policy.match_prefix_for_req().
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


MARKER = "# OPENCLAW_PREFIX_CACHE_LOOKUP_LOGGING"


def find_schedule_policy_path() -> pathlib.Path:
    try:
        spec = importlib.util.find_spec("sglang.srt.managers.schedule_policy")
    except Exception:
        spec = None
    if spec is not None and spec.origin is not None:
        return pathlib.Path(spec.origin)

    for entry in sys.path:
        if not entry:
            entry = "."
        candidate = (
            pathlib.Path(entry)
            / "sglang"
            / "srt"
            / "managers"
            / "schedule_policy.py"
        )
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate sglang.srt.managers.schedule_policy")


def patch_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False

    if "import json\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport json\n", 1)

    env_marker = '_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")'
    if env_marker not in text:
        raise RuntimeError("Could not find routing-key debug env marker")
    text = text.replace(
        env_marker,
        env_marker
        + '\n_PREFIX_CACHE_LOOKUP_DEBUG_LOG = get_bool_env_var("SGLANG_PREFIX_CACHE_DEBUG_LOG")',
        1,
    )

    helper_anchor = "IGNORE_EOS_RESERVE_TOKENS = 1\n"
    if helper_anchor not in text:
        raise RuntimeError("Could not find helper insertion anchor")
    helper = f'''

{MARKER}
def _openclaw_prefix_cache_node_id(node):
    return None if node is None else hex(id(node))


def _openclaw_log_prefix_cache_lookup(req, token_ids, match_result):
    if not _PREFIX_CACHE_LOOKUP_DEBUG_LOG:
        return

    input_token_len = len(token_ids)
    matched_prefix_len = len(match_result.device_indices)
    uncached_tokens = max(input_token_len - matched_prefix_len, 0)
    first_mismatch_token_position = (
        matched_prefix_len if matched_prefix_len < input_token_len else None
    )
    payload = {{
        "event": "cache_lookup",
        "rid": getattr(req, "rid", None),
        "input_token_len": input_token_len,
        "matched_prefix_len": matched_prefix_len,
        "matched_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_device_node
        ),
        "matched_host_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_host_node
        ),
        "cached_tokens": matched_prefix_len,
        "uncached_tokens": uncached_tokens,
        "first_mismatch_token_position": first_mismatch_token_position,
        "host_hit_length": match_result.host_hit_length,
        "cache_protected_len": match_result.cache_protected_len,
        "extra_key": repr(getattr(req, "extra_key", None)),
    }}
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
'''
    text = text.replace(helper_anchor, helper_anchor + helper, 1)

    call_anchor = """    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    return match_result
"""
    if call_anchor not in text:
        raise RuntimeError("Could not find match_prefix_for_req return anchor")
    text = text.replace(
        call_anchor,
        """    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    _openclaw_log_prefix_cache_lookup(req, token_ids, match_result)
    return match_result
""",
        1,
    )
    return text, True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the patch without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    target = find_schedule_policy_path()
    original = target.read_text(encoding="utf-8")
    patched, changed = patch_text(original)
    if not changed:
        print(f"Already patched: {target}")
        return 0

    compile(patched, str(target), "exec")
    if args.dry_run:
        print(f"Patch validates: {target}")
        return 0

    backup = target.with_suffix(target.suffix + ".openclaw-prefix-log.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
