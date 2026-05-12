# Local Mock SGLang Runtime Replay Smoke Test

Run directory:

```text
benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260513_012041/
```

Command:

```bash
python scripts/run_wildclaw_sglang_runtime_replay.py \
  --manifest benchmark_results/wildclaw_phase2/wildclaw_phase2_sglang_runtime_manifest.jsonl \
  --base-url http://127.0.0.1:18000/v1 \
  --metrics-url http://127.0.0.1:18000/metrics \
  --model mock-model \
  --repeat 2 \
  --max-tokens 64
```

Result:

```text
Rows: 24
Errors: 0
```

Replay-level summary:

| replay_index | rows | total_prompt_tokens | total_cached_tokens | total_prefill_tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 18161 | 0 | 18161 |
| 2 | 12 | 18161 | 18161 | 0 |

Condition-level summary:

| condition | rows | total_prompt_tokens | total_cached_tokens | total_prefill_tokens |
| --- | ---: | ---: | ---: | ---: |
| candidate_evidence_context | 6 | 10394 | 5197 | 5197 |
| file_based_chunks | 6 | 10052 | 5026 | 5026 |
| fixed_size_chunks | 6 | 10688 | 5344 | 5344 |
| v3_semantic_final_pass | 6 | 5188 | 2594 | 2594 |

Note: this is a local mock-server plumbing test, not a real SGLang runtime
measurement. It verifies that the replay script can send requests, read
OpenAI-compatible usage fields, scrape Prometheus-style metrics, and write
runtime CSV/JSON outputs. Real cached-token, prefill-token, TTFT, and latency
measurements still require a live SGLang server.
