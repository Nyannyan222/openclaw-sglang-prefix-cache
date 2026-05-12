# WildClawBench Framework Evaluation Run Summary

Run directory:

```text
benchmark_results/wildclaw_framework_eval_runs/wildclaw_framework_eval_20260512_212954
```

Backend:

```text
OpenAI-compatible Chat Completions, gpt-4o-mini
```

## Conditions

| Condition | Rows | Accuracy | Avg Evidence Usefulness | Avg Prompt Tokens | Avg Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate_evidence_context` | 3 | 0.00 | 2.67 / 5 | 2695.7 | 2.17 |
| `file_based_chunks` | 3 | 0.00 | 2.67 / 5 | 2628.7 | 2.14 |
| `fixed_size_chunks` | 3 | 0.00 | 2.67 / 5 | 2963.0 | 2.59 |
| `v3_semantic_final_pass` | 3 | 0.00 | 2.33 / 5 | 685.0 | 3.05 |

## Interpretation

This run successfully executes all 12 manifest rows, but none of the four
context policies solves the full WildClawBench tasks under the current runner.
That is expected because the runner instructs the model to use only the provided
context and does not give it live search/web tools.

Important observations:

- For task 2, every policy answers the local outdated `2 years` rule from
  `law12.pdf`. This confirms that V3 preserves the local conflict evidence, but
  it also shows that final correctness requires a web/current-law verification
  stage.
- For task 4, most policies correctly abstain with "Unable to confirm" because
  the provided context contains task constraints but no public webpage evidence
  for Python 3.12 or CPython PR #92517.
- For task 5, prompt-only context is insufficient. The V3 condition risks
  over-inference because it says it found the paper without supplying the actual
  expected title.

## Next Step

Add a retrieval/search execution stage before final answering:

```text
context policy -> framework search/retrieval -> answer with evidence trace
```

For the next run, the framework should be allowed to fetch or search external
evidence for tasks 2, 4, and 5, while still using V3 semantic sub-contexts as
the compact real-context input.

The key claim so far is not answer accuracy. The key claim is:

```text
V3 semantic sub-contexts preserve the useful local evidence with far fewer
prompt tokens, but task-level correctness requires tool-augmented retrieval for
current/public evidence.
```
