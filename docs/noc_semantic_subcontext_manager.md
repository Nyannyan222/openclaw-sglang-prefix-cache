# NOC Semantic Sub-context Manager Prototype

## Purpose

This prototype turns the WildClaw semantic sub-context experiments into a
formal context-management component for the new OpenClaw direction.

The component is still offline and metadata-level. It does not yet implement
runtime KV-cache reuse inside SGLang. Its job is to define the interfaces and
decisions needed before runtime integration:

1. represent semantic sub-contexts,
2. register and retrieve sub-contexts,
3. select request-relevant sub-contexts,
4. identify possible reuse candidates,
5. reject unsafe reuse by default.

The manager intentionally separates two questions:

- **Relevant:** should this sub-context be included in the current request's
  context window?
- **Reusable:** can this sub-context safely reuse another sub-context's evidence
  or future KV-cache block?

## Components

### `SubContext`

Canonical schema for one semantically independent context unit.

Important fields:

- `id`
- `text`
- `content_hash`
- `task_id`
- `category`
- `title`
- `question_or_objective`
- `source_spans`
- `expected_capability`
- quality signals such as independence, relevance, completeness, and redundancy

### `SubContextRegistry`

In-memory registry for loaded sub-contexts.

It supports lookup by:

- sub-context id,
- task id,
- category,
- exact normalized content hash.

This is the prototype version of a future persistent NOC sub-context store.

### `SubContextSelector`

Given a request profile, this component chooses a concise subset of relevant
sub-contexts.

Inputs:

- request query,
- task id,
- category,
- required capabilities,
- max number of contexts,
- max character budget.

Output:

- selected sub-contexts,
- selection score,
- selection reasons.

### `LexicalSimilarityJudge`

Offline similarity scorer used as a cheap first pass.

It computes:

- character shingle cosine,
- token Jaccard,
- combined lexical score,
- exact content hash match,
- same-task and same-category flags.

This component does not claim semantic equivalence. It only ranks possible
candidates before stricter embedding or LLM judging.

### `ReuseDecisionEngine`

Conservative reuse policy.

Reuse relation taxonomy:

| relation type | meaning | reuse eligibility |
| --- | --- | --- |
| `exact_duplicate` | content is nearly identical after normalization | yes |
| `near_duplicate` | different wording but potentially equivalent information | yes, after judge |
| `same_answer_utility` | either context can support the same answer/evidence role | yes |
| `partial_overlap` | some information overlaps, but evidence role may differ | no / maybe |
| `broad_topic` | same topic/domain but different use | no |
| `unrelated` | no meaningful relation | no |

This prevents topic-level similarity from being treated as true semantic reuse.
Embedding similarity finds candidates; an LLM judge or manual review separates
reusable evidence from merely related context.

## CLI

Run the prototype on an existing WildClaw semantic JSONL file:

```powershell
.\.venv\Scripts\python.exe scripts\run_noc_subcontext_manager.py `
  --semantic-jsonl benchmark_results\wildclaw_semantic_subcontext_pilot_v3\wildclaw_semantic_subcontext_pilot.jsonl `
  --output-dir benchmark_results\noc_subcontext_manager_smoke `
  --task-id 04_Search_Retrieval_task_2_conflicting_handling `
  --category 04_Search_Retrieval `
  --query "Evaluate statute of limitations and debt recovery evidence." `
  --required-capability retrieval `
  --required-capability reasoning `
  --same-category-only `
  --max-pairs 100
```

Outputs:

- `subcontext_registry_summary.json`
- `selected_subcontexts.jsonl`
- `reuse_decisions.jsonl`
- `reuse_decisions.csv`
- `noc_subcontext_manager_report.md`

## Current Role in the Final Project

This corresponds to Step 1 of the next implementation plan:

> Turn the semantic sub-context pipeline into a formal NOC context manager
> prototype.

It directly supports the final project goal of reducing context windows by
selecting only relevant sub-contexts for each LLM request.

It also prepares the interface for later runtime KV-cache work by giving every
sub-context a stable identity and conservative reuse decision.

## What It Does Not Do Yet

This prototype does not yet:

- run inside OpenClaw's planner/re-planner loop,
- dispatch requests to multiple LLM-backed agents,
- maintain multiple live KV caches,
- perform true non-prefix KV-cache reuse inside SGLang,
- replace backend LLM calls with local open-source LLM calls.

Those belong to later steps after this component interface is stable.
