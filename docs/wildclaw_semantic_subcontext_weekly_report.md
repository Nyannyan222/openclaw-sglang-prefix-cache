# WildClawBench Semantic Sub-context Weekly Report

## Overview

This week, I shifted the evaluation source from synthetic A/B/C contexts to
real WildClawBench task workspaces. The main goal was to build a more realistic
pipeline for identifying semantic sub-contexts and evaluating whether any of
them can be safely reused across tasks or prompts.

The work focused on three parts:

1. defining semantic sub-contexts,
2. finding reusable candidates,
3. avoiding incorrect reuse.

## 1. Defining Sub-contexts

The first step was to define what a sub-context means in this project. Instead
of treating context as arbitrary fixed-size chunks, I defined a sub-context as a
semantically independent unit extracted from a real task context. A valid
sub-context should contain enough information to be evaluated on its own, while
still preserving its source metadata.

I used WildClawBench real task data as the source. For each selected task, the
pipeline extracts:

- the task prompt,
- workspace files,
- source file paths,
- task id and category,
- extracted long context,
- leakage-filtered metadata.

The extractor intentionally excludes answer-like, grading, ground-truth, and
result files to avoid evaluation leakage.

After building the long context, I used LLM-based semantic segmentation to split
the context into smaller semantic sub-contexts. Each extracted sub-context keeps
metadata such as:

- sub-context id,
- task id,
- category,
- title or summary,
- source file/span information,
- character length,
- content hash,
- extraction method.

I also generated two baseline chunking methods for comparison:

- fixed-size chunking,
- file-based chunking.

This makes it possible to compare semantic segmentation against simpler
chunking strategies in terms of independence, relevance, completeness,
redundancy, and context reduction.

Main artifacts:

- `scripts/real_context_extractor.py`
- `scripts/llm_semantic_segmenter.py`
- `benchmark_results/wildclaw_semantic_subcontext_pilot_v3/`
- `benchmark_results/wildclaw_mixed_category_eval/`

## 2. Finding Reusable Candidates

After defining semantic sub-contexts, the next goal was to find whether any
sub-contexts are reusable. Here, reusable does not simply mean that two contexts
share the same topic. It means that two sub-contexts have sufficiently similar
semantic meaning or evidence utility that they may support the same answer or
reasoning role.

To search for reusable candidates, I implemented a semantic similarity discovery
pipeline. The pipeline uses multiple filtering stages:

1. lexical prefiltering,
2. embedding similarity,
3. LLM judge evaluation,
4. optional manual review.

The lexical prefilter is only used to cheaply reduce the search space. The main
semantic signal comes from embeddings and the LLM judge.

I also separated request relevance from reuse eligibility. A sub-context can be
relevant to the current request without being reusable as an equivalent context.
The reuse relation taxonomy is:

| relation type | meaning | reuse eligibility |
| --- | --- | --- |
| `exact_duplicate` | content is nearly identical after normalization | yes |
| `near_duplicate` | different wording but potentially equivalent information | yes, after judge |
| `same_answer_utility` | either context can support the same answer/evidence role | yes |
| `partial_overlap` | some information overlaps, but evidence role may differ | no / maybe |
| `broad_topic` | same topic/domain but different use | no |
| `unrelated` | no meaningful relation | no |

This makes the contribution clearer: embedding similarity finds candidates, and
the LLM judge separates reusable evidence from merely related context.

The strict matching criteria require:

- sufficient embedding cosine similarity,
- LLM judge score above the threshold,
- `same_answer_utility = true`.

This is stricter than ordinary topic similarity. For example, two Chinese legal
documents may both discuss regulation or civil law, but that does not mean they
can be reused for the same question.

I also built a WildClaw task/workspace selector to find better future source
tasks. The selector ranks tasks based on whether their workspace is likely to
contain repeated or near-repeated evidence. It checks signals such as:

- local workspace availability,
- number of usable workspace files,
- exact duplicate files,
- exact duplicate text segments,
- near-duplicate text segments,
- domain hints such as law, policy, safety, API docs, and instructions.

In the current local WildClawBench checkout, the selector found that
`04_Search_Retrieval_task_2_conflicting_handling` is the strongest candidate,
because it is the only task with substantial usable local workspace text.

Main artifacts:

- `scripts/find_semantic_similar_subcontexts.py`
- `scripts/select_wildclaw_similarity_tasks.py`
- `docs/semantic_subcontext_similarity_discovery.md`
- `docs/wildclaw_similarity_task_selector.md`
- `benchmark_results/semantic_similarity_discovery_openai_strict/`
- `benchmark_results/semantic_similarity_discovery_mixed_strict/`
- `benchmark_results/wildclaw_similarity_task_selection/`

## 3. Avoiding Incorrect Reuse

A key finding this week is that semantic reuse is risky if the matching rule is
too loose. Many sub-contexts look related at the topic level, but they are not
actually interchangeable.

For example, in the mixed-category strict run, the strongest embedding
neighbors were still rejected by the LLM judge. Several legal documents had
moderate embedding similarity because they shared a legal or regulatory domain,
but the judge labeled them as:

- `partial_overlap`,
- `broad_topic`,
- `unrelated`.

Most importantly, these pairs had `same_answer_utility = false`, meaning they
should not be reused as equivalent evidence.

This led to an important design decision: I stopped using the earlier
prefix-sharing prompt rewrite direction. That approach could make prompts share
the same prefix and improve SGLang prefix-cache hits, but it does not prove
that the original sub-contexts are semantically reusable. The current direction
is therefore focused on discovering genuine semantic similarity, not
manufacturing prefix-cache reuse.

Current strict results:

| experiment | pairs evaluated | semantic matches | groups | conclusion |
| --- | ---: | ---: | ---: | --- |
| Search Retrieval strict run | 45 | 0 | 0 | related contexts exist, but no safe reusable match |
| Mixed-category strict run | 66 | 0 | 0 | strongest pairs are broad-topic or partial-overlap |

This negative result is useful. It shows that the current pipeline can reject
unsafe reuse candidates instead of forcing reuse when the evidence does not
support it.

Main safety checks:

- exclude answer/grading/ground-truth files,
- require embedding similarity,
- require LLM judge agreement,
- require same-answer utility,
- separate true semantic similarity from broad topical similarity,
- avoid prompt rewriting methods that only create artificial prefix overlap.

## Summary

This week, I built a WildClawBench real-context semantic sub-context pipeline.
The work started from defining semantic sub-contexts, then moved to discovering
potential reusable candidates, and finally added stricter filters to avoid
incorrect reuse.

The main conclusion is that not all related contexts are reusable. The current
WildClawBench pilots contain many partial-overlap or broad-topic pairs, but the
strict semantic similarity pipeline has not yet found safe interchangeable
sub-contexts. This supports the need for a careful reuse mechanism based on
semantic utility rather than simple topic similarity or prefix overlap.

## Next Steps

The next step is to expand the task selection using the WildClaw task/workspace
selector. The goal is to find more tasks with repeated or near-repeated real
evidence, then rerun:

1. real context extraction,
2. LLM semantic segmentation,
3. strict semantic similarity discovery,
4. manual review of top candidates.

If a larger WildClawBench workspace becomes available, the selector should be
rerun first so that the next experiment focuses on tasks with stronger reuse
potential.
