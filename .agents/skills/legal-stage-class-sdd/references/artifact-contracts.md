# Artifact contracts

Read this reference while drafting or validating a design or plan.

## Approved design contract

An implementation-ready design must state:

- status, owner, date, and source context;
- problem, goals, non-goals, and affected experiment;
- system boundary, invariants, inputs, outputs, and failure behavior;
- components and interfaces affected, with compatibility and migration decisions;
- data split policy, reproducibility controls, observability, cost, and recovery concerns;
- acceptance criteria linked to one or more real-chain scenarios;
- unresolved decisions, which must be empty or explicitly deferred before approval.

For this repository, preserve these invariants unless the design explicitly supersedes them:

- pre-retrieval routing, during-retrieval tagging, and post-retrieval selection are independent interventions over the shared baseline;
- tuning uses validation data, never the reserved final evaluation split;
- every run records dataset split version, model and retriever versions, prompts or thresholds, seed, latency, cost, and relevant strategy parameters.

## Plan contract for `$executing-plans`

The plan must name its source design and include global constraints and review focus. Each task must contain:

- objective and dependencies;
- exact files to create, modify, or delete;
- interfaces consumed and produced, including data shapes or signatures;
- ordered checkbox steps in RED-GREEN-REFACTOR order where behavior changes;
- test code or precise assertions before implementation steps;
- an exact command for every verification step and an `Expected:` result;
- a whole-task verification command;
- a commit step when the executor requires task commits.

Tasks that share an interface must make that dependency explicit. The plan may not defer architectural choices or acceptance semantics to the implementation session.

## Required final end-to-end task

The final task depends on all implementation tasks and defines:

- environment and setup command;
- bounded but representative input data and deterministic seed;
- the public entry point used by a real user or experiment runner;
- all repository-owned stages that must remain real;
- any external provider sandbox or live boundary, including authorization and cost limits;
- exact execution command and expected exit status;
- expected ranking, metric, manifest, and log artifacts;
- assertions over artifact schema, provenance, and important semantic outcomes;
- cleanup and rerun behavior;
- evidence paths retained for review.

For LegalStageClass, the owned real chain normally means:

```text
experiment config
  -> dataset loading and validation
  -> index construction or loading
  -> BM25 and/or dense retrieval
  -> the selected independent intervention
  -> fusion or reranking
  -> metric computation
  -> rankings, metrics, cost/latency data, and run manifest
```

The end-to-end test must enter through the actual CLI or experiment entry point and inspect generated artifacts. It must not call internal functions directly to bypass configuration, I/O, indexing, retrieval, or evaluation. A compact checked-in fixture may replace the full STARD corpus for routine runs, but it must pass through the same parsers and pipeline code.

Repository-owned modules must not be mocked in this final path. External paid models should use a real sandbox or an explicitly authorized bounded live run. If that boundary cannot run, keep a separate offline contract or integration check, mark the live E2E gate incomplete, and do not report complete real-chain coverage.
