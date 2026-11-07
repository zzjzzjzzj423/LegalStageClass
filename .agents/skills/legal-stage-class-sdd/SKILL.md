---
name: legal-stage-class-sdd
description: Route LegalStageClass behavior changes through an approved EGM design, an executable plan, and verified plan execution.
---

# LegalStageClass spec-driven development

Use this skill to determine and run the next approved stage of a behavior-changing task in this repository. It coordinates the chosen upstream skills; it does not replace them.

## Determine the current stage

1. Read the applicable `AGENTS.md`, `README.md`, `docs/pipeline.md`, and relevant existing design or plan. Inspect the worktree before writing and preserve unrelated changes.
2. Identify a stable feature slug. Store the approved design at `docs/designs/<slug>.md` and its plan at `docs/plans/<slug>.md`.
3. Resume at the first incomplete gate below. Never infer approval from the presence of a draft.

## Gate 1: approved specification

For new behavior or a behavioral correction without an approved design, use the Google Elephant-Goldfish workflow:

1. Use `egm-brainstorm` when goals, boundaries, or trade-offs are unresolved.
2. Use `egm-draft` to write the design.
3. Use `egm-review` and `egm-resolve` for Goldfish findings when applicable.
4. Obtain explicit human approval and save the approved document under `docs/designs/`.

EGM skills are interactive and may require the user to select them explicitly. If they are not installed or cannot be invoked, report that exact blocker; do not draft a substitute specification under this skill.

Before approval, ensure the design contains the acceptance boundary and real-chain scenario from [artifact contracts](references/artifact-contracts.md). Specification work must not edit implementation code.

## Gate 2: executable plan

When an approved design exists but no approved plan exists, invoke `$spec-to-plan` with the design path and target `docs/plans/<slug>.md`.

Validate the result against [artifact contracts](references/artifact-contracts.md). The plan must be directly consumable by `$executing-plans`, including RED-GREEN steps, exact commands and expected results, cross-task interfaces, and a final real-chain end-to-end task. Repair the plan before approval if any required field is absent. Do not begin implementation while the plan is unapproved.

## Gate 3: implementation

When both artifacts are approved, invoke `$executing-plans` with the exact plan path. The design is the authority; the plan is its execution procedure. Use the executor's ledger, TDD, debugging, verification, and final review behavior without replacing it with an informal loop.

Testing is part of the plan rather than a fourth workflow stage. The final end-to-end task must exercise the repository's real owned path and produce inspectable evidence. A smaller representative dataset is allowed, but substituting mocks for owned pipeline stages is not.

If a paid model API, remote service, protected dataset, publish, push, or other external side effect is needed, stop at that action and request the required authorization. Record skipped live boundaries honestly; do not call the task fully end-to-end when they were not exercised.

## Keep artifacts aligned

If implementation reveals a changed architectural decision, invariant, experiment boundary, or acceptance criterion, stop execution and return to the design gate. Regenerate or update the plan after the design is approved. Mechanical plan corrections that do not change intended behavior may follow `$executing-plans` ruling and ledger rules.
