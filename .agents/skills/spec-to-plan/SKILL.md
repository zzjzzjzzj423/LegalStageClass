---
name: spec-to-plan
description: Convert a specification file into a step-by-step implementation plan for agents. Use when the user asks to create a plan, implementation plan, execution plan, or agent-ready plan from a spec path, with an optional custom output path.
---

# Spec To Plan

## Operating Principle

Act as a supervisor: route information between spec-understanding, plan-writing, and validation subagents, and resolve material ambiguity before finalizing the plan.

## Inputs

- Require a readable source spec path; stop and report the issue if it is missing or unreadable.
- Resolve the output path before delegation: default to `plan.md` beside the spec, use `plan.md` inside a provided directory, or use a provided file path as-is.

## Workflow

1. Spawn an understanding subagent with the spec path. Have it read the spec, inspect local project context when useful, and return:
   - its concise understanding of the required outcome,
   - relevant constraints, dependencies, risks, and likely implementation areas,
   - ambiguities, missing requirements, conflicts, and low-confidence assumptions,
   - clarifying questions, each with one recommended action and `2-3` other viable options.
2. Ask the user the material clarification questions, preserving the recommended action for each one. Return the answers to the understanding subagent and repeat until it can produce a final understanding packet with no unresolved material questions.
3. Spawn a planning subagent with the spec path, output path, final understanding packet, and accepted clarifications. Have it write the plan directly to the output path.
4. Spawn a validation subagent with the spec path, output path, final understanding packet, and accepted clarifications. Have it read the artifacts, inspect local project context when useful, and check that the plan fully adheres to the spec.
5. Resolve validation findings through the smallest necessary loop:
   - if the understanding packet or accepted clarifications already answer a finding, send that context to the planning subagent for revision;
   - if not, ask the user for clarification using the same recommended-action format, then update understanding and planning before validating again.
6. Repeat revision and validation until the validator reports no unresolved gaps, ambiguities, conflicts, or low-confidence areas.

## Plan Requirements

The planning subagent must create an agent-ready `plan.md` that includes:

- source spec path and output path,
- accepted clarifications and assumptions,
- ordered implementation steps with enough detail for another agent to execute,
- files, modules, systems, or docs likely to be touched when known,
- dependencies, sequencing constraints, and risk areas,
- validation steps, tests, or review checks required before considering the spec implemented.

Do not include unresolved questions in the final plan. If a question remains, continue the clarification loop instead of writing or finalizing the plan.

## Final Response

After validation passes, report the plan path, the subagent passes completed, and that no unresolved spec-to-plan questions remain.
