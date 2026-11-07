# Repository workflow

## Required development route

- For every change that adds or changes observable behavior, load `$legal-stage-class-sdd` before editing implementation code.
- Follow the repository stages in order: an approved EGM design in `docs/designs/`, a compatible plan in `docs/plans/`, then `$executing-plans`.
- Resume from the first incomplete stage. Do not silently replace a missing stage with an ad hoc chat plan or direct implementation.
- The EGM skills require explicit user interaction. If the required EGM, `$spec-to-plan`, or `$executing-plans` capability is unavailable, report the missing capability and stop before the affected stage.
- Typo-only documentation changes, comment-only changes, and behavior-preserving formatting are exempt. Tests, refactors, bug fixes, experiment changes, and configuration semantics are not exempt.

## Sources of truth

- Use `README.md` for project scope and `docs/pipeline.md` for experiment topology and implementation order.
- Treat an approved design as the authority for intended behavior. Treat its derived plan as the execution procedure.
- Keep the three classification interventions independent. Do not stack them into one experimental treatment unless an approved design explicitly changes that constraint.
- Never tune against the reserved final evaluation split.

## Verification

- Every implementation plan must end with a real-chain end-to-end acceptance task as defined by `$legal-stage-class-sdd`.
- Do not claim completion until the plan's focused checks and final end-to-end command have run successfully and their evidence has been inspected.
- Do not mock repository-owned stages in the final end-to-end path. Access to paid APIs, remote services, or protected datasets still requires the user's explicit authorization.
