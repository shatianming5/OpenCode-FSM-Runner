# Aider-FSM Runner (Overview)

This repo provides a small, auditable **closed-loop executor**:

1. Read `PLAN.md` + repo snapshot
2. Update the plan (model may only edit `PLAN.md`)
3. Execute exactly one `Next` step (model may not edit `PLAN.md` or `pipeline.yml`)
4. Verify via `pipeline.yml` (tests → deploy → benchmark → metrics)
5. If verification fails: fix or re-plan; optionally request `.aider_fsm/actions.yml`

The intent is to integrate with agent projects (e.g. RD-agent) where you need **repeatable post-training RL benchmark deployment + evaluation** with hard guardrails and artifacts.

## Key design constraints

- **Plan is machine-parseable**: `## Next` must contain exactly one unchecked `(STEP_ID=NNN)` item.
- **Pipeline is human-owned**: the runner reverts any model edits to the pipeline YAML.
- **Verification is deterministic**: runner executes commands and records artifacts (stdout/stderr, summaries).

## Files the runner uses (in the target repo)

- `PLAN.md`: plan + progress (machine-parseable format)
- `pipeline.yml` (optional): verification contract (see `docs/pipeline_spec.md`)
- `.aider_fsm/state.json`: runner state
- `.aider_fsm/logs/run_<id>.jsonl`: per-iteration logs
- `.aider_fsm/artifacts/<run_id>/...`: verification and actions artifacts

