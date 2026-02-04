# Metrics schema (recommended)

The runner validates that `evaluation.required_keys` exist in the JSON at `evaluation.metrics_path` (preferred),
or that `benchmark.required_keys` exist in the JSON at `benchmark.metrics_path`.
To make results comparable across repos (evaluation + rollout + post-training), this doc recommends a stable schema.

## File(s)

- `evaluation.metrics_path` or `benchmark.metrics_path` (required if you want metrics validation)
- Optional rollout artifact: `.aider_fsm/rollout.json`

## Minimal metrics JSON

Recommended minimum:

- `score`: number (or `eval.score`)
- `ts`: ISO timestamp string
- `run_id`: string (you can copy `AIDER_FSM_RUN_ID`)

## Recommended keys by area

### Evaluation

- `eval.score`: overall scalar score (primary KPI)
- `eval.details`: optional object/array for per-task breakdown

### Rollout (post-training RL)

- `rollout.ok`: boolean
- `rollout.n_episodes`: integer
- `rollout.success_rate`: number in `[0, 1]`
- `rollout.failures_by_type`: object mapping failure type -> count
- `rollout.avg_latency_ms`: number (optional)

### Training (post-training)

- `train.ok`: boolean
- `train.steps`: integer
- `train.wall_time_s`: number
- `train.loss`: number (optional)

## Notes

- Keep the metrics JSON small and stable; write full logs to artifacts.
- Use `AIDER_FSM_*` env vars to embed provenance in metrics and rollout artifacts.
