# Metrics schema (recommended)

The runner validates that `evaluation.required_keys` exist in the JSON at `evaluation.metrics_path` (preferred),
or that `benchmark.required_keys` exist in the JSON at `benchmark.metrics_path`.
To make results comparable across repos (evaluation + rollout + post-training), this doc recommends a stable schema.

## File(s)

- `evaluation.metrics_path` or `benchmark.metrics_path` (required if you want metrics validation)
- Rollout artifacts (recommended for post-training RL):
  - `.aider_fsm/rollout.json`
  - `rollout.json.paths.samples_jsonl` → JSONL samples file under `$AIDER_FSM_ARTIFACTS_DIR`

## Minimal metrics JSON

Recommended minimum:

- `ok`: boolean (true only when the run produced a real score)
- `score`: number (or `eval.score`)
- `ts`: ISO timestamp string
- `run_id`: string (you can copy `AIDER_FSM_RUN_ID`)

## Recommended keys by area

### Evaluation

- `eval.score`: overall scalar score (primary KPI)
- `eval.details`: optional object/array for per-task breakdown

### Rollout (post-training RL)

At minimum, rollout should produce:

- `.aider_fsm/rollout.json` (JSON object) with:
  - `ok`: boolean
  - `paths.samples_jsonl`: string path to a JSONL file

Samples JSONL schema (one JSON object per line; benchmark-agnostic):

- `prompt`: string
- `completion`: string
- `reward`: number
- `meta`: object (optional)

Additional recommended rollout keys (optional):

- `n_episodes`: integer
- `success_rate`: number in `[0, 1]`
- `failures_by_type`: object mapping failure type -> count
- `avg_latency_ms`: number

### Training (post-training)

- `train.ok`: boolean
- `train.steps`: integer
- `train.wall_time_s`: number
- `train.loss`: number (optional)

## Notes

- Keep the metrics JSON small and stable; write full logs to artifacts.
- Use `AIDER_FSM_*` env vars to embed provenance in metrics and rollout artifacts.
