# Verification suite (single-file, benchmark-agnostic)

This repo includes a **single Python file** you can use to validate that the programmatic `env` API can run end-to-end
on multiple targets *without writing any benchmark-specific runner code*:

- `examples/verify_suite_single_file.py`

It exercises the same shape you’d use in your own one-file training/eval loops:

1. `env.setup(target_url_or_path)`
2. `env.rollout_and_evaluation(llm, ...)`
3. `env.teardown()`

## What “full verification” means here

- **No benchmark-specific glue code** in this repo: targets are expected to provide a runnable contract via
  `pipeline.yml` + `.aider_fsm/stages/*.sh`.
- If a target has no `pipeline.yml`, the runner will attempt to scaffold it via OpenCode:
  - `--strict-opencode` (default): no runner prewrite/fallback, only OpenCode or repo-preexisting files count.
  - `--no-strict-opencode` (deprecated): kept for compatibility only; the runner no longer seeds or fallback-writes contract files.
- Evaluation prefers **repo-owned doc/CI hints** (README / docs / workflows) rather than hand-written start commands.

## Command (example: GSM8K + EvalPlus + MiniWoB++)

To fully validate the three public targets used during development:

```bash
python3 examples/verify_suite_single_file.py \
  --targets https://huggingface.co/datasets/openai/gsm8k \
  --targets https://github.com/evalplus/evalplus \
  --targets https://github.com/Farama-Foundation/miniwob-plusplus \
  --llm /abs/path/to/local_hf_model_dir \
  --eval-mode full \
  --require-samples \
  --no-strict-opencode \
  --repair-iters 5 \
  --opencode-model opencode/trinity-large-preview-free \
  --opencode-repair-model opencode/trinity-large-preview-free \
  --opencode-timeout-seconds 900 \
  --env AIDER_EVAL_LIMIT=1319 \
  --env AIDER_FSM_HINT_TIMEOUT_SECONDS=14400 \
  --env AIDER_FSM_MAX_CMD_SECONDS=14400 \
  --env AIDER_FSM_MAX_TOTAL_SECONDS=86400 \
  --env AIDER_FSM_HINT_MAX_ATTEMPTS=5
```

## Recommended verification matrix (smoke + full, strict)

Use this matrix for regression coverage:

1) **Smoke + strict**

```bash
python3 examples/verify_suite_single_file.py \
  --targets-file benchmarks.txt \
  --llm /abs/path/to/local_hf_model_dir \
  --eval-mode smoke \
  --require-samples \
  --strict-opencode
```

2) **Full + strict**

```bash
python3 examples/verify_suite_single_file.py \
  --targets-file benchmarks.txt \
  --llm /abs/path/to/local_hf_model_dir \
  --eval-mode full \
  --require-samples \
  --strict-opencode \
  --env AIDER_FSM_MAX_CMD_SECONDS=14400 \
  --env AIDER_FSM_HINT_TIMEOUT_SECONDS=14400
```

Notes:

- `--no-strict-opencode` is deprecated and does not enable any runner-written fallback contracts.

Notes:

- For HF dataset snapshots, `AIDER_EVAL_LIMIT` controls how many test rows the built-in HF rollout will sample.
  For GSM8K test split, the full row count is `1319`.
- Scores depend on the provided model. Using a tiny local model is fine for **contract validation** (plumbing),
  but will not produce strong benchmark accuracy.

## Output and artifacts

The suite prints a JSON summary like:

- `ok`: overall boolean
- `results[]`: per-target fields `rollout_ok`, `evaluation_ok`, `metrics` (when available), and an `artifacts_dir`

Each target clone also keeps full artifacts under:

- `<target>/.aider_fsm/artifacts/<run_id>/...`

These artifacts include full stdout/stderr tails, per-stage summaries, and the produced contract files:

- `.aider_fsm/runtime_env.json`
- `.aider_fsm/rollout.json` (+ referenced JSONL samples)
- `.aider_fsm/metrics.json`
- `.aider_fsm/hints_run.json` / `.aider_fsm/hints_used.json` (when hints are used)
- `scaffold/scaffold_provenance.json` (who wrote which contract files during scaffold)
- `repair_*/repair_provenance.json` (who wrote which files during repair)

## Example output (2026-02-06)

Example summary JSON from a full run on the three public targets above (using a tiny local HF model for contract validation):

```json
{
  "ok": true,
  "results": [
    {
      "target": "https://huggingface.co/datasets/openai/gsm8k",
      "ok": true,
      "rollout_ok": true,
      "evaluation_ok": true
    },
    {
      "target": "https://github.com/evalplus/evalplus",
      "ok": true,
      "rollout_ok": true,
      "evaluation_ok": true
    },
    {
      "target": "https://github.com/Farama-Foundation/miniwob-plusplus",
      "ok": true,
      "rollout_ok": true,
      "evaluation_ok": true
    }
  ]
}
```

## Troubleshooting

- If evaluation fails with dependency/version issues, add `.aider_fsm/bootstrap.yml` in the target repo to create an
  isolated environment (venv) and install compatible deps before running the hinted command.
  See `docs/bootstrap_spec.md`.
- If hint commands select the wrong tool from your global environment, ensure your bootstrap prepends a venv to `PATH`
  and keep `AIDER_FSM_HINT_LOGIN_SHELL` unset (default non-login shell).
