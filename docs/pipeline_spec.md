# `pipeline.yml` specification (v1)

`pipeline.yml` defines a **human-owned verification contract**. The runner executes stages in this order:

1. `auth` (optional)
2. `tests`
3. `deploy.setup` (optional)
4. `deploy.health` (optional)
5. `rollout` (optional)
6. `evaluation` (optional)
7. `benchmark` (optional)
8. `evaluation.metrics` / `benchmark.metrics` validation (optional)

See `examples/pipeline.example.yml` and `examples/pipeline.benchmark_skeleton.yml`.

## Top-level

- `version`: must be `1`
- `security`: command policy & timeouts
- `tests`: test commands (required unless you pass `--test-cmd`)
- `auth`: optional login steps
- `deploy`: optional deploy steps
- `rollout`: optional RL/post-training rollout steps
- `evaluation`: optional evaluation steps + metrics validation (preferred)
- `benchmark`: optional benchmark steps + metrics validation
- `artifacts`: output directory for run artifacts

## `security`

- `mode`: `safe` or `system`
- `allowlist`: optional list of regex patterns; if set, only matching commands are allowed
- `denylist`: optional list of regex patterns; always blocked
- `max_cmd_seconds`: optional per-command timeout
- `max_total_seconds`: optional total timeout across a stage

## `tests`

- `cmds`: list of shell commands
- `timeout_seconds`, `retries`
- `env`: mapping of env vars
- `workdir`: working directory (must be within repo)

## `auth` (optional)

- `steps` or `cmds`: list of shell commands
- `interactive`: if `true`, you must run the runner with `--unattended guided`
- `timeout_seconds`, `retries`, `env`, `workdir`

## `deploy` (optional)

- `setup_cmds`, `health_cmds`, `teardown_cmds`
- `teardown_policy`: `always|on_success|on_failure|never`
- `timeout_seconds`, `retries`, `env`, `workdir`
- `kubectl_dump`: optional debugging dump after verification

## `rollout` (optional)

- `run_cmds`
- `timeout_seconds`, `retries`, `env`, `workdir`

## `evaluation` (optional)

- `run_cmds`
- `metrics_path`: path to a JSON file produced by the evaluation (relative to repo)
- `required_keys`: list of keys that must exist in the metrics JSON
- `timeout_seconds`, `retries`, `env`, `workdir`

## `benchmark` (optional)

- `run_cmds`
- `metrics_path`: path to a JSON file produced by the benchmark (relative to repo)
- `required_keys`: list of keys that must exist in the metrics JSON
- `timeout_seconds`, `retries`, `env`, `workdir`

See `docs/metrics_schema.md` for a recommended stable schema across evaluation/rollout/training.

## Tooling bootstrap

This runner intentionally does **not** embed platform-specific tool installation or cluster creation.
If you need environment bootstrap steps, use `.aider_fsm/actions.yml` (see `examples/actions.example.yml`).
For repo-owned, always-on environment setup, use `.aider_fsm/bootstrap.yml` (see `docs/bootstrap_spec.md`).

## Runner-provided environment variables

For every executed stage command, the runner sets:

- `AIDER_FSM_STAGE`: stage name (e.g. `tests`, `deploy_setup`, `rollout`, `evaluation`, `benchmark`)
- `AIDER_FSM_ARTIFACTS_DIR`: stage artifacts directory (absolute path)
- `AIDER_FSM_REPO_ROOT`: repo root directory (absolute path)
