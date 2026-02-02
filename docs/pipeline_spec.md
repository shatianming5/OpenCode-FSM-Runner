# `pipeline.yml` specification (v1)

`pipeline.yml` defines a **human-owned verification contract**. The runner executes stages in this order:

1. `auth` (optional)
2. `tests`
3. `deploy.setup` (optional)
4. `deploy.health` (optional)
5. `benchmark` (optional)
6. `benchmark.metrics` validation (optional)

See `examples/pipeline.example.yml` and `examples/pipeline.rd_agent_rl_benchmark.yml`.

## Top-level

- `version`: must be `1`
- `security`: command policy & timeouts
- `tests`: test commands (required unless you pass `--test-cmd`)
- `auth`: optional login steps
- `deploy`: optional deploy steps
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

## `benchmark` (optional)

- `run_cmds`
- `metrics_path`: path to a JSON file produced by the benchmark (relative to repo)
- `required_keys`: list of keys that must exist in the metrics JSON
- `timeout_seconds`, `retries`, `env`, `workdir`

## Tooling bootstrap

This runner intentionally does **not** embed platform-specific tool installation or cluster creation.
If you need environment bootstrap steps, use `.aider_fsm/actions.yml` (see `examples/actions.example.yml`).

