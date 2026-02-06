# `.aider_fsm/bootstrap.yml` specification (v1)

`bootstrap.yml` defines **repo-owned environment setup** steps to run before verification.
It is intended to make “one-command contract runs” reproducible (e.g. create a venv, install deps, warm up caches).

Notes:

- The runner records bootstrap artifacts under `.aider_fsm/artifacts/<run_id>/...`.
- Commands are subject to the same security policy as pipeline/actions commands.
- In `--unattended strict` mode, likely-interactive commands are blocked.
- Bootstrap is intended for **environment preparation** (venv/deps/build caches). Do not put evaluation/test/benchmark runs
  (e.g. `pytest`, benchmark CLIs) into `bootstrap.yml`; keep those in the pipeline stages (especially `evaluation`).
- If you create a venv under `.aider_fsm/venv`, prefer installing via the venv interpreter (`.aider_fsm/venv/bin/python -m pip ...`)
  to avoid polluting global/user site-packages.

## Top-level

- `version`: must be `1`
- `cmds`: list of shell commands (optional; empty is allowed for env-only bootstrap)
  - Alias: `steps`
- `env`: mapping of env vars applied to bootstrap/pipeline/actions (optional)
  - Supports minimal expansion: `${VAR}` and `$VAR` are replaced from the current environment.
  - Use `$$` for a literal `$`.
- `workdir`: working directory (must be within repo; default: repo root)
- `timeout_seconds`: per-command timeout (optional)
- `retries`: retries per command (optional; default: 0)

## Artifacts

Bootstrap writes:

- `bootstrap.yml`: snapshot of the spec used
- `bootstrap_env.json`: applied env mapping (sensitive keys are redacted)
- `bootstrap_summary.json`: ok/failed_index/total_results
- `bootstrap_cmdXX_tryYY_*`: stdout/stderr/result per command attempt

## Example: uv + venv (recommended)

This pattern creates an isolated venv under `.aider_fsm/venv`, installs deps, then prepends it to `PATH`
so pipeline stages can run without hardcoding interpreter paths:

```yaml
version: 1
cmds:
  - uv venv .aider_fsm/venv
  - uv pip install -r requirements.txt
env:
  PATH: ".aider_fsm/venv/bin:$PATH"
```
