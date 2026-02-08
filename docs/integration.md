# Integrating a target repo (deploy + rollout + evaluation + benchmark)

The runner is intentionally **generic**. Integration is done through:

1. a `pipeline.yml` verification contract
2. a metrics JSON file produced by your benchmark
3. (optional) a `.aider_fsm/bootstrap.yml` for repo-owned environment setup

## Recommended starting point

- Copy `examples/pipeline.benchmark_skeleton.yml` into your target repo root as `pipeline.yml`
- Fill in:
  - `deploy.setup_cmds` / `deploy.health_cmds` (if you deploy anything)
  - `rollout.run_cmds` (optional) to generate rollouts/trajectories
  - `evaluation.run_cmds` to run evaluation and write `evaluation.metrics_path` (recommended)
  - `evaluation.required_keys` for your evaluation KPIs
  - (optional) `benchmark.run_cmds` for extra benchmark steps (if you use `benchmark.metrics_path`, it will also be validated)

## Environment bootstrap (recommended)

If your repo needs a reproducible environment, add `.aider_fsm/bootstrap.yml` (see `docs/bootstrap_spec.md` and
`examples/bootstrap.example.yml`). A common pattern is to create a venv under `.aider_fsm/venv` and prepend it to `PATH`.

## Metrics JSON contract

The runner expects a JSON **object** at `evaluation.metrics_path` (recommended) or `benchmark.metrics_path`.
It validates that all `evaluation.required_keys` / `benchmark.required_keys` are present (when configured).

Keep the metrics small and stable; write full raw logs into artifacts.

## Running

From the target repo root:

```bash
python3 /path/to/OpenCode-FSM-Runner/fsm_runner.py --repo . --pipeline pipeline.yml --goal "Run benchmark" --model openai/gpt-4o-mini
```

Artifacts are written under `.aider_fsm/artifacts/<run_id>/`.

## Optional: programmatic usage

If you want to drive `setup/rollout/evaluation` from a single Python training script (no benchmark-specific runner code),
see `docs/env_api.md` (recommended: `from runner import env as runner_env`; compatibility wrapper: `import env`).
