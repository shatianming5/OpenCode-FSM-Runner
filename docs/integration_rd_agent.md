# Integrating with RD-agent (post-training RL benchmark deploy + eval)

The runner is intentionally **generic**. RD-agent integration is done through:

1. a `pipeline.yml` contract
2. a metrics JSON file produced by your benchmark

## Recommended starting point

- Copy `examples/pipeline.rd_agent_rl_benchmark.yml` to your RD-agent repo as `pipeline.yml`
- Fill in:
  - `deploy.setup_cmds` / `deploy.health_cmds` (if you deploy anything)
  - `benchmark.run_cmds` to run the benchmark and write `benchmark.metrics_path`
  - `benchmark.required_keys` for your evaluation KPIs

## Metrics JSON contract

The runner expects a JSON **object** at `benchmark.metrics_path`.
It validates that all `benchmark.required_keys` are present.

Keep the metrics small and stable; write full raw logs into artifacts.

## Running

From the RD-agent repo root:

```bash
python3 /path/to/Aider_runner/fsm_runner.py --repo . --pipeline pipeline.yml --goal "Run post-training RL benchmark" --model openai/gpt-4o-mini
```

Artifacts are written under `.aider_fsm/artifacts/<run_id>/`.
