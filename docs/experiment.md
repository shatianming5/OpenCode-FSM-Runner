# Experiments

## Matrix
| ID | Name | Config/Script | Command | Metrics | Artifacts | Smoke | Full |
|---|---|---|---|---|---|---|---|
| Exp-001 | HF dataset contract run (GSM8K link) | `examples/verify_suite_single_file.py` | `python3 examples/verify_suite_single_file.py --targets https://huggingface.co/datasets/openai/gsm8k --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3` | `metrics.ok`, `metrics.score` | `.aider_fsm/artifacts/<run_id>/env_api/**`, `.aider_fsm/metrics.json`, `.aider_fsm/rollout.json` | [x] | [x] |
| Exp-002 | EvalPlus repo contract run | `examples/verify_suite_single_file.py` | `python3 examples/verify_suite_single_file.py --targets https://github.com/evalplus/evalplus --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3` | `metrics.ok`, `metrics.score` | `.aider_fsm/artifacts/<run_id>/env_api/**`, `.aider_fsm/metrics.json`, `.aider_fsm/rollout.json` | [x] | [x] |
| Exp-003 | MiniWoB++ repo contract run | `examples/verify_suite_single_file.py` | `python3 examples/verify_suite_single_file.py --targets https://github.com/Farama-Foundation/miniwob-plusplus --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3` | `metrics.ok`, `metrics.score` | `.aider_fsm/artifacts/<run_id>/env_api/**`, `.aider_fsm/metrics.json`, `.aider_fsm/rollout.json` | [x] | [x] |
| Exp-004 | 0.5B segment training + ordered target sweep | `runner/ml/train_and_benchmark.py` | `python3 -m runner.ml.train_and_benchmark --base-model Qwen/Qwen2.5-0.5B-Instruct --out-root /data/tiansha/aider_train_runs --targets-file benchmarks.txt --segments 2 --steps-per-segment 8 --opencode-model opencode/gpt-5-nano --require-samples --full-after-last` | per-target `rollout_ok`, `evaluation_ok`, `metrics.ok`, `metrics.score` | `/data/tiansha/aider_train_runs/train_and_benchmark_summary.json`, `/data/tiansha/aider_train_runs/seg_*/smoke/*`, `/data/tiansha/aider_train_runs/seg_*/full/*` | [x] | [x] |

## Latest Passing Runs (2026-02-06 UTC)
- Exp-001 smoke/full: passed (`.rd_queue/results/Exp-001-smoke.json`, `.rd_queue/results/Exp-001-full.json`), full metrics `ok=true`, `score=1.0`, `counts.samples=64`.
- Exp-002 smoke/full: passed (`.rd_queue/results/Exp-002-smoke.json`, `.rd_queue/results/Exp-002-full.json`), full metrics `ok=true`, `score=1.0`.
- Exp-003 smoke/full: passed (`.rd_queue/results/Exp-003-smoke.json`, `.rd_queue/results/Exp-003-full.json`), full metrics `ok=true`, `score=1.0`.
- Exp-004 full: passed (`.rd_queue/results/Exp-004-full.json`), summary at `/data/tiansha/aider_train_runs/train_and_benchmark_summary.json`.
- Exp-004 segment status:
  - segment 0 train: `ok=true`, `steps=8`, `last_loss=3.910231351852417`
  - segment 1 train: `ok=true`, `steps=8`, `last_loss=2.9190423488616943`
  - segment 0 smoke targets (gsm8k/evalplus/miniwob-plusplus): all `rollout_ok=true`, `evaluation_ok=true`, `metrics.ok=true`, `metrics.score=1.0`
  - segment 1 smoke targets (gsm8k/evalplus/miniwob-plusplus): all `rollout_ok=true`, `evaluation_ok=true`, `metrics.ok=true`, `metrics.score=1.0`
  - segment 1 full targets (gsm8k/evalplus/miniwob-plusplus): all `rollout_ok=true`, `evaluation_ok=true`, `metrics.ok=true`, `metrics.score=1.0`

## Details

### Exp-001: HF dataset contract run (GSM8K link)
- Goal: Verify benchmark-agnostic scaffold/repair can derive runnable rollout/evaluation from a HF dataset URL.
- Baseline: No benchmark-specific runner code; only `env.setup -> env.rollout -> env.evaluate`.
- Command:
  - Smoke: `python3 examples/verify_suite_single_file.py --targets https://huggingface.co/datasets/openai/gsm8k --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3`
  - Full: `python3 examples/verify_suite_single_file.py --targets https://huggingface.co/datasets/openai/gsm8k --llm deepseek-v3.2 --eval-mode full --require-samples --repair-iters 3 --env AIDER_FSM_MAX_CMD_SECONDS=14400`
- Resources: 1 CPU/GPU worker for contract run; runtime depends on target hint quality and network.
- Outputs: `.aider_fsm/rollout.json`, `.aider_fsm/metrics.json`, `.aider_fsm/hints_used.json`.
- Required metrics: `ok=true`, numeric `score`.
- Notes: This target has weaker upstream benchmark hints than code repos; failure should be logged as contract-gap evidence rather than patched with benchmark-specific code.

### Exp-002: EvalPlus repo contract run
- Goal: Ensure target-provided commands/hints can be recovered and executed for rollout/evaluation.
- Baseline: Same benchmark-agnostic API and repair loop as Exp-001.
- Command:
  - Smoke: `python3 examples/verify_suite_single_file.py --targets https://github.com/evalplus/evalplus --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3`
  - Full: `python3 examples/verify_suite_single_file.py --targets https://github.com/evalplus/evalplus --llm deepseek-v3.2 --eval-mode full --require-samples --repair-iters 3 --env AIDER_FSM_MAX_CMD_SECONDS=14400`
- Resources: 1 CPU/GPU worker; Full may require long timeout and larger disk cache.
- Outputs: `.aider_fsm/rollout.json`, `.aider_fsm/metrics.json`, `.aider_fsm/hints_used.json`.
- Required metrics: `ok=true`, numeric `score`, non-empty `used_anchors`.
- Notes: Exp-002 is expected to provide stronger doc/CI hints and should be the primary completeness signal for repo-based benchmarks.

### Exp-003: MiniWoB++ repo contract run
- Goal: Validate rollout/evaluation derivation for browser/task-interaction benchmark repos without bespoke adapters.
- Baseline: Same benchmark-agnostic setup as Exp-001/002.
- Command:
  - Smoke: `python3 examples/verify_suite_single_file.py --targets https://github.com/Farama-Foundation/miniwob-plusplus --llm deepseek-v3.2 --eval-mode smoke --require-samples --repair-iters 3`
  - Full: `python3 examples/verify_suite_single_file.py --targets https://github.com/Farama-Foundation/miniwob-plusplus --llm deepseek-v3.2 --eval-mode full --require-samples --repair-iters 3 --env AIDER_FSM_MAX_CMD_SECONDS=14400`
- Resources: 1 CPU/GPU worker; may need additional browser/system deps in target bootstrap.
- Outputs: `.aider_fsm/rollout.json`, `.aider_fsm/metrics.json`, `.aider_fsm/hints_used.json`.
- Required metrics: `ok=true`, numeric `score`, non-empty `used_anchors`.
- Notes: If blocked by missing system deps, resolve via target bootstrap contract only; do not add runner-side benchmark conditionals.

### Exp-004: 0.5B segment training + ordered target sweep
- Goal: During training, run ordered rollout+evaluation for each target after every segment.
- Baseline: `runner/ml/train_and_benchmark.py` now uses Method-2 calls (`env.setup`, `session.rollout`, `session.evaluate`, `env.teardown`) only.
- Command:
  - Smoke per segment: `python3 -m runner.ml.train_and_benchmark --base-model Qwen/Qwen2.5-0.5B-Instruct --out-root /data/tiansha/aider_train_runs --targets-file benchmarks.txt --segments 2 --steps-per-segment 8 --require-samples`
  - Full on final segment: add `--full-after-last`.
- Resources: GPU recommended for training; benchmark runs depend on target contracts.
- Outputs: `/data/tiansha/aider_train_runs/train_and_benchmark_summary.json` with ordered per-segment/per-target status.
- Required metrics:
  - For each target run: `rollout_ok=true`, `evaluation_ok=true`, `metrics.ok=true`.
  - Run order must match `benchmarks.txt` for every segment.
- Notes: No target identity branching is allowed in the orchestration script.

## Done Signals
- Each Exp has command-level reproducibility and explicit artifact paths.
- Method-2 API shape remains benchmark-agnostic and stable:
  - `import env`
  - `env_ = env.setup({...})`
  - `env_.rollout(...)`
  - `env_.evaluate(...)`
