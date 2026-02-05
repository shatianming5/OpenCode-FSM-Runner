# Programmatic `env` API (single-file training scripts)

This repo exposes a small **programmatic API** designed for “all code in one file” training/eval loops:

```python
import env

sess = env.setup("https://github.com/<owner>/<repo>")   # or a HF dataset URL
env.rollout("my-remote-model-name", session=sess)       # or a local HF model directory
env.evaluation(session=sess)
env.teardown(session=sess)
```

目标（中文）：

- 在**不写 benchmark-specific 硬编码**的前提下，让你可以在单个训练脚本里用 `setup/rollout/evaluation` 形式驱动任意 repo / benchmark / dataset。
- 最大化 OpenCode 的自主性：缺少 `pipeline.yml` 时由 OpenCode scaffold 合同；评测/测试命令尽量来自目标仓库的 README / docs / CI workflows（而不是 runner 手写启动逻辑）。

---

## Core idea: repo-owned contract (no hardcoding)

The runner itself stays benchmark-agnostic. The *target* repo is responsible for declaring how to run itself via a contract:

- `pipeline.yml` (v1): which stages to run (tests/deploy/rollout/evaluation/benchmark)
- `.aider_fsm/stages/*.sh`: stage scripts producing the expected JSON artifacts:
  - `.aider_fsm/runtime_env.json`
  - `.aider_fsm/rollout.json` + `rollout.json.paths.samples_jsonl`
  - `.aider_fsm/metrics.json` (must include `ok` and `score` when required)

If `pipeline.yml` is missing, `env.setup()` will **scaffold** a minimal contract via OpenCode, and/or seed a generic fallback
when `strict_opencode=False`.

See: `docs/pipeline_spec.md`, `docs/metrics_schema.md`.

---

## Public API

Top-level module: `env.py` (re-exports from `runner.env`).

Main calls:

- `env.setup(target, ...) -> EnvSession`
  - `target` can be:
    - local path
    - git URL / `owner/repo`
    - Hugging Face dataset URL: `https://huggingface.co/datasets/<namespace>/<name>`
- `env.deploy(llm, session=..., ...)`
- `env.rollout(llm, session=..., ...)`
- `env.evaluation(session=..., ...)`
- `env.rollout_and_evaluation(llm, session=..., ...)`
- `env.teardown(session=..., ...)`

### `llm` parameter (local vs remote)

`llm` can be either:

- **Local HF model directory**: an existing directory path.
  - Runner sets `AIDER_LLM_KIND=local_hf` and `AIDER_TRAINED_MODEL_DIR=/abs/path`.
- **Remote model id/name**: any other non-empty string.
  - Runner sets `AIDER_LLM_KIND=remote`, `AIDER_LLM_MODEL=<string>`, and also `OPENAI_MODEL=<string>`.
  - Endpoint/auth come from env (OpenAI-compatible): `OPENAI_API_KEY`, optional `OPENAI_API_BASE` / `OPENAI_BASE_URL`.

No endpoints, tokens, ports, or benchmark flags are hardcoded in the runner.

---

## Rollout contract (for RL/post-training)

To support generic post-training, rollout should produce:

- `.aider_fsm/rollout.json` (JSON object)
- `rollout.json.paths.samples_jsonl` → a JSONL file where each line includes:
  - `prompt` (string)
  - `completion` (string)
  - `reward` (number)

The example single-file post-training loop consumes this contract:

- `examples/train_rl_post_single.py`
  - `--dry-run` exercises only `env.setup -> env.rollout -> env.evaluation`
  - training mode performs a minimal PPO-style update on `(prompt, completion, reward)` samples

### Rollout validation (optional)

If you enable rollout contract validation (`require_samples=True` in code, or `--require-samples` in the verification suite),
the runner validates that `rollout.json.paths.samples_jsonl` exists and contains valid `(prompt, completion, reward)` JSONL lines.

For Hugging Face dataset snapshots (detected via `data/hf_manifest.json` + a test parquet with `question/answer` columns),
it also enforces:

- **Minimum sample count**: `min(AIDER_EVAL_LIMIT, test_rows)` (defaults: smoke=8, full=64 if `AIDER_EVAL_LIMIT` is unset)
- **Prompt diversity**: prevents trivial single-prompt rollouts
- **Prompt anchoring**: prompts must include real dataset question text (prevents synthetic unrelated tasks)

---

## Evaluation: doc/CI hints (maximize autonomy)

When a target repo has obvious “official” commands in its README/docs/CI, the runner extracts them and injects:

- `AIDER_FSM_HINTS_JSON`: JSON list of candidate commands
- `AIDER_FSM_HINT_ANCHORS_JSON`: JSON list of high-signal tokens

Then, evaluation is expected to run at least one hinted/official command and write:

- `.aider_fsm/hints_used.json` with:
  - `ok`: boolean (true only if a hinted/official command ran successfully)
  - `used_anchors`: list of anchors that prove usage
  - `commands`: attempted commands (recommended)
  - `reason`: required when `ok=false`

Generic helper used by fallback contracts:

- `runner.generic_evaluation` runs hints via `runner.hints_exec.run_hints()` and writes:
  - `.aider_fsm/hints_run.json` (debug details)
  - `.aider_fsm/hints_used.json`
  - `.aider_fsm/metrics.json`

Notes:

- For `pytest`-style hints, a non-zero exit is treated as a valid evaluation run if a summary can be parsed; score is derived
  from the parsed `(passed / (passed+failed+errors))`. This keeps the runner generic while still producing a numeric metric.
- If **no hints** are available and `AIDER_FSM_REQUIRE_HINTS` is not set, `runner.generic_evaluation` falls back to computing
  `score = average(reward)` from `rollout.json.paths.samples_jsonl`.

Offline preference (optional):

- Set `AIDER_FSM_PREFER_OFFLINE_HINTS=1` to prefer commands that can run without remote inference (e.g. `--samples ...`),
  and de-prioritize `--backend openai` hints.

Timeout overrides (optional):

- `AIDER_FSM_MAX_CMD_SECONDS=<int>`: override `pipeline.security.max_cmd_seconds` **at runtime** (useful for long-running “full” evals).
- `AIDER_FSM_MAX_TOTAL_SECONDS=<int>`: override `pipeline.security.max_total_seconds` at runtime.

---

## Verification suite (single file)

Use the generic suite to validate multiple targets end-to-end:

- `examples/verify_suite_single_file.py`

Example (three targets, “full” mode):

```bash
python3 examples/verify_suite_single_file.py \
  --targets https://huggingface.co/datasets/openai/gsm8k \
  --targets https://github.com/evalplus/evalplus \
  --targets https://github.com/Farama-Foundation/miniwob-plusplus \
  --llm deepseek-v3.2 \
  --eval-mode full \
  --require-samples \
  --repair-iters 0 \
  --no-strict-opencode \
  --env AIDER_EVAL_LIMIT=1319 \
  --env AIDER_FSM_PREFER_OFFLINE_HINTS=1 \
  --env AIDER_FSM_HINT_TIMEOUT_SECONDS=7200 \
  --env AIDER_FSM_MAX_CMD_SECONDS=14400 \
  --env AIDER_FSM_HINT_MAX_ATTEMPTS=1
```

This is intended as a “full pipeline validation” harness; the numeric score depends on what the target contract/hints produce.
