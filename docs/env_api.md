# Library API: `setup()` → `rollout()` → `evaluate()`

This repo exposes a small **programmatic API** designed for driving target repos end-to-end without benchmark-specific runner code.

Recommended (and only supported) library-style usage (import `runner_env`):

```python
import runner_env

sess = runner_env.setup("https://github.com/<owner>/<repo>")  # or a HF dataset URL
sess.rollout(llm="my-remote-model-name", mode="smoke")        # or a local HF model directory path
sess.evaluate(mode="smoke")                                   # uses the session's configured llm; auto-teardown
```

目标（中文）：

- 在**不写 benchmark-specific 硬编码**的前提下，让你可以在单个训练脚本里用 `setup/rollout/evaluate` 形式驱动任意 repo / benchmark / dataset。
- 最大化 OpenCode 的自主性：缺少 `pipeline.yml` 时由 OpenCode scaffold 合同；评测/测试命令尽量来自目标仓库的 README / docs / CI workflows（而不是 runner 手写启动逻辑）。
- 作为库（library）使用（硬需求）：只支持 `import runner_env`（推荐）或 `from runner import env as runner_env`（等价），并通过
  `sess = runner_env.setup(url)` → `sess.rollout(llm=...)` → `sess.evaluate()` 驱动闭环（`runner_env` 只是一个 import alias）。

---

## Core idea: repo-owned contract (no hardcoding)

The runner itself stays benchmark-agnostic. The *target* repo is responsible for declaring how to run itself via a contract:

- `pipeline.yml` (v1): which stages to run (tests/deploy/rollout/evaluation/benchmark)
- `.aider_fsm/stages/*.sh`: stage scripts producing the expected JSON artifacts:
  - `.aider_fsm/runtime_env.json`
  - `.aider_fsm/rollout.json` + `rollout.json.paths.samples_jsonl`
  - `.aider_fsm/metrics.json` (must include `ok` and `score` when required)

If `pipeline.yml` is missing, `runner_env.setup()` will scaffold a minimal contract via OpenCode:

- `strict_opencode=True` (default): runner does **not** prewrite/patch contract files; success requires OpenCode (or repo-preexisting files) to produce a valid contract.
- `strict_opencode=False` (deprecated): kept for compatibility only. The runner still does **not** prewrite/seed/fallback-write contract files.

Scaffold provenance is recorded at:

- `<artifacts>/scaffold/scaffold_provenance.json`
- `<artifacts>/repair_*/repair_provenance.json` (when repair runs)

See: `docs/pipeline_spec.md`, `docs/metrics_schema.md`.

---

## Public API

Only supported calls:

- `sess = runner_env.setup(target, ...) -> EnvSession`
- `sess.rollout(llm=..., ...)`
- `sess.evaluate(...)`

Notes:

- `rollout()` requires an explicit `llm=...`.
- `evaluate()` can reuse the session's configured LLM from `rollout()`, or accept `llm=...` as a convenience.
- `evaluate()` runs a best-effort teardown automatically at the end (no public teardown API).

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

### Rollout validation (optional)

If you enable rollout contract validation (`require_samples=True` in code, or `--require-samples` in the verification suite),
the runner validates that `rollout.json.paths.samples_jsonl` exists and contains valid `(prompt, completion, reward)` JSONL lines.

Additional generic sanity checks (all targets):

- If `rollout.json.counts.errors` is present and `errors >= samples`, the rollout is treated as invalid.
- If **all** samples have empty/whitespace-only `completion`, the rollout is treated as invalid.

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

Generic helper used by scaffolded contracts:

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

Shell execution note (optional):

- Hint commands are executed via a **non-login** shell by default (`bash -c ...`) so bootstrap PATH overrides
  like `.aider_fsm/venv/bin:$PATH` are preserved.
- If you explicitly need a login shell, set `AIDER_FSM_HINT_LOGIN_SHELL=1` (uses `bash -lc ...`).

Timeout overrides (optional):

- `AIDER_FSM_MAX_CMD_SECONDS=<int>`: override `pipeline.security.max_cmd_seconds` **at runtime** (useful for long-running “full” evals).
- `AIDER_FSM_MAX_TOTAL_SECONDS=<int>`: override `pipeline.security.max_total_seconds` at runtime.

Token cap (optional):

- `AIDER_FSM_MAX_TOKENS=<int>`: sets `max_tokens` used by the built-in `runner.generic_rollout` OpenAI-compatible requests.

---

## Verification suite (single file)
See `docs/verification.md` for the smoke/full-lite verification commands and evidence checklist.
