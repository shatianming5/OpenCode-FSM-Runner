# OpenCode-FSM Runner

A small, auditable **benchmark-agnostic** library for deploying and validating target repos via a repo-owned contract (`pipeline.yml` + `.aider_fsm/`).

中文：这是一个“目标仓库自带合同（pipeline.yml + .aider_fsm）”的通用执行与验收库。Runner 自身不写 benchmark-specific 逻辑。

## Install

Install OpenCode (CLI) first:

```bash
curl -fsSL https://opencode.ai/install | bash
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

- Provider credentials depend on your model choice, e.g.:
  - `OPENAI_API_KEY` (for `openai/...`)
  - `OPENAI_API_BASE` (optional; for OpenAI-compatible endpoints)
- OpenCode server auth (optional, if you use `opencode_url` in code):
  - `OPENCODE_SERVER_USERNAME`
  - `OPENCODE_SERVER_PASSWORD`

## Programmatic API (library)

Only supported entrypoints:

```python
import runner_env

sess = runner_env.setup("https://github.com/<owner>/<repo>")
sess.rollout(llm="deepseek-v3.2", mode="smoke", require_samples=True, repair_iters=0)
res = sess.evaluate(mode="smoke", repair_iters=0)
print(res.ok, res.metrics)
```

Notes:

- `sess.rollout()` requires an explicit `llm=...` (remote model id/name, or a local model dir path).
- `sess.evaluate()` can reuse the session LLM from `rollout()`, or accept `llm=...` as a convenience (for one-shot runs).
- `sess.evaluate()` performs a best-effort teardown automatically at the end (no public teardown API).

## OpenCode server (optional)

Start a server in your target repo root:

```bash
OPENCODE_SERVER_PASSWORD=... opencode serve --hostname 127.0.0.1 --port 4096
```

Then point `setup()` at it:

```python
import runner_env

sess = runner_env.setup(
    "https://github.com/<owner>/<repo>",
    opencode_url="http://127.0.0.1:4096",
    opencode_model="opencode/gpt-5-nano",
)
```

## Docs

- `docs/overview.md`
- `docs/env_api.md`
- `docs/pipeline_spec.md`
- `docs/bootstrap_spec.md`
- `docs/metrics_schema.md`
- `docs/security_model.md`
- `docs/integration.md`
- `docs/verification.md`

## Tests

```bash
pytest -q
```
