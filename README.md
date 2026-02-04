# OpenCode-FSM Runner

A small, auditable **closed-loop executor** driven by an OpenCode agent (via the OpenCode server API).
It is designed to integrate with agent projects for automation such as **benchmark deployment + evaluation** with hard guardrails.

Cycle:

1) snapshot repo + `PLAN.md`  
2) update plan (model may ONLY edit `PLAN.md`)  
3) execute exactly one `Next` step (model may NOT edit `PLAN.md` or `pipeline.yml`)  
4) verify via `pipeline.yml` (tests → deploy → rollout → evaluation → benchmark → metrics)  
5) pass → mark Done; fail → fix or re-plan; optionally request `.aider_fsm/actions.yml`  

中文：这是一个“计划-执行-验收”的闭环 runner，重点是可审计、可复现、可安全执行（适合作为 benchmark/deploy 验收框架）。

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
- Model selection:
  - Recommended: use `--model provider/model` (see `opencode models`)
  - Convenience: set `OPENAI_MODEL=<model_id>` (or `LITELLM_CHAT_MODEL`) and omit `--model`

## Run

Run from the **target repo root** (Git repo recommended for revert guards):

```bash
python3 fsm_runner.py --repo . --goal "你的目标" --test-cmd "pytest -q" --model myproxy/deepseek-v3.2
```

Or:

```bash
python3 -m runner --repo . --goal "你的目标" --test-cmd "pytest -q" --model myproxy/deepseek-v3.2
```

### Use an existing OpenCode server

Start a server in your target repo root:

```bash
OPENCODE_SERVER_PASSWORD=... opencode serve --hostname 127.0.0.1 --port 4096
```

Then run the runner with:

```bash
OPENCODE_SERVER_PASSWORD=... python3 -m runner --repo . --opencode-url http://127.0.0.1:4096
```

### Run a remote repo (auto-clone)

You can point `--repo` at a git URL; the runner will clone it to `/tmp/aider_fsm_targets/` by default:

```bash
python3 -m runner --repo https://github.com/evalplus/evalplus --goal "运行 evalplus smoke benchmark" --test-cmd "python -V"
```

Use `--clone-dir` to choose a different clone location. The runner will also load `.env` by default (disable with `--env-file ''`).

If the remote repo includes a root `pipeline.yml`, the runner will auto-load it. To require a pipeline (recommended for “contract runs”),
use `--require-pipeline`.

If a remote repo does **not** include `pipeline.yml`, the runner will auto-scaffold a minimal contract via OpenCode
(`pipeline.yml` + `.aider_fsm/`) so `--repo <url>` can run end-to-end. Disable with `--scaffold-contract off`.

Control OpenCode bash permissions during scaffolding via `--scaffold-opencode-bash` (default: `full`).

If `git clone` is blocked (common in restricted networks), GitHub HTTPS/SSH URLs will fall back to downloading a GitHub archive ZIP
(`main` then `master`) and extracting it locally.

### Deploy + benchmark (pipeline.yml)

Add a `pipeline.yml` to the target repo to include deploy/benchmark/metrics in the verification loop:

```bash
python3 fsm_runner.py --repo . --pipeline pipeline.yml
```

Notes:

- `pipeline.yml` is a **human-owned contract**. The runner will revert any model edits to it.
- Artifacts are written under `.aider_fsm/artifacts/<run_id>/` (override via `--artifacts-dir`).
- If you need interactive auth, set `pipeline.auth.interactive: true` and run with `--unattended guided`.
- `rollout` is an optional stage for post-training RL rollouts/trajectories (see `docs/pipeline_spec.md`).
- `evaluation` is an optional stage for evaluation/benchmark runs + metrics validation (see `docs/pipeline_spec.md`).

Examples:

- `examples/pipeline.example.yml`
- `examples/pipeline.benchmark_skeleton.yml`

### Repo-owned environment bootstrap (bootstrap.yml)

If your target repo needs a reproducible environment setup (e.g. venv + deps), add `.aider_fsm/bootstrap.yml`
(see `docs/bootstrap_spec.md` and `examples/bootstrap.example.yml`). The runner executes it before verification.

### Environment/tooling bootstrap (actions.yml)

If verification fails due to missing tools/config/auth, the model may write `.aider_fsm/actions.yml`.
The runner executes it (subject to security policy) and records artifacts, then deletes the file.

See `examples/actions.example.yml`.

## Docs

- `docs/overview.md`
- `docs/pipeline_spec.md`
- `docs/bootstrap_spec.md`
- `docs/metrics_schema.md`
- `docs/security_model.md`
- `docs/integration.md`

## Tests

```bash
pytest -q
```
