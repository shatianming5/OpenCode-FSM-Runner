# Verification (smoke + full-lite)

目标：证明库三入口可以在真实 repo benchmark 上跑通，并产出可审计证据。

只允许的公共入口：

```python
import runner_env

sess = runner_env.setup("https://...")
sess.rollout(llm="deepseek-v3.2", ...)
sess.evaluate(...)
```

## Preflight

- OpenCode 可用（用于缺少 `pipeline.yml` 时 scaffold/repair）：
  - `opencode` CLI 已安装
  - 若使用远端 OpenCode server：确保 `opencode_url` 可访问，并配置 `OPENCODE_SERVER_USERNAME/OPENCODE_SERVER_PASSWORD`
- 远端 LLM 可用（用于 rollout/evaluation 的实际推理）：
  - `OPENAI_API_KEY` 已设置
  - 可选：`OPENAI_API_BASE` / `OPENAI_BASE_URL`（OpenAI-compatible endpoint）

## Smoke (strict)

两个目标 repo（不跑 GSM8K）：

- `https://github.com/evalplus/evalplus`
- `https://github.com/Farama-Foundation/miniwob-plusplus`

建议用同一段脚本依次跑（严格模式 + 长超时 + require_samples）：

```bash
python3 - <<'PY'
import runner_env

TARGETS = [
    "https://github.com/evalplus/evalplus",
    "https://github.com/Farama-Foundation/miniwob-plusplus",
]

for t in TARGETS:
    sess = runner_env.setup(
        t,
        strict_opencode=True,
        unattended="strict",
        require_metrics=True,
        opencode_timeout_seconds=1200,
        opencode_retry_attempts=2,
    )
    r = sess.rollout(
        llm="deepseek-v3.2",
        mode="smoke",
        require_samples=True,
        repair_iters=3,
        env_overrides={
            "AIDER_FSM_MAX_CMD_SECONDS": "7200",
            "AIDER_FSM_HINT_TIMEOUT_SECONDS": "7200",
        },
    )
    e = sess.evaluate(
        mode="smoke",
        repair_iters=3,
        env_overrides={
            "AIDER_FSM_MAX_CMD_SECONDS": "7200",
            "AIDER_FSM_HINT_TIMEOUT_SECONDS": "7200",
        },
    )
    print("target:", t)
    print("rollout_ok:", bool(r.ok), "evaluation_ok:", bool(e.ok))
    print("metrics_ok:", bool(isinstance(e.metrics, dict) and e.metrics.get("ok") is True))
    print("metrics_score:", (e.metrics or {}).get("score") if isinstance(e.metrics, dict) else None)
PY
```

## Full-lite

定义：仍走 `mode="full"`，但通过上限缩短运行时间（避免完整 full 过长）。

建议用 `AIDER_EVAL_LIMIT` 控制样本规模，并保持长超时：

```bash
python3 - <<'PY'
import runner_env

TARGETS = [
    "https://github.com/evalplus/evalplus",
    "https://github.com/Farama-Foundation/miniwob-plusplus",
]

for t in TARGETS:
    sess = runner_env.setup(
        t,
        strict_opencode=True,
        unattended="strict",
        require_metrics=True,
        opencode_timeout_seconds=1800,
        opencode_retry_attempts=2,
    )
    r = sess.rollout(
        llm="deepseek-v3.2",
        mode="full",
        require_samples=True,
        repair_iters=3,
        env_overrides={
            "AIDER_EVAL_LIMIT": "16",
            "AIDER_FSM_MAX_CMD_SECONDS": "14400",
            "AIDER_FSM_HINT_TIMEOUT_SECONDS": "14400",
        },
    )
    e = sess.evaluate(
        mode="full",
        repair_iters=3,
        env_overrides={
            "AIDER_EVAL_LIMIT": "16",
            "AIDER_FSM_MAX_CMD_SECONDS": "14400",
            "AIDER_FSM_HINT_TIMEOUT_SECONDS": "14400",
        },
    )
    print("target:", t)
    print("rollout_ok:", bool(r.ok), "evaluation_ok:", bool(e.ok))
    print("metrics_ok:", bool(isinstance(e.metrics, dict) and e.metrics.get("ok") is True))
    print("metrics_score:", (e.metrics or {}).get("score") if isinstance(e.metrics, dict) else None)
PY
```

## Evidence checklist

每个 target 至少保留这些证据（用于审计与复现）：

- Scaffold/repair provenance：
  - `.aider_fsm/artifacts/*/scaffold/scaffold_provenance.json`
  - `.aider_fsm/artifacts/*/repair_*/repair_provenance.json`（若发生 repair）
  - 期望：`runner_written_count == 0`（严格模式下不接受 runner 代写合同）
- Runner artifacts：
  - `.aider_fsm/artifacts/<run_id>/env_api/**`
  - deploy/rollout/evaluation 各 stage 的 stdout/stderr 与 verification 结果（路径见 artifacts 目录结构）
- Contract outputs：
  - `.aider_fsm/rollout.json`
  - `.aider_fsm/metrics.json`
  - `.aider_fsm/hints_used.json`（当存在 hints 且 `AIDER_FSM_REQUIRE_HINTS=1` 时）
