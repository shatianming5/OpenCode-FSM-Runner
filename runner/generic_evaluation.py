from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Support running either as a module (`python -m runner.generic_evaluation`) or as a
# script (`python $AIDER_FSM_RUNNER_ROOT/runner/generic_evaluation.py`). The latter
# avoids module-name collisions with target repos that may contain their own `runner/`
# package.
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    # Ensure runner root is first on sys.path to avoid collisions like having
    # `$AIDER_FSM_RUNNER_ROOT/runner` earlier (which would import runner/runner.py
    # as a top-level module and break relative imports).
    root_s = str(_ROOT)
    try:
        while root_s in sys.path:
            sys.path.remove(root_s)
    except Exception:
        pass
    sys.path.insert(0, root_s)
    from runner.hints_exec import run_hints  # type: ignore
    from runner._util import _is_truthy, _read_json_object  # type: ignore
else:
    from .hints_exec import run_hints
    from ._util import _is_truthy, _read_json_object


def _write_json(path: Path, obj: dict) -> None:
    # 作用：内部符号：_write_json
    # 能否简略：否
    # 原因：规模≈3 行；引用次数≈18（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/generic_evaluation.py:36；类型=function；引用≈18；规模≈3行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reward_average_from_rollout(repo_root: Path) -> tuple[bool, float, dict[str, int] | None, str]:
    """Compute score as average `reward` across rollout samples (best-effort)."""
    # 作用：Compute score as average `reward` across rollout samples (best-effort).
    # 能否简略：否
    # 原因：规模≈55 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/generic_evaluation.py:50；类型=function；引用≈2；规模≈55行
    rollout_path = (repo_root / ".aider_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        return False, 0.0, None, f"missing_rollout_json: {rollout_path}"

    rollout = _read_json_object(rollout_path)
    if rollout is None:
        return False, 0.0, None, "rollout_json_not_object"

    paths = rollout.get("paths")
    if not isinstance(paths, dict):
        return False, 0.0, None, "rollout_json_missing_paths"

    raw = paths.get("samples_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        return False, 0.0, None, "rollout_json_missing_paths.samples_jsonl"

    samples_path = Path(raw.strip())
    if not samples_path.is_absolute():
        samples_path = (repo_root / samples_path).resolve()
    if not samples_path.exists():
        return False, 0.0, None, f"samples_jsonl_not_found: {samples_path}"

    total = 0
    reward_sum = 0.0
    bad = 0
    try:
        with samples_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    bad += 1
                    continue
                r = obj.get("reward")
                if not isinstance(r, (int, float)):
                    bad += 1
                    continue
                reward_sum += float(r)
                total += 1
    except Exception as e:
        return False, 0.0, None, f"failed_to_read_samples_jsonl: {e}"

    if total <= 0:
        return False, 0.0, {"samples": 0, "bad_lines": bad}, "no_valid_reward_samples"

    score = reward_sum / float(total)
    return True, float(score), {"samples": int(total), "bad_lines": int(bad)}, "reward_average"


def main() -> int:
    # 作用：内部符号：main
    # 能否简略：否
    # 原因：规模≈65 行；引用次数≈25（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/generic_evaluation.py:106；类型=function；引用≈25；规模≈65行
    repo_root = Path(os.environ.get("AIDER_FSM_REPO_ROOT") or ".").resolve()
    metrics_path = Path(os.environ.get("AIDER_FSM_METRICS_PATH") or ".aider_fsm/metrics.json")
    if not metrics_path.is_absolute():
        metrics_path = (repo_root / metrics_path).resolve()

    hints_path = (repo_root / ".aider_fsm" / "hints_used.json").resolve()
    hints_run_path = (repo_root / ".aider_fsm" / "hints_run.json").resolve()
    require_hints = _is_truthy(os.environ.get("AIDER_FSM_REQUIRE_HINTS"))

    try:
        timeout = int(os.environ.get("AIDER_FSM_HINT_TIMEOUT_SECONDS") or 600)
    except Exception:
        timeout = 600
    try:
        max_attempts = int(os.environ.get("AIDER_FSM_HINT_MAX_ATTEMPTS") or 3)
    except Exception:
        max_attempts = 3

    res = run_hints(repo=repo_root, max_attempts=max_attempts, timeout_seconds=timeout, env=dict(os.environ))
    try:
        _write_json(hints_run_path, dict(res) if isinstance(res, dict) else {"value": res})
    except Exception:
        pass
    ok = bool(res.get("ok") is True)
    score = float(res.get("score") or 0.0)
    hint_reason = str(res.get("reason") or "")

    hints_used = {
        "ok": bool(ok),
        "used_anchors": list(res.get("used_anchors") or []),
        "commands": [
            str(a.get("sanitized") or a.get("raw") or "").strip()
            for a in (res.get("attempts") or [])
            if isinstance(a, dict) and str(a.get("sanitized") or a.get("raw") or "").strip()
        ],
        "reason": "" if ok else str(res.get("reason") or "hint_command_failed"),
    }
    _write_json(hints_path, hints_used)

    # If there are no runnable hints, fall back to computing score from rollout samples.
    if not ok and hint_reason == "no_hints" and not require_hints:
        ok2, score2, counts2, reason2 = _reward_average_from_rollout(repo_root)
        metrics = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "ok": bool(ok2),
            "score": float(score2) if ok2 else 0.0,
            "reason": str(reason2),
        }
        if isinstance(counts2, dict):
            metrics["counts"] = dict(counts2)
        _write_json(metrics_path, metrics)
        return 0 if ok2 else 1

    metrics = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "ok": bool(ok),
        "score": score if ok else 0.0,
        "reason": hint_reason,
    }
    _write_json(metrics_path, metrics)

    if require_hints and not ok:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
