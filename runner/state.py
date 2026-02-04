from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def ensure_dirs(repo: Path) -> tuple[Path, Path, Path]:
    state_dir = repo / ".aider_fsm"
    logs_dir = state_dir / "logs"
    state_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    return state_dir, logs_dir, state_dir / "state.json"


def default_state(
    *, repo: Path, plan_rel: str, model: str, test_cmd: str, pipeline_rel: str | None = None
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "repo": str(repo),
        "plan_path": plan_rel,
        "pipeline_path": pipeline_rel or "",
        "model": model,
        "test_cmd": test_cmd,
        "iter_idx": 0,
        "fsm_state": "S0_BOOTSTRAP",
        "current_step_id": None,
        "current_step_text": None,
        "fix_attempts": 0,
        "last_bootstrap_rc": None,
        "last_rollout_rc": None,
        "last_test_rc": None,
        "last_deploy_setup_rc": None,
        "last_deploy_health_rc": None,
        "last_benchmark_rc": None,
        "last_evaluation_rc": None,
        "last_metrics_ok": None,
        "last_exit_reason": None,
        "updated_at": now_iso(),
    }


def load_state(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(defaults)
    if not isinstance(data, dict):
        return dict(defaults)
    merged = dict(defaults)
    merged.update(data)
    merged["updated_at"] = now_iso()
    return merged


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
