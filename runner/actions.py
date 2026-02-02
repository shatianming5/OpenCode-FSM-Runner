from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .paths import is_relative_to, resolve_config_path
from .pipeline_spec import PipelineSpec
from .security import cmd_allowed, looks_interactive, safe_env
from .subprocess_utils import run_cmd_capture, write_cmd_artifacts, write_json, write_text
from .types import CmdResult, StageResult


def load_actions_spec(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        raise ValueError(f"actions file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("actions.yml must be a YAML mapping (dict) at the top level")
    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported actions version: {version}")
    actions = data.get("actions") or []
    if actions is None:
        actions = []
    if not isinstance(actions, list) or not all(isinstance(a, dict) for a in actions):
        raise ValueError("actions.yml actions must be a list of mappings")
    return {"version": version, "actions": actions, "raw": raw}


def run_pending_actions(
    repo: Path,
    *,
    pipeline: PipelineSpec | None,
    unattended: str,
    actions_path: Path,
    artifacts_dir: Path,
    protected_paths: list[Path] | None = None,
) -> StageResult | None:
    if not actions_path.exists():
        return None

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec = load_actions_spec(actions_path)
    except Exception as e:
        res = CmdResult(cmd=f"parse {actions_path}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "actions_parse", res)
        return StageResult(ok=False, results=[res], failed_index=0)

    write_text(artifacts_dir / "actions.yml", spec["raw"])

    protected = {p.resolve() for p in (protected_paths or []) if p}

    env_base = dict(os.environ)
    env = safe_env(env_base, {}, unattended=unattended)
    workdir = repo

    results: list[CmdResult] = []
    failed_index: int | None = None

    for idx, action in enumerate(spec["actions"], start=1):
        action_id = str(action.get("id") or f"action-{idx:03d}").strip() or f"action-{idx:03d}"
        kind = str(action.get("kind") or "run_cmd").strip().lower()
        cmd = str(action.get("cmd") or "").strip()
        timeout_seconds = action.get("timeout_seconds")
        retries = int(action.get("retries") or 0)

        # 'install_tool' and 'start_service' are semantic aliases for 'run_cmd'.
        if kind not in ("run_cmd", "install_tool", "start_service", "write_file"):
            res = CmdResult(
                cmd=cmd or f"<{action_id}>",
                rc=2,
                stdout="",
                stderr=f"unsupported_action_kind: {kind}",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        if not cmd and kind != "write_file":
            res = CmdResult(
                cmd=f"<{action_id}>",
                rc=2,
                stdout="",
                stderr="missing_cmd",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        if kind == "write_file":
            path_raw = str(action.get("path") or "").strip()
            content = str(action.get("content") or "")
            if not path_raw:
                res = CmdResult(cmd=f"<{action_id}>", rc=2, stdout="", stderr="missing_path", timed_out=False)
                results.append(res)
                failed_index = len(results) - 1
                write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
                break
            try:
                out_path = resolve_config_path(repo, path_raw)
                if not is_relative_to(out_path, repo):
                    raise ValueError("path_outside_repo")
                if out_path in protected:
                    raise ValueError("path_is_protected")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
                res = CmdResult(cmd=f"write_file {out_path}", rc=0, stdout="", stderr="", timed_out=False)
            except Exception as e:
                res = CmdResult(cmd=f"write_file {path_raw}", rc=2, stdout="", stderr=str(e), timed_out=False)

            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            if res.rc != 0:
                failed_index = len(results) - 1
                break
            continue

        if unattended == "strict" and looks_interactive(cmd):
            res = CmdResult(
                cmd=cmd,
                rc=126,
                stdout="",
                stderr="likely_interactive_command_disallowed_in_strict_mode",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
        if not allowed:
            res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        eff_timeout: int | None = int(timeout_seconds) if timeout_seconds else None
        if pipeline and pipeline.security_max_cmd_seconds:
            eff_timeout = (
                int(pipeline.security_max_cmd_seconds)
                if eff_timeout is None
                else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
            )

        for attempt in range(1, retries + 2):
            res = run_cmd_capture(cmd, workdir, timeout_seconds=eff_timeout, env=env, interactive=False)
            results.append(res)
            write_cmd_artifacts(
                artifacts_dir,
                f"actions_{idx:02d}_{action_id}_try{attempt:02d}",
                res,
            )
            if res.rc == 0:
                break
        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            break

    ok = failed_index is None
    write_json(
        artifacts_dir / "actions_summary.json",
        {"ok": ok, "failed_index": failed_index, "total_results": len(results)},
    )

    try:
        actions_path.unlink()
    except Exception:
        # Best effort. Keeping the file may cause repeated execution; record a warning.
        write_text(artifacts_dir / "actions_warning.txt", f"failed to delete actions file: {actions_path}\n")

    return StageResult(ok=ok, results=results, failed_index=failed_index)

