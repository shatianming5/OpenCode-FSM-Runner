from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _file_meta(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except Exception:
        return {"exists": False}
    return {
        "exists": True,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def snapshot_contract_files(repo: Path) -> dict[str, dict[str, Any]]:
    """Snapshot contract-relevant files for provenance comparison.

    Keep this intentionally small:
    - We want provenance to explain "which contract files changed" without
      exploding to tens of thousands of entries (e.g. venv/site-packages) or
      capturing run artifacts that can be very large/noisy.
    """
    root = Path(repo).resolve()
    out: dict[str, dict[str, Any]] = {}
    out["pipeline.yml"] = _file_meta((root / "pipeline.yml").resolve())
    aider = (root / ".aider_fsm").resolve()
    if aider.exists():
        # Single-file contract inputs/outputs.
        for rel in (
            "bootstrap.yml",
            "runtime_env.json",
            "rollout.json",
            "metrics.json",
            "hints_used.json",
            "hints_run.json",
        ):
            p = (aider / rel).resolve()
            if p.exists() and p.is_file():
                out[p.relative_to(root).as_posix()] = _file_meta(p)

        # Stage scripts are the main contract surface.
        stages = (aider / "stages").resolve()
        if stages.exists():
            for p in sorted(stages.rglob("*")):
                if not p.is_file():
                    continue
                out[p.relative_to(root).as_posix()] = _file_meta(p)
    return out


def _normalize_rel_to_repo(repo: Path, raw_path: str) -> str | None:
    try:
        p = Path(str(raw_path or "")).expanduser()
    except Exception:
        return None
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo).as_posix()
    except Exception:
        return None


def extract_tool_written_paths(*, repo: Path, tool_trace: list[dict[str, Any]] | None) -> set[str]:
    """Collect repo-relative file paths written via tool calls."""
    root = Path(repo).resolve()
    writes: set[str] = set()
    for turn in list(tool_trace or []):
        results = turn.get("results")
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or item.get("kind") or "").strip().lower()
            ok = bool(item.get("ok"))
            # OpenCode can write files via either `<write ...>` or `<edit ...>` tool calls.
            if tool not in ("write", "edit") or not ok:
                continue
            rel = _normalize_rel_to_repo(root, str(item.get("filePath") or ""))
            if rel:
                writes.add(rel)
    return writes


def _status(before: dict[str, Any] | None, after: dict[str, Any] | None) -> str:
    b_exists = bool((before or {}).get("exists"))
    a_exists = bool((after or {}).get("exists"))
    if not b_exists and not a_exists:
        return "absent"
    if not b_exists and a_exists:
        return "created"
    if b_exists and not a_exists:
        return "deleted"
    if (before or {}).get("sha256") != (after or {}).get("sha256"):
        return "modified"
    return "unchanged"


def changed_paths(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for rel in set(before.keys()) | set(after.keys()):
        if _status(before.get(rel), after.get(rel)) != "unchanged":
            out.add(rel)
    return out


def build_contract_provenance_report(
    *,
    repo: Path,
    purpose: str,
    strict_opencode: bool,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    tool_trace: list[dict[str, Any]] | None,
    runner_written_paths: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(repo).resolve()
    agent_write_paths = extract_tool_written_paths(repo=root, tool_trace=tool_trace)
    runner_write_paths = set(runner_written_paths or set())
    entries: list[dict[str, Any]] = []
    changed_count = 0
    for rel in sorted(set(before.keys()) | set(after.keys())):
        b = before.get(rel, {"exists": False})
        a = after.get(rel, {"exists": False})
        st = _status(b, a)
        if st == "absent":
            continue
        if st != "unchanged":
            changed_count += 1
        source = "repo_preexisting"
        if st != "unchanged":
            if rel in runner_write_paths:
                source = "runner_prewrite_or_fallback"
            elif rel in agent_write_paths:
                source = "opencode_tool_write"
            else:
                # Most often from OpenCode-executed bash commands or side effects.
                source = "opencode_bash_or_side_effect"
        entries.append(
            {
                "path": rel,
                "status": st,
                "source": source,
                "before": b,
                "after": a,
            }
        )

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "repo": str(root),
        "purpose": str(purpose or ""),
        "strict_opencode": bool(strict_opencode),
        "summary": {
            "tracked_files": len(entries),
            "changed_files": int(changed_count),
            "runner_written_count": len(runner_write_paths),
            "opencode_tool_write_count": len(agent_write_paths),
        },
        "tool_trace": list(tool_trace or []),
        "runner_written_paths": sorted(runner_write_paths),
        "opencode_tool_written_paths": sorted(agent_write_paths),
        "files": entries,
    }


def dump_provenance(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
