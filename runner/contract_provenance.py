from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def snapshot_contract_files(repo: Path) -> dict[str, dict[str, Any]]:
    """Snapshot contract-relevant files for provenance comparison.

    Keep this intentionally small:
    - We want provenance to explain "which contract files changed" without
      exploding to tens of thousands of entries (e.g. venv/site-packages) or
      capturing run artifacts that can be very large/noisy.
    """
    # 作用：Snapshot contract-relevant files for provenance comparison.
    # 能否简略：否
    # 原因：规模≈34 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/contract_provenance.py:30；类型=function；引用≈10；规模≈34行
    root = Path(repo).resolve()
    out: dict[str, dict[str, Any]] = {}
    try:
        data = (root / "pipeline.yml").resolve().read_bytes()
    except Exception:
        out["pipeline.yml"] = {"exists": False}
    else:
        out["pipeline.yml"] = {
            "exists": True,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
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
                try:
                    data = p.read_bytes()
                except Exception:
                    out[p.relative_to(root).as_posix()] = {"exists": False}
                else:
                    out[p.relative_to(root).as_posix()] = {
                        "exists": True,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }

        # Stage scripts are the main contract surface.
        stages = (aider / "stages").resolve()
        if stages.exists():
            for p in sorted(stages.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    data = p.read_bytes()
                except Exception:
                    out[p.relative_to(root).as_posix()] = {"exists": False}
                else:
                    out[p.relative_to(root).as_posix()] = {
                        "exists": True,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
    return out


def extract_tool_written_paths(*, repo: Path, tool_trace: list[dict[str, Any]] | None) -> set[str]:
    """Collect repo-relative file paths written via tool calls."""
    # 作用：Collect repo-relative file paths written via tool calls.
    # 能否简略：否
    # 原因：规模≈20 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/contract_provenance.py:75；类型=function；引用≈2；规模≈20行
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
            raw_path = str(item.get("filePath") or "")
            try:
                p = Path(raw_path).expanduser()
            except Exception:
                continue
            if not p.is_absolute():
                p = (root / p).resolve()
            else:
                p = p.resolve()
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = None
            if rel:
                writes.add(rel)
    return writes


def changed_paths(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
    # 作用：内部符号：changed_paths
    # 能否简略：否
    # 原因：规模≈6 行；引用次数≈3（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/contract_provenance.py:110；类型=function；引用≈3；规模≈6行
    out: set[str] = set()
    for rel in set(before.keys()) | set(after.keys()):
        b = before.get(rel)
        a = after.get(rel)
        b_exists = bool((b or {}).get("exists"))
        a_exists = bool((a or {}).get("exists"))
        if not b_exists and not a_exists:
            st = "absent"
        elif not b_exists and a_exists:
            st = "created"
        elif b_exists and not a_exists:
            st = "deleted"
        elif (b or {}).get("sha256") != (a or {}).get("sha256"):
            st = "modified"
        else:
            st = "unchanged"
        if st != "unchanged":
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
    # 作用：内部符号：build_contract_provenance_report
    # 能否简略：否
    # 原因：规模≈58 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/contract_provenance.py:127；类型=function；引用≈7；规模≈58行
    root = Path(repo).resolve()
    agent_write_paths = extract_tool_written_paths(repo=root, tool_trace=tool_trace)
    runner_write_paths = set(runner_written_paths or set())
    entries: list[dict[str, Any]] = []
    changed_count = 0
    for rel in sorted(set(before.keys()) | set(after.keys())):
        b = before.get(rel, {"exists": False})
        a = after.get(rel, {"exists": False})
        b_exists = bool((b or {}).get("exists"))
        a_exists = bool((a or {}).get("exists"))
        if not b_exists and not a_exists:
            st = "absent"
        elif not b_exists and a_exists:
            st = "created"
        elif b_exists and not a_exists:
            st = "deleted"
        elif (b or {}).get("sha256") != (a or {}).get("sha256"):
            st = "modified"
        else:
            st = "unchanged"
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
    # 作用：内部符号：dump_provenance
    # 能否简略：否
    # 原因：规模≈3 行；引用次数≈5（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/contract_provenance.py:178；类型=function；引用≈5；规模≈3行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
