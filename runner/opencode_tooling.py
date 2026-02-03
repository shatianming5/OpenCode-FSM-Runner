from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .security import cmd_allowed, looks_interactive, safe_env
from .subprocess_utils import STDIO_TAIL_CHARS, run_cmd_capture, tail


@dataclass(frozen=True)
class ToolCall:
    kind: str  # bash | file
    start: int
    payload: dict[str, Any] | str


@dataclass(frozen=True)
class ToolResult:
    kind: str
    ok: bool
    detail: dict[str, Any]


_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]+)?\n(?P<body>.*?)\n```", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<(?P<tag>bash|read|edit)>(?P<body>.*?)</(?P=tag)>", re.DOTALL | re.IGNORECASE)
_SELF_CLOSING_RE = re.compile(r"<(?P<tag>bash|read|write)\s+(?P<attrs>[^/>]*?)/>", re.DOTALL | re.IGNORECASE)
_ATTR_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*"(?P<val>[^"]*)"')


def _try_json(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    try:
        data = json.loads(s)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from OpenCode-style outputs.

    Supported patterns:
    - ```json {\"filePath\": \"...\", \"content\": \"...\"}```  -> file write
    - ```json {\"filePath\": \"...\"}```                      -> file read
    - ```bash\\nbash\\n{\"command\": \"...\"}```              -> bash command
    - <tool_call><bash>{...}</bash></tool_call>
    - <tool_call><edit>{...}</edit></tool_call>
    - <tool_call><read>{...}</read></tool_call>
    """
    calls: list[ToolCall] = []

    # <bash command="..." /> / <write filePath="..." content="..." /> / <read filePath="..." />
    for m in _SELF_CLOSING_RE.finditer(text):
        tag = (m.group("tag") or "").strip().lower()
        attrs_raw = m.group("attrs") or ""
        attrs: dict[str, str] = {}
        for am in _ATTR_RE.finditer(attrs_raw):
            key = (am.group("key") or "").strip()
            val = am.group("val") or ""
            attrs[key] = val
        if tag == "bash" and attrs.get("command"):
            calls.append(
                ToolCall(
                    kind="bash",
                    start=m.start(),
                    payload={"command": attrs.get("command", ""), "description": attrs.get("description", "")},
                )
            )
        if tag == "read" and attrs.get("filePath"):
            calls.append(ToolCall(kind="file", start=m.start(), payload={"filePath": attrs["filePath"]}))
        if tag == "write" and attrs.get("filePath"):
            payload: dict[str, Any] = {"filePath": attrs["filePath"]}
            if "content" in attrs:
                payload["content"] = attrs["content"].replace("\\n", "\n")
            calls.append(ToolCall(kind="file", start=m.start(), payload=payload))

    for m in _TOOL_CALL_RE.finditer(text):
        inner = m.group("body") or ""
        for tm in _TAG_RE.finditer(inner):
            tag = (tm.group("tag") or "").strip().lower()
            body = (tm.group("body") or "").strip()
            data = _try_json(body)
            if not data:
                continue
            if tag == "bash" and isinstance(data.get("command"), str):
                calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
            if tag in ("read", "edit") and isinstance(data.get("filePath"), str):
                calls.append(ToolCall(kind="file", start=m.start(), payload=data))

    for m in _FENCE_RE.finditer(text):
        lang = (m.group("lang") or "").strip().lower()
        body = (m.group("body") or "").strip()

        if lang == "json":
            data = _try_json(body)
            if data and isinstance(data.get("filePath"), str):
                calls.append(ToolCall(kind="file", start=m.start(), payload=data))
            continue

        if lang == "bash":
            lines = body.splitlines()
            if not lines:
                continue
            if lines[0].strip().lower() != "bash":
                continue
            data = _try_json("\n".join(lines[1:]))
            if data and isinstance(data.get("command"), str):
                calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
            continue

    calls.sort(key=lambda c: c.start)
    # De-dup identical calls that can appear via nested tags + fences.
    uniq: list[ToolCall] = []
    seen: set[str] = set()
    for c in calls:
        key = json.dumps({"kind": c.kind, "payload": c.payload}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _is_env_like(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env":
        return True
    if name.startswith(".env."):
        return True
    if name.endswith(".env"):
        return True
    if ".env." in name:
        return True
    return False


def _within_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def _sanitized_env(*, unattended: str) -> dict[str, str]:
    base: dict[str, str] = {}
    for k in ("PATH", "HOME", "LANG", "LC_ALL", "TERM"):
        v = os.environ.get(k)
        if v:
            base[k] = v
    # Never pass obvious secret env vars to tool commands.
    for k, v in os.environ.items():
        ku = k.upper()
        if ku.endswith("_KEY") or ku.endswith("_TOKEN") or ku.endswith("_PASSWORD") or ku.endswith("_SECRET"):
            continue
        if ku in ("OPENAI_API_KEY", "OPENCODE_SERVER_PASSWORD", "OPENCODE_SERVER_USERNAME"):
            continue
        # Keep PATH/HOME/etc only.
    return safe_env(base, {}, unattended=unattended)


def _restricted_bash_allowed(cmd: str, *, repo: Path) -> tuple[bool, str | None]:
    s = cmd.strip()
    if not s:
        return False, "empty_command"

    # No shell metacharacters in restricted mode.
    if any(ch in s for ch in (";", "|", "&", ">", "<")):
        return False, "blocked_shell_metacharacters"

    try:
        argv = shlex.split(s)
    except ValueError:
        return False, "blocked_unparseable_command"
    if not argv:
        return False, "empty_command"

    prog = argv[0]
    args = argv[1:]

    if prog == "ls":
        return True, None

    if prog in ("rg", "grep"):
        return True, None

    if prog == "git":
        if not args:
            return False, "blocked_git_without_subcommand"
        if args[0] not in ("status", "diff", "log", "show"):
            return False, "blocked_git_subcommand"
        return True, None

    if prog == "cat":
        if not args:
            return False, "blocked_cat_without_path"
        # Allow cat on repo-relative files only.
        for raw in args:
            p = Path(raw)
            if p.is_absolute() or ".." in p.parts:
                return False, "blocked_cat_non_repo_path"
            abs_p = (repo / p).resolve()
            if not _within_root(repo, abs_p):
                return False, "blocked_cat_non_repo_path"
            if _is_env_like(abs_p):
                return False, "blocked_cat_env_file"
        return True, None

    return False, "blocked_by_restricted_bash_mode"


@dataclass(frozen=True)
class ToolPolicy:
    repo: Path
    plan_path: Path
    pipeline_path: Path | None
    purpose: str
    bash_mode: str
    unattended: str

    def allow_file_read(self, path: Path) -> tuple[bool, str | None]:
        if _is_env_like(path):
            return False, "reading_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"
        return True, None

    def allow_file_write(self, path: Path) -> tuple[bool, str | None]:
        if _is_env_like(path):
            return False, "writing_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"

        p = str(self.purpose or "").strip().lower()
        if p.startswith("plan_update") or p in ("mark_done", "block_step"):
            if path == self.plan_path:
                return True, None
            return False, "plan_update_allows_only_plan_md"

        if p == "execute_step":
            if path == self.plan_path:
                return False, "execute_step_disallows_plan_md"
            if self.pipeline_path and path == self.pipeline_path:
                return False, "execute_step_disallows_pipeline_yml"
            return True, None

        if p.startswith("fix_or_replan"):
            if self.pipeline_path and path == self.pipeline_path:
                return False, "fix_or_replan_disallows_pipeline_yml"
            return True, None

        return True, None

    def allow_bash(self, cmd: str) -> tuple[bool, str | None]:
        cmd = cmd.strip()
        if not cmd:
            return False, "empty_command"

        if str(self.bash_mode or "restricted").strip().lower() != "full":
            return _restricted_bash_allowed(cmd, repo=self.repo)

        allowed, reason = cmd_allowed(cmd, pipeline=None)
        if not allowed:
            return False, reason or "blocked"
        if str(self.unattended or "").strip().lower() == "strict" and looks_interactive(cmd):
            return False, "likely_interactive_command_disallowed_in_strict_mode"
        return True, None


def execute_tool_calls(
    calls: Iterable[ToolCall],
    *,
    repo: Path,
    policy: ToolPolicy,
) -> list[ToolResult]:
    results: list[ToolResult] = []

    for call in calls:
        if call.kind == "file":
            data = call.payload if isinstance(call.payload, dict) else {}
            file_path_raw = str(data.get("filePath") or "").strip()
            content = data.get("content")

            if not file_path_raw:
                results.append(ToolResult(kind="file", ok=False, detail={"error": "missing_filePath"}))
                continue

            file_path = Path(file_path_raw).expanduser()
            if not file_path.is_absolute():
                file_path = (repo / file_path).resolve()
            else:
                file_path = file_path.resolve()

            if content is None:
                ok, reason = policy.allow_file_read(file_path)
                if not ok:
                    results.append(
                        ToolResult(
                            kind="read",
                            ok=False,
                            detail={"filePath": str(file_path), "error": reason or "blocked"},
                        )
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="read", ok=False, detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                raw = file_path.read_text(encoding="utf-8", errors="replace")
                results.append(
                    ToolResult(
                        kind="read",
                        ok=True,
                        detail={"filePath": str(file_path), "content": tail(raw, 20000)},
                    )
                )
                continue

            if not isinstance(content, str):
                results.append(
                    ToolResult(kind="write", ok=False, detail={"filePath": str(file_path), "error": "invalid_content"})
                )
                continue

            ok, reason = policy.allow_file_write(file_path)
            if not ok:
                results.append(
                    ToolResult(
                        kind="write",
                        ok=False,
                        detail={"filePath": str(file_path), "error": reason or "blocked"},
                    )
                )
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8", errors="replace")
            results.append(
                ToolResult(kind="write", ok=True, detail={"filePath": str(file_path), "bytes": len(content)})
            )
            continue

        if call.kind == "bash":
            data = call.payload if isinstance(call.payload, dict) else {}
            cmd = str(data.get("command") or "").strip()
            ok, reason = policy.allow_bash(cmd)
            if not ok:
                results.append(ToolResult(kind="bash", ok=False, detail={"command": cmd, "error": reason or "blocked"}))
                continue

            env = _sanitized_env(unattended=str(policy.unattended or "strict"))
            res = run_cmd_capture(cmd, repo, timeout_seconds=60, env=env, interactive=False)
            results.append(
                ToolResult(
                    kind="bash",
                    ok=(res.rc == 0),
                    detail={
                        "command": cmd,
                        "rc": res.rc,
                        "timed_out": res.timed_out,
                        "stdout": tail(res.stdout or "", STDIO_TAIL_CHARS),
                        "stderr": tail(res.stderr or "", STDIO_TAIL_CHARS),
                    },
                )
            )
            continue

        results.append(ToolResult(kind=str(call.kind), ok=False, detail={"error": "unsupported_tool"}))

    return results


def format_tool_results(results: list[ToolResult]) -> str:
    payload = [r.detail | {"tool": r.kind, "ok": r.ok} for r in results]
    return (
        "Tool results (executed by the runner). Continue by either issuing more tool calls or responding normally.\n\n"
        "```tool_result\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
