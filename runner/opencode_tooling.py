from __future__ import annotations

import html
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
    """中文说明：
    - 含义：从 assistant 输出中解析出来的“工具调用”描述。
    - 内容：kind 表示类型（bash/file），start 表示在原文本中的起始位置（用于排序），payload 是解析出的参数（如 command/filePath/content）。
    - 可简略：否（tool-call 是 agent 执行闭环的基础协议）。
    """

    kind: str  # bash | file
    start: int
    payload: dict[str, Any] | str


@dataclass(frozen=True)
class ToolResult:
    """中文说明：
    - 含义：Runner 执行单个 tool-call 的结果（用于回灌给 agent）。
    - 内容：包含执行是否成功、以及标准化 detail（例如 read 的 content，bash 的 rc/stdout/stderr）。
    - 可简略：否（tool loop 的关键数据结构；影响可审计性与可恢复性）。
    """

    kind: str
    ok: bool
    detail: dict[str, Any]


_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]+)?\n(?P<body>.*?)\n```", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<(?P<tag>bash|read|edit)>(?P<body>.*?)</(?P=tag)>", re.DOTALL | re.IGNORECASE)
_ATTR_TAG_START_RE = re.compile(r"<(?P<tag>bash|read|write|edit)\b", re.IGNORECASE)
_ATTR_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]*")
_INLINE_TOOL_JSON_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<tag>bash|read|write|edit)\s*(?P<json>\{)",
    re.IGNORECASE,
)


def _xml_unescape(text: str) -> str:
    """Unescape common XML/HTML entities in tool-call payloads.

    OpenCode tool-calls are XML-like. Model outputs often escape `<`/`&` in file content
    (e.g. `<<` becomes `&lt;&lt;`). We want the literal characters in written files.
    """
    # Some model outputs are double-escaped (`&amp;amp;`), so unescape a few times
    # until stable. Keep a small cap to avoid pathological expansion.
    s = str(text or "")
    for _ in range(3):
        s2 = html.unescape(s)
        if s2 == s:
            break
        s = s2
    return s


def _decode_attr_value(raw: str) -> str:
    """Decode minimal escapes inside XML-like attribute values.

    Keep unknown escapes intact (e.g. ``\\n``) so later logic can decide whether
    to interpret them.
    """
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in {'"', "'", "\\"}:
                out.append(nxt)
            else:
                out.append("\\")
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_attrs(attrs_raw: str) -> dict[str, str]:
    """Parse XML-like key/value attributes with quoted values."""
    attrs: dict[str, str] = {}
    i = 0
    n = len(attrs_raw)
    while i < n:
        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n:
            break

        km = _ATTR_NAME_RE.match(attrs_raw, i)
        if not km:
            i += 1
            continue
        key = km.group(0)
        i = km.end()

        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n or attrs_raw[i] != "=":
            continue
        i += 1

        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n:
            break

        quote = attrs_raw[i]
        if quote not in {'"', "'"}:
            # Best-effort fallback for unquoted values.
            start = i
            while i < n and not attrs_raw[i].isspace():
                i += 1
            attrs[key] = attrs_raw[start:i]
            continue
        i += 1

        buf: list[str] = []
        while i < n:
            ch = attrs_raw[i]
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(attrs_raw[i + 1])
                i += 2
                continue
            if ch == quote:
                i += 1
                break
            buf.append(ch)
            i += 1

        attrs[key] = _decode_attr_value("".join(buf))
    return attrs


def _extract_attr_loose(attrs_raw: str, key: str) -> str | None:
    """Best-effort attribute extractor for malformed/self-closing tags.

    Handles cases where quoted values contain unescaped quotes that break the
    strict parser, especially `content='...f'...'`.
    """
    km = re.search(rf"\b{re.escape(key)}\s*=\s*(['\"])", attrs_raw, flags=re.IGNORECASE)
    if not km:
        return None
    quote = km.group(1)
    start = km.end()
    if start >= len(attrs_raw):
        return None

    if key.lower() == "content":
        end = attrs_raw.rfind(quote)
        if end <= start:
            return None
        return _decode_attr_value(attrs_raw[start:end])

    i = start
    escaped = False
    while i < len(attrs_raw):
        ch = attrs_raw[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == quote:
            return _decode_attr_value(attrs_raw[start:i])
        i += 1
    return None


def _try_json(text: str) -> dict[str, Any] | None:
    """中文说明：
    - 含义：尝试把字符串解析为 JSON object（dict）。
    - 内容：解析失败或非 dict 则返回 None；用于从 fenced code block / tag body 中提取工具参数。
    - 可简略：可能（内部 helper；但集中处理能减少重复 try/except）。
    """
    s = text.strip()
    if not s:
        return None
    try:
        data = json.loads(s)
    except Exception:
        # Common model bug: invalid JSON escapes inside string values.
        #
        # Example (invalid JSON): `"content":"... \\${VAR} ..."` because `\$` is not a valid
        # JSON escape. Models often include `\${...}` when trying to prevent shell expansion.
        # Fix by doubling any backslash that does not introduce a valid JSON escape sequence,
        # but only when inside JSON double-quoted strings.
        def _repair_invalid_string_escapes(raw: str) -> str:
            valid_next = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
            out: list[str] = []
            in_string = False
            escaped = False
            i = 0
            while i < len(raw):
                ch = raw[i]
                if not in_string:
                    out.append(ch)
                    if ch == '"':
                        in_string = True
                    i += 1
                    continue

                if escaped:
                    out.append(ch)
                    escaped = False
                    i += 1
                    continue

                if ch == "\\":
                    if i + 1 >= len(raw):
                        out.append("\\\\")
                        i += 1
                        continue
                    nxt = raw[i + 1]
                    if nxt in valid_next:
                        out.append("\\")
                        escaped = True
                        i += 1
                        continue
                    # Invalid escape sequence: preserve the backslash as a literal char.
                    out.append("\\\\")
                    i += 1
                    continue

                out.append(ch)
                if ch == '"':
                    in_string = False
                i += 1
            return "".join(out)

        repaired_escapes = _repair_invalid_string_escapes(s)
        if repaired_escapes != s:
            try:
                data = json.loads(repaired_escapes)
                return data if isinstance(data, dict) else None
            except Exception:
                pass

        # Common model bug: missing opening quote around JSON keys, e.g. `{filePath": "..."}`
        # or `,filePath":` (has trailing quote but missing the leading one).
        repaired = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)"\s*:', r'\1"\2":', s)
        if repaired != s:
            try:
                data = json.loads(repaired)
                return data if isinstance(data, dict) else None
            except Exception:
                # If key-quote repair succeeded but invalid escapes remain, try both together.
                repaired2 = _repair_invalid_string_escapes(repaired)
                if repaired2 != repaired:
                    try:
                        data = json.loads(repaired2)
                        return data if isinstance(data, dict) else None
                    except Exception:
                        return None
                return None
        return None
    return data if isinstance(data, dict) else None


def _extract_json_object(text: str, start: int) -> tuple[str, int] | None:
    """Extract a balanced JSON object from `text[start:]` where `text[start] == '{'`."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1
    return None


def _find_tag_gt(text: str, start: int) -> int:
    """Find the end of an XML-like opening tag, respecting quoted attrs."""
    i = start
    quote: str | None = None
    escaped = False
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == ">":
            return i
        i += 1
    return -1


def _iter_attr_tags(text: str) -> Iterable[tuple[str, int, str, str]]:
    """Yield (tag, start, attrs_raw, body) for `<bash|read|write|edit ...>` tags.

    Uses a linear scanner to avoid catastrophic regex backtracking on malformed
    long assistant outputs.
    """
    lower = text.lower()
    pos = 0
    while True:
        m = _ATTR_TAG_START_RE.search(text, pos)
        if not m:
            break
        tag = (m.group("tag") or "").strip().lower()
        start = m.start()
        gt = _find_tag_gt(text, m.end())
        if gt < 0:
            pos = m.end()
            continue

        raw_open = text[m.end() : gt]
        raw_open_rstrip = raw_open.rstrip()
        self_closing = raw_open_rstrip.endswith("/")
        attrs_raw = raw_open_rstrip[:-1].rstrip() if self_closing else raw_open.strip()

        body = ""
        end = gt + 1
        if not self_closing:
            close = f"</{tag}>"
            close_idx = lower.find(close, gt + 1)
            if close_idx < 0:
                pos = gt + 1
                continue
            body = text[gt + 1 : close_idx]
            end = close_idx + len(close)

        yield tag, start, attrs_raw, body
        pos = end


def parse_tool_calls(text: str) -> list[ToolCall]:
    """中文说明：
    - 含义：从 OpenCode 风格的 assistant 输出中提取 tool-calls。
    - 内容：支持 fenced ```json```/```bash```、以及 `<tool_call><bash/read/edit>...</...></tool_call>` 等多种格式；返回按出现顺序排序并去重后的 ToolCall 列表。
    - 可简略：否（这是 runner 与 agent 的“可执行协议解析器”；简化易导致兼容性回退）。

    ---

    English (original intent):
    Extract tool calls from OpenCode-style outputs.

    Supported patterns:
    - ```json {\"filePath\": \"...\", \"content\": \"...\"}```  -> file write
    - ```json {\"filePath\": \"...\"}```                      -> file read
    - ```bash\\nbash\\n{\"command\": \"...\"}```              -> bash command
    - <tool_call><bash>{...}</bash></tool_call>
    - <tool_call><edit>{...}</edit></tool_call>
    - <tool_call><read>{...}</read></tool_call>
    - read{...} / write\n{...} / bash{...}
    """
    calls: list[ToolCall] = []

    # Compatibility: some models output malformed tags like `bash<command="..."/>`
    # (missing the leading `<bash .../>`). Normalize these so the scanner can parse them.
    text = re.sub(r"(?i)(?<!<)\b(bash|read|write|edit)\s*<", r"<\1 ", text)

    # Parse attribute tags with a linear scanner. This avoids catastrophic regex
    # backtracking on malformed long tool output.
    for tag, start, attrs_raw, body in _iter_attr_tags(text):
        attrs = _parse_attrs(attrs_raw)
        # Fill missing attrs from malformed quoting (best effort).
        if "command" not in attrs:
            cmd = _extract_attr_loose(attrs_raw, "command")
            if cmd is not None:
                attrs["command"] = cmd
        if "filePath" not in attrs:
            fp = _extract_attr_loose(attrs_raw, "filePath")
            if fp is not None:
                attrs["filePath"] = fp
        if tag == "write":
            ct = _extract_attr_loose(attrs_raw, "content")
            if ct is not None:
                prev = attrs.get("content")
                if not isinstance(prev, str) or len(ct) > len(prev):
                    attrs["content"] = ct
        if tag == "edit":
            os_ = _extract_attr_loose(attrs_raw, "oldString")
            if os_ is not None:
                prev = attrs.get("oldString")
                if not isinstance(prev, str) or len(os_) > len(prev):
                    attrs["oldString"] = os_
            ns_ = _extract_attr_loose(attrs_raw, "newString")
            if ns_ is not None:
                prev = attrs.get("newString")
                if not isinstance(prev, str) or len(ns_) > len(prev):
                    attrs["newString"] = ns_

        if tag == "bash" and attrs.get("command"):
            calls.append(
                ToolCall(
                    kind="bash",
                    start=start,
                    payload={"command": attrs.get("command", ""), "description": attrs.get("description", "")},
                )
            )
        if tag == "read" and attrs.get("filePath"):
            calls.append(ToolCall(kind="file", start=start, payload={"filePath": attrs["filePath"]}))
        if tag == "write" and attrs.get("filePath"):
            payload2: dict[str, Any] = {"filePath": attrs["filePath"]}
            if body:
                payload2["content"] = body
            elif "content" in attrs:
                payload2["content"] = attrs["content"].replace("\\n", "\n")
            calls.append(ToolCall(kind="file", start=start, payload=payload2))
        if tag == "edit" and attrs.get("filePath"):
            payload2 = {"filePath": attrs["filePath"]}
            if body:
                payload2["content"] = body
            if "oldString" in attrs:
                payload2["oldString"] = attrs["oldString"].replace("\\n", "\n")
            if "newString" in attrs:
                payload2["newString"] = attrs["newString"].replace("\\n", "\n")
            calls.append(ToolCall(kind="file", start=start, payload=payload2))

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

        # Compatibility: some agents emit `<tool_call>{...json...}</tool_call>` without inner `<read>/<bash>` tags.
        data2 = _try_json(inner)
        if data2:
            # Some models wrap tool calls as `{ "name": "write", "arguments": { ... } }`.
            raw_name = data2.get("name") or data2.get("tool") or data2.get("toolName")
            raw_args = data2.get("arguments") or data2.get("args")
            if isinstance(raw_name, str) and isinstance(raw_args, dict):
                name = raw_name.strip().lower()
                args = raw_args
                if name == "bash" and isinstance(args.get("command"), str):
                    calls.append(ToolCall(kind="bash", start=m.start(), payload=args))
                if name in ("read", "write", "edit") and isinstance(args.get("filePath"), str):
                    calls.append(ToolCall(kind="file", start=m.start(), payload=args))
            else:
                if isinstance(data2.get("command"), str):
                    calls.append(ToolCall(kind="bash", start=m.start(), payload=data2))
                if isinstance(data2.get("filePath"), str):
                    calls.append(ToolCall(kind="file", start=m.start(), payload=data2))

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

    # Compatibility: some models emit plain inline calls like `read{...}` or `write\n{...}`.
    for m in _INLINE_TOOL_JSON_RE.finditer(text):
        tag = (m.group("tag") or "").strip().lower()
        json_start = int(m.start("json"))
        extracted = _extract_json_object(text, json_start)
        if not extracted:
            continue
        raw_json, _ = extracted
        data = _try_json(raw_json)
        if not data:
            continue
        if tag == "bash" and isinstance(data.get("command"), str):
            calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
        if tag in ("read", "write", "edit") and isinstance(data.get("filePath"), str):
            calls.append(ToolCall(kind="file", start=m.start(), payload=data))

    calls.sort(key=lambda c: c.start)
    # If strict + loose parsing captured the same tag start, keep the richer payload.
    merged: list[ToolCall] = []
    for c in calls:
        if merged and merged[-1].start == c.start and merged[-1].kind == c.kind:
            prev = merged[-1]
            prev_size = len(json.dumps(prev.payload, sort_keys=True, ensure_ascii=False))
            cur_size = len(json.dumps(c.payload, sort_keys=True, ensure_ascii=False))
            if cur_size >= prev_size:
                merged[-1] = c
            continue
        merged.append(c)
    calls = merged

    # De-dup identical calls that can appear via nested tags + fences.
    uniq: list[ToolCall] = []
    seen: set[str] = set()
    for c in calls:
        key = json.dumps({"kind": c.kind, "payload": c.payload}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # XML-like tool calls often escape `<`/`&` inside attribute values and tag bodies.
    # Decode them so written files contain the intended literal characters.
    for c in uniq:
        if not isinstance(c.payload, dict):
            continue
        for k in ("filePath", "content", "oldString", "newString", "command", "description"):
            v = c.payload.get(k)
            if isinstance(v, str) and v:
                c.payload[k] = _xml_unescape(v)
    return uniq


def _is_env_like(path: Path) -> bool:
    """中文说明：
    - 含义：判断某个路径是否类似 dotenv（可能包含敏感信息）。
    - 内容：匹配 `.env`/`.env.*`/`*.env` 等命名；用于阻止 agent 读取/写入这些文件。
    - 可简略：可能（启发式；可按实际项目策略调整）。
    """
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
    """中文说明：
    - 含义：判断 target 是否位于 root 目录树内（越界保护）。
    - 内容：通过 `relative_to` 实现；用于限制 tool-call 读写只能发生在 repo 内。
    - 可简略：可能（与 `runner.paths.is_relative_to` 类似，可考虑合并）。
    """
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def _sanitized_env(*, unattended: str) -> dict[str, str]:
    """中文说明：
    - 含义：为 tool-call 中的 bash 命令构造一个“尽量不泄密”的环境变量集合。
    - 内容：只传递 PATH/HOME/LANG 等少量基础变量；显式过滤 *_KEY/*_TOKEN/*_PASSWORD 等；再应用 strict 模式的 safe_env 默认值。
    - 可简略：否（避免 secret 泄漏到不受信任命令的关键安全措施）。
    """
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
    """中文说明：
    - 含义：restricted bash 模式下，对命令做白名单式解析与校验。
    - 内容：禁止 shell 元字符（`;|&><`）；只允许少量只读/诊断命令（ls/rg/grep、git status/diff/log/show、cat repo 内非 env 文件）。
    - 可简略：否（这是 tool-call 执行层的核心安全边界）。
    """
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
    """中文说明：
    - 含义：对 tool-call 的访问控制策略（文件读写 + bash 执行）。
    - 内容：结合 repo 根目录、PLAN/pipeline 路径、purpose（plan_update/execute/scaffold 等）、bash_mode（restricted/full）与 unattended 模式来做精细化限制。
    - 可简略：否（安全与权限分层的核心）。
    """

    repo: Path
    plan_path: Path
    pipeline_path: Path | None
    purpose: str
    bash_mode: str
    unattended: str

    def allow_file_read(self, path: Path) -> tuple[bool, str | None]:
        """中文说明：
        - 含义：判断是否允许读取该文件。
        - 内容：禁止读取 dotenv；禁止 repo 外路径。
        - 可简略：可能（规则较少；但建议保留集中入口）。
        """
        if _is_env_like(path):
            return False, "reading_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"
        return True, None

    def allow_file_write(self, path: Path) -> tuple[bool, str | None]:
        """中文说明：
        - 含义：判断是否允许写入该文件。
        - 内容：按 purpose 分层：
          - scaffold_contract：只允许写 `pipeline.yml` 与 `.aider_fsm/**`
          - plan_update/mark_done/block_step：只允许写 `PLAN.md`
          - execute_step：禁止写 PLAN/pipeline
          - fix_or_replan：禁止写 pipeline
        - 可简略：否（权限隔离的关键；简化易导致越权写入）。
        """
        if _is_env_like(path):
            return False, "writing_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"

        p = str(self.purpose or "").strip().lower()
        if p == "scaffold_contract":
            pipeline = (self.repo / "pipeline.yml").resolve()
            aider = (self.repo / ".aider_fsm").resolve()
            if path == pipeline:
                return True, None
            if _within_root(aider, path):
                return True, None
            return False, "scaffold_contract_allows_only_pipeline_yml_and_aider_fsm"
        if p == "repair_contract":
            aider = (self.repo / ".aider_fsm").resolve()
            if _within_root(aider, path):
                return True, None
            return False, "repair_contract_allows_only_aider_fsm"
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
        """中文说明：
        - 含义：判断是否允许执行某条 bash 命令。
        - 内容：restricted 模式走 `_restricted_bash_allowed`；full 模式则复用 `security.cmd_allowed` 与 strict 的交互阻断。
        - 可简略：否（命令执行安全边界）。
        """
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
    """中文说明：
    - 含义：逐条执行解析出的 tool-calls，并返回可回灌给 agent 的结果列表。
    - 内容：
      - file：支持 read（无 content）与 write（有 content），并应用 ToolPolicy 约束
      - bash：执行命令（固定 timeout=60s），并对 stdout/stderr 截断；执行环境使用 `_sanitized_env`
    - 可简略：否（tool loop 的执行内核；影响可控性与安全）。
    """
    results: list[ToolResult] = []

    for call in calls:
        if call.kind == "file":
            data = call.payload if isinstance(call.payload, dict) else {}
            file_path_raw = _xml_unescape(str(data.get("filePath") or "")).strip()
            content = data.get("content")

            if not file_path_raw:
                results.append(ToolResult(kind="file", ok=False, detail={"error": "missing_filePath"}))
                continue

            file_path = Path(file_path_raw).expanduser()
            if not file_path.is_absolute():
                file_path = (repo / file_path).resolve()
            else:
                file_path = file_path.resolve()

            # Support OpenCode `<edit filePath="..." oldString="..." newString="..." />` tags.
            if content is None and ("oldString" in data or "newString" in data):
                old = data.get("oldString")
                new = data.get("newString")
                if isinstance(old, str) and old:
                    old = _xml_unescape(old)
                if isinstance(new, str) and new:
                    new = _xml_unescape(new)
                if new is None:
                    new_s = ""
                elif isinstance(new, str):
                    new_s = new
                else:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={"filePath": str(file_path), "error": "invalid_newString"},
                        )
                    )
                    continue

                # If oldString is empty/missing, treat as full replacement write.
                if old is None or (isinstance(old, str) and old == ""):
                    ok, reason = policy.allow_file_write(file_path)
                    if not ok:
                        results.append(
                            ToolResult(
                                kind="edit",
                                ok=False,
                                detail={"filePath": str(file_path), "error": reason or "blocked"},
                            )
                        )
                        continue
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(new_s, encoding="utf-8", errors="replace")
                    except Exception as e:
                        results.append(
                            ToolResult(
                                kind="edit",
                                ok=False,
                                detail={
                                    "filePath": str(file_path),
                                    "error": "write_failed",
                                    "exception": type(e).__name__,
                                    "message": str(e)[:200],
                                },
                            )
                        )
                        continue
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=True,
                            detail={"filePath": str(file_path), "bytes": len(new_s), "mode": "replace"},
                        )
                    )
                    continue

                if not isinstance(old, str):
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={"filePath": str(file_path), "error": "invalid_oldString"})
                    )
                    continue

                ok, reason = policy.allow_file_write(file_path)
                if not ok:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={"filePath": str(file_path), "error": reason or "blocked"},
                        )
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={
                                "filePath": str(file_path),
                                "error": "read_failed",
                                "exception": type(e).__name__,
                                "message": str(e)[:200],
                            },
                        )
                    )
                    continue
                matches = raw.count(old)
                if matches <= 0:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={"filePath": str(file_path), "error": "oldString_not_found"},
                        )
                    )
                    continue
                if matches != 1:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={"filePath": str(file_path), "error": "oldString_not_unique", "matches": matches},
                        )
                    )
                    continue
                updated = raw.replace(old, new_s, 1)
                try:
                    file_path.write_text(updated, encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(
                            kind="edit",
                            ok=False,
                            detail={
                                "filePath": str(file_path),
                                "error": "write_failed",
                                "exception": type(e).__name__,
                                "message": str(e)[:200],
                            },
                        )
                    )
                    continue
                results.append(
                    ToolResult(
                        kind="edit",
                        ok=True,
                        detail={"filePath": str(file_path), "bytes": len(updated), "mode": "replace_once"},
                    )
                )
                continue

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
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(
                            kind="read",
                            ok=False,
                            detail={
                                "filePath": str(file_path),
                                "error": "read_failed",
                                "exception": type(e).__name__,
                                "message": str(e)[:200],
                            },
                        )
                    )
                    continue
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
            content = _xml_unescape(content)

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

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8", errors="replace")
            except Exception as e:
                results.append(
                    ToolResult(
                        kind="write",
                        ok=False,
                        detail={
                            "filePath": str(file_path),
                            "error": "write_failed",
                            "exception": type(e).__name__,
                            "message": str(e)[:200],
                        },
                    )
                )
                continue
            results.append(
                ToolResult(kind="write", ok=True, detail={"filePath": str(file_path), "bytes": len(content)})
            )
            continue

        if call.kind == "bash":
            data = call.payload if isinstance(call.payload, dict) else {}
            cmd = _xml_unescape(str(data.get("command") or "")).strip()
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
    """中文说明：
    - 含义：把 tool 执行结果包装成 OpenCode 期望的 ` ```tool_result ... ``` ` 文本格式。
    - 内容：将结果列表序列化为 JSON，并提示 agent 继续发起 tool-calls 或给出最终回复。
    - 可简略：可能（格式协议相对固定；但保留集中函数便于未来调整兼容性）。
    """
    payload = [r.detail | {"tool": r.kind, "ok": r.ok} for r in results]
    return (
        "Tool results (executed by the runner). Continue by either issuing more tool calls or responding normally.\n\n"
        "```tool_result\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
