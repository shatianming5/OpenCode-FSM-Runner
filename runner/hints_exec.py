from __future__ import annotations

import configparser
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import cmd_allowed
from ._util import _is_truthy, _parse_json_str_list


def _read_hints_file(path: Path) -> list[str]:
    # 作用：内部符号：_read_hints_file
    # 能否简略：是
    # 原因：规模≈12 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:38；类型=function；引用≈2；规模≈12行
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _find_latest_scaffold_hints_file(repo: Path) -> Path | None:
    # 作用：内部符号：_find_latest_scaffold_hints_file
    # 能否简略：是
    # 原因：规模≈10 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:52；类型=function；引用≈2；规模≈10行
    repo = Path(repo).resolve()
    root = (repo / ".aider_fsm" / "artifacts").resolve()
    if not root.exists():
        return None
    candidates = list(root.glob("*/scaffold/scaffold_command_hints.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]


_FLAG_VALUE_RE = re.compile(r"(?P<flag>--[A-Za-z0-9_.-]+)\s+(?P<val>(?:\"[^\"]*\"|'[^']*'|\S+))")


def _replace_flag_value(cmd: str, *, flag: str, new_value: str) -> str:
    # 作用：内部符号：_replace_flag_value
    # 能否简略：部分
    # 原因：规模≈17 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:67；类型=function；引用≈4；规模≈17行
    flag = flag.strip()
    if not flag:
        return cmd
    if not new_value:
        return cmd

    def repl(m: re.Match[str]) -> str:
        # 作用：内部符号：_replace_flag_value.repl
        # 能否简略：是
        # 原因：规模≈8 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:74；类型=function；引用≈2；规模≈8行
        if m.group("flag") != flag:
            return m.group(0)
        v = new_value
        # Quote the value if it has whitespace.
        if any(ch.isspace() for ch in v):
            v = json.dumps(v)
        return f"{flag} {v}"

    return _FLAG_VALUE_RE.sub(repl, cmd)


_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_ANGLE_GROUP_RE = re.compile(r"<[^>]+>")
_GHA_EXPR_RE = re.compile(r"\$\{\{\s*([^}]+)\s*\}\}")
_PIPE_TO_BASH_RE = re.compile(r"(?i)\b(?:curl|wget)\b[^\n]*\|[^\n]*\bbash\b")
_DOTTED_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+$")
_DOCKER_LINE_RE = re.compile(r"(?im)^\s*docker\s+")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_BUILTINS = {
    ":",
    ".",
    "alias",
    "bg",
    "break",
    "builtin",
    "cd",
    "command",
    "continue",
    "dirs",
    "echo",
    "eval",
    "exec",
    "exit",
    "export",
    "false",
    "fg",
    "hash",
    "help",
    "history",
    "jobs",
    "kill",
    "local",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "source",
    "test",
    "times",
    "trap",
    "true",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "wait",
}


_PY_MAJOR_MINOR_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")


def _as_major_minor(raw: str | None) -> str:
    # 作用：内部符号：_as_major_minor
    # 能否简略：是
    # 原因：规模≈14 行；引用次数≈5（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈5；规模≈14行
    s = str(raw or "").strip()
    if not s:
        return ""
    m = _PY_MAJOR_MINOR_RE.search(s)
    if not m:
        return ""
    try:
        major = int(m.group("major"))
        minor = int(m.group("minor"))
    except Exception:
        return ""
    if major <= 0 or minor < 0:
        return ""
    return f"{major}.{minor}"


def _infer_repo_python_pin(repo: Path) -> str:
    """Infer a repo's preferred Python major.minor from common version pin files."""
    # 作用：Infer a repo's preferred Python major.minor from common version pin files.
    # 能否简略：部分
    # 原因：多来源探测（.python-version/.tool-versions）；属于兼容性策略的一部分；规模≈33 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈33行
    repo = Path(repo).resolve()
    for rel in (".python-version", "runtime.txt"):
        p = (repo / rel).resolve()
        try:
            if not p.exists() or not p.is_file():
                continue
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                mm = _as_major_minor(line)
                if mm:
                    return mm
                break
        except Exception:
            continue

    p = (repo / ".tool-versions").resolve()
    try:
        if p.exists() and p.is_file():
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("python"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    mm = _as_major_minor(parts[1])
                    if mm:
                        return mm
    except Exception:
        pass
    return ""


def _infer_repo_requires_python(repo: Path) -> str:
    """Infer a repo's `requires-python` style spec (best-effort)."""
    # 作用：Infer a repo's `requires-python` style spec (best-effort).
    # 能否简略：部分
    # 原因：覆盖 pyproject/setup.cfg/setup.py 三类常见声明；用于选择可用解释器；规模≈63 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈63行
    repo = Path(repo).resolve()
    pyproject = (repo / "pyproject.toml").resolve()
    if pyproject.exists() and pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                proj = data.get("project")
                if isinstance(proj, dict):
                    rp = proj.get("requires-python")
                    if isinstance(rp, str) and rp.strip():
                        return rp.strip()
                tool = data.get("tool")
                if isinstance(tool, dict):
                    poetry = tool.get("poetry")
                    if isinstance(poetry, dict):
                        deps = poetry.get("dependencies")
                        if isinstance(deps, dict):
                            py = deps.get("python")
                            if isinstance(py, str) and py.strip():
                                return py.strip()
        except Exception:
            pass

    setup_cfg = (repo / "setup.cfg").resolve()
    if setup_cfg.exists() and setup_cfg.is_file():
        try:
            cp = configparser.ConfigParser()
            cp.read(setup_cfg, encoding="utf-8")
            if cp.has_option("options", "python_requires"):
                v = str(cp.get("options", "python_requires") or "").strip()
                if v:
                    return v
        except Exception:
            pass

    setup_py = (repo / "setup.py").resolve()
    if setup_py.exists() and setup_py.is_file():
        try:
            text = setup_py.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"(?i)python_requires\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", text)
            if m:
                v = str(m.group(1) or "").strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def _best_python_minor_from_spec(spec: str, *, candidates: list[str]) -> str:
    """Pick the highest major.minor in candidates that satisfies a python spec string."""
    # 作用：Pick the highest major.minor in candidates that satisfies a python spec string.
    # 能否简略：部分
    # 原因：优先用 packaging 精确匹配；缺失时回退到启发式；用于 uv/venv 选择；规模≈46 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈46行
    s = str(spec or "").strip()
    if not s:
        return ""
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
        from packaging.version import Version  # type: ignore

        ss = SpecifierSet(s)
        for mm in candidates:
            try:
                if Version(f"{mm}.0") in ss:
                    return mm
            except Exception:
                continue
    except Exception:
        pass

    # Heuristic fallback when packaging isn't available or the spec isn't PEP 440 (e.g. Poetry's ^3.11).
    # Try explicit major.minor mentions first.
    for mm in candidates:
        if mm in s:
            return mm
    # Try simple upper-bound patterns like "<3.13" -> choose 3.12, etc.
    m = re.search(r"<\\s*(\\d+)\\.(\\d+)", s)
    if m:
        try:
            major = int(m.group(1))
            minor = int(m.group(2))
        except Exception:
            major = 0
            minor = 0
        if major > 0:
            want = f"{major}.{max(0, minor - 1)}"
            if want in candidates:
                return want
    return ""


def _infer_uv_python_candidates(repo: Path, *, env: dict[str, str]) -> list[str]:
    """Infer a list of uv `--python` requests to try (most preferred first)."""
    # 作用：Infer a list of uv `--python` requests to try (most preferred first).
    # 能否简略：否
    # 原因：把“显式配置 + repo 元数据 + 安全默认值”整合为统一策略；避免写死 py311；规模≈48 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈48行
    env2 = dict(env or {})
    out: list[str] = []

    raw_candidates = _parse_json_str_list(env2.get("AIDER_FSM_HINT_UV_PYTHON_CANDIDATES_JSON"))
    if raw_candidates:
        out.extend([c.strip() for c in raw_candidates if isinstance(c, str) and c.strip()])
    else:
        single = str(env2.get("AIDER_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
        if single:
            out.append(single)

    if not out:
        pinned = _infer_repo_python_pin(repo)
        if pinned:
            out.append(pinned)

    if not out:
        spec = _infer_repo_requires_python(repo)
        if spec:
            # Prefer newer stable minors first when we need to choose.
            prefer = ["3.12", "3.11", "3.10", "3.9", "3.8"]
            picked = _best_python_minor_from_spec(spec, candidates=prefer)
            if picked:
                out.append(picked)
            else:
                # As a last resort, pick any explicit X.Y mention.
                mm = _as_major_minor(spec)
                if mm:
                    out.append(mm)

    if not out and sys.version_info >= (3, 13):
        # Safe defaults when running under very new Python versions: prefer a stable minor
        # with broad wheel availability.
        out.extend(["3.12", "3.11"])

    # Deduplicate while preserving order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for v in out:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def _canonical_base_url(url: str | None) -> str:
    # 作用：内部符号：_canonical_base_url
    # 能否简略：是
    # 原因：规模≈8 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:141；类型=function；引用≈3；规模≈8行
    s = str(url or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")
    if s.endswith("/v1"):
        return s[: -len("/v1")]
    return s


def _first_command_line(cmd: str) -> str:
    # 作用：内部符号：_first_command_line
    # 能否简略：否
    # 原因：规模≈6 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:151；类型=function；引用≈6；规模≈6行
    for raw in str(cmd or "").splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _extract_cli_flag_value(cmd: str, flag: str) -> str:
    # 作用：内部符号：_extract_cli_flag_value
    # 能否简略：部分
    # 原因：规模≈17 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:159；类型=function；引用≈4；规模≈17行
    line = _first_command_line(cmd)
    if not line:
        return ""
    try:
        parts = shlex.split(line, posix=True)
    except Exception:
        return ""
    i = 0
    while i < len(parts):
        tok = str(parts[i] or "")
        if tok == flag and i + 1 < len(parts):
            return str(parts[i + 1] or "").strip()
        if tok.startswith(flag + "="):
            return str(tok.split("=", 1)[1] or "").strip()
        i += 1
    return ""


def _extract_cli_flag_value_any(cmd: str, flags: list[str]) -> str:
    # 作用：内部符号：_extract_cli_flag_value_any
    # 能否简略：部分
    # 原因：规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:178；类型=function；引用≈4；规模≈6行
    for flag in list(flags or []):
        v = _extract_cli_flag_value(cmd, str(flag))
        if v:
            return v
    return ""


def _hint_backend(cmd: str) -> str:
    # 作用：内部符号：_hint_backend
    # 能否简略：是
    # 原因：规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:186；类型=function；引用≈3；规模≈9行
    backend = _extract_cli_flag_value(cmd, "--backend").strip().lower()
    if backend:
        return backend
    line = _first_command_line(cmd).lower()
    # Generic heuristic: evaluator-style CLIs with model+dataset often default to OpenAI backend.
    if ".evaluate" in line and "--dataset" in line and "--model" in line:
        return "openai"
    return ""


def _is_remote_openai_hint(cmd: str) -> bool:
    # 作用：内部符号：_is_remote_openai_hint
    # 能否简略：是
    # 原因：规模≈2 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:197；类型=function；引用≈3；规模≈2行
    return _hint_backend(cmd) == "openai"


def _contains_openai_auth_error(text: str) -> bool:
    # 作用：内部符号：_contains_openai_auth_error
    # 能否简略：是
    # 原因：规模≈11 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:201；类型=function；引用≈2；规模≈11行
    low = str(text or "").lower()
    needles = (
        "invalid_api_key",
        "incorrect api key provided",
        "authenticationerror",
        "error code: 401",
        "status': 401",
        "status: 401",
    )
    return any(n in low for n in needles)


_SCORE_TOKEN_RE = re.compile(
    r"(?i)\b(?P<key>pass@1\+?|accuracy|score)\b[^0-9%]{0,12}(?P<val>\d+(?:\.\d+)?)(?P<pct>\s*%)?"
)


def _normalize_score(value: float, *, had_percent: bool) -> float | None:
    # 作用：内部符号：_normalize_score
    # 能否简略：是
    # 原因：规模≈10 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:219；类型=function；引用≈3；规模≈10行
    v = float(value)
    if had_percent:
        v = v / 100.0
    # Heuristic: if a tool prints accuracy like "75" (meaning 75%), normalize to [0,1].
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return None
    return float(v)


def _extract_score_from_text(text: str) -> tuple[float | None, str]:
    """Best-effort score extraction from stdout/stderr (generic, benchmark-agnostic)."""
    # 作用：Best-effort score extraction from stdout/stderr (generic, benchmark-agnostic).
    # 能否简略：部分
    # 原因：规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:232；类型=function；引用≈2；规模≈22行
    t = str(text or "")
    # Strip ANSI sequences so regexes match more reliably.
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    last: dict[str, tuple[float, bool]] = {}
    for m in _SCORE_TOKEN_RE.finditer(t):
        key = str(m.group("key") or "").strip().lower()
        raw = str(m.group("val") or "").strip()
        had_pct = bool((m.group("pct") or "").strip())
        try:
            val = float(raw)
        except Exception:
            continue
        last[key] = (val, had_pct)
    for key in ("pass@1", "pass@1+", "accuracy", "score"):
        if key in last:
            val, had_pct = last[key]
            norm = _normalize_score(val, had_percent=had_pct)
            if norm is not None:
                return norm, f"text:{key}"
    return None, "no_score_in_text"


def _extract_score_from_json_obj(obj: object) -> tuple[float | None, str]:
    """Best-effort score extraction from JSON-like objects."""
    # 作用：Best-effort score extraction from JSON-like objects.
    # 能否简略：部分
    # 原因：规模≈25 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:256；类型=function；引用≈2；规模≈25行

    def rec(x: object) -> list[tuple[str, float]]:
        # 作用：内部符号：_extract_score_from_json_obj.rec
        # 能否简略：否
        # 原因：规模≈12 行；引用次数≈4（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/hints_exec.py:258；类型=function；引用≈4；规模≈12行
        out: list[tuple[str, float]] = []
        if isinstance(x, dict):
            for k, v in x.items():
                kk = str(k or "").strip().lower()
                if isinstance(v, (int, float)):
                    out.append((kk, float(v)))
                out.extend(rec(v))
        elif isinstance(x, list):
            for it in x:
                out.extend(rec(it))
        return out

    pairs = rec(obj)
    # Prefer pass@1, then accuracy, then score.
    for needle in ("pass@1", "pass_at_1", "pass@1+", "pass_at_1_plus", "accuracy", "score"):
        for k, v in reversed(pairs):
            if needle in k:
                norm = _normalize_score(v, had_percent=False)
                if norm is not None:
                    return norm, f"json:{needle}"
    return None, "no_score_in_json"


def _extract_score_from_json_file(path: Path) -> tuple[float | None, str]:
    # 作用：内部符号：_extract_score_from_json_file
    # 能否简略：是
    # 原因：规模≈6 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:282；类型=function；引用≈2；规模≈6行
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return None, f"metrics_json_parse_failed:{e}"
    return _extract_score_from_json_obj(data)


def _candidate_metrics_paths(cmd: str, *, repo: Path, workdir: Path | None = None) -> list[Path]:
    """Infer likely output paths for evaluation metrics from a hint command.

    NOTE: some hint commands are executed from an artifacts workdir (not the repo root)
    to avoid polluting the repo and to sidestep permission issues caused by docker-created
    root-owned output directories. For relative paths, prefer resolving against that
    execution workdir when provided.
    """
    # 作用：Infer likely output paths for evaluation metrics from a hint command.
    # 能否简略：部分
    # 原因：规模≈37 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:297；类型=function；引用≈2；规模≈37行
    repo = Path(repo).resolve()
    base = Path(workdir).resolve() if workdir is not None else repo
    out: list[Path] = []
    out_dir = _extract_cli_flag_value_any(
        cmd,
        [
            "--output-dir",
            "--output_dir",
            "--out-dir",
            "--out_dir",
            "--outdir",
            "--results-dir",
            "--results_dir",
        ],
    )
    if out_dir:
        p = Path(out_dir.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        for name in ("metrics.json", "results.json", "summary.json"):
            out.append((p / name).resolve())
    # Also allow commands that directly write a metrics file.
    out_path = _extract_cli_flag_value_any(cmd, ["--metrics", "--metrics-path", "--metrics_path"])
    if out_path:
        p = Path(out_path.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p.resolve())
    return out


def _hint_runtime_compatible(*, cmd: str, env: dict[str, str], strict_compat: bool) -> tuple[bool, str]:
    # 作用：内部符号：_hint_runtime_compatible
    # 能否简略：是
    # 原因：规模≈19 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:329；类型=function；引用≈2；规模≈19行
    if not strict_compat:
        return True, "ok"
    low = _first_command_line(cmd).lower()
    if not low:
        return False, "empty"

    backend = _hint_backend(cmd)
    llm_kind = str(env.get("AIDER_LLM_KIND") or "").strip().lower()
    if llm_kind == "remote" and backend and backend != "openai":
        if "--samples" not in low:
            return False, f"backend_mismatch:{backend}"

    runtime_base = _canonical_base_url(env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE"))
    hinted_base = _canonical_base_url(_extract_cli_flag_value_any(cmd, ["--base-url", "--base_url"]))
    if runtime_base and hinted_base and runtime_base != hinted_base:
        return False, "base_url_mismatch"

    return True, "ok"


def normalize_hint_command(cmd: str, *, env: dict[str, str]) -> tuple[str, str | None]:
    """Normalize a doc-derived command hint into something runnable.

    Returns (sanitized_cmd, skip_reason). If skip_reason is not None, callers should skip it.
    """
    # 作用：Normalize a doc-derived command hint into something runnable.
    # 能否简略：部分
    # 原因：规模≈184 行；引用次数≈6（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:354；类型=function；引用≈6；规模≈184行
    s = str(cmd or "").strip()
    if not s:
        return "", "empty"

    # Strip common prompt prefixes that appear in docs (best-effort).
    # Only strip `$` when followed by whitespace to avoid breaking `$HOME/foo` style paths.
    cleaned: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("> "):
            line = line[2:].lstrip()
        if line.startswith("$") and len(line) >= 2 and line[1].isspace():
            line = line[2:].lstrip()
        if line.startswith(">>> "):
            line = line[4:].lstrip()
        if line.startswith("... "):
            line = line[4:].lstrip()
        cleaned.append(line)
    s = "\n".join(cleaned).strip()
    if not s:
        return "", "empty_after_sanitize"

    # Replace bracketed option groups like [a|b|c] -> a (first option).
    def bracket_repl(m: re.Match[str]) -> str:
        # 作用：内部符号：normalize_hint_command.bracket_repl
        # 能否简略：是
        # 原因：规模≈6 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:360；类型=function；引用≈2；规模≈6行
        inner = m.group(1)
        if "|" in inner:
            return inner.split("|", 1)[0].strip()
        # Keep unresolved placeholders as-is; we'll skip later if still bracketed.
        return m.group(0)

    s2 = _BRACKET_GROUP_RE.sub(bracket_repl, s)
    # Remove angle placeholders like <TENSOR_PARALLEL_SIZE>
    s2 = _ANGLE_GROUP_RE.sub("", s2)

    # Replace common GitHub Actions expressions (e.g., matrix python versions) with local defaults.
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    def gha_repl(m: re.Match[str]) -> str:
        # 作用：内部符号：normalize_hint_command.gha_repl
        # 能否简略：是
        # 原因：规模≈5 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:374；类型=function；引用≈2；规模≈5行
        inner = str(m.group(1) or "").strip().lower()
        if "matrix.python-version" in inner or "matrix.python_version" in inner or "python-version" in inner:
            return py_ver
        return ""

    s2 = _GHA_EXPR_RE.sub(gha_repl, s2)

    # Replace model/base-url flags with env-provided values when available.
    model = (env.get("AIDER_LLM_MODEL") or env.get("OPENAI_MODEL") or "").strip()
    base_url = (env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL") or "").strip()
    if model:
        s2 = _replace_flag_value(s2, flag="--model", new_value=model)
    if base_url:
        s2 = _replace_flag_value(s2, flag="--base-url", new_value=base_url)
        s2 = _replace_flag_value(s2, flag="--base_url", new_value=base_url)

    # Normalize whitespace but preserve newlines for multi-command scripts.
    s2 = re.sub(r"[ \t]+", " ", s2)
    lines: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    s2 = "\n".join(lines).strip()
    if not s2:
        return "", "empty_after_sanitize"

    # Best-effort: rewrite console-script style invocations like `pkg.module ...`
    # into `python -m pkg.module ...` when the entrypoint isn't available.
    #
    # This is generic and avoids benchmark-specific hardcoding (common in README/CI).
    py = (env.get("AIDER_FSM_PYTHON") or env.get("PYTHON") or "python3").strip() or "python3"
    # When the pipeline provides a repo-relative python path (e.g. `.aider_fsm/venv/bin/python`),
    # make it absolute so hints can run from any working directory.
    repo_root = str(env.get("AIDER_FSM_REPO_ROOT") or "").strip()
    if repo_root and ("/" in py or py.startswith((".", "~"))):
        try:
            p = Path(py).expanduser()
            if not p.is_absolute():
                cand = (Path(repo_root).expanduser().resolve() / p).resolve()
                if cand.exists():
                    py = str(cand)
        except Exception:
            pass

    def _rewrite_line(line: str) -> str:
        # 作用：内部符号：normalize_hint_command._rewrite_line
        # 能否简略：是
        # 原因：规模≈15 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:421；类型=function；引用≈2；规模≈15行
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            return line
        if not parts:
            return line
        first = str(parts[0] or "").strip()
        if not first or "/" in first or first.startswith((".", "~")):
            return line
        if _DOTTED_MODULE_RE.fullmatch(first) and shutil.which(first) is None:
            rest = " ".join(shlex.quote(str(p)) for p in parts[1:])
            base = f"{shlex.quote(py)} -m {shlex.quote(first)}"
            return f"{base} {rest}".strip() if rest else base
        return line

    s2 = "\n".join([_rewrite_line(line) for line in s2.splitlines() if line.strip()]).strip()

    py_is_path = ("/" in py) or py.startswith((".", "~"))

    def _maybe_rewrite_python_tools(line: str) -> str:
        """Best-effort: route python/pip/pytest invocations to the repo-provided interpreter.

        This avoids accidentally picking up a global interpreter (e.g. Py3.13) when the
        repo is bootstrapped with a specific venv under `.aider_fsm/`.
        """
        # 作用：内部符号：normalize_hint_command._maybe_rewrite_python_tools
        # 能否简略：部分
        # 原因：提升 hints 对“任意 python 脚本入口”的稳定性（不仅是 pytest）；但需要谨慎 shlex 解析与重建；规模≈57 行
        # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈57行
        if not py_is_path:
            return line
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            return line
        if not parts:
            return line

        def _is_py(tok: str) -> bool:
            # 作用：内部符号：normalize_hint_command._maybe_rewrite_python_tools._is_py
            # 能否简略：是
            # 原因：小型谓词；集中处理 python/python3/python3.11 等模式；规模≈7 行
            # 证据：位置=runner/hints_exec.py；类型=function；引用≈2；规模≈7行
            t = str(tok or "").strip()
            if t in ("python", "python3"):
                return True
            return bool(re.fullmatch(r"python\\d+(?:\\.\\d+)?", t))

        def _is_pip(tok: str) -> bool:
            # 作用：内部符号：normalize_hint_command._maybe_rewrite_python_tools._is_pip
            # 能否简略：是
            # 原因：小型谓词；集中处理 pip/pip3/pip3.11 等模式；规模≈7 行
            # 证据：位置=runner/hints_exec.py；类型=function；引用≈2；规模≈7行
            t = str(tok or "").strip()
            if t in ("pip", "pip3"):
                return True
            return bool(re.fullmatch(r"pip\\d+(?:\\.\\d+)?", t))

        prefix: list[str] = []
        i = 0
        while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
            prefix.append(str(parts[i] or ""))
            i += 1
        if i < len(parts) and str(parts[i] or "") == "env":
            prefix.append("env")
            i += 1
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                prefix.append(str(parts[i] or ""))
                i += 1
        if i >= len(parts):
            return line

        cmd0 = str(parts[i] or "")
        rest = [str(x or "") for x in parts[i + 1 :]]

        if cmd0 != py and _is_py(cmd0):
            cmd0 = py
        elif _is_pip(cmd0):
            cmd0 = py
            rest = ["-m", "pip"] + rest
        elif cmd0 == "pytest":
            cmd0 = py
            rest = ["-m", "pytest"] + rest

        return " ".join(shlex.quote(x) for x in (prefix + [cmd0] + rest)).strip()

    s2 = "\n".join([_maybe_rewrite_python_tools(line) for line in s2.splitlines() if line.strip()]).strip()

    def _looks_like_fire_cli(line: str) -> bool:
        """Heuristic: console-script entrypoints like `pkg.subcmd` are often python-fire CLIs."""
        # 作用：Heuristic: console-script entrypoints like `pkg.subcmd` are often python-fire CLIs.
        # 能否简略：部分
        # 原因：规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/hints_exec.py:440；类型=function；引用≈2；规模≈26行
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            return False
        if not parts:
            return False
        # Skip leading env assignments, e.g. `FOO=1 cmd ...`.
        i = 0
        while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
            i += 1
        if i >= len(parts):
            return False
        first = str(parts[i] or "").strip()
        if not first:
            return False
        if _DOTTED_MODULE_RE.fullmatch(first):
            return True
        # Also accept `python -m pkg.mod ...`.
        if first in ("python", "python3", shlex.split(shlex.quote(py))[0]):
            if i + 2 < len(parts) and str(parts[i + 1]) == "-m":
                mod = str(parts[i + 2] or "").strip()
                if _DOTTED_MODULE_RE.fullmatch(mod):
                    return True
        return False

    def _maybe_normalize_fire_flag_aliases(line: str) -> str:
        # 作用：内部符号：normalize_hint_command._maybe_normalize_fire_flag_aliases
        # 能否简略：是
        # 原因：规模≈18 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:466；类型=function；引用≈2；规模≈18行
        if not _looks_like_fire_cli(line):
            return line
        aliases = {
            "--base-url": "--base_url",
            "--n-samples": "--n_samples",
            "--id-range": "--id_range",
            "--i-just-wanna-run": "--i_just_wanna_run",
            "--test-details": "--test_details",
            "--base-only": "--base_only",
            "--output-file": "--output_file",
            "--min-time-limit": "--min_time_limit",
            "--gt-time-limit-factor": "--gt_time_limit_factor",
        }
        out = line
        for old, new in aliases.items():
            out = out.replace(old, new)
        return out

    def _maybe_bound_openai_codegen_eval(line: str) -> str:
        # Best-effort: bound expensive "codegen + evaluate" hint commands using AIDER_EVAL_LIMIT.
        #
        # Generic heuristic: commands that include `--backend openai` + `--model` + `--dataset`
        # and do NOT include `--samples` are likely to trigger large numbers of API calls.
        # For python-fire style CLIs, `--n_samples 1` is commonly supported.
        #
        # NOTE: we intentionally do NOT inject `--id_range` here. Some evaluators use
        # `--id_range` only for code generation while still expecting a full dataset
        # during evaluation, which can cause hard failures (e.g., "Missing problems in samples").
        # 作用：内部符号：normalize_hint_command._maybe_bound_openai_codegen_eval
        # 能否简略：部分
        # 原因：规模≈30 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/hints_exec.py:494；类型=function；引用≈2；规模≈30行
        low = line.lower()
        if "--samples" in low or " -s " in f" {low} ":
            return line
        if ("--backend openai" not in low) and ("--backend=openai" not in low):
            return line
        if "--model" not in low:
            return line
        if "--dataset" not in low:
            return line
        if (".evaluate" not in low) and (".codegen" not in low):
            return line

        parts = line.split()
        has_n_samples = any(p.startswith("--n_samples") for p in parts)

        suffix: list[str] = []
        if not has_n_samples:
            suffix.extend(["--n_samples", "1"])

        return (line + (" " + " ".join(suffix) if suffix else "")).strip()

    s2 = "\n".join([_maybe_bound_openai_codegen_eval(_maybe_normalize_fire_flag_aliases(line)) for line in s2.splitlines()])

    # Some repos recommend `pytest -n ...` (xdist), but the plugin is often missing in
    # minimal environments. Strip xdist-only flags by default to improve compatibility.
    strip_pytest_n = _is_truthy(env.get("AIDER_FSM_HINT_STRIP_PYTEST_N", "1"))
    if strip_pytest_n:
        def _strip_pytest_xdist_flags(line: str) -> str:
            # 作用：内部符号：normalize_hint_command._strip_pytest_xdist_flags
            # 能否简略：是
            # 原因：小型兼容性改写；避免 xdist 缺失导致 `pytest -n` 直接报错（runner 更通用）
            # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈40行
            try:
                parts = shlex.split(line, posix=True)
            except Exception:
                return line
            if not parts:
                return line
            if "pytest" not in parts:
                return line
            out: list[str] = []
            i = 0
            while i < len(parts):
                tok = str(parts[i] or "")
                if tok == "-n":
                    # Drop `-n <N|auto>`; keep the next token if it looks like another flag.
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("-n="):
                    i += 1
                    continue
                if tok == "--dist":
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("--dist="):
                    i += 1
                    continue
                out.append(tok)
                i += 1
            return " ".join(shlex.quote(x) for x in out)

        s2 = "\n".join([_strip_pytest_xdist_flags(line) for line in s2.splitlines()])

    # If the command still contains bracket placeholders, it's likely not directly runnable.
    tmp = re.sub(r"\[\s*\d+\s*,\s*\d+\s*\]", "", s2)
    if "[" in tmp and "]" in tmp:
        return s2, "unresolved_brackets"
    if "<" in s2 and ">" in s2:
        return s2, "unresolved_angle_placeholders"

    # Block common unsafe installer patterns from docs.
    if _PIPE_TO_BASH_RE.search(s2):
        return s2, "blocked_pipe_to_bash"

    # Apply the runner's safe-mode denylist as a generic guardrail.
    allowed, reason = cmd_allowed(s2, pipeline=None)
    if not allowed:
        return s2, reason or "blocked_by_policy"
    return s2, None


@dataclass(frozen=True)
class HintAttempt:
    # 作用：内部符号：HintAttempt
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:537；类型=class；引用≈9；规模≈9行
    raw: str
    sanitized: str
    rc: int
    seconds: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class HintProbe:
    # 作用：内部符号：HintProbe
    # 能否简略：否
    # 原因：规模≈6 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:549；类型=class；引用≈7；规模≈6行
    raw: str
    sanitized: str
    ok: bool | None
    reason: str
    priority: int


def _tail(text: str, n: int) -> str:
    # 作用：内部符号：_tail
    # 能否简略：否
    # 原因：规模≈5 行；引用次数≈11（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:557；类型=function；引用≈11；规模≈5行
    t = str(text or "")
    if len(t) <= n:
        return t
    return t[-n:]


def _docker_available(*, env: dict[str, str]) -> tuple[bool, str]:
    """Best-effort check for a usable local Docker daemon.

    This is intentionally generic and only used to avoid spending hint attempts on
    guaranteed-failing docker commands (e.g. Docker Desktop / Colima not running).
    """
    # 作用：Best-effort check for a usable local Docker daemon.
    # 能否简略：部分
    # 原因：规模≈25 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:574；类型=function；引用≈2；规模≈25行
    if shutil.which("docker") is None:
        return False, "docker_not_found"
    try:
        res = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=env,
        )
    except Exception as e:
        return False, f"docker_info_failed: {e}"
    if int(res.returncode) != 0:
        tail = (res.stderr or res.stdout or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        return False, tail or f"docker_info_rc={res.returncode}"
    return True, "ok"


def _extract_invoked_command(parts: list[str]) -> tuple[str, list[str]]:
    # 作用：内部符号：_extract_invoked_command
    # 能否简略：是
    # 原因：规模≈18 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:596；类型=function；引用≈2；规模≈18行
    i = 0
    n = len(parts)
    while i < n:
        tok = str(parts[i] or "").strip()
        if not tok:
            i += 1
            continue
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        if tok == "env":
            i += 1
            while i < n and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            continue
        return tok, parts[i:]
    return "", []


def _probe_hint_command(
    *,
    cmd: str,
    repo: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool | None, str]:
    """Best-effort non-mutating probe for hint runnability."""
    # 作用：Best-effort non-mutating probe for hint runnability.
    # 能否简略：否
    # 原因：规模≈110 行；引用次数≈3（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:623；类型=function；引用≈3；规模≈110行
    text = str(cmd or "").strip()
    if not text:
        return False, "empty"

    first_line = ""
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            first_line = s
            break
    if not first_line:
        return False, "empty"

    try:
        parts = shlex.split(first_line, posix=True)
    except Exception:
        return None, "probe_shlex_failed"
    if not parts:
        return False, "empty"

    invoked, tail_parts = _extract_invoked_command(parts)
    if not invoked:
        return None, "probe_no_invoked_command"

    tok = str(invoked).strip()
    if not tok:
        return None, "probe_no_invoked_command"

    # Shell wrappers and scripts are hard to probe cheaply without side effects.
    if tok in ("bash", "sh", "zsh", "fish"):
        return None, "probe_shell_wrapper"
    if tok in _SHELL_BUILTINS:
        return None, "probe_shell_builtin"
    if "/" in tok or tok.startswith((".", "~")):
        return True, "ok"

    # Best-effort: if a hint references an explicit samples file, skip it early when missing.
    # This avoids spending attempts on obviously failing commands (common in README snippets).
    if tok != "docker":
        samples = ""
        i = 0
        while i < len(parts):
            t = str(parts[i] or "").strip()
            if t in ("--samples", "-s") and i + 1 < len(parts):
                samples = str(parts[i + 1] or "").strip()
                break
            if t.startswith("--samples="):
                samples = str(t.split("=", 1)[1] or "").strip()
                break
            i += 1
        if samples and not samples.startswith("-"):
            sp = Path(samples)
            if not sp.is_absolute():
                sp = (repo / sp).resolve()
            try:
                if not sp.exists():
                    return False, f"samples_not_found:{sp}"
            except Exception:
                pass

    # Validate python module entrypoints (`python -m xxx`) up front.
    py_names = {
        "python",
        "python3",
        Path(str(env.get("AIDER_FSM_PYTHON") or "")).name.strip(),
        Path(str(env.get("PYTHON") or "")).name.strip(),
    }
    py_names = {x for x in py_names if x}
    if tok in py_names:
        if len(tail_parts) >= 3 and str(tail_parts[1]) == "-m":
            module = str(tail_parts[2] or "").strip()
            if module:
                probe_py = (
                    str(env.get("AIDER_FSM_PYTHON") or "").strip()
                    or str(env.get("PYTHON") or "").strip()
                    or tok
                    or "python3"
                )
                code = (
                    "import importlib.util, sys; "
                    "m = (sys.argv[1] if len(sys.argv) > 1 else '').strip(); "
                    "sys.exit(0 if (m and importlib.util.find_spec(m) is not None) else 3)"
                )
                try:
                    res = subprocess.run(
                        [probe_py, "-c", code, module],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=min(max(2, int(timeout_seconds)), 8),
                        cwd=str(repo),
                        env=env,
                    )
                except Exception as e:
                    return None, f"probe_module_check_failed:{e}"
                if int(res.returncode) != 0:
                    return False, f"module_not_found:{module}"
                return True, "ok"

    if shutil.which(tok) is None:
        return False, f"binary_not_found:{tok}"
    return True, "ok"


def _matched_anchors(text: str, *, anchors: list[str]) -> list[str]:
    # 作用：内部符号：_matched_anchors
    # 能否简略：部分
    # 原因：规模≈16 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:728；类型=function；引用≈4；规模≈16行
    if not anchors:
        return []
    low = str(text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for raw in anchors:
        a = str(raw or "").strip()
        if not a:
            continue
        if a in seen:
            continue
        if a.lower() in low:
            seen.add(a)
            out.append(a)
    return out


def run_hints(
    *,
    repo: Path,
    max_attempts: int = 3,
    timeout_seconds: int = 600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    # 作用：内部符号：run_hints
    # 能否简略：否
    # 原因：规模≈617 行；引用次数≈16（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:752；类型=function；引用≈16；规模≈617行
    repo = Path(repo).resolve()
    env2 = dict(env or os.environ)

    raw_hints = _parse_json_str_list(env2.get("AIDER_FSM_HINTS_JSON"))
    if not raw_hints:
        hints_file = _find_latest_scaffold_hints_file(repo)
        if hints_file is not None:
            raw_hints = _read_hints_file(hints_file)

    anchors = _parse_json_str_list(env2.get("AIDER_FSM_HINT_ANCHORS_JSON"))
    used_anchors: list[str] = []

    kind = (env2.get("AIDER_LLM_KIND") or "").strip().lower()
    prefer_offline = _is_truthy(env2.get("AIDER_FSM_PREFER_OFFLINE_HINTS"))
    login_shell = _is_truthy(env2.get("AIDER_FSM_HINT_LOGIN_SHELL"))
    strict_compat = _is_truthy(env2.get("AIDER_FSM_HINT_STRICT_COMPAT", "1"))
    require_real_score = _is_truthy(env2.get("AIDER_FSM_REQUIRE_REAL_SCORE"))
    auto_uv_venv = _is_truthy(env2.get("AIDER_FSM_HINT_AUTO_UV", env2.get("AIDER_FSM_HINT_AUTO_UV_PY311", "1")))
    artifacts_dir_s = str(env2.get("AIDER_FSM_ARTIFACTS_DIR") or "").strip()
    artifacts_dir: Path | None = None
    if artifacts_dir_s:
        try:
            p = Path(artifacts_dir_s).expanduser()
            artifacts_dir = p.resolve() if p.is_absolute() else (repo / p).resolve()
        except Exception:
            artifacts_dir = None

    uv_py_candidates = _infer_uv_python_candidates(repo, env=env2)
    uv_py_env_default = str(env2.get("AIDER_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
    if not uv_py_env_default and sys.version_info >= (3, 13) and uv_py_candidates:
        # When the runner itself is executed under a very new Python, steer uv toward a
        # stable minor with broad wheel availability, unless the caller already pinned it.
        uv_py_env_default = uv_py_candidates[0]

    uv_hint_env: dict[str, str] | None = None
    uv_hint_env_py: str = ""

    def _looks_like_py_incompat_build_failure(text: str) -> bool:
        # 作用：内部符号：run_hints._looks_like_py_incompat_build_failure
        # 能否简略：部分
        # 原因：启发式判断编译/轮子构建失败（常见于 Py3.13 C 扩展）；用于触发一次性回退重试
        # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈15行
        low = str(text or "").lower()
        if not low:
            return False
        if "greenlet" in low and ("cframe" in low or "_pycframe" in low or "failed to build" in low):
            return True
        if "failed building wheel for" in low or "could not build wheels for" in low:
            return True
        if "subprocess-exited-with-error" in low and ("error:" in low or "failed" in low):
            return True
        if "failed to build installable wheels" in low and ("pyproject.toml" in low or "greenlet" in low):
            return True
        return False

    def _ensure_uv_hint_env() -> tuple[dict[str, str] | None, str, str]:
        # 作用：内部符号：run_hints._ensure_uv_hint_env
        # 能否简略：否
        # 原因：把“失败后切 uv venv（按候选 Python 版本）重试”的策略封装到一个点；避免写死 py311
        # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈95行
        nonlocal uv_hint_env, uv_hint_env_py
        if uv_hint_env is not None:
            return uv_hint_env, "cached", uv_hint_env_py
        if not auto_uv_venv:
            return None, "disabled", ""
        if shutil.which("uv") is None:
            return None, "uv_not_found", ""
        if not uv_py_candidates:
            return None, "no_uv_python_candidates", ""

        raw_venv_dir = str(env2.get("AIDER_FSM_HINT_UV_VENV_DIR") or "").strip()
        uv_try = [uv_py_candidates[0]] if raw_venv_dir else list(uv_py_candidates)

        last_err = ""
        for py_req in uv_try:
            try:
                m = re.match(r"^\\s*(\\d+)\\.(\\d+)\\s*$", py_req)
                tag = f"py{m.group(1)}{m.group(2)}" if m else "py"
            except Exception:
                tag = "py"

            if raw_venv_dir:
                venv_dir = Path(raw_venv_dir).expanduser()
                if not venv_dir.is_absolute():
                    venv_dir = (repo / venv_dir).resolve()
            else:
                venv_dir = (repo / ".aider_fsm" / f"venv_hints_{tag}").resolve()
            py_bin = (venv_dir / "bin" / "python").absolute()
            try:
                venv_dir.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                res = subprocess.run(
                    ["uv", "venv", "--allow-existing", "--seed", "pip", "--python", py_req, str(venv_dir)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(repo),
                    env=env2,
                )
            except Exception as e:
                last_err = f"uv_venv_failed:{e}"
                continue
            try:
                rc_i = int(getattr(res, "returncode", 1))
            except Exception:
                rc_i = 1
            if rc_i != 0:
                tail = _tail(
                    str(getattr(res, "stderr", "") or "") + "\n" + str(getattr(res, "stdout", "") or ""),
                    2500,
                )
                last_err = f"uv_venv_failed_rc={getattr(res, 'returncode', None)}:{tail}"
                continue

            envx = dict(env2)
            old_path = str(envx.get("PATH") or "")
            envx["PATH"] = str((venv_dir / "bin").absolute()) + (os.pathsep + old_path if old_path else "")
            envx["VIRTUAL_ENV"] = str(venv_dir.absolute())
            envx["AIDER_FSM_PYTHON"] = str(py_bin)
            envx["PYTHON"] = str(py_bin)
            envx.setdefault("UV_PYTHON", str(py_req).strip())
            uv_hint_env = envx
            uv_hint_env_py = str(py_req).strip()
            return uv_hint_env, "ok", uv_hint_env_py

        return None, last_err or "uv_venv_failed", ""

    def _priority(raw: str) -> int:
        # 作用：内部符号：run_hints._priority
        # 能否简略：部分
        # 原因：规模≈34 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/hints_exec.py:779；类型=function；引用≈3；规模≈34行
        s = str(raw or "").lower()
        p = 0
        # Prefer commands that actually *run* evaluations/tests over setup/build steps.
        if "pytest" in s:
            p += 50
        if "evaluate" in s or "evaluation" in s:
            p += 20
        if "benchmark" in s:
            p += 10
        has_eval = ("pytest" in s) or ("evaluate" in s) or ("evaluation" in s) or ("benchmark" in s)
        if "docker build" in s and not has_eval:
            p -= 5
        if ("pip install" in s or "poetry install" in s or "conda install" in s) and not has_eval:
            p -= 10
        if prefer_offline:
            if " --samples" in f" {s} ":
                p += 15
            if " --backend openai" in s or " backend openai" in s:
                p -= 30
        else:
            if " --backend openai" in s or " backend openai" in s:
                p += 20
            if "openai" in s and kind == "remote":
                p += 5
        if s.startswith("docker "):
            p += 3
        if "vllm" in s and kind == "remote":
            p -= 5
        if "anthropic" in s or "bedrock" in s or "google" in s:
            p -= 5
        if "ollama" in s:
            p -= 3
        return p

    raw_hints.sort(key=_priority, reverse=True)

    attempts: list[HintAttempt] = []
    probes: list[HintProbe] = []
    chosen: str | None = None
    ok = False
    score = 0.0
    reason = ""
    rc0_no_score = False
    docker_status: tuple[bool, str] | None = None
    hint_work_root: Path | None = None

    def _looks_like_openai_codegen_eval(cmd: str) -> bool:
        # Heuristic: `.evaluate/.codegen` CLIs with `--backend openai` + `--model` + `--dataset`
        # and without `--samples` are likely to generate outputs under a default relative root.
        # 作用：内部符号：run_hints._looks_like_openai_codegen_eval
        # 能否简略：是
        # 原因：规模≈15 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:828；类型=function；引用≈3；规模≈15行
        low = _first_command_line(cmd).lower()
        if not low:
            return False
        if "--samples" in low or " -s " in f" {low} ":
            return False
        if ("--backend openai" not in low) and ("--backend=openai" not in low):
            return False
        if "--model" not in low or "--dataset" not in low:
            return False
        if (".evaluate" not in low) and (".codegen" not in low):
            return False
        return True

    def _hint_workdir(cmd: str, *, attempt_no: int) -> Path:
        # 作用：内部符号：run_hints._hint_workdir
        # 能否简略：是
        # 原因：规模≈15 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:842；类型=function；引用≈2；规模≈15行
        nonlocal hint_work_root
        # Default: run hints from repo root.
        if artifacts_dir is None:
            return repo
        # Use isolated workdirs for docker hints so any container-created output dirs
        # (often owned by root) do not contaminate the repo or break subsequent hints.
        if _DOCKER_LINE_RE.search(cmd) or _looks_like_openai_codegen_eval(cmd):
            if hint_work_root is None:
                hint_work_root = (artifacts_dir / "hints_workdir").resolve()
                hint_work_root.mkdir(parents=True, exist_ok=True)
            wd = (hint_work_root / f"attempt_{attempt_no:02d}").resolve()
            wd.mkdir(parents=True, exist_ok=True)
            return wd
        return repo

    def _maybe_prepare_dataset_override(cmd: str, *, workdir: Path, env: dict[str, str]) -> dict[str, str]:
        """Best-effort: create a small dataset override file for smoke/full-lite runs.

        Some evaluators read `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` to override
        their dataset JSONL source. When `AIDER_EVAL_LIMIT` is set, we can generate an
        override file with the first N tasks to bound runtime without depending on
        evaluator-specific CLI flags.
        """
        # 作用：Best-effort: create a small dataset override file for smoke/full-lite runs.
        # 能否简略：部分
        # 原因：规模≈123 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/hints_exec.py:865；类型=function；引用≈2；规模≈123行
        try:
            lim = int(str(env.get("AIDER_EVAL_LIMIT") or "").strip() or 0)
        except Exception:
            lim = 0
        if lim <= 0:
            return env

        dataset = _extract_cli_flag_value(cmd, "--dataset").strip().lower()
        if dataset not in ("humaneval", "mbpp"):
            return env

        override_var = "HUMANEVAL_OVERRIDE_PATH" if dataset == "humaneval" else "MBPP_OVERRIDE_PATH"
        if str(env.get(override_var) or "").strip():
            return env

        # Identify the evaluator package from either:
        # - `python -m pkg.mod ...` (module execution)
        # - `pkg.mod ...` / `/path/to/pkg.mod ...` (console-script style)
        line = _first_command_line(cmd)
        if not line:
            return env
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            return env
        module = ""
        if "-m" in parts:
            try:
                module = str(parts[parts.index("-m") + 1] or "").strip()
            except Exception:
                module = ""
        if not module:
            # Skip leading env assignments, e.g. `FOO=1 cmd ...`.
            i = 0
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            if i < len(parts):
                first = str(parts[i] or "").strip()
                # Allow either `pkg.mod` or `/abs/path/pkg.mod`.
                cand = os.path.basename(first)
                if _DOTTED_MODULE_RE.fullmatch(cand):
                    module = cand
                elif _DOTTED_MODULE_RE.fullmatch(first):
                    module = first
        if not module or "." not in module:
            return env
        pkg = module.split(".", 1)[0].strip()
        if not pkg:
            return env

        out_path = (Path(workdir) / f"{dataset}_override_{lim}.jsonl").resolve()
        # If the file already exists, reuse it.
        try:
            if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 0:
                env2 = dict(env)
                env2[override_var] = str(out_path)
                return env2
        except Exception:
            pass

        # Prefer the runner-provided python when available.
        py_exec = str(env.get("AIDER_FSM_PYTHON") or env.get("PYTHON") or sys.executable).strip() or sys.executable
        try:
            p = Path(py_exec).expanduser()
            if not p.is_absolute() and ("/" in py_exec or py_exec.startswith((".", "~"))):
                cand = (repo / p).resolve()
                if cand.exists():
                    py_exec = str(cand)
        except Exception:
            pass

        code = r"""
import importlib
import json
import sys
from pathlib import Path

pkg = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
dataset = (sys.argv[2] if len(sys.argv) > 2 else "").strip().lower()
out = Path(sys.argv[3] if len(sys.argv) > 3 else "").expanduser().resolve()
limit = int(sys.argv[4] if len(sys.argv) > 4 else "0")
if not pkg or not out or limit <= 0:
    raise SystemExit(2)

if dataset == "humaneval":
    dm = importlib.import_module(pkg + ".data.humaneval")
    src = dm._ready_human_eval_plus_path()
elif dataset == "mbpp":
    dm = importlib.import_module(pkg + ".data.mbpp")
    src = dm._ready_mbpp_plus_path()
else:
    raise SystemExit(3)

seen = set()
out.parent.mkdir(parents=True, exist_ok=True)
with open(src, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        tid = obj.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        if tid in seen:
            continue
        seen.add(tid)
        g.write(s + "\n")
        if len(seen) >= limit:
            break
if len(seen) <= 0:
    raise SystemExit(4)
"""
        try:
            res = subprocess.run(
                [py_exec, "-c", code, pkg, dataset, str(out_path), str(lim)],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(repo),
                env=env,
            )
            if int(res.returncode) == 0 and out_path.exists():
                env2 = dict(env)
                env2[override_var] = str(out_path)
                return env2
        except Exception:
            return env
        return env

    def _parse_pytest_counts(text: str) -> tuple[int, int, int] | None:
        """Parse (passed, failed, errors) from pytest output (best-effort)."""
        # 作用：Parse (passed, failed, errors) from pytest output (best-effort).
        # 能否简略：部分
        # 原因：规模≈24 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/hints_exec.py:983；类型=function；引用≈3；规模≈24行
        t = str(text or "")
        # Strip ANSI color codes so regexes can match reliably.
        t = re.sub(r"\x1b\[[0-9;]*m", "", t)
        passed_ms = list(re.finditer(r"(?i)\b(\d+)\s+passed\b", t))
        failed_ms = list(re.finditer(r"(?i)\b(\d+)\s+failed\b", t))
        errors_ms = list(re.finditer(r"(?i)\b(\d+)\s+error(?:s)?\b", t))
        try:
            passed = int(passed_ms[-1].group(1)) if passed_ms else 0
        except Exception:
            passed = 0
        try:
            failed = int(failed_ms[-1].group(1)) if failed_ms else 0
        except Exception:
            failed = 0
        try:
            errors = int(errors_ms[-1].group(1)) if errors_ms else 0
        except Exception:
            errors = 0
        total = passed + failed + errors
        if total <= 0:
            return None
        return passed, failed, errors

    candidates: list[dict[str, Any]] = []
    probe_timeout = min(max(3, int(timeout_seconds)), 20)
    seen_sanitized: set[str] = set()

    # Phase 1: normalize + probe runnability without executing hinted workloads.
    for raw in raw_hints:
        priority = _priority(raw)
        sanitized, skip_reason = normalize_hint_command(raw, env=env2)
        if skip_reason is not None:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=skip_reason,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip_reason, priority=priority))
            continue

        key = str(sanitized or "").strip()
        if key in seen_sanitized:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason="duplicate_sanitized_hint",
                )
            )
            probes.append(
                HintProbe(
                    raw=raw,
                    sanitized=sanitized,
                    ok=False,
                    reason="duplicate_sanitized_hint",
                    priority=priority,
                )
            )
            continue
        seen_sanitized.add(key)

        compat_ok, compat_reason = _hint_runtime_compatible(cmd=sanitized, env=env2, strict_compat=strict_compat)
        if not compat_ok:
            skip = f"incompatible_hint: {compat_reason}"
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=skip,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
            continue

        if _DOCKER_LINE_RE.search(sanitized):
            if docker_status is None:
                docker_status = _docker_available(env=env2)
            if not docker_status[0]:
                skip = f"docker_unavailable: {docker_status[1]}"
                attempts.append(
                    HintAttempt(
                        raw=raw,
                        sanitized=sanitized,
                        rc=0,
                        seconds=0.0,
                        timed_out=False,
                        stdout_tail="",
                        stderr_tail="",
                        skip_reason=skip,
                    )
                )
                probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
                continue

        probe_ok, probe_reason = _probe_hint_command(
            cmd=sanitized,
            repo=repo,
            env=env2,
            timeout_seconds=probe_timeout,
        )
        probes.append(
            HintProbe(
                raw=raw,
                sanitized=sanitized,
                ok=probe_ok,
                reason=str(probe_reason or ""),
                priority=priority,
            )
        )
        candidates.append(
            {
                "raw": raw,
                "sanitized": sanitized,
                "priority": int(priority),
                "probe_ok": probe_ok,
                "probe_reason": str(probe_reason or ""),
            }
        )

    def _probe_rank(v: bool | None) -> int:
        # 作用：内部符号：run_hints._probe_rank
        # 能否简略：是
        # 原因：规模≈6 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:1120；类型=function；引用≈2；规模≈6行
        if v is True:
            return 2
        if v is None:
            return 1
        return 0

    candidates.sort(key=lambda x: (_probe_rank(x.get("probe_ok")), int(x.get("priority") or 0)), reverse=True)

    def _hint_kind(sanitized: str) -> str:
        # 作用：内部符号：run_hints._hint_kind
        # 能否简略：是
        # 原因：规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/hints_exec.py:1132；类型=function；引用≈1；规模≈11行
        low = str(sanitized or "").lower()
        if "pytest" in low:
            return "pytest"
        if "pip install" in low or "poetry install" in low or "conda install" in low:
            return "install"
        if low.lstrip().startswith("docker "):
            return "docker"
        if low.lstrip().startswith("git "):
            return "git"
        return "other"

    # Prefer a diverse first-attempt set so we don't burn the whole attempt budget
    # on a single category (e.g. many pytest hints failing for missing deps) while
    # skipping simpler setup hints that might succeed (e.g. `pip install pkg`).
    picked: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for want in ("pytest", "install", "docker"):
        for i, cand in enumerate(candidates):
            if i in picked:
                continue
            if _hint_kind(str(cand.get("sanitized") or "")) == want:
                ordered.append(cand)
                picked.add(i)
                break
    for i, cand in enumerate(candidates):
        if i in picked:
            continue
        ordered.append(cand)

    # Phase 2: execute best candidates only (bounded by max_attempts).
    executed = 0
    openai_auth_failed = False
    for cand in ordered:
        if executed >= int(max(0, max_attempts)):
            break

        probe_ok = cand.get("probe_ok")
        probe_reason = str(cand.get("probe_reason") or "")
        raw = str(cand.get("raw") or "")
        sanitized = str(cand.get("sanitized") or "")

        if openai_auth_failed and _is_remote_openai_hint(sanitized):
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason="skipped_after_openai_auth_failure",
                )
            )
            continue

        if probe_ok is False:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=f"probe_failed: {probe_reason or 'unrunnable'}",
                )
            )
            continue

        workdir = _hint_workdir(sanitized, attempt_no=int(executed) + 1)
        metrics_paths = _candidate_metrics_paths(sanitized, repo=repo, workdir=workdir if workdir != repo else None)
        if require_real_score:
            metrics_paths.append((repo / ".aider_fsm" / "metrics.json").resolve())
        pre_mtimes: dict[Path, float] = {}
        for p in metrics_paths:
            try:
                if p.exists():
                    pre_mtimes[p] = float(p.stat().st_mtime)
            except Exception:
                continue

        t0 = time.monotonic()
        timed_out = False
        def _exec(cmd: str, *, env: dict[str, str]) -> tuple[int, str, str, bool]:
            # 作用：内部符号：run_hints._exec
            # 能否简略：否
            # 原因：集中处理超时/解码/返回码，保证每次 hint 执行的行为一致且可审计
            # 证据：位置=runner/hints_exec.py；类型=function；引用≈2；规模≈25行
            nonlocal timed_out
            try:
                bash_args = ["bash", "-lc", cmd] if login_shell else ["bash", "-c", cmd]
                res = subprocess.run(
                    bash_args,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_seconds),
                    cwd=str(workdir),
                    env=env,
                )
                return int(res.returncode), (res.stdout or ""), (res.stderr or ""), False
            except subprocess.TimeoutExpired as e:
                timed_out = True
                out = getattr(e, "stdout", "") or ""
                err = getattr(e, "stderr", "") or ""
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="replace")
                return 124, str(out), str(err), True

        # IMPORTANT: prefer a non-login shell by default so bootstrap PATH overrides
        # (e.g. `.aider_fsm/venv/bin:$PATH`) are preserved. Login shells frequently
        # reset PATH via /etc/profile and can accidentally select global tools.
        env3: dict[str, str] = env2
        if uv_py_env_default:
            env3 = dict(env3)
            env3.setdefault("UV_PYTHON", uv_py_env_default)
        if workdir != repo:
            # When running from an artifacts workdir, keep the repo root importable
            # for `python -m pkg.mod` style hints that rely on local modules.
            if env3 is env2:
                env3 = dict(env3)
            repo_s = str(repo)
            pp = str(env3.get("PYTHONPATH") or "")
            parts = [p for p in pp.split(os.pathsep) if p]
            if repo_s not in parts:
                env3["PYTHONPATH"] = pp + (os.pathsep if pp else "") + repo_s
        if workdir != repo and _looks_like_openai_codegen_eval(sanitized):
            env3 = _maybe_prepare_dataset_override(sanitized, workdir=workdir, env=env3)

        rc, out, err, this_timed_out = _exec(sanitized, env=env3)

        # If we see a Python/C-extension build failure (common on Py3.13), retry once
        # inside a uv-managed venv with a more compatible Python.
        if (
            rc != 0
            and not this_timed_out
            and auto_uv_venv
            and _looks_like_py_incompat_build_failure((_tail(out, 20000) + "\n" + _tail(err, 20000)))
        ):
            tail_text = (_tail(out, 20000) + "\n" + _tail(err, 20000)).lower()
            if sys.version_info >= (3, 13) or ("cp313" in tail_text) or ("python 3.13" in tail_text) or ("py3.13" in tail_text):
                env_uv, prep_reason, prep_py = _ensure_uv_hint_env()
                if env_uv is not None:
                    rc2, out2, err2, _ = _exec(sanitized, env=env_uv)
                    # Keep the retry result, but also surface that a retry happened.
                    out = out2
                    extra = f"{prep_reason}" + (f" python={prep_py}" if prep_py else "")
                    err = f"(retry_uv_venv: {extra})\n{err2}"
                    rc = rc2

        executed += 1
        dt = time.monotonic() - t0
        attempts.append(
            HintAttempt(
                raw=raw,
                sanitized=sanitized,
                rc=rc,
                seconds=float(dt),
                timed_out=timed_out,
                stdout_tail=_tail(out, 4000),
                stderr_tail=_tail(err, 4000),
                skip_reason=None,
            )
        )

        low_cmd = sanitized.lower()

        if rc == 0 and require_real_score:
            # Prefer deterministic file outputs when available, otherwise parse stdout/stderr.
            extracted: float | None = None
            source = ""

            # Pytest: use passed/total as a concrete score (even when rc==0).
            if "pytest" in low_cmd:
                tail_text = _tail(out, 20000) + "\n" + _tail(err, 20000)
                counts = _parse_pytest_counts(tail_text)
                if counts is not None:
                    passed, failed, errors = counts
                    total = max(1, passed + failed + errors)
                    extracted = float(passed) / float(total)
                    source = f"pytest_counts: passed={passed} failed={failed} errors={errors}"

            if extracted is None:
                for p in metrics_paths:
                    try:
                        if not p.exists():
                            continue
                        mt = float(p.stat().st_mtime)
                        if p in pre_mtimes and mt <= pre_mtimes[p] + 1e-6:
                            continue
                    except Exception:
                        continue
                    val, src = _extract_score_from_json_file(p)
                    if val is not None:
                        extracted = float(val)
                        source = f"file:{p.name}:{src}"
                        break

            if extracted is None:
                val, src = _extract_score_from_text(_tail(out, 20000) + "\n" + _tail(err, 20000))
                if val is not None:
                    extracted = float(val)
                    source = src

            if extracted is not None:
                chosen = sanitized
                ok = True
                score = float(extracted)
                reason = str(source or "ok")
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

            rc0_no_score = True
            # Continue trying other hints to find one that yields a parseable score.
            continue

        if rc == 0:
            chosen = sanitized
            ok = True
            score = 1.0
            reason = ""
            used_anchors = _matched_anchors(sanitized, anchors=anchors)
            break

        # Some evaluation commands (notably `pytest`) intentionally return non-zero
        # when checks fail, but still produce useful numeric metrics. Treat these as
        # a "successful run" and derive score from the outputs.
        if "pytest" in low_cmd:
            tail_text = _tail(out, 20000) + "\n" + _tail(err, 20000)
            counts = _parse_pytest_counts(tail_text)
            if counts is not None:
                passed, failed, errors = counts
                total = max(1, passed + failed + errors)
                chosen = sanitized
                ok = True
                score = float(passed) / float(total)
                reason = f"pytest_nonzero_exit: passed={passed} failed={failed} errors={errors}"
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

        if _is_remote_openai_hint(sanitized):
            if _contains_openai_auth_error((_tail(out, 12000) + "\n" + _tail(err, 12000))):
                openai_auth_failed = True

    if not ok:
        if not raw_hints:
            reason = "no_hints"
        elif require_real_score and rc0_no_score:
            reason = "all_hints_no_real_score"
        elif openai_auth_failed:
            reason = "all_hints_auth_failed_or_unrunnable"
        elif candidates and not any((c.get("probe_ok") is not False) for c in candidates):
            reason = "all_hints_unrunnable"
        elif any(a.skip_reason == "unresolved_brackets" for a in attempts):
            reason = "all_hints_unresolved_or_failed"
        else:
            reason = "all_hints_failed"

    return {
        "ok": bool(ok),
        "score": float(score) if ok else 0.0,
        "chosen_command": chosen,
        "used_anchors": used_anchors,
        "executed_attempts": int(executed),
        "probes": [
            {
                "raw": p.raw,
                "sanitized": p.sanitized,
                "ok": p.ok,
                "reason": p.reason,
                "priority": p.priority,
            }
            for p in probes
        ],
        "attempts": [
            {
                "raw": a.raw,
                "sanitized": a.sanitized,
                "rc": a.rc,
                "seconds": a.seconds,
                "timed_out": a.timed_out,
                "stdout_tail": a.stdout_tail,
                "stderr_tail": a.stderr_tail,
                "skip_reason": a.skip_reason,
            }
            for a in attempts
        ],
        "reason": reason,
    }


def main() -> int:
    # 作用：内部符号：main
    # 能否简略：否
    # 原因：规模≈12 行；引用次数≈25（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:1365；类型=function；引用≈25；规模≈12行
    import argparse

    ap = argparse.ArgumentParser(description="Run doc-derived hint commands with best-effort sanitization.")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    args = ap.parse_args()

    repo_root = Path(os.environ.get("AIDER_FSM_REPO_ROOT") or ".").resolve()
    res = run_hints(repo=repo_root, max_attempts=int(args.max_attempts), timeout_seconds=int(args.timeout_seconds))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
