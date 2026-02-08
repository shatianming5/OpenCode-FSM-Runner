from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import resolve_workdir
from .pipeline_spec import PipelineSpec
from .security import cmd_allowed, looks_interactive, safe_env
from .subprocess_utils import read_text_if_exists, run_cmd_capture, write_cmd_artifacts, write_json, write_text
from .types import CmdResult, StageResult


@dataclass(frozen=True)
class BootstrapSpec:
    """中文说明：
    - 含义：`.aider_fsm/bootstrap.yml` 的解析后结构（v1）。
    - 内容：描述在 pipeline 验收前要执行的命令（可为空）以及要应用的 env/workdir/timeout/retries。
    - 可简略：否（bootstrap 是复现/同机调用的重要能力；结构化对象便于测试与演进）。
    """

    version: int = 1
    cmds: list[str] = None  # type: ignore[assignment]
    env: dict[str, str] = None  # type: ignore[assignment]
    workdir: str | None = None
    timeout_seconds: int | None = None
    retries: int = 0

    def __post_init__(self) -> None:
        """中文说明：
        - 含义：dataclass 初始化后把 None 字段归一化为可用默认值。
        - 内容：将 `cmds/env` 从 None 变为 `[]/{}`，避免调用点反复做空值处理。
        - 可简略：可能（也可在类型定义处避免 None；但当前写法更直观）。
        """
        if self.cmds is None:
            object.__setattr__(self, "cmds", [])
        if self.env is None:
            object.__setattr__(self, "env", {})


@dataclass(frozen=True)
class BootstrapLoadResult:
    """Structured bootstrap parse output with non-fatal normalization warnings."""

    spec: BootstrapSpec
    raw: str
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            object.__setattr__(self, "warnings", [])


_VAR_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_BARE_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_DOLLAR_PLACEHOLDER = "\x00DOLLAR\x00"


def _expand_env_value(raw: str, env: dict[str, str]) -> str:
    """中文说明：
    - 含义：对 bootstrap.yml 的 env 值做最小化的 shell 风格变量展开。
    - 内容：支持 `${VAR}` 与 `$VAR`，并把 `$$` 转义成字面 `$`；不执行 shell，不做命令替换。
    - 可简略：否（这是 bootstrap env 的关键特性；且实现刻意保持“安全且最小”）。
    """
    # Minimal shell-style expansion for env values:
    # - supports ${VAR} and $VAR
    # - "$$" escapes to literal "$"
    s = str(raw or "")
    s = s.replace("$$", _DOLLAR_PLACEHOLDER)

    def _brace(m: re.Match[str]) -> str:
        """中文说明：
        - 含义：`${VAR}` 形式的替换回调函数。
        - 内容：从 `env` 取出变量值；不存在则返回空字符串。
        - 可简略：是（可与 `_bare` 合并成一个回调；但拆开更直观）。
        """
        return str(env.get(m.group(1)) or "")

    def _bare(m: re.Match[str]) -> str:
        """中文说明：
        - 含义：`$VAR` 形式的替换回调函数。
        - 内容：从 `env` 取出变量值；不存在则返回空字符串。
        - 可简略：是（可与 `_brace` 合并；保留两个函数主要是可读性）。
        """
        return str(env.get(m.group(1)) or "")

    s = _VAR_BRACE_RE.sub(_brace, s)
    s = _VAR_BARE_RE.sub(_bare, s)
    return s.replace(_DOLLAR_PLACEHOLDER, "$")


def _apply_env_mapping(base: dict[str, str], mapping: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """中文说明：
    - 含义：把 bootstrap.yml 的 env 映射应用到基底环境，并返回 (新环境, 实际应用的键值)。
    - 内容：按顺序扩展变量（后面的值可引用前面刚写入的变量）；用于后续 stage 命令执行与结果记录。
    - 可简略：否（和 `_expand_env_value` 配合构成 bootstrap env 语义）。
    """
    env = dict(base)
    applied: dict[str, str] = {}
    for k, v in (mapping or {}).items():
        key = str(k or "").strip()
        if not key:
            continue
        value = _expand_env_value(str(v or ""), env)
        env[key] = value
        applied[key] = value
    return env, applied


def _is_sensitive_key(key: str) -> bool:
    """中文说明：
    - 含义：粗略判断一个环境变量 key 是否可能包含敏感信息（用于日志脱敏）。
    - 内容：按 key 名包含 KEY/TOKEN/SECRET/PASSWORD 等子串来判断。
    - 可简略：可能（启发式判断；可按实际需要调整或迁移到统一的 secret 管理）。
    """
    k = (key or "").upper()
    return any(x in k for x in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "PWD"))


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    """中文说明：
    - 含义：将 env mapping 中可能敏感的值替换为 `***redacted***`。
    - 内容：用于写入 bootstrap artifacts，避免把 token/key 直接落盘。
    - 可简略：否（安全与合规需要；至少要保留脱敏机制）。
    """
    out: dict[str, str] = {}
    for k, v in (env or {}).items():
        if _is_sensitive_key(k):
            out[str(k)] = "***redacted***"
        else:
            out[str(k)] = "" if v is None else str(v)
    return out


def _as_optional_int(raw: Any, *, field: str, min_value: int | None = None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(s)
    except Exception as e:
        raise ValueError(f"bootstrap.{field} must be an integer") from e
    if min_value is not None and n < int(min_value):
        raise ValueError(f"bootstrap.{field} must be >= {int(min_value)}")
    return n


def _coerce_cmds(raw: Any, *, field_name: str, warnings: list[str]) -> list[str]:
    out: list[str] = []

    if raw is None:
        return out

    if isinstance(raw, str):
        s = raw.strip()
        if s:
            warnings.append(f"bootstrap.{field_name}_coerced_from_string")
            return [s]
        return out

    if isinstance(raw, dict):
        cmd = raw.get("cmd")
        if cmd is None:
            cmd = raw.get("run")
        if cmd is None:
            cmd = raw.get("command")
        if isinstance(cmd, str) and cmd.strip():
            warnings.append(f"bootstrap.{field_name}_mapping_coerced_to_single_cmd")
            return [cmd.strip()]
        raise ValueError(
            f"bootstrap.{field_name} mapping must contain one of keys: cmd, run, command "
            "(with non-empty string value)"
        )

    if not isinstance(raw, list):
        raise ValueError(f"bootstrap.{field_name} must be a list of non-empty strings")

    for i, item in enumerate(raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
            continue
        if isinstance(item, dict):
            cmd = item.get("cmd")
            if cmd is None:
                cmd = item.get("run")
            if cmd is None:
                cmd = item.get("command")
            if isinstance(cmd, str) and cmd.strip():
                out.append(cmd.strip())
                warnings.append(f"bootstrap.{field_name}[{i}]_mapping_coerced_via_run")
                continue
        raise ValueError(
            f"bootstrap.{field_name}[{i}] must be a non-empty string "
            "or mapping with cmd/run/command"
        )
    return out


def _normalize_bootstrap_mapping(data: dict[str, Any], *, warnings: list[str]) -> dict[str, Any]:
    out = dict(data or {})
    boot = out.get("boot")
    if isinstance(boot, dict):
        top_level_keys = {"cmds", "steps", "env", "workdir", "cwd", "timeout_seconds", "timeout", "retries", "retry"}
        has_top_level_spec = any(k in out for k in top_level_keys)
        if not has_top_level_spec:
            merged = dict(boot)
            for k, v in out.items():
                if k == "boot":
                    continue
                merged[k] = v
            out = merged
            warnings.append("bootstrap.boot_mapping_unwrapped")
        else:
            warnings.append("bootstrap.boot_mapping_ignored_due_to_top_level_fields")
    return out


def load_bootstrap_spec_with_diagnostics(path: Path) -> BootstrapLoadResult:
    """Load bootstrap.yml and tolerate common scaffold formatting variants.

    This keeps parsing strict enough for safety while handling typical agent outputs
    (e.g. `boot:` wrapper, `steps` alias, mapping-style step items).
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"bootstrap file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("bootstrap.yml must be a YAML mapping (dict) at the top level")

    warnings: list[str] = []
    obj = _normalize_bootstrap_mapping(data, warnings=warnings)

    version = int(obj.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported bootstrap version: {version}")

    cmds_src = obj.get("cmds")
    if cmds_src is None and obj.get("steps") is not None:
        cmds_src = obj.get("steps")
        warnings.append("bootstrap.steps_alias_used")
    cmds = _coerce_cmds(cmds_src, field_name="cmds", warnings=warnings)

    env = obj.get("env") or {}
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError("bootstrap.env must be a mapping")
    env_out: dict[str, str] = {}
    for k, v in env.items():
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        env_out[ks] = "" if v is None else str(v)

    workdir_raw = obj.get("workdir")
    if workdir_raw is None and obj.get("cwd") is not None:
        workdir_raw = obj.get("cwd")
        warnings.append("bootstrap.cwd_alias_used")
    workdir = str(workdir_raw).strip() if workdir_raw else None

    timeout_raw = obj.get("timeout_seconds")
    if timeout_raw is None and obj.get("timeout") is not None:
        timeout_raw = obj.get("timeout")
        warnings.append("bootstrap.timeout_alias_used")
    timeout_seconds = _as_optional_int(timeout_raw, field="timeout_seconds", min_value=1)

    retries_raw = obj.get("retries")
    if retries_raw is None and obj.get("retry") is not None:
        retries_raw = obj.get("retry")
        warnings.append("bootstrap.retry_alias_used")
    retries = _as_optional_int(retries_raw, field="retries", min_value=0)
    if retries is None:
        retries = 0

    spec = BootstrapSpec(
        version=version,
        cmds=[c.strip() for c in cmds if c.strip()],
        env=env_out,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
        retries=int(retries),
    )
    return BootstrapLoadResult(spec=spec, raw=raw, warnings=warnings)


def load_bootstrap_spec(path: Path) -> tuple[BootstrapSpec, str]:
    """中文说明：
    - 含义：读取并解析 `.aider_fsm/bootstrap.yml`（v1），返回 (BootstrapSpec, raw_text)。
    - 内容：校验 version/cmds/env/workdir 等字段类型；支持 `steps` 作为 `cmds` 别名；返回 raw 用于 artifacts 固化。
    - 可简略：可能（薄封装；但集中校验便于维护与测试）。
    """
    loaded = load_bootstrap_spec_with_diagnostics(path)
    return loaded.spec, loaded.raw


def run_bootstrap(
    repo: Path,
    *,
    bootstrap_path: Path,
    pipeline: PipelineSpec | None,
    unattended: str,
    artifacts_dir: Path,
) -> tuple[StageResult, dict[str, str]]:
    """中文说明：
    - 含义：执行 bootstrap.yml，并返回 (bootstrap 阶段结果, 应用后的 env 变量映射)。
    - 内容：记录 bootstrap.yml/raw/env（脱敏）与每条命令的 artifacts；执行命令受同一套安全策略约束；返回的 applied_env 通常用于把 venv PATH 等注入后续 stages。
    - 可简略：否（这是“一键可复现运行”的关键组成）。

    ---

    English (original intent):
    Run bootstrap commands and return (stage_result, applied_env).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        loaded = load_bootstrap_spec_with_diagnostics(bootstrap_path)
        spec, raw = loaded.spec, loaded.raw
        if loaded.warnings:
            write_json(artifacts_dir / "bootstrap_parse_warnings.json", {"warnings": list(loaded.warnings)})
    except Exception as e:
        res = CmdResult(cmd=f"parse {bootstrap_path}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_parse", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), {}

    write_text(artifacts_dir / "bootstrap.yml", raw)

    # Seed commonly-used variables for env expansion.
    env_base = dict(os.environ)
    env_base["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())

    env_for_cmds, applied_env = _apply_env_mapping(env_base, spec.env)
    env_for_cmds = safe_env(env_for_cmds, {}, unattended=unattended)
    env_for_cmds["AIDER_FSM_STAGE"] = "bootstrap"
    env_for_cmds["AIDER_FSM_ARTIFACTS_DIR"] = str(artifacts_dir.resolve())
    env_for_cmds["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())
    write_json(artifacts_dir / "bootstrap_env.json", _redact_env(applied_env))

    try:
        workdir = resolve_workdir(repo, spec.workdir)
    except Exception as e:
        res = CmdResult(cmd=f"resolve_workdir {spec.workdir}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_workdir_error", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), applied_env

    results: list[CmdResult] = []
    failed_index: int | None = None

    # No cmds is valid: env-only bootstrap.
    for cmd_idx, raw_cmd in enumerate(spec.cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
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
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
        if not allowed:
            res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        eff_timeout = spec.timeout_seconds
        if pipeline and pipeline.security_max_cmd_seconds:
            eff_timeout = (
                int(pipeline.security_max_cmd_seconds)
                if eff_timeout is None
                else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
            )

        for attempt in range(1, int(spec.retries) + 2):
            res = run_cmd_capture(cmd, workdir, timeout_seconds=eff_timeout, env=env_for_cmds, interactive=False)
            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try{attempt:02d}", res)
            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            break

    ok = failed_index is None
    write_json(artifacts_dir / "bootstrap_summary.json", {"ok": ok, "failed_index": failed_index, "total_results": len(results)})

    return StageResult(ok=ok, results=results, failed_index=failed_index), applied_env
