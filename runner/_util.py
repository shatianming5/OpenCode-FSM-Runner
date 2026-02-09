from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _is_truthy(value: str | None) -> bool:
    # 作用：内部符号：_is_truthy
    # 能否简略：否
    # 原因：规模≈3 行；引用次数≈13（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/_util.py:9；类型=function；引用≈13；规模≈3行
    v = str(value or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _parse_json_str_list(raw: str | None) -> list[str]:
    # 作用：内部符号：_parse_json_str_list
    # 能否简略：否
    # 原因：规模≈17 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/_util.py:14；类型=function；引用≈10；规模≈17行
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s:
            out.append(s)
    return out


def _read_json_object(path: Path) -> dict[str, Any] | None:
    # 作用：内部符号：_read_json_object
    # 能否简略：否
    # 原因：规模≈8 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/_util.py:33；类型=function；引用≈10；规模≈8行
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _find_hf_test_parquet(repo_root: Path) -> Path | None:
    """Find a Hugging Face dataset test split parquet file (best-effort)."""
    # 作用：Find a Hugging Face dataset test split parquet file (best-effort).
    # 能否简略：否
    # 原因：规模≈18 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/_util.py:44；类型=function；引用≈7；规模≈18行
    repo_root = Path(repo_root).resolve()

    # Common HF snapshot layout includes `main/test-00000-of-00001.parquet`.
    p0 = (repo_root / "main" / "test-00000-of-00001.parquet").resolve()
    if p0.exists():
        return p0

    cands: list[Path] = []
    for p in repo_root.rglob("test-*.parquet"):
        try:
            if p.is_file():
                cands.append(p.resolve())
        except Exception:
            continue
    cands.sort()
    return cands[0] if cands else None


def _ensure_openai_v1_base(base_url: str) -> str:
    # 作用：内部符号：_ensure_openai_v1_base
    # 能否简略：否
    # 原因：规模≈5 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/_util.py:79；类型=function；引用≈7；规模≈5行
    b = str(base_url or "").strip().rstrip("/")
    if not b:
        return ""
    return b if b.endswith("/v1") else b + "/v1"
