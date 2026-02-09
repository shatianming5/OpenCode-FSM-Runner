from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    root_s = str(_ROOT)
    try:
        while root_s in sys.path:
            sys.path.remove(root_s)
    except Exception:
        pass
    sys.path.insert(0, root_s)
    from runner._util import (  # type: ignore
        _ensure_openai_v1_base,
        _find_hf_test_parquet,
        _parse_json_str_list,
        _read_json_object,
    )
else:
    from ._util import _ensure_openai_v1_base, _find_hf_test_parquet, _parse_json_str_list, _read_json_object


def _now_iso() -> str:
    # 作用：内部符号：_now_iso
    # 能否简略：是
    # 原因：规模≈2 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:16；类型=function；引用≈3；规模≈2行
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_text(path: Path, *, max_chars: int) -> str:
    # 作用：内部符号：_read_text
    # 能否简略：是
    # 原因：规模≈8 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:20；类型=function；引用≈2；规模≈8行
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _read_hf_manifest(repo_root: Path) -> dict[str, Any] | None:
    # 作用：内部符号：_read_hf_manifest
    # 能否简略：是
    # 原因：规模≈5 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:59；类型=function；引用≈2；规模≈5行
    p = (repo_root / "data" / "hf_manifest.json").resolve()
    if not p.exists():
        return None
    return _read_json_object(p)


def _resolve_openai_base(repo_root: Path) -> str:
    # 作用：内部符号：_resolve_openai_base
    # 能否简略：部分
    # 原因：规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/generic_rollout.py:83；类型=function；引用≈2；规模≈27行
    base = (os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    if base:
        return base.rstrip("/")

    runtime_path = (os.environ.get("AIDER_RUNTIME_ENV_PATH") or "").strip()
    if runtime_path:
        p = Path(runtime_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            inf = obj.get("inference")
            if isinstance(inf, dict):
                b = str(inf.get("openai_base_url") or "").strip()
                if b:
                    return b.rstrip("/")
            svc = obj.get("service")
            if isinstance(svc, dict):
                b2 = str(svc.get("base_url") or "").strip()
                if b2:
                    return b2.rstrip("/")

    return "https://api.openai.com/v1"


def _env_int(name: str) -> int | None:
    # 作用：内部符号：_env_int
    # 能否简略：部分
    # 原因：规模≈9 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/generic_rollout.py:119；类型=function；引用≈5；规模≈9行
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except Exception:
        return None
    return n if n > 0 else None


def _chat_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout_seconds: int,
    max_tokens: int | None = None,
) -> str:
    # 作用：内部符号：_chat_completion
    # 能否简略：部分
    # 原因：规模≈37 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/generic_rollout.py:138；类型=function；引用≈3；规模≈37行
    url = _ensure_openai_v1_base(base_url) + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if max_tokens is None:
        max_tokens = _env_int("AIDER_FSM_MAX_TOKENS") or 256
    max_tokens = max(1, int(max_tokens))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return content if isinstance(content, str) else ""


_RE_NUM = re.compile(r"-?\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:/\\d+(?:\\.\\d+)?)?")


_RE_FINAL_LINE = re.compile(r"(?im)^\\s*final\\s*[:：]\\s*(?P<ans>.+?)\\s*$")


def _norm_number_str(s: str) -> str:
    # 作用：内部符号：_norm_number_str
    # 能否简略：部分
    # 原因：规模≈2 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/generic_rollout.py:175；类型=function；引用≈4；规模≈2行
    return str(s or "").strip().replace(",", "")


def _to_fraction(s: str) -> Fraction | None:
    # 作用：内部符号：_to_fraction
    # 能否简略：是
    # 原因：规模≈6 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:179；类型=function；引用≈3；规模≈6行
    ss = _norm_number_str(s)
    try:
        return Fraction(ss)
    except Exception:
        return None


def _extract_final_line(text: str) -> str:
    # 作用：内部符号：_extract_final_line
    # 能否简略：是
    # 原因：规模≈14 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:187；类型=function；引用≈3；规模≈14行
    t = str(text or "").strip()
    if not t:
        return ""
    m = _RE_FINAL_LINE.search(t)
    if m:
        ans = str(m.group("ans") or "").strip()
        if ans:
            return ans
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return t
    last = lines[-1]
    return re.sub(r"(?i)^final\\s*[:：]\\s*", "", last).strip()


def _norm_answer_str(s: str) -> str:
    # 作用：内部符号：_norm_answer_str
    # 能否简略：是
    # 原因：规模≈4 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:203；类型=function；引用≈3；规模≈4行
    t = str(s or "").strip().lower()
    t = re.sub(r"\\s+", " ", t)
    return t


def _extract_last_number(text: str) -> str:
    # 作用：内部符号：_extract_last_number
    # 能否简略：是
    # 原因：规模≈5 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:209；类型=function；引用≈3；规模≈5行
    nums = _RE_NUM.findall(str(text or ""))
    if not nums:
        return ""
    return str(nums[-1] or "").strip()


def _answers_match(pred_text: str, gold_text: str) -> bool:
    # 作用：内部符号：_answers_match
    # 能否简略：是
    # 原因：规模≈14 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/generic_rollout.py:216；类型=function；引用≈2；规模≈14行
    pred = _extract_final_line(pred_text)
    gold = _extract_final_line(gold_text)

    pred_num = _extract_last_number(pred)
    gold_num = _extract_last_number(gold)
    if pred_num and gold_num:
        fp = _to_fraction(pred_num)
        fg = _to_fraction(gold_num)
        if fp is not None and fg is not None:
            return fp == fg
        return _norm_number_str(pred_num) == _norm_number_str(gold_num)

    return _norm_answer_str(pred) == _norm_answer_str(gold)


def _maybe_rollout_hf_qa_parquet(
    repo_root: Path,
    *,
    artifacts_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    mode: str,
    limit: int,
) -> tuple[bool, dict[str, Any]]:
    """Best-effort: if repo_root looks like an HF dataset snapshot, generate QA rollout samples.

    This intentionally avoids dataset-id hardcoding. It only triggers when:
    - `data/hf_manifest.json` exists (repo_resolver HF snapshot marker)
    - a test parquet exists
    - the parquet has columns: `question`, `answer`
    """
    # 作用：Best-effort: if repo_root looks like an HF dataset snapshot, generate QA rollout samples.
    # 能否简略：否
    # 原因：规模≈102 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/generic_rollout.py:248；类型=function；引用≈2；规模≈102行
    if _read_hf_manifest(repo_root) is None:
        return False, {}

    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return False, {"reason": "hf_manifest_present_but_no_test_parquet"}

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        return False, {"reason": f"pyarrow_unavailable: {e}"}

    try:
        table = pq.read_table(parquet_path, columns=["question", "answer"])
    except Exception as e:
        return False, {"reason": f"failed_to_read_parquet: {e}"}

    try:
        if int(limit) > 0 and getattr(table, "num_rows", 0) and int(table.num_rows) > int(limit):
            table = table.slice(0, int(limit))
    except Exception:
        pass

    try:
        rows = table.to_pylist()
    except Exception as e:
        return False, {"reason": f"failed_to_convert_parquet: {e}"}

    if not rows:
        return False, {"reason": "no_rows_in_test_parquet"}

    template = (
        "Answer the following question. You may include reasoning, but put the final answer on the last line as:\n"
        "FINAL: <answer>\n\n"
        "{prompt}"
    )

    samples_path = (artifacts_dir / f"rollout_samples_{int(time.time())}.jsonl").resolve()
    correct = 0
    lines = 0
    errors: list[str] = []

    with samples_path.open("w", encoding="utf-8") as out:
        for r in rows:
            if not isinstance(r, dict):
                continue
            q = str(r.get("question") or "").strip()
            a = str(r.get("answer") or "").strip()
            if not q or not a:
                continue

            prompt = template.format(prompt=q)
            try:
                completion = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout_seconds=120,
                )
            except Exception as e:
                errors.append(str(e))
                completion = ""

            is_ok = _answers_match(completion, a)

            reward = 1.0 if is_ok else 0.0
            correct += int(is_ok)
            lines += 1
            out.write(json.dumps({"prompt": prompt, "completion": completion, "reward": reward}, ensure_ascii=False) + "\n")

    if lines <= 0:
        return False, {"reason": "no_usable_rows_for_rollout"}

    rollout = {
        "ts": _now_iso(),
        "ok": True,
        "mode": mode,
        "counts": {"samples": lines, "correct": correct, "errors": len(errors)},
        "dataset": {"parquet": str(parquet_path)},
        "paths": {"samples_jsonl": str(samples_path)},
    }
    if errors:
        rollout["errors"] = errors[:3]
    return True, rollout


def _build_prompts(repo_root: Path, *, n: int) -> list[str]:
    # 作用：内部符号：_build_prompts
    # 能否简略：部分
    # 原因：规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/generic_rollout.py:336；类型=function；引用≈2；规模≈27行
    hints = _parse_json_str_list(os.environ.get("AIDER_FSM_HINTS_JSON"))
    prompts: list[str] = []
    for h in hints[: max(0, n)]:
        prompts.append(f"Explain how to run this command and what it does:\n\n{h}")
    if len(prompts) >= n:
        return prompts[:n]

    readme = ""
    for cand in ("README.md", "readme.md", "README.txt"):
        p = (repo_root / cand).resolve()
        if p.exists():
            readme = _read_text(p, max_chars=6000)
            break
    if readme:
        chunks = [c.strip() for c in readme.split("\n\n") if c.strip()]
        random.shuffle(chunks)
        for c in chunks:
            if len(prompts) >= n:
                break
            excerpt = c[:800]
            prompts.append(f"Answer based on this repo excerpt:\n\n{excerpt}")
    if prompts:
        return prompts[:n]

    repo_name = repo_root.name
    return [f"Describe the purpose of the repository `{repo_name}` and how to get started."][:n]


def main() -> int:
    # 作用：内部符号：main
    # 能否简略：否
    # 原因：规模≈71 行；引用次数≈25（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/generic_rollout.py:365；类型=function；引用≈25；规模≈71行
    repo_root = Path(os.environ.get("AIDER_FSM_REPO_ROOT") or ".").resolve()
    artifacts_dir = Path(os.environ.get("AIDER_FSM_ARTIFACTS_DIR") or (repo_root / ".aider_fsm" / "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = (repo_root / artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mode = (os.environ.get("AIDER_EVAL_MODE") or "smoke").strip().lower() or "smoke"
    try:
        limit = int(os.environ.get("AIDER_EVAL_LIMIT") or (64 if mode == "full" else 8))
    except Exception:
        limit = 8
    limit = max(1, int(limit))

    base_url = _resolve_openai_base(repo_root)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
    model = (os.environ.get("AIDER_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "").strip()
    if not model:
        raise SystemExit("missing_model: set AIDER_LLM_MODEL or OPENAI_MODEL")

    # HF dataset snapshot support: if detected, generate QA samples with rewards.
    ok_ds, ds_rollout = _maybe_rollout_hf_qa_parquet(
        repo_root,
        artifacts_dir=artifacts_dir,
        base_url=base_url,
        api_key=api_key,
        model=model,
        mode=mode,
        limit=limit,
    )
    if ok_ds:
        rollout_path = (repo_root / ".aider_fsm" / "rollout.json").resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.write_text(json.dumps(ds_rollout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    # Default: generic, bounded "repo understanding" prompts (cap for safety).
    prompt_limit = max(1, min(int(limit), 32))
    prompts = _build_prompts(repo_root, n=prompt_limit)
    samples_path = (artifacts_dir / f"rollout_samples_{int(time.time())}.jsonl").resolve()

    samples_written = 0
    errors_count = 0
    with samples_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            try:
                completion = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout_seconds=30,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                completion = ""
                errors_count += 1
            obj = {"prompt": prompt, "completion": completion, "reward": 0.0}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            samples_written += 1

    rollout_path = (repo_root / ".aider_fsm" / "rollout.json").resolve()
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout = {
        "ts": _now_iso(),
        "ok": True,
        "mode": mode,
        "counts": {"samples": samples_written, "errors": int(errors_count)},
        "paths": {"samples_jsonl": str(samples_path)},
    }
    rollout_path.write_text(json.dumps(rollout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
