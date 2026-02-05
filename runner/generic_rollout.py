from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_text(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _parse_json_list(raw: str | None) -> list[str]:
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
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _read_hf_manifest(repo_root: Path) -> dict[str, Any] | None:
    p = (repo_root / "data" / "hf_manifest.json").resolve()
    if not p.exists():
        return None
    return _read_json_object(p)


def _find_test_parquet(repo_root: Path) -> Path | None:
    """Find a Hugging Face dataset test split parquet file (best-effort)."""
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


def _resolve_openai_base(repo_root: Path) -> str:
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


def _ensure_v1(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _chat_completion(*, base_url: str, api_key: str | None, model: str, prompt: str, timeout_seconds: int) -> str:
    url = _ensure_v1(base_url) + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
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


_RE_BOXED = re.compile(r"\\\\boxed\{([^}]*)\}")
_RE_NUM = re.compile(r"-?\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:/\\d+(?:\\.\\d+)?)?")


def _norm_number_str(s: str) -> str:
    return str(s or "").strip().replace(",", "")


def _to_fraction(s: str) -> Fraction | None:
    ss = _norm_number_str(s)
    try:
        return Fraction(ss)
    except Exception:
        return None


def _extract_pred_final(text: str) -> str:
    t = str(text or "")
    m = _RE_BOXED.search(t)
    if m:
        return str(m.group(1) or "").strip()
    nums = _RE_NUM.findall(t)
    if nums:
        return str(nums[-1] or "").strip()
    return ""


def _extract_gold_final(answer_text: str) -> str:
    a = str(answer_text or "").strip()
    if "####" in a:
        return a.split("####")[-1].strip()
    toks = a.split()
    return toks[-1].strip() if toks else ""


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
    if _read_hf_manifest(repo_root) is None:
        return False, {}

    parquet_path = _find_test_parquet(repo_root)
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
        "Solve the following math problem efficiently and clearly. The last line\n"
        "of your response should be of the following format: 'Therefore, the final\n"
        "answer is: $\\\\boxed{{ANSWER}}$. I hope it is correct' (without quotes)\n"
        "where ANSWER is just the final number or expression that solves the\n"
        "problem.\n\n"
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

            gold = _extract_gold_final(a)
            pred = _extract_pred_final(completion)
            fg = _to_fraction(gold)
            fp = _to_fraction(pred)
            is_ok = False
            if fg is not None and fp is not None:
                is_ok = fg == fp
            else:
                is_ok = _norm_number_str(pred) == _norm_number_str(gold)

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
    hints = _parse_json_list(os.environ.get("AIDER_FSM_HINTS_JSON"))
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
            obj = {"prompt": prompt, "completion": completion, "reward": 0.0}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            samples_written += 1

    rollout_path = (repo_root / ".aider_fsm" / "rollout.json").resolve()
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout = {
        "ts": _now_iso(),
        "ok": True,
        "mode": mode,
        "counts": {"samples": samples_written},
        "paths": {"samples_jsonl": str(samples_path)},
    }
    rollout_path.write_text(json.dumps(rollout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
