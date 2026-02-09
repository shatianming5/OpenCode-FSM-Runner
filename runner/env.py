from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import _ensure_openai_v1_base, _find_hf_test_parquet, _read_json_object
from .contract_hints import suggest_contract_hints
from .contract_repair import repair_contract
from .eval_audit import (
    audit_eval_script_for_hardcoded_nonzero_score,
    audit_eval_script_has_real_execution,
    audit_eval_script_mentions_any_anchor,
)
from .env_local import (
    DeployCallResult,
    EnvHandle,
    EvaluationCallResult,
    RolloutCallResult,
    deploy as _deploy,
    deploy_teardown as _deploy_teardown,
    evaluate as _evaluate,
    open_env,
    rollout as _rollout,
    rollout_and_evaluate as _rollout_and_evaluate,
    with_runtime_env_path,
)

__all__ = ["EnvSession", "setup"]


def _now_run_id() -> str:
    # 作用：内部符号：_now_run_id
    # 能否简略：是
    # 原因：规模≈2 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/env.py:37；类型=function；引用≈2；规模≈2行
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_path(p: str | Path) -> Path:
    # 作用：内部符号：_resolve_path
    # 能否简略：部分
    # 原因：规模≈2 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:41；类型=function；引用≈4；规模≈2行
    return Path(str(p)).expanduser().resolve()


def _resolve_llm(llm: str | Path) -> tuple[str, Path | None, str | None]:
    """Resolve an LLM reference to either a local model dir or a remote model id.

    - Local: an existing directory path (Path or str).
    - Remote: any other non-empty string (passed through as-is).
    """
    # 作用：Resolve an LLM reference to either a local model dir or a remote model id.
    # 能否简略：部分
    # 原因：规模≈34 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:50；类型=function；引用≈2；规模≈34行
    if isinstance(llm, Path):
        p = llm.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"llm_path_not_found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"llm_not_directory: {p}")
        return "local_hf", p, None

    s = str(llm or "").strip()
    if not s:
        raise ValueError("empty_llm")

    # Treat explicit path-like strings as local paths (error if missing).
    if s.startswith(("/", "./", "../", "~")):
        p = Path(s).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"llm_path_not_found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"llm_not_directory: {p}")
        return "local_hf", p, None

    # If it exists as a relative directory, treat it as local.
    p2 = Path(s)
    if p2.exists() and p2.is_dir():
        return "local_hf", p2.expanduser().resolve(), None

    # Otherwise: remote model id/name, passed through.
    return "remote", None, s


def _resolve_run_root(repo: Path, *, run_id: str, artifacts_dir: Path | None) -> Path:
    # 作用：内部符号：_resolve_run_root
    # 能否简略：部分
    # 原因：规模≈7 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:81；类型=function；引用≈4；规模≈7行
    if artifacts_dir is not None:
        out = _resolve_path(artifacts_dir)
    else:
        out = (repo / ".aider_fsm" / "artifacts" / run_id / "env_api").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _inject_openai_base_compat(overrides: dict[str, str]) -> None:
    """Keep OpenAI-compatible base URL aliases in sync for downstream tools.

    Some downstream libraries read `OPENAI_BASE_URL`, while others read
    `OPENAI_API_BASE`. We normalize to include both keys when either is present
    in env overrides or inherited process env.
    """
    # 作用：Keep OpenAI-compatible base URL aliases in sync for downstream tools.
    # 能否简略：是
    # 原因：规模≈18 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/env.py:113；类型=function；引用≈2；规模≈18行
    base = (
        str(overrides.get("OPENAI_BASE_URL") or "").strip()
        or str(overrides.get("OPENAI_API_BASE") or "").strip()
        or str(os.environ.get("OPENAI_BASE_URL") or "").strip()
        or str(os.environ.get("OPENAI_API_BASE") or "").strip()
    )
    if not base:
        return
    base = _ensure_openai_v1_base(base)
    overrides.setdefault("OPENAI_BASE_URL", base)
    overrides.setdefault("OPENAI_API_BASE", base)


def _runtime_openai_config(runtime_env_path: Path) -> tuple[str | None, str | None]:
    """Best-effort: derive (OPENAI_API_BASE, OPENAI_MODEL) from runtime_env.json."""
    # 作用：Best-effort: derive (OPENAI_API_BASE, OPENAI_MODEL) from runtime_env.json.
    # 能否简略：部分
    # 原因：规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:128；类型=function；引用≈2；规模≈22行
    p = Path(runtime_env_path).expanduser().resolve()
    obj = _read_json_object(p)
    if obj is None:
        return None, None

    base = ""
    model = ""

    inf = obj.get("inference")
    if isinstance(inf, dict):
        base = str(inf.get("openai_base_url") or inf.get("base_url") or "").strip()
        model = str(inf.get("model") or "").strip()

    if not base:
        svc = obj.get("service")
        if isinstance(svc, dict):
            base = str(svc.get("base_url") or "").strip()

    base_v1 = _ensure_openai_v1_base(base) if base else ""
    return (base_v1 or None), (model or None)


def _verification_errors_summary(verify: Any) -> str:
    """Best-effort: surface pipeline verification errors for contract repair prompts."""
    # 作用：Best-effort: surface pipeline verification errors for contract repair prompts.
    # 能否简略：是
    # 原因：规模≈14 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/env.py:152；类型=function；引用≈3；规模≈14行
    if verify is None:
        return ""
    parts: list[str] = []
    failed_stage = getattr(verify, "failed_stage", None)
    if isinstance(failed_stage, str) and failed_stage.strip():
        parts.append(f"verify.failed_stage: {failed_stage.strip()}")
    metrics_errors = getattr(verify, "metrics_errors", None)
    if isinstance(metrics_errors, list):
        cleaned = [str(x).strip() for x in metrics_errors if str(x).strip()]
        if cleaned:
            parts.append("verify.metrics_errors:\n" + "\n".join([f"- {x}" for x in cleaned]))
    return "\n".join(parts).strip()


def _hf_parquet_qa_rows(repo_root: Path) -> int | None:
    """If repo_root is an HF snapshot with a QA test parquet, return its row count."""
    # 作用：If repo_root is an HF snapshot with a QA test parquet, return its row count.
    # 能否简略：部分
    # 原因：规模≈30 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:185；类型=function；引用≈2；规模≈30行
    repo_root = Path(repo_root).resolve()
    if not (repo_root / "data" / "hf_manifest.json").exists():
        return None
    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None
    try:
        schema_names = set(str(n) for n in (pf.schema.names or []))
    except Exception:
        schema_names = set()
    if not {"question", "answer"}.issubset(schema_names):
        return None
    try:
        meta = pf.metadata
        if meta is not None:
            n = int(meta.num_rows)
            return n if n > 0 else None
    except Exception:
        return None
    return None


def _hf_parquet_qa_question_samples(repo_root: Path, *, max_questions: int = 20) -> list[str] | None:
    """If repo_root is an HF QA snapshot, return up to N sample questions from the test parquet."""
    # 作用：If repo_root is an HF QA snapshot, return up to N sample questions from the test parquet.
    # 能否简略：部分
    # 原因：规模≈39 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:217；类型=function；引用≈2；规模≈39行
    repo_root = Path(repo_root).resolve()
    if not (repo_root / "data" / "hf_manifest.json").exists():
        return None
    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None
    try:
        schema_names = set(str(n) for n in (pf.schema.names or []))
    except Exception:
        schema_names = set()
    if "question" not in schema_names:
        return None

    n = max(1, int(max_questions))
    out: list[str] = []
    try:
        for batch in pf.iter_batches(batch_size=n, columns=["question"]):
            try:
                arr = batch.column(0).to_pylist()
            except Exception:
                arr = []
            for q in arr:
                s = str(q or "").strip()
                if s:
                    out.append(s)
            break
    except Exception:
        return None
    return out[:n] if out else None


def _validate_rollout_samples(
    repo: Path,
    rollout_path: Path | None,
    *,
    mode: str,
    eval_limit: int | None,
) -> tuple[bool, str]:
    """Validate that rollout produced a usable samples JSONL reference.

    Minimal contract enforced for all targets:
    - rollout.json exists and is a JSON object
    - rollout.json.paths.samples_jsonl exists and points to an existing JSONL file
    - the JSONL file contains at least one valid sample line with keys: prompt, completion, reward

    Additional enforcement for Hugging Face QA dataset snapshots (best-effort):
    - if `data/hf_manifest.json` exists AND a test parquet with columns (question, answer) exists,
      require that the samples JSONL contains at least `min(eval_limit, test_rows)` valid samples
      (defaults match `runner.generic_rollout`: smoke=8, full=64 when eval_limit is not set).
      Also require some prompt diversity to avoid trivial placeholder rollouts.
      Also require prompts to be anchored to dataset questions (to avoid synthetic "fake" tasks).
    """
    # 作用：Validate that rollout produced a usable samples JSONL reference.
    # 能否简略：部分
    # 原因：规模≈157 行；引用次数≈9（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:277；类型=function；引用≈9；规模≈157行
    repo = Path(repo).resolve()
    p = (rollout_path or (repo / ".aider_fsm" / "rollout.json")).resolve()
    if not p.exists():
        return False, f"missing_rollout_json: {p}"
    obj = _read_json_object(p)
    if obj is None:
        return False, "rollout_json_not_object"

    # Best-effort: if rollout explicitly reports that all sample attempts errored, fail fast.
    counts = obj.get("counts")
    if isinstance(counts, dict):
        raw_samples = counts.get("samples")
        raw_errors = counts.get("errors")
        try:
            n_samples = int(raw_samples) if isinstance(raw_samples, (int, float, str)) else None
        except Exception:
            n_samples = None
        try:
            n_errors = int(raw_errors) if isinstance(raw_errors, (int, float, str)) else None
        except Exception:
            n_errors = None
        if (
            isinstance(n_samples, int)
            and isinstance(n_errors, int)
            and n_samples > 0
            and n_errors > 0
            and n_errors >= n_samples
        ):
            return False, f"rollout_counts_all_errors: errors={n_errors} samples={n_samples}"

    paths = obj.get("paths")
    if not isinstance(paths, dict):
        return False, "rollout_json_missing_paths"
    raw = paths.get("samples_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        return False, "rollout_json_missing_paths.samples_jsonl"
    samples_path = Path(raw.strip())
    if not samples_path.is_absolute():
        samples_path = (repo / samples_path).resolve()
    if not samples_path.exists():
        return False, f"samples_jsonl_not_found: {samples_path}"
    try:
        if samples_path.stat().st_size <= 0:
            return False, "samples_jsonl_empty"
    except Exception:
        pass

    # Determine whether we should enforce a minimum sample count (HF QA parquet snapshots).
    mode2 = str(mode or "").strip().lower() or "smoke"
    default_limit = 64 if mode2 == "full" else 8
    try:
        lim = int(eval_limit) if eval_limit is not None else int(default_limit)
    except Exception:
        lim = int(default_limit)
    lim = max(1, int(lim))

    qa_rows = _hf_parquet_qa_rows(repo)
    expected_min = min(int(qa_rows), int(lim)) if isinstance(qa_rows, int) and qa_rows > 0 else None
    distinct_target = 1 if not expected_min or expected_min <= 1 else min(10, int(expected_min))
    qa_questions = _hf_parquet_qa_question_samples(repo, max_questions=20) if expected_min is not None else None

    def _norm_ws(s: str) -> str:
        # 作用：内部符号：_validate_rollout_samples._norm_ws
        # 能否简略：是
        # 原因：规模≈2 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/env.py:339；类型=function；引用≈3；规模≈2行
        return " ".join(str(s or "").strip().lower().split())

    qa_q_norms: list[str] = []
    if isinstance(qa_questions, list):
        seen: set[str] = set()
        for q in qa_questions:
            qn = _norm_ws(q)
            if not qn:
                continue
            if qn in seen:
                continue
            seen.add(qn)
            qa_q_norms.append(qn)
    qa_anchor_target = min(5, len(qa_q_norms)) if qa_q_norms else 0
    matched_anchors: set[str] = set()

    # Parse samples (bounded by the contract and, when applicable, by expected_min).
    valid = 0
    nonempty_completions = 0
    distinct_prompts: set[str] = set()
    to_scan = int(expected_min or 1)
    # Keep a cap for prompt diversity scanning (does not affect counting).
    diversity_scan_cap = max(1, min(200, to_scan))
    try:
        with samples_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    item = json.loads(s)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                completion = item.get("completion")
                reward = item.get("reward")
                if isinstance(prompt, str) and isinstance(completion, str) and isinstance(reward, (int, float)):
                    valid += 1
                    if completion.strip():
                        nonempty_completions += 1
                    if len(distinct_prompts) < diversity_scan_cap:
                        distinct_prompts.add(prompt)
                        if qa_anchor_target > 0 and len(matched_anchors) < qa_anchor_target:
                            pn = _norm_ws(prompt)
                            for qn in qa_q_norms:
                                if qn and qn in pn:
                                    matched_anchors.add(qn)
                                    break
                    if expected_min is None and nonempty_completions >= 1:
                        return True, "ok"
                    if (
                        expected_min is not None
                        and valid >= int(expected_min)
                        and nonempty_completions >= 1
                        and len(distinct_prompts) >= int(distinct_target)
                        and (qa_anchor_target <= 0 or len(matched_anchors) >= int(qa_anchor_target))
                    ):
                        return True, "ok"
    except Exception as e:
        return False, f"failed_to_read_samples_jsonl: {e}"

    if valid <= 0:
        return False, "samples_jsonl_has_no_valid_samples"
    if nonempty_completions <= 0:
        return False, "samples_jsonl_all_empty_completions"
    if expected_min is not None and valid < int(expected_min):
        return False, f"hf_qa_samples_too_few: expected>={expected_min} got={valid}"
    if expected_min is not None and len(distinct_prompts) < int(distinct_target):
        return False, f"hf_qa_prompts_not_diverse: expected>={distinct_target} got={len(distinct_prompts)}"
    if expected_min is not None and qa_anchor_target > 0 and len(matched_anchors) < int(qa_anchor_target):
        return False, f"hf_qa_prompts_not_anchored: expected>={qa_anchor_target} got={len(matched_anchors)}"
    return True, "ok"


@dataclass
class EnvSession:
    """A small programmatic wrapper for `runner.env_local`.

    中文说明：
    - 含义：提供最小闭环三入口：`setup()` -> `sess.rollout(llm=...)` -> `sess.evaluate()`
    - 内容：只负责 orchestration 与自动 repair；不写任何 benchmark-specific 逻辑。
    """
    # 作用：A small programmatic wrapper for `runner.env_local`.
    # 能否简略：部分
    # 原因：规模≈421 行；引用次数≈6（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/env.py:423；类型=class；引用≈6；规模≈421行

    env: EnvHandle
    run_id: str
    unattended: str
    opencode_model: str
    opencode_repair_model: str
    opencode_url: str
    opencode_timeout_seconds: int
    require_metrics: bool
    command_hints: list[str]
    hint_anchors: list[str]
    opencode_retry_attempts: int = 2
    opencode_retry_backoff_seconds: float = 2.0
    opencode_context_length: int | None = None
    opencode_max_prompt_chars: int | None = None
    audit: str = "on"  # on|off|warn-only
    runtime_env_path: Path | None = None
    llm_kind: str = ""
    llm_model: str | None = None
    trained_model_dir: Path | None = None

    def _audit_mode(self) -> str:
        # 作用：内部符号：EnvSession._audit_mode
        # 能否简略：部分
        # 原因：规模≈5 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/env.py:445；类型=method；引用≈4；规模≈5行
        m = str(self.audit or "on").strip().lower() or "on"
        if m not in ("on", "off", "warn-only"):
            return "on"
        return m

    def _set_llm(self, llm: str | Path) -> None:
        # 作用：内部符号：EnvSession._set_llm
        # 能否简略：是
        # 原因：规模≈5 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/env.py:451；类型=method；引用≈1；规模≈5行
        kind, model_dir, model = _resolve_llm(llm)
        self.llm_kind = kind
        self.trained_model_dir = model_dir
        self.llm_model = model

    def _apply_llm_overrides(self, overrides: dict[str, str]) -> None:
        """Force a consistent LLM contract into env vars (no hardcoded paths/endpoints)."""
        # 作用：Force a consistent LLM contract into env vars (no hardcoded paths/endpoints).
        # 能否简略：是
        # 原因：规模≈20 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/env.py:458；类型=method；引用≈2；规模≈20行
        kind = str(self.llm_kind or "").strip() or "local_hf"
        if kind == "remote":
            if not self.llm_model:
                raise ValueError("missing_llm: call deploy/rollout with llm=model_id first")
            overrides["AIDER_LLM_KIND"] = "remote"
            overrides["AIDER_LLM_MODEL"] = str(self.llm_model)
            overrides.setdefault("OPENAI_MODEL", str(self.llm_model))
            overrides.pop("AIDER_TRAINED_MODEL_DIR", None)
            return

        if self.trained_model_dir is None:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_dir first")
        overrides["AIDER_LLM_KIND"] = "local_hf"
        overrides["AIDER_TRAINED_MODEL_DIR"] = str(self.trained_model_dir)
        model_id = str(self.trained_model_dir.name or "").strip()
        if model_id:
            overrides.setdefault("OPENAI_MODEL", model_id)
        overrides.pop("AIDER_LLM_MODEL", None)

    def _base_overrides(self, *, mode: str, extra: dict[str, str] | None) -> dict[str, str]:
        # 作用：内部符号：EnvSession._base_overrides
        # 能否简略：部分
        # 原因：规模≈27 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/env.py:478；类型=method；引用≈3；规模≈27行
        out = dict(extra or {})
        _inject_openai_base_compat(out)
        out.setdefault("AIDER_FSM_RUN_ID", str(self.run_id))
        out.setdefault("AIDER_EVAL_MODE", str(mode or "smoke").strip() or "smoke")
        if self.command_hints:
            # When doc hints exist, require evaluation scripts to run at least one
            # official/hinted command and record usage (see runner/pipeline_verify.py).
            out.setdefault("AIDER_FSM_REQUIRE_HINTS", "1")
            try:
                out.setdefault("AIDER_FSM_HINTS_JSON", json.dumps(list(self.command_hints[:20]), ensure_ascii=False))
            except Exception:
                out.setdefault("AIDER_FSM_HINTS_JSON", "[]")
            try:
                out.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", json.dumps(list(self.hint_anchors[:20]), ensure_ascii=False))
            except Exception:
                out.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", "[]")
        if self.llm_kind:
            out.setdefault("AIDER_LLM_KIND", str(self.llm_kind))
        if self.llm_kind == "remote" and self.llm_model:
            out.setdefault("AIDER_LLM_MODEL", str(self.llm_model))
            out.setdefault("OPENAI_MODEL", str(self.llm_model))
        if self.trained_model_dir is not None:
            out.setdefault("AIDER_TRAINED_MODEL_DIR", str(self.trained_model_dir))
        if self.runtime_env_path is not None:
            out.update(with_runtime_env_path(self.runtime_env_path))
        return out

    def _apply_runtime_env_inference_overrides(self, overrides: dict[str, str]) -> None:
        # 作用：内部符号：EnvSession._apply_runtime_env_inference_overrides
        # 能否简略：是
        # 原因：规模≈17 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/env.py:506；类型=method；引用≈3；规模≈17行
        if self.runtime_env_path is None:
            return
        base, model = _runtime_openai_config(self.runtime_env_path)
        if base:
            explicit = any(
                str(overrides.get(k) or "").strip() for k in ("OPENAI_API_BASE", "OPENAI_BASE_URL")
            )
            # For local_hf, runtime_env.json is the source of truth even if the base env is set (e.g. .env).
            if self.llm_kind == "local_hf" or not explicit:
                overrides["OPENAI_API_BASE"] = base
                overrides["OPENAI_BASE_URL"] = base
        if model:
            overrides.setdefault("OPENAI_MODEL", model)
            overrides.setdefault("AIDER_LLM_MODEL", model)
        if self.llm_kind == "local_hf":
            overrides.setdefault("OPENAI_API_KEY", str(overrides.get("OPENAI_API_KEY") or "local"))

    def _maybe_teardown(self, *, run_root: Path, overrides: dict[str, str]) -> None:
        # 作用：内部符号：EnvSession._maybe_teardown
        # 能否简略：是
        # 原因：规模≈10 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/env.py:524；类型=method；引用≈3；规模≈10行
        try:
            _deploy_teardown(
                self.env,
                artifacts_dir=(run_root / "deploy_teardown"),
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
            )
        except Exception:
            pass

    def rollout(
        self,
        llm: str | Path,
        *,
        mode: str = "smoke",
        require_samples: bool = False,
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> RolloutCallResult:
        # 作用：内部符号：EnvSession.rollout
        # 能否简略：否
        # 原因：公共 API/关键编排点；规模≈117 行；引用次数≈3（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/env.py:544；类型=method；引用≈3；规模≈117行
        self._set_llm(llm)

        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            rollout_dir = (run_root / f"rollout_attempt_{attempt+1:02d}").resolve()
            contract_err = ""
            overrides = self._base_overrides(mode=mode, extra=env_overrides)
            self._apply_llm_overrides(overrides)

            deploy_res = _deploy(
                self.env,
                artifacts_dir=deploy_dir,
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            if not deploy_res.ok:
                if attempt >= int(max(0, repair_iters)):
                    # Return a rollout-shaped failure with deploy verification attached.
                    return RolloutCallResult(ok=False, artifacts_dir=rollout_dir, rollout_path=None, verify=deploy_res.verify)
                repair_contract(
                    repo=self.env.repo,
                    model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                    opencode_url=str(self.opencode_url or ""),
                    unattended=str(self.unattended or "strict"),
                    artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                    failed_stage=str(getattr(deploy_res.verify, "failed_stage", "") or "deploy"),
                    deploy_artifacts_dir=deploy_dir,
                    rollout_eval_artifacts_dir=rollout_dir,
                    llm_kind=str(self.llm_kind or ""),
                    llm_model=str(self.llm_model or ""),
                    command_hints=self.command_hints,
                    extra_context="",
                    timeout_seconds=int(self.opencode_timeout_seconds or 300),
                    retry_attempts=int(self.opencode_retry_attempts or 0),
                    retry_backoff_seconds=float(self.opencode_retry_backoff_seconds or 0.0),
                    context_length=(int(self.opencode_context_length) if self.opencode_context_length is not None else None),
                    max_prompt_chars=(int(self.opencode_max_prompt_chars) if self.opencode_max_prompt_chars is not None else None),
                )
                continue

            self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
            overrides.update(with_runtime_env_path(self.runtime_env_path))
            self._apply_runtime_env_inference_overrides(overrides)

            rollout_res = _rollout(
                self.env,
                artifacts_dir=rollout_dir,
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            if rollout_res.ok and bool(require_samples):
                raw_limit = overrides.get("AIDER_EVAL_LIMIT")
                try:
                    lim = int(str(raw_limit).strip()) if str(raw_limit or "").strip() else None
                except Exception:
                    lim = None
                ok_samples, reason = _validate_rollout_samples(
                    self.env.repo,
                    rollout_res.rollout_path,
                    mode=str(mode or "smoke"),
                    eval_limit=lim,
                )
                if not ok_samples:
                    try:
                        (rollout_dir / "rollout_contract_error.txt").write_text(str(reason) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    contract_err = str(reason)
                    rollout_res = RolloutCallResult(
                        ok=False,
                        artifacts_dir=rollout_res.artifacts_dir,
                        rollout_path=rollout_res.rollout_path,
                        verify=rollout_res.verify,
                    )

            if rollout_res.ok:
                return rollout_res

            if attempt >= int(max(0, repair_iters)):
                return rollout_res

            self._maybe_teardown(run_root=run_root / f"teardown_attempt_{attempt+1:02d}", overrides=overrides)
            repair_contract(
                repo=self.env.repo,
                model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                opencode_url=str(self.opencode_url or ""),
                unattended=str(self.unattended or "strict"),
                artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                failed_stage="rollout",
                deploy_artifacts_dir=deploy_dir,
                rollout_eval_artifacts_dir=rollout_dir,
                llm_kind=str(self.llm_kind or ""),
                llm_model=str(self.llm_model or ""),
                command_hints=self.command_hints,
                extra_context=(contract_err or ("rollout_contract_invalid" if require_samples else "")),
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
                retry_attempts=int(self.opencode_retry_attempts or 0),
                retry_backoff_seconds=float(self.opencode_retry_backoff_seconds or 0.0),
                context_length=(int(self.opencode_context_length) if self.opencode_context_length is not None else None),
                max_prompt_chars=(int(self.opencode_max_prompt_chars) if self.opencode_max_prompt_chars is not None else None),
            )

        return rollout_res  # pragma: no cover

    def _evaluation(
        self,
        *,
        mode: str = "smoke",
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> EvaluationCallResult:
        # 作用：内部符号：EnvSession._evaluation
        # 能否简略：部分
        # 原因：规模≈164 行；引用次数≈1（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/env.py:660；类型=method；引用≈1；规模≈164行
        if not self.llm_kind:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_dir|model_id first")

        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            roll_eval_dir = (run_root / f"rollout_evaluation_attempt_{attempt+1:02d}").resolve()
            eval_dir = (run_root / f"evaluation_attempt_{attempt+1:02d}").resolve()

            overrides = self._base_overrides(mode=mode, extra=env_overrides)
            self._apply_llm_overrides(overrides)

            combined = ""
            if self.runtime_env_path is not None and attempt == 0:
                # Fast path: reuse the existing deploy and only run evaluation.
                overrides.update(with_runtime_env_path(self.runtime_env_path))
                self._apply_runtime_env_inference_overrides(overrides)
                eval_res = _evaluate(
                    self.env,
                    artifacts_dir=eval_dir,
                    env_overrides=overrides,
                    unattended=str(self.unattended or "strict"),
                    run_bootstrap_first=True,
                )
                if not eval_res.ok:
                    combined = _verification_errors_summary(eval_res.verify)
                if eval_res.ok:
                    if bool(self.require_metrics) and isinstance(eval_res.metrics, dict) and eval_res.metrics.get("ok") is not True:
                        combined = "metrics.ok_not_true"
                        try:
                            (eval_dir / "metrics_contract_error.txt").write_text("metrics.ok_not_true\n", encoding="utf-8")
                        except Exception:
                            pass
                    elif bool(self.require_metrics) and self._audit_mode() != "off":
                        audit_issue = audit_eval_script_for_hardcoded_nonzero_score(self.env.repo)
                        audit_issue2 = audit_eval_script_has_real_execution(self.env.repo, extra_markers=self.hint_anchors)
                        audit_issue3 = audit_eval_script_mentions_any_anchor(self.env.repo, self.hint_anchors)
                        combined = "\n\n".join([x for x in (audit_issue, audit_issue2, audit_issue3) if x]).strip()
                        if combined:
                            try:
                                (eval_dir / "evaluation_audit_error.txt").write_text(combined + "\n", encoding="utf-8")
                            except Exception:
                                pass
                    if not combined or self._audit_mode() == "warn-only":
                        return eval_res
                    eval_res = EvaluationCallResult(
                        ok=False,
                        artifacts_dir=eval_res.artifacts_dir,
                        metrics_path=eval_res.metrics_path,
                        metrics=eval_res.metrics,
                        verify=eval_res.verify,
                    )
            else:
                # Repair path: run a full deploy -> rollout -> evaluation pass.
                deploy_res = _deploy(
                    self.env,
                    artifacts_dir=deploy_dir,
                    env_overrides=overrides,
                    unattended=str(self.unattended or "strict"),
                    run_bootstrap_first=True,
                )
                if not deploy_res.ok:
                    if attempt >= int(max(0, repair_iters)):
                        return EvaluationCallResult(
                            ok=False,
                            artifacts_dir=roll_eval_dir,
                            metrics_path=None,
                            metrics=None,
                            verify=deploy_res.verify,
                        )
                    repair_contract(
                        repo=self.env.repo,
                        model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                        opencode_url=str(self.opencode_url or ""),
                        unattended=str(self.unattended or "strict"),
                        artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                        failed_stage=str(getattr(deploy_res.verify, "failed_stage", "") or "deploy"),
                        deploy_artifacts_dir=deploy_dir,
                        rollout_eval_artifacts_dir=roll_eval_dir,
                        llm_kind=str(self.llm_kind or ""),
                        llm_model=str(self.llm_model or ""),
                        command_hints=self.command_hints,
                        extra_context="",
                        timeout_seconds=int(self.opencode_timeout_seconds or 300),
                        retry_attempts=int(self.opencode_retry_attempts or 0),
                        retry_backoff_seconds=float(self.opencode_retry_backoff_seconds or 0.0),
                        context_length=(int(self.opencode_context_length) if self.opencode_context_length is not None else None),
                        max_prompt_chars=(int(self.opencode_max_prompt_chars) if self.opencode_max_prompt_chars is not None else None),
                    )
                    continue

                self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
                overrides.update(with_runtime_env_path(self.runtime_env_path))
                self._apply_runtime_env_inference_overrides(overrides)
                _rollout_res, eval_res = _rollout_and_evaluate(
                    self.env,
                    artifacts_dir=roll_eval_dir,
                    env_overrides=overrides,
                    unattended=str(self.unattended or "strict"),
                    run_bootstrap_first=True,
                )
                if not eval_res.ok:
                    combined = _verification_errors_summary(eval_res.verify)
                if eval_res.ok:
                    if bool(self.require_metrics) and isinstance(eval_res.metrics, dict) and eval_res.metrics.get("ok") is not True:
                        combined = "metrics.ok_not_true"
                        try:
                            (roll_eval_dir / "metrics_contract_error.txt").write_text("metrics.ok_not_true\n", encoding="utf-8")
                        except Exception:
                            pass
                    elif bool(self.require_metrics) and self._audit_mode() != "off":
                        audit_issue = audit_eval_script_for_hardcoded_nonzero_score(self.env.repo)
                        audit_issue2 = audit_eval_script_has_real_execution(self.env.repo, extra_markers=self.hint_anchors)
                        audit_issue3 = audit_eval_script_mentions_any_anchor(self.env.repo, self.hint_anchors)
                        combined = "\n\n".join([x for x in (audit_issue, audit_issue2, audit_issue3) if x]).strip()
                        if combined:
                            try:
                                (roll_eval_dir / "evaluation_audit_error.txt").write_text(combined + "\n", encoding="utf-8")
                            except Exception:
                                pass
                    if not combined or self._audit_mode() == "warn-only":
                        return eval_res
                    eval_res = EvaluationCallResult(
                        ok=False,
                        artifacts_dir=eval_res.artifacts_dir,
                        metrics_path=eval_res.metrics_path,
                        metrics=eval_res.metrics,
                        verify=eval_res.verify,
                    )

            if attempt >= int(max(0, repair_iters)):
                return eval_res

            self._maybe_teardown(run_root=run_root / f"teardown_attempt_{attempt+1:02d}", overrides=overrides)
            repair_contract(
                repo=self.env.repo,
                model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                opencode_url=str(self.opencode_url or ""),
                unattended=str(self.unattended or "strict"),
                artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                failed_stage="evaluation",
                deploy_artifacts_dir=deploy_dir if deploy_dir.exists() else run_root,
                rollout_eval_artifacts_dir=roll_eval_dir if roll_eval_dir.exists() else eval_dir,
                llm_kind=str(self.llm_kind or ""),
                llm_model=str(self.llm_model or ""),
                command_hints=self.command_hints,
                extra_context=combined,
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
                retry_attempts=int(self.opencode_retry_attempts or 0),
                retry_backoff_seconds=float(self.opencode_retry_backoff_seconds or 0.0),
                context_length=(int(self.opencode_context_length) if self.opencode_context_length is not None else None),
                max_prompt_chars=(int(self.opencode_max_prompt_chars) if self.opencode_max_prompt_chars is not None else None),
            )

        return eval_res  # pragma: no cover

    def evaluate(
        self,
        llm: str | Path | None = None,
        *,
        mode: str = "smoke",
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> EvaluationCallResult:
        # 作用：内部符号：EnvSession.evaluate
        # 能否简略：否
        # 原因：公共 API/关键编排点；规模≈20 行；引用次数≈4（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/env.py:825；类型=method；引用≈4；规模≈20行
        if llm is not None:
            self._set_llm(llm)

        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)
        try:
            res = self._evaluation(
                mode=mode,
                env_overrides=env_overrides,
                artifacts_dir=artifacts_dir,
                repair_iters=repair_iters,
            )
            return res
        finally:
            overrides = self._base_overrides(mode=mode, extra=env_overrides)
            self._maybe_teardown(run_root=run_root / "final_teardown", overrides=overrides)


def setup(
    target: str | Path,
    *,
    clones_dir: Path | None = None,
    pipeline_rel: str = "pipeline.yml",
    require_metrics: bool = True,
    audit: str = "on",
    opencode_model: str = "",
    opencode_repair_model: str | None = None,
    opencode_url: str = "",
    unattended: str = "strict",
    opencode_timeout_seconds: int = 300,
    opencode_repair_timeout_seconds: int | None = None,
    opencode_retry_attempts: int = 2,
    opencode_retry_backoff_seconds: float = 2.0,
    opencode_session_recover_attempts: int | None = None,
    opencode_session_recover_backoff_seconds: float | None = None,
    opencode_context_length: int | None = None,
    opencode_max_prompt_chars: int | None = None,
    opencode_bash: str = "restricted",
    scaffold_opencode_bash: str = "restricted",
    strict_opencode: bool = True,
    artifacts_dir: Path | None = None,
) -> EnvSession:
    """Open an environment handle for a target repo/url and ensure a runnable contract exists.

    中文说明：
    - 含义：你训练脚本的入口：支持本地路径、git URL、HF dataset URL。
    - 内容：若缺少 pipeline.yml，则通过 OpenCode scaffold 合同（pipeline.yml + `.aider_fsm/**`）。
    - 约束：本函数不包含任何 benchmark-specific hardcoding。
    """
    # 作用：Open an environment handle for a target repo/url and ensure a runnable contract exists.
    # 能否简略：否
    # 原因：公共 API/关键编排点；规模≈83 行；引用次数≈21（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/env.py:870；类型=function；引用≈21；规模≈83行
    clones_dir = _resolve_path(clones_dir) if clones_dir is not None else None
    artifacts_dir = _resolve_path(artifacts_dir) if artifacts_dir is not None else None
    env_handle: EnvHandle = open_env(
        target,
        clones_dir=clones_dir,
        pipeline_rel=str(pipeline_rel or "pipeline.yml").strip() or "pipeline.yml",
        require_pipeline=True,
        scaffold_contract="opencode",
        scaffold_require_metrics=bool(require_metrics),
        model=str(opencode_model or ""),
        opencode_url=str(opencode_url or ""),
        opencode_timeout_seconds=int(opencode_timeout_seconds or 300),
        opencode_retry_attempts=int(opencode_retry_attempts or 0),
        opencode_retry_backoff_seconds=float(opencode_retry_backoff_seconds or 0.0),
        opencode_session_recover_attempts=(
            int(opencode_session_recover_attempts) if opencode_session_recover_attempts is not None else None
        ),
        opencode_session_recover_backoff_seconds=(
            float(opencode_session_recover_backoff_seconds)
            if opencode_session_recover_backoff_seconds is not None
            else None
        ),
        opencode_context_length=(int(opencode_context_length) if opencode_context_length is not None else None),
        opencode_max_prompt_chars=(int(opencode_max_prompt_chars) if opencode_max_prompt_chars is not None else None),
        opencode_bash=str(opencode_bash or "restricted"),
        scaffold_opencode_bash=str(scaffold_opencode_bash or "restricted"),
        unattended=str(unattended or "strict"),
        artifacts_dir=artifacts_dir,
        # Runner no longer prewrites/fallback-writes scaffold contracts.
        # Keep these flags for compatibility/provenance labeling only.
        seed_stage_skeleton=not bool(strict_opencode),
        write_fallback_pipeline_yml=not bool(strict_opencode),
    )
    hints = suggest_contract_hints(env_handle.repo)
    session = EnvSession(
        env=env_handle,
        run_id=_now_run_id(),
        unattended=str(unattended or "strict"),
        opencode_model=str(opencode_model or ""),
        opencode_repair_model=str(opencode_repair_model or opencode_model or ""),
        opencode_url=str(opencode_url or ""),
        opencode_timeout_seconds=int(opencode_repair_timeout_seconds or opencode_timeout_seconds or 300),
        require_metrics=bool(require_metrics),
        command_hints=list(hints.commands or []),
        hint_anchors=list(hints.anchors or []),
        opencode_retry_attempts=int(opencode_retry_attempts or 0),
        opencode_retry_backoff_seconds=float(opencode_retry_backoff_seconds or 0.0),
        opencode_context_length=(int(opencode_context_length) if opencode_context_length is not None else None),
        opencode_max_prompt_chars=(int(opencode_max_prompt_chars) if opencode_max_prompt_chars is not None else None),
        audit=str(audit or "on"),
    )
    return session
