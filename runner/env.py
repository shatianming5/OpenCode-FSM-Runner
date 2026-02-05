from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_path(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def _resolve_llm(llm: str | Path) -> tuple[str, Path | None, str | None]:
    """Resolve an LLM reference to either a local model dir or a remote model id.

    - Local: an existing directory path (Path or str).
    - Remote: any other non-empty string (passed through as-is).
    """
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
    if artifacts_dir is not None:
        out = _resolve_path(artifacts_dir)
    else:
        out = (repo / ".aider_fsm" / "artifacts" / run_id / "env_api").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _verification_errors_summary(verify: Any) -> str:
    """Best-effort: surface pipeline verification errors for contract repair prompts."""
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


def _validate_rollout_samples(repo: Path, rollout_path: Path | None) -> tuple[bool, str]:
    """Validate that rollout produced a usable samples JSONL reference.

    This is benchmark-agnostic and only enforces the minimal contract required by RL training:
    - rollout.json exists and is a JSON object
    - rollout.json.paths.samples_jsonl exists and points to an existing JSONL file
    - the JSONL file contains at least one valid sample line with keys: prompt, completion, reward
    """
    repo = Path(repo).resolve()
    p = (rollout_path or (repo / ".aider_fsm" / "rollout.json")).resolve()
    if not p.exists():
        return False, f"missing_rollout_json: {p}"
    obj = _read_json_object(p)
    if obj is None:
        return False, "rollout_json_not_object"
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

    # Parse a bounded number of lines to find at least one valid sample.
    try:
        with samples_path.open("r", encoding="utf-8", errors="replace") as f:
            checked = 0
            for line in f:
                if checked >= 200:
                    break
                checked += 1
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
                    return True, "ok"
    except Exception as e:
        return False, f"failed_to_read_samples_jsonl: {e}"

    return False, "samples_jsonl_has_no_valid_samples"


@dataclass
class EnvSession:
    """A small programmatic wrapper for `runner.env_local`.

    中文说明：
    - 含义：提供你训练脚本需要的 `setup/deploy/rollout/evaluation/teardown` 调用形态。
    - 内容：只负责 orchestration 与自动 repair；不写任何 benchmark-specific 逻辑。
    """

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
    audit: str = "on"  # on|off|warn-only
    runtime_env_path: Path | None = None
    llm_kind: str = ""
    llm_model: str | None = None
    trained_model_dir: Path | None = None

    def _audit_mode(self) -> str:
        m = str(self.audit or "on").strip().lower() or "on"
        if m not in ("on", "off", "warn-only"):
            return "on"
        return m

    def _set_llm(self, llm: str | Path) -> None:
        kind, model_dir, model = _resolve_llm(llm)
        self.llm_kind = kind
        self.trained_model_dir = model_dir
        self.llm_model = model

    def _apply_llm_overrides(self, overrides: dict[str, str]) -> None:
        """Force a consistent LLM contract into env vars (no hardcoded paths/endpoints)."""
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
        overrides.pop("AIDER_LLM_MODEL", None)

    def _base_overrides(self, *, mode: str, extra: dict[str, str] | None) -> dict[str, str]:
        out = dict(extra or {})
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

    def _maybe_teardown(self, *, run_root: Path, overrides: dict[str, str]) -> None:
        try:
            _deploy_teardown(
                self.env,
                artifacts_dir=(run_root / "deploy_teardown"),
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
            )
        except Exception:
            return

    def deploy(
        self,
        llm: str | Path,
        *,
        mode: str = "smoke",
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> DeployCallResult:
        self._set_llm(llm)
        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            roll_eval_dir = (run_root / f"rollout_evaluation_attempt_{attempt+1:02d}").resolve()
            overrides = self._base_overrides(mode=mode, extra=env_overrides)
            self._apply_llm_overrides(overrides)

            res = _deploy(
                self.env,
                artifacts_dir=deploy_dir,
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            if res.ok:
                self.runtime_env_path = res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
                return res

            if attempt >= int(max(0, repair_iters)):
                return res

            repair_contract(
                repo=self.env.repo,
                model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                opencode_url=str(self.opencode_url or ""),
                unattended=str(self.unattended or "strict"),
                artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                failed_stage="deploy",
                deploy_artifacts_dir=deploy_dir,
                rollout_eval_artifacts_dir=roll_eval_dir,
                command_hints=self.command_hints,
                extra_context="",
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
            )

        return res  # pragma: no cover

    def rollout(
        self,
        llm: str | Path | None = None,
        *,
        mode: str = "smoke",
        require_samples: bool = False,
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> RolloutCallResult:
        if llm is not None:
            self._set_llm(llm)
        if not self.llm_kind:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_dir|model_id first")

        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            rollout_dir = (run_root / f"rollout_attempt_{attempt+1:02d}").resolve()
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
                    failed_stage="deploy",
                    deploy_artifacts_dir=deploy_dir,
                    rollout_eval_artifacts_dir=rollout_dir,
                    command_hints=self.command_hints,
                    extra_context="",
                    timeout_seconds=int(self.opencode_timeout_seconds or 300),
                )
                continue

            self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
            overrides.update(with_runtime_env_path(self.runtime_env_path))

            rollout_res = _rollout(
                self.env,
                artifacts_dir=rollout_dir,
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            if rollout_res.ok and bool(require_samples):
                ok_samples, reason = _validate_rollout_samples(self.env.repo, rollout_res.rollout_path)
                if not ok_samples:
                    try:
                        (rollout_dir / "rollout_contract_error.txt").write_text(str(reason) + "\n", encoding="utf-8")
                    except Exception:
                        pass
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
                command_hints=self.command_hints,
                extra_context=("rollout_contract_invalid: missing_samples_jsonl" if require_samples else ""),
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
            )

        return rollout_res  # pragma: no cover

    def evaluation(
        self,
        *,
        mode: str = "smoke",
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> EvaluationCallResult:
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
                        failed_stage="deploy",
                        deploy_artifacts_dir=deploy_dir,
                        rollout_eval_artifacts_dir=roll_eval_dir,
                        command_hints=self.command_hints,
                        extra_context="",
                        timeout_seconds=int(self.opencode_timeout_seconds or 300),
                    )
                    continue

                self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
                overrides.update(with_runtime_env_path(self.runtime_env_path))
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
                command_hints=self.command_hints,
                extra_context=combined,
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
            )

        return eval_res  # pragma: no cover

    def rollout_and_evaluation(
        self,
        llm: str | Path | None = None,
        *,
        mode: str = "smoke",
        require_samples: bool = False,
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
        repair_iters: int = 3,
    ) -> tuple[RolloutCallResult, EvaluationCallResult]:
        if llm is not None:
            self._set_llm(llm)
        if not self.llm_kind:
            raise ValueError("missing_llm: call deploy/rollout with llm=model_dir|model_id first")

        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)
        overrides = self._base_overrides(mode=mode, extra=env_overrides)
        self._apply_llm_overrides(overrides)

        last_rollout: RolloutCallResult | None = None
        last_eval: EvaluationCallResult | None = None

        for attempt in range(int(max(0, repair_iters)) + 1):
            deploy_dir = (run_root / f"deploy_attempt_{attempt+1:02d}").resolve()
            roll_eval_dir = (run_root / f"rollout_evaluation_attempt_{attempt+1:02d}").resolve()

            deploy_res = _deploy(
                self.env,
                artifacts_dir=deploy_dir,
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            if not deploy_res.ok:
                last_rollout = RolloutCallResult(ok=False, artifacts_dir=roll_eval_dir, rollout_path=None, verify=deploy_res.verify)
                last_eval = EvaluationCallResult(ok=False, artifacts_dir=roll_eval_dir, metrics_path=None, metrics=None, verify=deploy_res.verify)
                if attempt >= int(max(0, repair_iters)):
                    return last_rollout, last_eval
                repair_contract(
                    repo=self.env.repo,
                    model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                    opencode_url=str(self.opencode_url or ""),
                    unattended=str(self.unattended or "strict"),
                    artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                    failed_stage="deploy",
                    deploy_artifacts_dir=deploy_dir,
                    rollout_eval_artifacts_dir=roll_eval_dir,
                    command_hints=self.command_hints,
                    extra_context="",
                    timeout_seconds=int(self.opencode_timeout_seconds or 300),
                )
                continue

            self.runtime_env_path = deploy_res.runtime_env_path or (self.env.repo / ".aider_fsm" / "runtime_env.json").resolve()
            overrides2 = dict(overrides)
            overrides2.update(with_runtime_env_path(self.runtime_env_path))

            rollout_res, eval_res = _rollout_and_evaluate(
                self.env,
                artifacts_dir=roll_eval_dir,
                env_overrides=overrides2,
                unattended=str(self.unattended or "strict"),
                run_bootstrap_first=True,
            )
            combined = ""
            if not eval_res.ok:
                combined = _verification_errors_summary(eval_res.verify)
            if eval_res.ok and bool(self.require_metrics) and isinstance(eval_res.metrics, dict) and eval_res.metrics.get("ok") is not True:
                combined = "metrics.ok_not_true"
                try:
                    (roll_eval_dir / "metrics_contract_error.txt").write_text("metrics.ok_not_true\n", encoding="utf-8")
                except Exception:
                    pass
                eval_res = EvaluationCallResult(
                    ok=False,
                    artifacts_dir=eval_res.artifacts_dir,
                    metrics_path=eval_res.metrics_path,
                    metrics=eval_res.metrics,
                    verify=eval_res.verify,
                )
            if rollout_res.ok and eval_res.ok and bool(self.require_metrics) and self._audit_mode() != "off":
                audit_issue = audit_eval_script_for_hardcoded_nonzero_score(self.env.repo)
                audit_issue2 = audit_eval_script_has_real_execution(self.env.repo, extra_markers=self.hint_anchors)
                audit_issue3 = audit_eval_script_mentions_any_anchor(self.env.repo, self.hint_anchors)
                combined = "\n\n".join([x for x in (audit_issue, audit_issue2, audit_issue3) if x]).strip()
                if combined:
                    try:
                        (roll_eval_dir / "evaluation_audit_error.txt").write_text(combined + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    if self._audit_mode() != "warn-only":
                        eval_res = EvaluationCallResult(
                            ok=False,
                            artifacts_dir=eval_res.artifacts_dir,
                            metrics_path=eval_res.metrics_path,
                            metrics=eval_res.metrics,
                            verify=eval_res.verify,
                        )
            if rollout_res.ok and bool(require_samples):
                ok_samples, reason = _validate_rollout_samples(self.env.repo, rollout_res.rollout_path)
                if not ok_samples:
                    try:
                        (roll_eval_dir / "rollout_contract_error.txt").write_text(str(reason) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    rollout_res = RolloutCallResult(
                        ok=False,
                        artifacts_dir=rollout_res.artifacts_dir,
                        rollout_path=rollout_res.rollout_path,
                        verify=rollout_res.verify,
                    )
            last_rollout, last_eval = rollout_res, eval_res
            if rollout_res.ok and eval_res.ok:
                return rollout_res, eval_res

            if attempt >= int(max(0, repair_iters)):
                return rollout_res, eval_res

            self._maybe_teardown(run_root=run_root / f"teardown_attempt_{attempt+1:02d}", overrides=overrides2)
            repair_contract(
                repo=self.env.repo,
                model=str(self.opencode_repair_model or self.opencode_model or "").strip(),
                opencode_url=str(self.opencode_url or ""),
                unattended=str(self.unattended or "strict"),
                artifacts_dir=(run_root / f"repair_{attempt+1:02d}").resolve(),
                failed_stage=("rollout" if not rollout_res.ok else "evaluation"),
                deploy_artifacts_dir=deploy_dir,
                rollout_eval_artifacts_dir=roll_eval_dir,
                command_hints=self.command_hints,
                extra_context=(
                    combined
                    or ("rollout_contract_invalid: missing_samples_jsonl" if (require_samples and not rollout_res.ok) else "")
                ),
                timeout_seconds=int(self.opencode_timeout_seconds or 300),
            )

        assert last_rollout is not None and last_eval is not None  # pragma: no cover
        return last_rollout, last_eval

    def teardown(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
        artifacts_dir: Path | None = None,
    ) -> bool:
        overrides = self._base_overrides(mode=str((env_overrides or {}).get("AIDER_EVAL_MODE") or "smoke"), extra=env_overrides)
        run_root = _resolve_run_root(self.env.repo, run_id=self.run_id, artifacts_dir=artifacts_dir)
        return bool(
            _deploy_teardown(
                self.env,
                artifacts_dir=(run_root / "deploy_teardown").resolve(),
                env_overrides=overrides,
                unattended=str(self.unattended or "strict"),
            )
        )


_DEFAULT_SESSION: EnvSession | None = None


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
    opencode_bash: str = "restricted",
    scaffold_opencode_bash: str = "full",
    strict_opencode: bool = True,
    artifacts_dir: Path | None = None,
) -> EnvSession:
    """Open an environment handle for a target repo/url and ensure a runnable contract exists.

    中文说明：
    - 含义：你训练脚本的入口：支持本地路径、git URL、HF dataset URL。
    - 内容：若缺少 pipeline.yml，则通过 OpenCode scaffold 合同（pipeline.yml + `.aider_fsm/**`）。
    - 约束：本函数不包含任何 benchmark-specific hardcoding。
    """
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
        opencode_bash=str(opencode_bash or "restricted"),
        scaffold_opencode_bash=str(scaffold_opencode_bash or "full"),
        unattended=str(unattended or "strict"),
        artifacts_dir=artifacts_dir,
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
        audit=str(audit or "on"),
    )

    global _DEFAULT_SESSION
    _DEFAULT_SESSION = session
    return session


def _require_session(session: EnvSession | None) -> EnvSession:
    s = session or _DEFAULT_SESSION
    if s is None:
        raise RuntimeError("env_not_setup: call env.setup(...) first")
    return s


def deploy(
    llm: str | Path,
    *,
    session: EnvSession | None = None,
    mode: str = "smoke",
    env_overrides: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
    repair_iters: int = 3,
) -> DeployCallResult:
    return _require_session(session).deploy(
        llm,
        mode=mode,
        env_overrides=env_overrides,
        artifacts_dir=artifacts_dir,
        repair_iters=repair_iters,
    )


def rollout(
    llm: str | Path | None = None,
    *,
    session: EnvSession | None = None,
    mode: str = "smoke",
    require_samples: bool = False,
    env_overrides: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
    repair_iters: int = 3,
) -> RolloutCallResult:
    return _require_session(session).rollout(
        llm,
        mode=mode,
        require_samples=require_samples,
        env_overrides=env_overrides,
        artifacts_dir=artifacts_dir,
        repair_iters=repair_iters,
    )


def evaluation(
    *,
    session: EnvSession | None = None,
    mode: str = "smoke",
    env_overrides: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
    repair_iters: int = 3,
) -> EvaluationCallResult:
    return _require_session(session).evaluation(
        mode=mode,
        env_overrides=env_overrides,
        artifacts_dir=artifacts_dir,
        repair_iters=repair_iters,
    )


def rollout_and_evaluation(
    llm: str | Path | None = None,
    *,
    session: EnvSession | None = None,
    mode: str = "smoke",
    require_samples: bool = False,
    env_overrides: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
    repair_iters: int = 3,
) -> tuple[RolloutCallResult, EvaluationCallResult]:
    return _require_session(session).rollout_and_evaluation(
        llm,
        mode=mode,
        require_samples=require_samples,
        env_overrides=env_overrides,
        artifacts_dir=artifacts_dir,
        repair_iters=repair_iters,
    )


def teardown(
    *,
    session: EnvSession | None = None,
    env_overrides: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
) -> bool:
    return _require_session(session).teardown(env_overrides=env_overrides, artifacts_dir=artifacts_dir)


def current_session() -> EnvSession | None:
    return _DEFAULT_SESSION
