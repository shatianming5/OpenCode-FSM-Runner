from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

from .agent_client import AgentClient
from .bootstrap import run_bootstrap
from .opencode_client import OpenCodeClient
from .prompts import make_scaffold_contract_prompt
from .pipeline_spec import PipelineSpec, load_pipeline_spec
from .pipeline_verify import run_pipeline_verification
from .repo_resolver import prepare_repo
from .subprocess_utils import tail, write_text
from .types import VerificationResult


@dataclass(frozen=True)
class EnvHandle:
    repo: Path
    pipeline_path: Path
    pipeline: PipelineSpec


@dataclass(frozen=True)
class RolloutCallResult:
    ok: bool
    artifacts_dir: Path
    rollout_path: Path | None
    verify: VerificationResult


@dataclass(frozen=True)
class EvaluationCallResult:
    ok: bool
    artifacts_dir: Path
    metrics_path: Path | None
    metrics: dict | None
    verify: VerificationResult


def _default_artifacts_dir(repo: Path, *, prefix: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return (repo / ".aider_fsm" / "artifacts" / run_id / prefix).resolve()


def _list_opencode_models() -> list[str]:
    if not shutil.which("opencode"):
        return []
    try:
        res = subprocess.run(
            ["opencode", "models"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if res.returncode != 0:
        return []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    models: list[str] = []
    for raw in (res.stdout or "").splitlines():
        line = ansi.sub("", raw).strip()
        if not line or "/" not in line:
            continue
        models.append(line)
    return models


def _resolve_model(raw_model: str) -> str:
    s = str(raw_model or "").strip()
    if not s:
        candidates = _list_opencode_models()
        if "openai/gpt-4o-mini" in candidates:
            return "openai/gpt-4o-mini"
        if "opencode/gpt-5-nano" in candidates:
            return "opencode/gpt-5-nano"
        if candidates:
            return candidates[0]
        return "openai/gpt-4o-mini"
    if "/" in s:
        return s
    candidates = _list_opencode_models()
    matches = [m for m in candidates if m.split("/", 1)[1] == s]
    if matches:
        for m in matches:
            if m.startswith("myproxy/"):
                return m
        return matches[0]
    return f"openai/{s}"


def _validate_scaffolded_pipeline(p: PipelineSpec, *, require_metrics: bool) -> list[str]:
    missing: list[str] = []
    if p.security_max_cmd_seconds is None or int(p.security_max_cmd_seconds) <= 0:
        missing.append("security.max_cmd_seconds")

    if not require_metrics:
        return missing

    required = {"score"}
    eval_ok = (
        bool(list(p.evaluation_run_cmds or []))
        and bool(str(p.evaluation_metrics_path or "").strip())
        and required.issubset(set(p.evaluation_required_keys or []))
    )
    bench_ok = (
        bool(list(p.benchmark_run_cmds or []))
        and bool(str(p.benchmark_metrics_path or "").strip())
        and required.issubset(set(p.benchmark_required_keys or []))
    )
    if eval_ok or bench_ok:
        return missing

    eval_touched = bool(
        (p.evaluation_run_cmds or []) or str(p.evaluation_metrics_path or "").strip() or (p.evaluation_required_keys or [])
    )
    bench_touched = bool(
        (p.benchmark_run_cmds or []) or str(p.benchmark_metrics_path or "").strip() or (p.benchmark_required_keys or [])
    )

    if eval_touched or not bench_touched:
        if not list(p.evaluation_run_cmds or []):
            missing.append("evaluation.run_cmds")
        if not str(p.evaluation_metrics_path or "").strip():
            missing.append("evaluation.metrics_path")
        if not required.issubset(set(p.evaluation_required_keys or [])):
            missing.append("evaluation.required_keys (missing: score)")

    if bench_touched:
        if not list(p.benchmark_run_cmds or []):
            missing.append("benchmark.run_cmds")
        if not str(p.benchmark_metrics_path or "").strip():
            missing.append("benchmark.metrics_path")
        if not required.issubset(set(p.benchmark_required_keys or [])):
            missing.append("benchmark.required_keys (missing: score)")

    return missing


def open_env(
    repo: str | Path,
    *,
    clones_dir: Path | None = None,
    pipeline_rel: str = "pipeline.yml",
    require_pipeline: bool = True,
    scaffold_contract: str = "opencode",  # off|opencode
    scaffold_require_metrics: bool = True,
    model: str = "",
    opencode_url: str = "",
    opencode_timeout_seconds: int = 300,
    opencode_bash: str = "restricted",
    scaffold_opencode_bash: str = "full",
    unattended: str = "strict",
    artifacts_dir: Path | None = None,
    agent: AgentClient | None = None,
) -> EnvHandle:
    prepared = prepare_repo(str(repo), clones_dir=clones_dir)
    repo_root = prepared.repo.resolve()
    pipeline_path = (repo_root / str(pipeline_rel)).resolve()
    if not pipeline_path.exists():
        mode = str(scaffold_contract or "off").strip().lower() or "off"
        if mode not in ("off", "opencode"):
            mode = "off"
        if mode == "opencode":
            out_dir = (artifacts_dir or _default_artifacts_dir(repo_root, prefix="scaffold")).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            created_agent = False
            try:
                if agent is None:
                    created_agent = True
                    agent = OpenCodeClient(
                        repo=repo_root,
                        plan_rel="PLAN.md",
                        pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                        model=_resolve_model(model),
                        base_url=(str(opencode_url or "").strip() or None),
                        timeout_seconds=int(opencode_timeout_seconds or 300),
                        bash_mode=str(opencode_bash or "restricted"),
                        scaffold_bash_mode=str(scaffold_opencode_bash or "full"),
                        unattended=str(unattended or "strict"),
                        server_log_path=out_dir / "opencode_server.log",
                        session_title=f"{repo_root.name}:scaffold",
                        username=(str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode")
                        if str(opencode_url or "").strip()
                        else None,
                        password=(str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip() or None)
                        if str(opencode_url or "").strip()
                        else None,
                    )
                res = agent.run(
                    make_scaffold_contract_prompt(
                        repo_root,
                        pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                        require_metrics=bool(scaffold_require_metrics),
                    ),
                    fsm_state="S0_SCAFFOLD",
                    iter_idx=0,
                    purpose="scaffold_contract",
                )
                write_text(out_dir / "scaffold_agent_result.txt", tail(res.assistant_text or "", 20000) + "\n")
            finally:
                if created_agent and agent is not None:
                    try:
                        agent.close()
                    except Exception:
                        pass

            if not pipeline_path.exists():
                write_text(out_dir / "scaffold_error.txt", "scaffold_contract_failed: missing_pipeline_yml\n")
                raise RuntimeError(f"scaffold_contract_failed: pipeline not created: {pipeline_path}")

            parsed = load_pipeline_spec(pipeline_path)
            missing = _validate_scaffolded_pipeline(parsed, require_metrics=bool(scaffold_require_metrics))
            if missing:
                write_text(
                    out_dir / "scaffold_error.txt",
                    "scaffold_contract_failed: incomplete_pipeline_yml\n" + "\n".join([f"- {x}" for x in missing]) + "\n",
                )
                raise RuntimeError(f"scaffold_contract_failed: incomplete pipeline.yml: {missing}")
        elif require_pipeline:
            raise FileNotFoundError(f"pipeline not found: {pipeline_path}")
        else:
            pipeline = PipelineSpec()
            return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)
    pipeline = load_pipeline_spec(pipeline_path)
    return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)


def _merge_env(base: dict[str, str], overrides: dict[str, str] | None) -> dict[str, str]:
    out = dict(base or {})
    if overrides:
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


def _stage_only_pipeline(
    pipeline: PipelineSpec,
    *,
    rollout: bool,
    evaluation: bool,
    benchmark: bool,
    env_overrides: dict[str, str] | None,
) -> PipelineSpec:
    env_overrides = env_overrides or {}
    return PipelineSpec(
        # security
        security_mode=str(pipeline.security_mode or "safe"),
        security_allowlist=list(pipeline.security_allowlist or []),
        security_denylist=list(pipeline.security_denylist or []),
        security_max_cmd_seconds=pipeline.security_max_cmd_seconds,
        security_max_total_seconds=pipeline.security_max_total_seconds,
        # rollout
        rollout_run_cmds=list(pipeline.rollout_run_cmds or []) if rollout else [],
        rollout_timeout_seconds=pipeline.rollout_timeout_seconds if rollout else None,
        rollout_retries=int(pipeline.rollout_retries or 0) if rollout else 0,
        rollout_env=_merge_env(pipeline.rollout_env, env_overrides) if rollout else {},
        rollout_workdir=pipeline.rollout_workdir if rollout else None,
        # evaluation
        evaluation_run_cmds=list(pipeline.evaluation_run_cmds or []) if evaluation else [],
        evaluation_timeout_seconds=pipeline.evaluation_timeout_seconds if evaluation else None,
        evaluation_retries=int(pipeline.evaluation_retries or 0) if evaluation else 0,
        evaluation_env=_merge_env(pipeline.evaluation_env, env_overrides) if evaluation else {},
        evaluation_workdir=pipeline.evaluation_workdir if evaluation else None,
        evaluation_metrics_path=pipeline.evaluation_metrics_path if evaluation else None,
        evaluation_required_keys=list(pipeline.evaluation_required_keys or []) if evaluation else [],
        # benchmark (optional)
        benchmark_run_cmds=list(pipeline.benchmark_run_cmds or []) if benchmark else [],
        benchmark_timeout_seconds=pipeline.benchmark_timeout_seconds if benchmark else None,
        benchmark_retries=int(pipeline.benchmark_retries or 0) if benchmark else 0,
        benchmark_env=_merge_env(pipeline.benchmark_env, env_overrides) if benchmark else {},
        benchmark_workdir=pipeline.benchmark_workdir if benchmark else None,
        benchmark_metrics_path=pipeline.benchmark_metrics_path if benchmark else None,
        benchmark_required_keys=list(pipeline.benchmark_required_keys or []) if benchmark else [],
    )


def rollout(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
) -> RolloutCallResult:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout")).resolve()
    p = _stage_only_pipeline(env.pipeline, rollout=True, evaluation=False, benchmark=False, env_overrides=env_overrides)
    bootstrap_path = (env.repo / ".aider_fsm" / "bootstrap.yml").resolve()
    if run_bootstrap_first and bootstrap_path.exists():
        bootstrap_stage, applied_env = run_bootstrap(
            env.repo,
            bootstrap_path=bootstrap_path,
            pipeline=p,
            unattended=str(unattended or "strict"),
            artifacts_dir=artifacts_dir / "bootstrap",
        )
        if not bootstrap_stage.ok:
            verify = VerificationResult(ok=False, failed_stage="bootstrap", bootstrap=bootstrap_stage, metrics_errors=[])
            return RolloutCallResult(ok=False, artifacts_dir=artifacts_dir, rollout_path=None, verify=verify)
        old_values = {str(k): os.environ.get(str(k)) for k in (applied_env or {}).keys()}
        try:
            for k, v in (applied_env or {}).items():
                os.environ[str(k)] = str(v)
            verify = run_pipeline_verification(
                env.repo,
                pipeline=p,
                tests_cmds=["echo tests_skipped"],
                artifacts_dir=artifacts_dir,
                unattended=str(unattended or "strict"),
            )
            verify = replace(verify, bootstrap=bootstrap_stage)
        finally:
            for k, old in old_values.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    else:
        verify = run_pipeline_verification(
            env.repo,
            pipeline=p,
            tests_cmds=["echo tests_skipped"],
            artifacts_dir=artifacts_dir,
            unattended=str(unattended or "strict"),
        )
    rollout_path = (env.repo / ".aider_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        rollout_path = None
    return RolloutCallResult(ok=bool(verify.ok), artifacts_dir=artifacts_dir, rollout_path=rollout_path, verify=verify)


def evaluate(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
) -> EvaluationCallResult:
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, rollout=False, evaluation=True, benchmark=False, env_overrides=env_overrides)
    bootstrap_path = (env.repo / ".aider_fsm" / "bootstrap.yml").resolve()
    if run_bootstrap_first and bootstrap_path.exists():
        bootstrap_stage, applied_env = run_bootstrap(
            env.repo,
            bootstrap_path=bootstrap_path,
            pipeline=p,
            unattended=str(unattended or "strict"),
            artifacts_dir=artifacts_dir / "bootstrap",
        )
        if not bootstrap_stage.ok:
            verify = VerificationResult(ok=False, failed_stage="bootstrap", bootstrap=bootstrap_stage, metrics_errors=[])
            return EvaluationCallResult(
                ok=False, artifacts_dir=artifacts_dir, metrics_path=None, metrics=None, verify=verify
            )
        old_values = {str(k): os.environ.get(str(k)) for k in (applied_env or {}).keys()}
        try:
            for k, v in (applied_env or {}).items():
                os.environ[str(k)] = str(v)
            verify = run_pipeline_verification(
                env.repo,
                pipeline=p,
                tests_cmds=["echo tests_skipped"],
                artifacts_dir=artifacts_dir,
                unattended=str(unattended or "strict"),
            )
            verify = replace(verify, bootstrap=bootstrap_stage)
        finally:
            for k, old in old_values.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    else:
        verify = run_pipeline_verification(
            env.repo,
            pipeline=p,
            tests_cmds=["echo tests_skipped"],
            artifacts_dir=artifacts_dir,
            unattended=str(unattended or "strict"),
        )
    metrics_path = Path(str(verify.metrics_path)).resolve() if getattr(verify, "metrics_path", None) else None
    metrics = getattr(verify, "metrics", None)
    return EvaluationCallResult(
        ok=bool(verify.ok),
        artifacts_dir=artifacts_dir,
        metrics_path=metrics_path,
        metrics=metrics,
        verify=verify,
    )


def rollout_and_evaluate(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
) -> tuple[RolloutCallResult, EvaluationCallResult]:
    # One verification pass runs rollout then evaluation (and validates evaluation metrics).
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout_evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, rollout=True, evaluation=True, benchmark=False, env_overrides=env_overrides)
    bootstrap_path = (env.repo / ".aider_fsm" / "bootstrap.yml").resolve()
    if run_bootstrap_first and bootstrap_path.exists():
        bootstrap_stage, applied_env = run_bootstrap(
            env.repo,
            bootstrap_path=bootstrap_path,
            pipeline=p,
            unattended=str(unattended or "strict"),
            artifacts_dir=artifacts_dir / "bootstrap",
        )
        if not bootstrap_stage.ok:
            verify = VerificationResult(ok=False, failed_stage="bootstrap", bootstrap=bootstrap_stage, metrics_errors=[])
            return (
                RolloutCallResult(ok=False, artifacts_dir=artifacts_dir, rollout_path=None, verify=verify),
                EvaluationCallResult(ok=False, artifacts_dir=artifacts_dir, metrics_path=None, metrics=None, verify=verify),
            )
        old_values = {str(k): os.environ.get(str(k)) for k in (applied_env or {}).keys()}
        try:
            for k, v in (applied_env or {}).items():
                os.environ[str(k)] = str(v)
            verify = run_pipeline_verification(
                env.repo,
                pipeline=p,
                tests_cmds=["echo tests_skipped"],
                artifacts_dir=artifacts_dir,
                unattended=str(unattended or "strict"),
            )
            verify = replace(verify, bootstrap=bootstrap_stage)
        finally:
            for k, old in old_values.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    else:
        verify = run_pipeline_verification(
            env.repo,
            pipeline=p,
            tests_cmds=["echo tests_skipped"],
            artifacts_dir=artifacts_dir,
            unattended=str(unattended or "strict"),
        )

    rollout_path = (env.repo / ".aider_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        rollout_path = None
    metrics_path = Path(str(verify.metrics_path)).resolve() if getattr(verify, "metrics_path", None) else None
    metrics = getattr(verify, "metrics", None)

    return (
        RolloutCallResult(ok=bool(verify.ok), artifacts_dir=artifacts_dir, rollout_path=rollout_path, verify=verify),
        EvaluationCallResult(
            ok=bool(verify.ok),
            artifacts_dir=artifacts_dir,
            metrics_path=metrics_path,
            metrics=metrics,
            verify=verify,
        ),
    )


def with_env_vars(extra_env: dict[str, str]) -> dict[str, str]:
    """Convenience helper for callers that want to pass environment variables into stages.

    Example:
      rollout_and_evaluate(env, env_overrides=with_env_vars({"OPENCODE_MODEL": "opencode/gpt-5-nano"}))
    """

    return _merge_env({}, extra_env)
