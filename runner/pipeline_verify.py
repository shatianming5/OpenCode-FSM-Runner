from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

from .paths import resolve_workdir
from .pipeline_spec import PipelineSpec
from .security import cmd_allowed, looks_interactive, safe_env
from .subprocess_utils import (
    STDIO_TAIL_CHARS,
    read_text_if_exists,
    run_cmd_capture,
    tail,
    write_cmd_artifacts,
    write_json,
    write_text,
)
from .types import CmdResult, StageResult, VerificationResult


def stage_rc(stage: StageResult | None) -> int | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is None:
        return stage.results[-1].rc
    if 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index].rc
    return stage.results[-1].rc


def stage_failed_cmd(stage: StageResult | None) -> CmdResult | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is not None and 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index]
    return stage.results[-1]


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"failed_to_read: {e}"
    try:
        data = json.loads(raw)
    except Exception as e:
        return None, f"invalid_json: {e}"
    if not isinstance(data, dict):
        return None, "metrics_json_not_object"
    return data, None


def _validate_metrics(metrics: dict[str, Any], required_keys: list[str]) -> list[str]:
    missing: list[str] = []
    for k in required_keys:
        if k not in metrics:
            missing.append(k)
    return missing


def _dump_kubectl(
    out_dir: Path,
    repo: Path,
    *,
    namespace: str | None,
    label_selector: str | None,
    include_logs: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds: list[tuple[str, str]] = [
        ("kubectl_get_nodes", "kubectl get nodes -o wide"),
        ("kubectl_get_namespaces", "kubectl get namespaces"),
        ("kubectl_get_pods", "kubectl get pods -A -o wide"),
        ("kubectl_get_all", "kubectl get all -A -o wide"),
        ("kubectl_get_events", "kubectl get events -A --sort-by=.metadata.creationTimestamp"),
    ]
    for prefix, cmd in cmds:
        res = run_cmd_capture(cmd, repo, timeout_seconds=60)
        write_cmd_artifacts(out_dir, prefix, res)

    if include_logs and label_selector:
        if namespace:
            cmd = (
                f"kubectl logs -n {shlex.quote(namespace)} -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        else:
            cmd = (
                f"kubectl logs --all-namespaces -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        res = run_cmd_capture(cmd, repo, timeout_seconds=120)
        write_cmd_artifacts(out_dir, "kubectl_logs", res)


def _run_stage(
    repo: Path,
    *,
    stage: str,
    cmds: list[str],
    workdir: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
    retries: int,
    interactive: bool,
    unattended: str,
    pipeline: PipelineSpec | None,
    artifacts_dir: Path,
) -> StageResult:
    stage = stage.strip() or "stage"
    env2 = dict(env)
    env2["AIDER_FSM_STAGE"] = stage
    env2["AIDER_FSM_ARTIFACTS_DIR"] = str(artifacts_dir.resolve())
    env2["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())
    results: list[CmdResult] = []
    started = time.monotonic()

    for cmd_idx, raw_cmd in enumerate(cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
            continue

        for attempt in range(1, int(retries) + 2):
            if pipeline and pipeline.security_max_total_seconds:
                elapsed = time.monotonic() - started
                if elapsed > float(pipeline.security_max_total_seconds):
                    res = CmdResult(
                        cmd=cmd,
                        rc=124,
                        stdout="",
                        stderr=f"max_total_seconds_exceeded: {pipeline.security_max_total_seconds}",
                        timed_out=True,
                    )
                    results.append(res)
                    failed_index = len(results) - 1
                    write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)
                    write_cmd_artifacts(artifacts_dir, stage, res)
                    write_json(
                        artifacts_dir / f"{stage}_summary.json",
                        {"ok": False, "failed_index": failed_index, "total_results": len(results)},
                    )
                    return StageResult(ok=False, results=results, failed_index=failed_index)

            allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
            if not allowed:
                res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            elif unattended == "strict" and looks_interactive(cmd):
                res = CmdResult(
                    cmd=cmd,
                    rc=126,
                    stdout="",
                    stderr="likely_interactive_command_disallowed_in_strict_mode",
                    timed_out=False,
                )
            else:
                eff_timeout = timeout_seconds
                if pipeline and pipeline.security_max_cmd_seconds:
                    eff_timeout = (
                        int(pipeline.security_max_cmd_seconds)
                        if eff_timeout is None
                        else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
                    )
                res = run_cmd_capture(
                    cmd,
                    workdir,
                    timeout_seconds=eff_timeout,
                    env=env2,
                    interactive=bool(interactive and unattended == "guided"),
                )

            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)

            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, stage, results[-1])
            write_json(
                artifacts_dir / f"{stage}_summary.json",
                {"ok": False, "failed_index": failed_index, "total_results": len(results)},
            )
            return StageResult(ok=False, results=results, failed_index=failed_index)

    if results:
        write_cmd_artifacts(artifacts_dir, stage, results[-1])
    write_json(
        artifacts_dir / f"{stage}_summary.json",
        {"ok": True, "failed_index": None, "total_results": len(results)},
    )
    return StageResult(ok=True, results=results, failed_index=None)


def run_pipeline_verification(
    repo: Path,
    *,
    pipeline: PipelineSpec | None,
    tests_cmds: list[str],
    artifacts_dir: Path,
    unattended: str = "strict",
) -> VerificationResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ok = False
    failed_stage: str | None = None
    auth_res: StageResult | None = None
    tests_res: StageResult | None = None
    deploy_setup_res: StageResult | None = None
    deploy_health_res: StageResult | None = None
    rollout_res: StageResult | None = None
    eval_res: StageResult | None = None
    bench_res: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] = []

    teardown_cmds = list(pipeline.deploy_teardown_cmds or []) if pipeline else []
    teardown_policy = (pipeline.deploy_teardown_policy if pipeline else "never").lower()
    kubectl_dump_enabled = bool(pipeline and pipeline.kubectl_dump_enabled)

    def _teardown_allowed(success: bool) -> bool:
        if not teardown_cmds:
            return False
        if teardown_policy == "never":
            return False
        if teardown_policy == "always":
            return True
        if teardown_policy == "on_success":
            return success
        if teardown_policy == "on_failure":
            return not success
        return False

    env_base = dict(os.environ)

    def _workdir_or_fail(stage: str, raw: str | None) -> tuple[Path, StageResult | None]:
        try:
            return resolve_workdir(repo, raw), None
        except Exception as e:
            err = CmdResult(cmd=f"resolve_workdir {raw}", rc=2, stdout="", stderr=str(e), timed_out=False)
            write_cmd_artifacts(artifacts_dir, f"{stage}_workdir_error", err)
            return repo, StageResult(ok=False, results=[err], failed_index=0)

    try:
        if pipeline and pipeline.auth_cmds:
            auth_env = safe_env(env_base, pipeline.auth_env, unattended=unattended)
            auth_workdir, auth_wd_err = _workdir_or_fail("auth", pipeline.auth_workdir)
            if auth_wd_err is not None:
                failed_stage = "auth"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_wd_err,
                    metrics_errors=metrics_errors,
                )
            auth_res = _run_stage(
                repo,
                stage="auth",
                cmds=pipeline.auth_cmds,
                workdir=auth_workdir,
                env=auth_env,
                timeout_seconds=pipeline.auth_timeout_seconds,
                retries=pipeline.auth_retries,
                interactive=bool(pipeline.auth_interactive),
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not auth_res.ok:
                failed_stage = "auth"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    metrics_errors=metrics_errors,
                )

        tests_env = safe_env(env_base, pipeline.tests_env if pipeline else {}, unattended=unattended)
        tests_workdir, tests_wd_err = _workdir_or_fail("tests", pipeline.tests_workdir if pipeline else None)
        if tests_wd_err is not None:
            failed_stage = "tests"
            return VerificationResult(
                ok=False,
                failed_stage=failed_stage,
                auth=auth_res,
                tests=tests_wd_err,
                metrics_errors=metrics_errors,
            )
        tests_timeout = pipeline.tests_timeout_seconds if pipeline else None
        tests_retries = pipeline.tests_retries if pipeline else 0
        tests_res = _run_stage(
            repo,
            stage="tests",
            cmds=tests_cmds,
            workdir=tests_workdir,
            env=tests_env,
            timeout_seconds=tests_timeout,
            retries=tests_retries,
            interactive=False,
            unattended=unattended,
            pipeline=pipeline,
            artifacts_dir=artifacts_dir,
        )
        if not tests_res.ok:
            failed_stage = "tests"
            return VerificationResult(
                ok=False,
                failed_stage=failed_stage,
                auth=auth_res,
                tests=tests_res,
                metrics_errors=metrics_errors,
            )

        if pipeline and pipeline.deploy_setup_cmds:
            deploy_env = safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, deploy_wd_err = _workdir_or_fail("deploy_setup", pipeline.deploy_workdir)
            if deploy_wd_err is not None:
                failed_stage = "deploy_setup"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_wd_err,
                    metrics_errors=metrics_errors,
                )
            deploy_timeout = pipeline.deploy_timeout_seconds
            deploy_setup_res = _run_stage(
                repo,
                stage="deploy_setup",
                cmds=pipeline.deploy_setup_cmds,
                workdir=deploy_workdir,
                env=deploy_env,
                timeout_seconds=deploy_timeout,
                retries=pipeline.deploy_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not deploy_setup_res.ok:
                failed_stage = "deploy_setup"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    metrics_errors=metrics_errors,
                )

        if pipeline and pipeline.deploy_health_cmds:
            deploy_env = safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, deploy_wd_err = _workdir_or_fail("deploy_health", pipeline.deploy_workdir)
            if deploy_wd_err is not None:
                failed_stage = "deploy_health"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_wd_err,
                    metrics_errors=metrics_errors,
                )
            deploy_timeout = pipeline.deploy_timeout_seconds
            deploy_health_res = _run_stage(
                repo,
                stage="deploy_health",
                cmds=pipeline.deploy_health_cmds,
                workdir=deploy_workdir,
                env=deploy_env,
                timeout_seconds=deploy_timeout,
                retries=pipeline.deploy_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not deploy_health_res.ok:
                failed_stage = "deploy_health"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    metrics_errors=metrics_errors,
                )

        if pipeline and pipeline.rollout_run_cmds:
            rollout_env = safe_env(env_base, pipeline.rollout_env, unattended=unattended)
            rollout_workdir, rollout_wd_err = _workdir_or_fail("rollout", pipeline.rollout_workdir)
            if rollout_wd_err is not None:
                failed_stage = "rollout"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_wd_err,
                    metrics_errors=metrics_errors,
                )
            rollout_timeout = pipeline.rollout_timeout_seconds
            rollout_res = _run_stage(
                repo,
                stage="rollout",
                cmds=pipeline.rollout_run_cmds,
                workdir=rollout_workdir,
                env=rollout_env,
                timeout_seconds=rollout_timeout,
                retries=pipeline.rollout_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not rollout_res.ok:
                failed_stage = "rollout"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    metrics_errors=metrics_errors,
                )

        if pipeline and pipeline.evaluation_run_cmds:
            eval_env = safe_env(env_base, pipeline.evaluation_env, unattended=unattended)
            eval_workdir, eval_wd_err = _workdir_or_fail("evaluation", pipeline.evaluation_workdir)
            if eval_wd_err is not None:
                failed_stage = "evaluation"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_wd_err,
                    metrics_errors=metrics_errors,
                )
            eval_timeout = pipeline.evaluation_timeout_seconds
            eval_res = _run_stage(
                repo,
                stage="evaluation",
                cmds=pipeline.evaluation_run_cmds,
                workdir=eval_workdir,
                env=eval_env,
                timeout_seconds=eval_timeout,
                retries=pipeline.evaluation_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not eval_res.ok:
                failed_stage = "evaluation"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    metrics_errors=metrics_errors,
                )

        if pipeline and pipeline.evaluation_metrics_path:
            mpath = Path(pipeline.evaluation_metrics_path).expanduser()
            if not mpath.is_absolute():
                mpath = repo / mpath
            eval_metrics_path = str(mpath)
            if not mpath.exists():
                failed_stage = "metrics"
                metrics_errors.append(f"evaluation.metrics_file_missing: {mpath}")
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    benchmark=bench_res,
                    metrics_path=eval_metrics_path,
                    metrics_errors=metrics_errors,
                )

            # Always snapshot the produced metrics file for reproducibility, even if it is invalid JSON.
            write_text(artifacts_dir / "metrics_evaluation.json", read_text_if_exists(mpath))
            if metrics_path is None:
                write_text(artifacts_dir / "metrics.json", read_text_if_exists(mpath))

            eval_metrics, err = _read_json(mpath)
            if err:
                failed_stage = "metrics"
                metrics_errors.append(f"evaluation.{err}")
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    benchmark=bench_res,
                    metrics_path=eval_metrics_path,
                    metrics=eval_metrics,
                    metrics_errors=metrics_errors,
                )

            missing = _validate_metrics(eval_metrics or {}, pipeline.evaluation_required_keys)
            if missing:
                failed_stage = "metrics"
                metrics_errors.append("evaluation.missing_keys: " + ", ".join(missing))
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    benchmark=bench_res,
                    metrics_path=eval_metrics_path,
                    metrics=eval_metrics,
                    metrics_errors=metrics_errors,
                )

            # Prefer evaluation metrics as the "primary" metrics payload.
            if metrics_path is None:
                metrics_path = eval_metrics_path
                metrics = eval_metrics

        if pipeline and pipeline.benchmark_run_cmds:
            bench_env = safe_env(env_base, pipeline.benchmark_env, unattended=unattended)
            bench_workdir, bench_wd_err = _workdir_or_fail("benchmark", pipeline.benchmark_workdir)
            if bench_wd_err is not None:
                failed_stage = "benchmark"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    benchmark=bench_wd_err,
                    metrics_errors=metrics_errors,
                )
            bench_timeout = pipeline.benchmark_timeout_seconds
            bench_res = _run_stage(
                repo,
                stage="benchmark",
                cmds=pipeline.benchmark_run_cmds,
                workdir=bench_workdir,
                env=bench_env,
                timeout_seconds=bench_timeout,
                retries=pipeline.benchmark_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not bench_res.ok:
                failed_stage = "benchmark"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    rollout=rollout_res,
                    evaluation=eval_res,
                    benchmark=bench_res,
                    metrics_errors=metrics_errors,
                )

            if pipeline.benchmark_metrics_path:
                mpath = Path(pipeline.benchmark_metrics_path).expanduser()
                if not mpath.is_absolute():
                    mpath = repo / mpath
                bench_metrics_path = str(mpath)
                if not mpath.exists():
                    failed_stage = "metrics"
                    metrics_errors.append(f"metrics_file_missing: {mpath}")
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        rollout=rollout_res,
                        evaluation=eval_res,
                        benchmark=bench_res,
                        metrics_path=bench_metrics_path,
                        metrics_errors=metrics_errors,
                    )

                # Always snapshot the produced metrics file for reproducibility, even if it is invalid JSON.
                write_text(artifacts_dir / "metrics_benchmark.json", read_text_if_exists(mpath))
                if metrics_path is None:
                    write_text(artifacts_dir / "metrics.json", read_text_if_exists(mpath))

                bench_metrics, err = _read_json(mpath)
                if err:
                    failed_stage = "metrics"
                    metrics_errors.append(err)
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        rollout=rollout_res,
                        evaluation=eval_res,
                        benchmark=bench_res,
                        metrics_path=bench_metrics_path,
                        metrics=bench_metrics,
                        metrics_errors=metrics_errors,
                    )

                missing = _validate_metrics(bench_metrics or {}, pipeline.benchmark_required_keys)
                if missing:
                    failed_stage = "metrics"
                    metrics_errors.append("missing_keys: " + ", ".join(missing))
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        rollout=rollout_res,
                        evaluation=eval_res,
                        benchmark=bench_res,
                        metrics_path=bench_metrics_path,
                        metrics=bench_metrics,
                        metrics_errors=metrics_errors,
                    )

                # Back-compat: if no evaluation metrics were produced, use benchmark metrics as primary.
                if metrics_path is None:
                    metrics_path = bench_metrics_path
                    metrics = bench_metrics

        ok = True
        return VerificationResult(
            ok=True,
            failed_stage=None,
            auth=auth_res,
            tests=tests_res,
            deploy_setup=deploy_setup_res,
            deploy_health=deploy_health_res,
            rollout=rollout_res,
            evaluation=eval_res,
            benchmark=bench_res,
            metrics_path=metrics_path,
            metrics=metrics,
            metrics_errors=metrics_errors,
        )
    finally:
        if kubectl_dump_enabled:
            _dump_kubectl(
                artifacts_dir / "kubectl",
                repo,
                namespace=(pipeline.kubectl_dump_namespace if pipeline else None),
                label_selector=(pipeline.kubectl_dump_label_selector if pipeline else None),
                include_logs=bool(pipeline and pipeline.kubectl_dump_include_logs),
            )

        if pipeline and _teardown_allowed(ok):
            deploy_env = safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, td_wd_err = _workdir_or_fail("deploy_teardown", pipeline.deploy_workdir)
            if td_wd_err is not None:
                write_text(artifacts_dir / "deploy_teardown_warning.txt", "skip teardown due to invalid workdir\n")
            else:
                td = _run_stage(
                    repo,
                    stage="deploy_teardown",
                    cmds=teardown_cmds,
                    workdir=deploy_workdir,
                    env=deploy_env,
                    timeout_seconds=pipeline.deploy_timeout_seconds,
                    retries=0,
                    interactive=False,
                    unattended=unattended,
                    pipeline=pipeline,
                    artifacts_dir=artifacts_dir,
                )
                # keep the most recent command output for backwards-compatible filenames
                if td.results:
                    write_cmd_artifacts(artifacts_dir, "deploy_teardown", td.results[-1])


def fmt_stage_tail(prefix: str, stage: StageResult | None) -> str:
    res = stage_failed_cmd(stage)
    if res is None:
        return ""
    return (
        f"[{prefix}_RC]\n{res.rc}\n\n"
        f"[{prefix}_STDOUT_TAIL]\n{tail(res.stdout, STDIO_TAIL_CHARS)}\n\n"
        f"[{prefix}_STDERR_TAIL]\n{tail(res.stderr, STDIO_TAIL_CHARS)}\n\n"
    )
