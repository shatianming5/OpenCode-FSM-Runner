from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .paths import resolve_workdir
from .pipeline_spec import PipelineSpec
from .security import audit_bash_script, cmd_allowed, looks_interactive, safe_env
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
from ._util import _is_truthy, _parse_json_str_list


def stage_rc(stage: StageResult | None) -> int | None:
    """中文说明：
    - 含义：提取某个 stage 的“代表性返回码”（用于 summary/state）。
    - 内容：优先取 failed_index 对应命令的 rc；否则取最后一次执行的 rc；若无结果则返回 None。
    - 可简略：可能（纯辅助函数；但集中实现避免边界处理分散）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈13 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:33；类型=function；引用≈1；规模≈13行
    if stage is None or not stage.results:
        return None
    if stage.failed_index is None:
        return stage.results[-1].rc
    if 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index].rc
    return stage.results[-1].rc


def stage_failed_cmd(stage: StageResult | None) -> CmdResult | None:
    """中文说明：
    - 含义：获取某个 stage 的“失败命令”或“最后命令”的 CmdResult。
    - 内容：如果 failed_index 存在则返回对应结果；否则返回最后一次结果；用于展示错误尾部与生成修复 prompt。
    - 可简略：可能（小工具；但使得错误输出逻辑统一）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈11 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:48；类型=function；引用≈2；规模≈11行
    if stage is None or not stage.results:
        return None
    if stage.failed_index is not None and 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index]
    return stage.results[-1]


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """中文说明：
    - 含义：读取并解析 metrics JSON 文件。
    - 内容：要求是 JSON object（dict）；失败时返回 (None, 错误原因字符串)。
    - 可简略：可能（内部 helper；但把错误信息标准化很有用）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈17 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:61；类型=function；引用≈10；规模≈17行
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
    """中文说明：
    - 含义：校验 metrics dict 是否包含 required_keys。
    - 内容：返回缺失 key 列表；为空表示通过。
    - 可简略：可能（非常小的 helper；但逻辑集中更清晰）。
    """
    # 作用：中文说明：
    # 能否简略：是
    # 原因：规模≈11 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/pipeline_verify.py:80；类型=function；引用≈3；规模≈11行
    missing: list[str] = []
    for k in required_keys:
        if k not in metrics:
            missing.append(k)
    return missing


def _validate_hints_used(repo: Path, *, expected_anchors: list[str]) -> tuple[bool, str]:
    """Validate `.aider_fsm/hints_used.json` when hint execution is required."""
    # 作用：Validate `.aider_fsm/hints_used.json` when hint execution is required.
    # 能否简略：部分
    # 原因：规模≈31 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/pipeline_verify.py:113；类型=function；引用≈2；规模≈31行
    repo = Path(repo).resolve()
    path = (repo / ".aider_fsm" / "hints_used.json").resolve()
    if not path.exists():
        return False, f"missing_hints_used_json: {path}"
    data, err = _read_json(path)
    if err:
        return False, f"hints_used_json_{err}"
    assert isinstance(data, dict)  # _read_json guarantees dict on success
    if data.get("ok") is not True:
        return False, "hints_used.ok_not_true"

    used = data.get("used_anchors")
    if not isinstance(used, list) or not used:
        return False, "hints_used.used_anchors_missing_or_empty"
    used_clean = [str(x).strip() for x in used if isinstance(x, str) and str(x).strip()]
    if not used_clean:
        return False, "hints_used.used_anchors_invalid"

    exp = [str(x).strip() for x in (expected_anchors or []) if str(x).strip()]
    if exp:
        if not any(u in exp for u in used_clean):
            return False, "hints_used.no_expected_anchor"

    commands = data.get("commands")
    if commands is not None:
        if not isinstance(commands, list) or not any(isinstance(x, str) and x.strip() for x in commands):
            return False, "hints_used.commands_invalid"

    return True, "ok"


def _validate_hints_run(repo: Path) -> tuple[bool, str]:
    """Validate `.aider_fsm/hints_run.json` when a real (parseable) score is required."""
    # 作用：Validate `.aider_fsm/hints_run.json` when a real (parseable) score is required.
    # 能否简略：部分
    # 原因：规模≈35 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/pipeline_verify.py:146；类型=function；引用≈2；规模≈35行
    repo = Path(repo).resolve()
    path = (repo / ".aider_fsm" / "hints_run.json").resolve()
    if not path.exists():
        return False, f"missing_hints_run_json: {path}"
    data, err = _read_json(path)
    if err:
        return False, f"hints_run_json_{err}"
    assert isinstance(data, dict)  # _read_json guarantees dict on success

    if data.get("ok") is not True:
        return False, "hints_run.ok_not_true"

    chosen = data.get("chosen_command")
    if not isinstance(chosen, str) or not chosen.strip():
        return False, "hints_run.chosen_command_missing_or_empty"

    executed = data.get("executed_attempts")
    try:
        executed_i = int(executed)
    except Exception:
        executed_i = 0
    if executed_i <= 0:
        return False, "hints_run.executed_attempts_not_positive"

    score = data.get("score")
    try:
        score_f = float(score)
    except Exception:
        return False, "hints_run.score_invalid"
    if score_f < 0.0 or score_f > 1.0:
        return False, "hints_run.score_out_of_range"

    return True, "ok"


def _dump_kubectl(
    out_dir: Path,
    repo: Path,
    *,
    namespace: str | None,
    label_selector: str | None,
    include_logs: bool,
) -> None:
    """中文说明：
    - 含义：在 deploy 阶段结束后导出 k8s 调试信息（可选）。
    - 内容：执行一组 `kubectl get ...` 并写入 artifacts；可按 label_selector 导出相关 pod 的日志（tail=2000）。
    - 可简略：是（纯增强诊断能力；不影响核心闭环）。
    """
    # 作用：中文说明：
    # 能否简略：部分
    # 原因：规模≈38 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/pipeline_verify.py:194；类型=function；引用≈2；规模≈38行
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
    """中文说明：
    - 含义：执行某个 stage 的一组命令，并把每次尝试的输出写入 artifacts。
    - 内容：为每个命令支持 retries；应用安全策略（cmd_allowed/looks_interactive）；应用 per-cmd 与 per-stage 总超时；落盘 `<stage>_cmdXX_tryYY_*` 与 `<stage>_summary.json`。
    - 可简略：否（这是 pipeline 执行器的核心；与审计/安全/重试/超时强绑定）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈106 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:240；类型=function；引用≈9；规模≈106行
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
                ok, audit_reason = audit_bash_script(cmd, repo=repo, workdir=workdir, pipeline=pipeline)
                if not ok:
                    res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=audit_reason or "blocked_by_script_audit", timed_out=False)
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
    """中文说明：
    - 含义：按 pipeline 合同执行一次完整的验收（verification）。
    - 内容：
      - 顺序：auth(可选) → tests → deploy.setup(可选) → deploy.health(可选) → rollout(可选) → evaluation(可选) → benchmark(可选)
      - metrics：若配置了 evaluation/benchmark 的 metrics_path + required_keys，则读取并校验 JSON object 的 key 是否齐全
      - artifacts：为每个 stage 写入 cmd/stdout/stderr/result/summary，并在需要时执行 deploy_teardown 与 kubectl dump
    - 可简略：否（runner 的“验收契约”核心入口；删改会影响行为与兼容性）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈675 行；引用次数≈29（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:345；类型=function；引用≈29；规模≈675行
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
        """中文说明：
        - 含义：根据 teardown_policy 判断是否允许执行 deploy teardown。
        - 内容：综合 teardown_cmds 是否为空以及 `always/on_success/on_failure/never` 策略做布尔判断。
        - 可简略：是（纯 helper；可内联到调用点）。
        """
        # 作用：中文说明：
        # 能否简略：是
        # 原因：规模≈17 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/pipeline_verify.py:369；类型=function；引用≈2；规模≈17行
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
    runner_root = Path(__file__).resolve().parents[1]
    env_base.setdefault("AIDER_FSM_RUNNER_ROOT", str(runner_root))
    env_base.setdefault("AIDER_FSM_PYTHON", sys.executable)
    # Ensure the runner repo root is first on PYTHONPATH.
    #
    # Why: some environments (or target repos) may already have a `.../runner` path
    # earlier on sys.path, which can cause Python to import `runner/runner.py` as a
    # top-level module named `runner` (breaking relative imports). Prepending the
    # runner repo root avoids that collision.
    existing_pp = str(env_base.get("PYTHONPATH") or "")
    parts = [p for p in existing_pp.split(os.pathsep) if p]
    root_s = str(runner_root)
    parts = [p for p in parts if p != root_s]
    env_base["PYTHONPATH"] = root_s + (os.pathsep + os.pathsep.join(parts) if parts else "")

    # Generic timeout overrides (no benchmark-specific logic).
    #
    # Why: Some "full" evaluations (doc-hinted commands) can legitimately take >2h.
    # Let callers extend caps without editing the repo-owned `pipeline.yml` contract.
    #
    # Sources: allow both the *base* environment and per-stage env injections (env_overrides)
    # since programmatic callers often pass overrides via `pipeline.*_env`.
    def _env_get(name: str) -> str | None:
        # 作用：内部符号：run_pipeline_verification._env_get
        # 能否简略：是
        # 原因：规模≈18 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/pipeline_verify.py:405；类型=function；引用≈2；规模≈18行
        v = env_base.get(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if pipeline is None:
            return None
        for m in (
            pipeline.auth_env,
            pipeline.tests_env,
            pipeline.deploy_env,
            pipeline.rollout_env,
            pipeline.evaluation_env,
            pipeline.benchmark_env,
        ):
            vv = m.get(name) if isinstance(m, dict) else None
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
        return None

    def _env_int(name: str) -> int | None:
        # 作用：内部符号：run_pipeline_verification._env_int
        # 能否简略：部分
        # 原因：规模≈9 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/pipeline_verify.py:424；类型=function；引用≈5；规模≈9行
        raw = _env_get(name)
        if raw is None:
            return None
        try:
            n = int(str(raw).strip())
        except Exception:
            return None
        return n if n > 0 else None

    if pipeline is not None:
        max_cmd = _env_int("AIDER_FSM_MAX_CMD_SECONDS")
        max_total = _env_int("AIDER_FSM_MAX_TOTAL_SECONDS")
        if max_cmd is not None or max_total is not None:
            pipeline = replace(
                pipeline,
                security_max_cmd_seconds=max_cmd if max_cmd is not None else pipeline.security_max_cmd_seconds,
                security_max_total_seconds=max_total if max_total is not None else pipeline.security_max_total_seconds,
            )

    def _workdir_or_fail(stage: str, raw: str | None) -> tuple[Path, StageResult | None]:
        """中文说明：
        - 含义：解析某个 stage 的 workdir；失败时返回可序列化的 StageResult 错误。
        - 内容：用 `resolve_workdir` 计算实际路径；异常时写 artifacts，并返回 (repo, failed_stage_result)。
        - 可简略：可能（抽出 helper 能统一错误格式与落盘；也可拆成 try/except 重复代码）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈12 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/pipeline_verify.py:449；类型=function；引用≈9；规模≈12行
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
            if _is_truthy(eval_env.get("AIDER_FSM_REQUIRE_HINTS")):
                expected = _parse_json_str_list(eval_env.get("AIDER_FSM_HINT_ANCHORS_JSON"))
                ok_hints, hint_reason = _validate_hints_used(repo, expected_anchors=expected)
                if not ok_hints:
                    try:
                        write_text(artifacts_dir / "hints_requirement_error.txt", hint_reason + "\n")
                    except Exception:
                        pass
                    metrics_errors.append(f"evaluation.hints_requirement_failed: {hint_reason}")
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
                ok_run, run_reason = _validate_hints_run(repo)
                if not ok_run:
                    try:
                        write_text(artifacts_dir / "hints_run_requirement_error.txt", run_reason + "\n")
                    except Exception:
                        pass
                    metrics_errors.append(f"evaluation.hints_run_requirement_failed: {run_reason}")
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

            # Convention: if the contract requires an explicit boolean `ok`, treat `ok != true` as failure.
            # This helps prevent "placeholder success" where scripts exit 0 but write fallback metrics.
            if "ok" in (pipeline.evaluation_required_keys or []):
                if eval_metrics.get("ok") is not True:
                    failed_stage = "metrics"
                    metrics_errors.append("evaluation.ok_not_true")
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

                if "ok" in (pipeline.benchmark_required_keys or []):
                    if bench_metrics.get("ok") is not True:
                        failed_stage = "metrics"
                        metrics_errors.append("benchmark.ok_not_true")
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
    """中文说明：
    - 含义：将某个 stage 的“失败/最后一次命令”的 stdout/stderr 尾部格式化为 prompt 片段。
    - 内容：输出 `[PREFIX_RC] / [PREFIX_STDOUT_TAIL] / [PREFIX_STDERR_TAIL]` 三段，长度受 `STDIO_TAIL_CHARS` 限制。
    - 可简略：可能（只用于构建提示词；但集中格式化便于一致性与测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈14 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_verify.py:1012；类型=function；引用≈10；规模≈14行
    res = stage_failed_cmd(stage)
    if res is None:
        return ""
    return (
        f"[{prefix}_RC]\n{res.rc}\n\n"
        f"[{prefix}_STDOUT_TAIL]\n{tail(res.stdout, STDIO_TAIL_CHARS)}\n\n"
        f"[{prefix}_STDERR_TAIL]\n{tail(res.stderr, STDIO_TAIL_CHARS)}\n\n"
    )
