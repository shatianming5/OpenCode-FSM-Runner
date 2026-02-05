from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .actions import run_pending_actions
from .agent_client import AgentClient, AgentResult
from .bootstrap import run_bootstrap
from .contract_hints import suggest_contract_hints
from .env_local import _ensure_contract_stage_skeleton, _write_fallback_pipeline_yml
from .opencode_client import OpenCodeClient
from .pipeline_spec import PipelineSpec, load_pipeline_spec
from .pipeline_verify import fmt_stage_tail, run_pipeline_verification, stage_rc
from .plan_format import ensure_plan_file, parse_plan
from .paths import relpath_or_none
from .prompts import (
    make_block_step_prompt,
    make_execute_prompt,
    make_fix_or_replan_prompt,
    make_mark_done_prompt,
    make_plan_update_prompt,
    make_scaffold_contract_prompt,
)
from .snapshot import build_snapshot, get_git_changed_files, git_checkout, non_plan_changes
from .state import append_jsonl, default_state, ensure_dirs, now_iso, load_state, save_state
from .subprocess_utils import read_text_if_exists, run_cmd, tail, write_json, write_text
from .scaffold_validation import validate_scaffolded_files, validate_scaffolded_pipeline
from .types import VerificationResult


@dataclass(frozen=True)
class RunnerConfig:
    """中文说明：
    - 含义：Runner 的运行配置（CLI 解析后的集中参数对象）。
    - 内容：包含 repo、模型、计划路径、pipeline 位置与解析结果、tests 命令、artifacts 目录、迭代/修复上限、unattended 模式，以及 scaffold 合同相关开关。
    - 可简略：可能（字段较多；可拆分为“核心配置 + 可选特性配置”，但需谨慎保持 CLI/测试兼容）。
    """

    repo: Path
    goal: str
    model: str
    plan_rel: str
    pipeline_abs: Path | None
    pipeline_rel: str | None
    pipeline: PipelineSpec | None
    tests_cmds: list[str]
    effective_test_cmd: str
    artifacts_base: Path
    seed_files: list[str]
    max_iters: int
    max_fix: int
    unattended: str
    preflight_only: bool
    opencode_url: str
    opencode_timeout_seconds: int
    opencode_bash: str
    scaffold_opencode_bash: str = "full"
    tests_from_user: bool = False
    require_pipeline: bool = False
    scaffold_contract: str = "off"  # off|opencode
    scaffold_require_metrics: bool = True


def _load_seed_block(seed_files: list[str]) -> str:
    """中文说明：
    - 含义：把 `--seed` 文件内容打包成一个注入到 prompt 的文本块。
    - 内容：读取每个 seed 文件并截断尾部；形成 `[SEEDS]` 块供 agent 参考（常用于约束/背景/操作指南）。
    - 可简略：可能（seed 是可选能力；若不需要可移除整个 seed 机制）。
    """
    if not seed_files:
        return ""
    blocks: list[str] = []
    for raw in seed_files:
        p = Path(str(raw)).expanduser()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not text.strip():
            continue
        blocks.append(f"[SEED_FILE]\npath: {p}\n---\n{tail(text, 20000)}\n")
    if not blocks:
        return ""
    return "[SEEDS]\n" + "\n".join(blocks)


def _probe_versions(repo: Path) -> dict[str, Any]:
    """中文说明：
    - 含义：收集运行环境/工具版本信息并写入 artifacts（用于可复现与诊断）。
    - 内容：记录 python 版本、git head（若可用）、以及 docker/kubectl/helm 等工具版本（best-effort）。
    - 可简略：是（纯诊断信息；不影响核心闭环）。
    """
    def _cmd(name: str, cmd: str) -> dict[str, Any]:
        """中文说明：
        - 含义：执行一个版本探测命令并返回结构化结果。
        - 内容：返回 rc/out_tail/err_tail；用于构造 versions.json 的 tools 列表。
        - 可简略：可能（仅 probe_versions 内部使用；但独立函数便于复用该结构）。
        """
        rc, out, err = run_cmd(cmd, repo)
        return {"name": name, "cmd": cmd, "rc": rc, "out_tail": out, "err_tail": err}

    git_head = _cmd("git_head", "git rev-parse HEAD")
    if git_head["rc"] != 0:
        git_head = _cmd("git_head", "echo -n ''")

    return {
        "ts": now_iso(),
        "python": {"version": sys.version, "executable": sys.executable},
        "git": {"head": (git_head["out_tail"] or "").strip()},
        # Optional tools: do not assume they're installed.
        "tools": [
            _cmd("docker", "docker version"),
            _cmd("kubectl", "kubectl version --client"),
            _cmd("helm", "helm version"),
        ],
    }


def _agent_run(
    agent: AgentClient, text: str, *, log_path: Path, iter_idx: int, fsm_state: str, event: str
) -> AgentResult:
    """中文说明：
    - 含义：对 agent 的一次调用（带结构化 jsonl 日志）。
    - 内容：在调用前记录 prompt 预览；调用后记录 assistant 回复预览；便于离线审计与重放调试。
    - 可简略：否（可观测性关键；否则很难定位 agent 行为问题）。
    """
    append_jsonl(
        log_path,
        {
            "ts": now_iso(),
            "iter_idx": iter_idx,
            "fsm_state": fsm_state,
            "event": event,
            "phase": "prompt",
            "prompt_preview": tail(text, 1200),
        },
    )
    res = agent.run(text, fsm_state=fsm_state, iter_idx=iter_idx, purpose=event)
    append_jsonl(
        log_path,
        {
            "ts": now_iso(),
            "iter_idx": iter_idx,
            "fsm_state": fsm_state,
            "event": event,
            "phase": "response",
            "assistant_preview": tail(res.assistant_text or "", 1200),
        },
    )
    return res


def run(config: RunnerConfig, *, agent: AgentClient | None = None) -> int:
    """中文说明：
    - 含义：运行 OpenCode-FSM runner 的主闭环（或 preflight）。
    - 内容：
      - 可选：scaffold 合同（pipeline.yml + .aider_fsm/）并校验其满足最小要求
      - 可选：preflight-only（执行一次 bootstrap+pipeline 验收后退出）
      - 闭环：snapshot → plan_update(只改 PLAN) → execute_step(只做 Next) → verify(pipeline) → fix_or_replan/mark_done/block_step
      - 全程写入 `.aider_fsm/artifacts/<run_id>/...` 与 `.aider_fsm/logs/run_<id>.jsonl`
    - 可简略：否（项目核心主流程）。
    """
    repo = config.repo
    plan_abs = (repo / config.plan_rel).resolve()

    state_dir, logs_dir, state_path = ensure_dirs(repo)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = logs_dir / f"run_{run_id}.jsonl"

    artifacts_run_dir = config.artifacts_base / run_id
    artifacts_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_run_dir / "versions.json", _probe_versions(repo))
    os.environ["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())
    os.environ["AIDER_FSM_RUN_ID"] = str(run_id)
    os.environ["AIDER_FSM_ARTIFACTS_BASE"] = str(config.artifacts_base.resolve())

    # Contract guardrail: if repo docs contain runnable evaluation hints, require the contract
    # to execute at least one hinted/official command and record usage (or fail with ok=false).
    hints = suggest_contract_hints(repo)
    if hints.commands:
        os.environ.setdefault("AIDER_FSM_REQUIRE_HINTS", "1")
        try:
            os.environ.setdefault("AIDER_FSM_HINTS_JSON", json.dumps(list(hints.commands[:20]), ensure_ascii=False))
        except Exception:
            os.environ.setdefault("AIDER_FSM_HINTS_JSON", "[]")
        try:
            os.environ.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", json.dumps(list(hints.anchors[:20]), ensure_ascii=False))
        except Exception:
            os.environ.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", "[]")
        try:
            write_text(artifacts_run_dir / "command_hints.txt", "\n".join(hints.commands) + "\n")
            write_text(artifacts_run_dir / "command_hint_anchors.txt", "\n".join(hints.anchors) + "\n")
        except Exception:
            pass
    bootstrap_path = (repo / ".aider_fsm" / "bootstrap.yml").resolve()

    # If the repo does not provide a pipeline contract, optionally scaffold a minimal one.
    scaffold_mode = str(config.scaffold_contract or "off").strip().lower() or "off"
    created_agent = False

    def _ensure_agent(*, pipeline_rel_override: str | None = None) -> AgentClient:
        """中文说明：
        - 含义：懒加载/复用 AgentClient（OpenCodeClient）。
        - 内容：若外部传入 agent 则复用；否则按 config 启动/连接 OpenCode server，创建并缓存 agent；支持在 scaffold 阶段覆盖 pipeline_rel。
        - 可简略：否（避免重复启动 server/重复创建 session；并承载外部 server 认证逻辑）。
        """
        nonlocal agent, created_agent, config
        if agent is not None:
            return agent

        oc_username = None
        oc_password = None
        if str(config.opencode_url or "").strip():
            oc_username = str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode"
            oc_password = str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip()

        agent = OpenCodeClient(
            repo=repo,
            plan_rel=config.plan_rel,
            pipeline_rel=str(pipeline_rel_override).strip() if pipeline_rel_override else config.pipeline_rel,
            model=config.model,
            base_url=(str(config.opencode_url or "").strip() or None),
            timeout_seconds=int(config.opencode_timeout_seconds or 300),
            bash_mode=str(config.opencode_bash or "restricted"),
            scaffold_bash_mode=str(config.scaffold_opencode_bash or "full"),
            unattended=str(config.unattended or "strict"),
            server_log_path=artifacts_run_dir / "opencode_server.log",
            session_title=f"{repo.name}:{run_id}",
            username=oc_username,
            password=oc_password,
        )
        created_agent = True
        return agent

    need_scaffold = config.pipeline_abs is None and not (repo / "pipeline.yml").exists() and scaffold_mode == "opencode"

    if need_scaffold and scaffold_mode == "opencode":
        # Agent-first contract scaffolding: let OpenCode write `pipeline.yml` + `.aider_fsm/`.
        scaffold_err = ""
        try:
            scaffold_agent = _ensure_agent(pipeline_rel_override="pipeline.yml")
            _ensure_contract_stage_skeleton(repo)
            hints = suggest_contract_hints(repo)
            if hints.commands:
                write_text(artifacts_run_dir / "scaffold_command_hints.txt", "\n".join(hints.commands) + "\n")
            res = _agent_run(
                scaffold_agent,
                make_scaffold_contract_prompt(
                    repo,
                    pipeline_rel="pipeline.yml",
                    require_metrics=bool(config.scaffold_require_metrics),
                    command_hints=hints.commands,
                ),
                log_path=log_path,
                iter_idx=0,
                fsm_state="S0_SCAFFOLD",
                event="scaffold_contract",
            )
            write_text(artifacts_run_dir / "scaffold_agent_result.txt", tail(res.assistant_text or "", 20000) + "\n")
        except Exception as e:
            scaffold_err = tail(str(e), 4000)
            write_text(artifacts_run_dir / "scaffold_agent_error.txt", scaffold_err + "\n")
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "iter_idx": 0,
                    "fsm_state": "S0_SCAFFOLD",
                    "event": "scaffold_contract_error",
                    "error": tail(str(e), 2000),
                },
            )

        # Fallback: if the model forgot tool calls and pipeline.yml is still missing, write a minimal pipeline.yml
        # referencing `.aider_fsm/stages/*.sh` so downstream repair can proceed.
        pipeline_path = (repo / "pipeline.yml").resolve()
        if not pipeline_path.exists():
            try:
                _write_fallback_pipeline_yml(repo, pipeline_rel="pipeline.yml", require_metrics=bool(config.scaffold_require_metrics))
                write_text(artifacts_run_dir / "scaffold_fallback_used.txt", "wrote_fallback_pipeline_yml\n")
            except Exception:
                pass

        # Validate pipeline parseability (and minimal contract requirements). If missing/invalid, fail fast.
        pipeline_ok = False
        pipeline_validation_reason = ""
        if pipeline_path.exists():
            try:
                parsed = load_pipeline_spec(pipeline_path)
                missing_fields = validate_scaffolded_pipeline(parsed, require_metrics=bool(config.scaffold_require_metrics))
                missing_files = validate_scaffolded_files(repo)
                if missing_fields or missing_files:
                    pipeline_validation_reason = "missing_scaffold_requirements"
                    write_text(
                        artifacts_run_dir / "scaffold_agent_pipeline_validation_error.txt",
                        "Pipeline is parseable but does not meet scaffold requirements:\n"
                        + "\n".join([f"- {x}" for x in missing_fields])
                        + ("\n" if missing_fields else "")
                        + "\n".join([f"- missing_file: {x}" for x in missing_files])
                        + ("\n" if missing_files else ""),
                    )
                    append_jsonl(
                        log_path,
                        {
                            "ts": now_iso(),
                            "iter_idx": 0,
                            "fsm_state": "S0_SCAFFOLD",
                            "event": "scaffold_agent_pipeline_validation_error",
                            "pipeline_path": str(pipeline_path),
                            "missing_fields": missing_fields,
                            "missing_files": missing_files,
                        },
                    )
                else:
                    pipeline_ok = True
            except Exception as e:
                write_text(
                    artifacts_run_dir / "scaffold_agent_pipeline_parse_error.txt",
                    tail(str(e), 4000) + "\n",
                )
                append_jsonl(
                    log_path,
                    {
                        "ts": now_iso(),
                        "iter_idx": 0,
                        "fsm_state": "S0_SCAFFOLD",
                        "event": "scaffold_agent_pipeline_parse_error",
                        "pipeline_path": str(pipeline_path),
                        "error": tail(str(e), 2000),
                    },
                )
        if not pipeline_ok:
            reason = "missing_pipeline_yml"
            if pipeline_path.exists():
                reason = "invalid_or_incomplete_pipeline_yml"
            if pipeline_validation_reason:
                reason = f"{reason}; {pipeline_validation_reason}"
            if scaffold_err:
                reason = f"{reason}; scaffold_contract_error"
            if pipeline_path.exists():
                write_text(artifacts_run_dir / "scaffold_agent_pipeline.yml", read_text_if_exists(pipeline_path))
            write_text(
                artifacts_run_dir / "scaffold_error.txt",
                "scaffold_contract_failed: opencode did not produce a valid pipeline contract.\n"
                f"reason: {reason}\n",
            )
            print(
                "ERROR: `--scaffold-contract opencode` did not produce a valid pipeline.yml contract. "
                "See artifacts for details (scaffold_agent_result.txt / scaffold_*_error.txt).",
                file=sys.stderr,
            )
            if created_agent and agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass
            return 2

    # Load pipeline after scaffolding (or when provided).
    pipeline_abs = config.pipeline_abs
    pipeline_rel = config.pipeline_rel
    if pipeline_abs is None:
        default_pipeline = (repo / "pipeline.yml").resolve()
        if default_pipeline.exists():
            pipeline_abs = default_pipeline
            pipeline_rel = relpath_or_none(pipeline_abs, repo)

    pipeline = config.pipeline
    if pipeline_abs and pipeline_abs.exists():
        try:
            pipeline = load_pipeline_spec(pipeline_abs)
        except Exception as e:
            pipeline = None
            write_text(artifacts_run_dir / "pipeline_parse_error.txt", tail(str(e), 4000) + "\n")
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "iter_idx": 0,
                    "fsm_state": "S0_SCAFFOLD",
                    "event": "pipeline_parse_error",
                    "pipeline_path": str(pipeline_abs),
                    "error": tail(str(e), 2000),
                },
            )

    tests_cmds = list(config.tests_cmds or [])
    effective_test_cmd = config.effective_test_cmd
    if not bool(config.tests_from_user) and pipeline and pipeline.tests_cmds:
        tests_cmds = list(pipeline.tests_cmds)
        effective_test_cmd = " && ".join(tests_cmds)

    config = replace(
        config,
        pipeline_abs=pipeline_abs,
        pipeline_rel=pipeline_rel,
        pipeline=pipeline,
        tests_cmds=tests_cmds,
        effective_test_cmd=effective_test_cmd,
    )

    if config.require_pipeline and (config.pipeline_abs is None or config.pipeline is None):
        print(
            "ERROR: pipeline.yml not found or invalid. "
            "Provide --pipeline or add a root pipeline.yml (version: 1), or use --scaffold-contract opencode.",
            file=sys.stderr,
        )
        return 2

    if config.pipeline_abs and config.pipeline_abs.exists():
        write_text(artifacts_run_dir / "pipeline.yml", read_text_if_exists(config.pipeline_abs))

    if config.preflight_only:
        try:
            preflight_dir = artifacts_run_dir / "preflight"
            bootstrap_stage = None
            if bootstrap_path.exists():
                bootstrap_stage, applied_env = run_bootstrap(
                    repo,
                    bootstrap_path=bootstrap_path,
                    pipeline=config.pipeline,
                    unattended=config.unattended,
                    artifacts_dir=preflight_dir,
                )
                for k, v in (applied_env or {}).items():
                    os.environ[str(k)] = str(v)

                if not bootstrap_stage.ok:
                    verify = VerificationResult(
                        ok=False,
                        failed_stage="bootstrap",
                        bootstrap=bootstrap_stage,
                        metrics_errors=[],
                    )
                else:
                    verify = run_pipeline_verification(
                        repo,
                        pipeline=config.pipeline,
                        tests_cmds=config.tests_cmds,
                        artifacts_dir=preflight_dir,
                        unattended=config.unattended,
                    )
                    verify = replace(verify, bootstrap=bootstrap_stage)
            else:
                verify = run_pipeline_verification(
                    repo,
                    pipeline=config.pipeline,
                    tests_cmds=config.tests_cmds,
                    artifacts_dir=preflight_dir,
                    unattended=config.unattended,
                )
            write_json(
                preflight_dir / "summary.json",
                {
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                    "bootstrap_rc": stage_rc(verify.bootstrap),
                    "auth_rc": stage_rc(verify.auth),
                    "test_rc": stage_rc(verify.tests),
                    "deploy_setup_rc": stage_rc(verify.deploy_setup),
                    "deploy_health_rc": stage_rc(verify.deploy_health),
                    "rollout_rc": stage_rc(verify.rollout),
                    "evaluation_rc": stage_rc(verify.evaluation),
                    "benchmark_rc": stage_rc(verify.benchmark),
                    "metrics_path": verify.metrics_path,
                    "metrics_errors": verify.metrics_errors or [],
                },
            )
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "iter_idx": 0,
                    "fsm_state": "PREFLIGHT",
                    "event": "preflight_verify",
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                },
            )
            return 0 if verify.ok else 3
        finally:
            if created_agent and agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass

    ensure_plan_file(plan_abs, config.goal, config.effective_test_cmd, pipeline=config.pipeline)
    os.chdir(repo)
    seed_block = _load_seed_block(list(config.seed_files or []))

    defaults = default_state(
        repo=repo,
        plan_rel=config.plan_rel,
        pipeline_rel=config.pipeline_rel,
        model=config.model,
        test_cmd=config.effective_test_cmd,
    )
    state = load_state(state_path, defaults)
    save_state(state_path, state)

    bootstrap_ok = False

    def _verify_with_bootstrap(*, iter_artifacts_dir: Path) -> VerificationResult:
        """中文说明：
        - 含义：执行一次“先 bootstrap（最多一次成功后复用）再 pipeline 验收”的组合流程。
        - 内容：bootstrap 成功后设置 bootstrap_ok=True，后续迭代不再重复跑；若 bootstrap 失败则直接返回 failed_stage=bootstrap；否则执行 run_pipeline_verification 并把 bootstrap 结果附加到 VerificationResult。
        - 可简略：可能（组合函数；可以内联到主循环，但会让 run() 更难读）。
        """
        nonlocal bootstrap_ok
        bootstrap_stage = None
        if bootstrap_path.exists() and not bootstrap_ok:
            bootstrap_stage, applied_env = run_bootstrap(
                repo,
                bootstrap_path=bootstrap_path,
                pipeline=config.pipeline,
                unattended=config.unattended,
                artifacts_dir=iter_artifacts_dir,
            )
            for k, v in (applied_env or {}).items():
                os.environ[str(k)] = str(v)
            if bootstrap_stage.ok:
                bootstrap_ok = True
            else:
                return VerificationResult(
                    ok=False,
                    failed_stage="bootstrap",
                    bootstrap=bootstrap_stage,
                    metrics_errors=[],
                )

        verify = run_pipeline_verification(
            repo,
            pipeline=config.pipeline,
            tests_cmds=config.tests_cmds,
            artifacts_dir=iter_artifacts_dir,
            unattended=config.unattended,
        )
        if bootstrap_stage is not None:
            verify = replace(verify, bootstrap=bootstrap_stage)
        return verify

    agent = _ensure_agent()
    try:
        for iter_idx in range(1, config.max_iters + 1):
            state["iter_idx"] = iter_idx
            state["fsm_state"] = "S1_SNAPSHOT"
            save_state(state_path, state)

            snapshot, snapshot_text = build_snapshot(repo, plan_abs, pipeline_abs)
            if seed_block:
                snapshot_text = snapshot_text + "\n" + seed_block + "\n"
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "snapshot",
                    "snapshot": {k: snapshot[k] for k in ("repo", "plan_path", "env_probe", "git")},
                },
            )

            # S2: plan update (only PLAN.md)
            state["fsm_state"] = "S2_PLAN_UPDATE"
            save_state(state_path, state)
            plan_text: str | None = None
            parsed: dict[str, Any] | None = None
            next_step: dict[str, str] | None = None
            backlog_open = 0

            for attempt in range(1, 4):
                extra = ""
                if attempt > 1:
                    extra = "Previous attempt produced invalid output: ONLY edit PLAN.md and keep Next to exactly one item."

                pipeline_before = read_text_if_exists(pipeline_abs) if pipeline_abs else ""
                _agent_run(
                    agent,
                    make_plan_update_prompt(snapshot_text, config.effective_test_cmd, extra=extra),
                    log_path=log_path,
                    iter_idx=iter_idx,
                    fsm_state=state["fsm_state"],
                    event=f"plan_update_attempt_{attempt}",
                )
                if pipeline_abs and read_text_if_exists(pipeline_abs) != pipeline_before:
                    write_text(pipeline_abs, pipeline_before)
                    append_jsonl(
                        log_path,
                        {
                            "ts": now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "plan_update_touched_pipeline_reverted",
                            "pipeline_path": str(pipeline_abs),
                        },
                    )

                changed = get_git_changed_files(repo)
                if changed is not None:
                    illegal = non_plan_changes(changed, config.plan_rel)
                    if illegal:
                        rc, out, err = git_checkout(repo, illegal)
                        append_jsonl(
                            log_path,
                            {
                                "ts": now_iso(),
                                "iter_idx": iter_idx,
                                "fsm_state": state["fsm_state"],
                                "event": "plan_update_revert_non_plan_files",
                                "illegal_files": illegal,
                                "git_checkout_rc": rc,
                                "git_checkout_out": out,
                                "git_checkout_err": err,
                            },
                        )
                        if attempt == 3:
                            state["last_exit_reason"] = "PLAN_UPDATE_TOUCHED_CODE"
                            state["updated_at"] = now_iso()
                            save_state(state_path, state)
                            return 2
                        continue

                plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
                parsed = parse_plan(plan_text)

                if parsed["errors"]:
                    append_jsonl(
                        log_path,
                        {
                            "ts": now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "plan_parse_error",
                            "errors": parsed["errors"],
                        },
                    )
                    if attempt == 3:
                        state["last_exit_reason"] = "PLAN_PARSE_ERROR"
                        state["updated_at"] = now_iso()
                        save_state(state_path, state)
                        return 2
                    continue

                next_step = parsed["next_step"]
                backlog_open = int(parsed["backlog_open_count"])
                if next_step is None and backlog_open > 0:
                    append_jsonl(
                        log_path,
                        {
                            "ts": now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "plan_inconsistent_missing_next",
                            "backlog_open_count": backlog_open,
                        },
                    )
                    if attempt == 3:
                        state["last_exit_reason"] = "MISSING_NEXT_STEP"
                        state["updated_at"] = now_iso()
                        save_state(state_path, state)
                        return 2
                    continue

                break

            if next_step is None:
                # No next/backlog: verify once; if failing, ask to add a repair step.
                state["fsm_state"] = "S4_VERIFY"
                save_state(state_path, state)
                iter_artifacts_dir = artifacts_run_dir / f"iter_{iter_idx:04d}"
                verify = _verify_with_bootstrap(iter_artifacts_dir=iter_artifacts_dir)
                state["last_test_rc"] = stage_rc(verify.tests)
                state["last_deploy_setup_rc"] = stage_rc(verify.deploy_setup)
                state["last_deploy_health_rc"] = stage_rc(verify.deploy_health)
                state["last_rollout_rc"] = stage_rc(verify.rollout)
                state["last_evaluation_rc"] = stage_rc(verify.evaluation)
                state["last_benchmark_rc"] = stage_rc(verify.benchmark)
                state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
                if verify.bootstrap is not None:
                    state["last_bootstrap_rc"] = stage_rc(verify.bootstrap)
                save_state(state_path, state)
                append_jsonl(
                    log_path,
                    {
                        "ts": now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "verify_pipeline_no_steps",
                        "ok": verify.ok,
                        "failed_stage": verify.failed_stage,
                        "artifacts_dir": str(iter_artifacts_dir),
                    },
                )
                if verify.ok:
                    state["last_exit_reason"] = "DONE"
                    state["updated_at"] = now_iso()
                    save_state(state_path, state)
                    return 0

                snapshot2, snapshot_text2 = build_snapshot(repo, plan_abs, pipeline_abs)
                failure_tail = (
                    f"FAILED_STAGE={verify.failed_stage}\n"
                    f"ARTIFACTS_DIR={iter_artifacts_dir}\n\n"
                    + fmt_stage_tail("BOOTSTRAP", verify.bootstrap)
                    + fmt_stage_tail("AUTH", verify.auth)
                    + fmt_stage_tail("TESTS", verify.tests)
                    + fmt_stage_tail("DEPLOY_SETUP", verify.deploy_setup)
                    + fmt_stage_tail("DEPLOY_HEALTH", verify.deploy_health)
                    + fmt_stage_tail("ROLLOUT", verify.rollout)
                    + fmt_stage_tail("EVALUATION", verify.evaluation)
                    + fmt_stage_tail("BENCHMARK", verify.benchmark)
                    + (f"METRICS_ERRORS={verify.metrics_errors}\n" if (verify.metrics_errors or []) else "")
                )
                _agent_run(
                    agent,
                    make_plan_update_prompt(
                        snapshot_text2,
                        config.effective_test_cmd,
                        extra=(
                            "Verification failed but Next/Backlog is empty. Add ONE smallest Next step to fix.\n"
                            f"{failure_tail}"
                        ),
                    ),
                    log_path=log_path,
                    iter_idx=iter_idx,
                    fsm_state="S2_PLAN_UPDATE",
                    event="plan_update_due_to_failing_tests_no_steps",
                )
                continue

            # S3: execute
            state["fsm_state"] = "S3_EXECUTE_STEP"
            state["current_step_id"] = next_step["id"]
            state["current_step_text"] = next_step["text"]
            save_state(state_path, state)

            plan_before = plan_text or ""
            pipeline_before = read_text_if_exists(pipeline_abs) if pipeline_abs else ""
            _agent_run(
                agent,
                make_execute_prompt(snapshot_text, next_step),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="execute_step",
            )

            plan_after = plan_abs.read_text(encoding="utf-8", errors="replace") if plan_abs.exists() else ""
            pipeline_after = read_text_if_exists(pipeline_abs) if pipeline_abs else ""

            touched_plan = plan_after != plan_before
            touched_pipeline = bool(pipeline_abs and pipeline_after != pipeline_before)

            if touched_plan:
                # Guard: execution must not change PLAN.md; revert by content (works even without git).
                plan_abs.write_text(plan_before, encoding="utf-8")
                append_jsonl(
                    log_path,
                    {
                        "ts": now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "execute_touched_plan_reverted",
                    },
                )
            if touched_pipeline and pipeline_abs:
                write_text(pipeline_abs, pipeline_before)
                append_jsonl(
                    log_path,
                    {
                        "ts": now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "execute_touched_pipeline_reverted",
                        "pipeline_path": str(pipeline_abs),
                    },
                )
            if touched_plan or touched_pipeline:
                continue

            # S4: verify
            state["fsm_state"] = "S4_VERIFY"
            save_state(state_path, state)
            iter_artifacts_dir = artifacts_run_dir / f"iter_{iter_idx:04d}"
            verify = _verify_with_bootstrap(iter_artifacts_dir=iter_artifacts_dir)
            write_json(
                iter_artifacts_dir / "summary.json",
                {
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                    "bootstrap_rc": stage_rc(verify.bootstrap),
                    "auth_rc": stage_rc(verify.auth),
                    "test_rc": stage_rc(verify.tests),
                    "deploy_setup_rc": stage_rc(verify.deploy_setup),
                    "deploy_health_rc": stage_rc(verify.deploy_health),
                    "rollout_rc": stage_rc(verify.rollout),
                    "evaluation_rc": stage_rc(verify.evaluation),
                    "benchmark_rc": stage_rc(verify.benchmark),
                    "metrics_path": verify.metrics_path,
                    "metrics_errors": verify.metrics_errors or [],
                },
            )
            state["last_test_rc"] = stage_rc(verify.tests)
            state["last_deploy_setup_rc"] = stage_rc(verify.deploy_setup)
            state["last_deploy_health_rc"] = stage_rc(verify.deploy_health)
            state["last_rollout_rc"] = stage_rc(verify.rollout)
            state["last_evaluation_rc"] = stage_rc(verify.evaluation)
            state["last_benchmark_rc"] = stage_rc(verify.benchmark)
            state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
            if verify.bootstrap is not None:
                state["last_bootstrap_rc"] = stage_rc(verify.bootstrap)
            save_state(state_path, state)
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "verify_pipeline",
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                    "artifacts_dir": str(iter_artifacts_dir),
                },
            )

            # S5: decide
            state["fsm_state"] = "S5_DECIDE"
            save_state(state_path, state)

            if verify.ok:
                state["fix_attempts"] = 0
                save_state(state_path, state)
                _agent_run(
                    agent,
                    make_mark_done_prompt(next_step),
                    log_path=log_path,
                    iter_idx=iter_idx,
                    fsm_state=state["fsm_state"],
                    event="mark_done",
                )
                continue

            state["fix_attempts"] = int(state.get("fix_attempts") or 0) + 1
            save_state(state_path, state)
            if int(state["fix_attempts"]) <= int(config.max_fix):
                _agent_run(
                    agent,
                    make_fix_or_replan_prompt(
                        next_step,
                        verify,
                        tests_cmds=config.tests_cmds,
                        artifacts_dir=iter_artifacts_dir,
                    ),
                    log_path=log_path,
                    iter_idx=iter_idx,
                    fsm_state=state["fsm_state"],
                    event=f"fix_or_replan_attempt_{state['fix_attempts']}",
                )
                actions_stage = run_pending_actions(
                    repo,
                    pipeline=config.pipeline,
                    unattended=config.unattended,
                    actions_path=repo / ".aider_fsm" / "actions.yml",
                    artifacts_dir=iter_artifacts_dir / "actions",
                    protected_paths=[p for p in (plan_abs, pipeline_abs) if p],
                )
                if actions_stage is not None:
                    append_jsonl(
                        log_path,
                        {
                            "ts": now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "actions_executed",
                            "ok": actions_stage.ok,
                            "actions_rc": stage_rc(actions_stage),
                        },
                    )
            else:
                state["fix_attempts"] = 0
                save_state(state_path, state)
                last_failure = (
                    f"FAILED_STAGE={verify.failed_stage}\n"
                    f"ARTIFACTS_DIR={iter_artifacts_dir}\n\n"
                    + fmt_stage_tail("AUTH", verify.auth)
                    + fmt_stage_tail("TESTS", verify.tests)
                    + fmt_stage_tail("DEPLOY_SETUP", verify.deploy_setup)
                    + fmt_stage_tail("DEPLOY_HEALTH", verify.deploy_health)
                    + fmt_stage_tail("ROLLOUT", verify.rollout)
                    + fmt_stage_tail("EVALUATION", verify.evaluation)
                    + fmt_stage_tail("BENCHMARK", verify.benchmark)
                    + (f"METRICS_ERRORS={verify.metrics_errors}\n" if (verify.metrics_errors or []) else "")
                )
                _agent_run(
                    agent,
                    make_block_step_prompt(next_step, last_failure),
                    log_path=log_path,
                    iter_idx=iter_idx,
                    fsm_state=state["fsm_state"],
                    event="block_step",
                )

        state["last_exit_reason"] = "MAX_ITERS"
        state["updated_at"] = now_iso()
        save_state(state_path, state)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        raise
    except Exception as e:
        append_jsonl(
            log_path,
            {
                "ts": now_iso(),
                "iter_idx": int(state.get("iter_idx") or 0),
                "fsm_state": str(state.get("fsm_state") or ""),
                "event": "agent_error",
                "error": tail(str(e), 2000),
            },
        )
        state["last_exit_reason"] = "AGENT_ERROR"
        state["updated_at"] = now_iso()
        save_state(state_path, state)
        return 2
    finally:
        if created_agent:
            try:
                agent.close()
            except Exception:
                pass
