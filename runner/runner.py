from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import run_pending_actions
from .agent_client import AgentClient, AgentResult
from .opencode_client import OpenCodeClient
from .pipeline_spec import PipelineSpec
from .pipeline_verify import fmt_stage_tail, run_pipeline_verification, stage_rc
from .plan_format import ensure_plan_file, parse_plan
from .prompts import (
    make_block_step_prompt,
    make_execute_prompt,
    make_fix_or_replan_prompt,
    make_mark_done_prompt,
    make_plan_update_prompt,
)
from .snapshot import build_snapshot, get_git_changed_files, git_checkout, non_plan_changes
from .state import append_jsonl, default_state, ensure_dirs, now_iso, load_state, save_state
from .subprocess_utils import read_text_if_exists, run_cmd, tail, write_json, write_text


@dataclass(frozen=True)
class RunnerConfig:
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


def _load_seed_block(seed_files: list[str]) -> str:
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
    def _cmd(name: str, cmd: str) -> dict[str, Any]:
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
    repo = config.repo
    plan_abs = (repo / config.plan_rel).resolve()
    pipeline_abs = config.pipeline_abs

    state_dir, logs_dir, state_path = ensure_dirs(repo)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = logs_dir / f"run_{run_id}.jsonl"

    artifacts_run_dir = config.artifacts_base / run_id
    artifacts_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_run_dir / "versions.json", _probe_versions(repo))
    if pipeline_abs and pipeline_abs.exists():
        write_text(artifacts_run_dir / "pipeline.yml", read_text_if_exists(pipeline_abs))

    if config.preflight_only:
        preflight_dir = artifacts_run_dir / "preflight"
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
                "auth_rc": stage_rc(verify.auth),
                "test_rc": stage_rc(verify.tests),
                "deploy_setup_rc": stage_rc(verify.deploy_setup),
                "deploy_health_rc": stage_rc(verify.deploy_health),
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

    created_agent = False
    if agent is None:
        oc_username = None
        oc_password = None
        if str(config.opencode_url or "").strip():
            oc_username = str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode"
            oc_password = str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip()

        agent = OpenCodeClient(
            repo=repo,
            plan_rel=config.plan_rel,
            pipeline_rel=config.pipeline_rel,
            model=config.model,
            base_url=(str(config.opencode_url or "").strip() or None),
            timeout_seconds=int(config.opencode_timeout_seconds or 300),
            bash_mode=str(config.opencode_bash or "restricted"),
            unattended=str(config.unattended or "strict"),
            server_log_path=artifacts_run_dir / "opencode_server.log",
            session_title=f"{repo.name}:{run_id}",
            username=oc_username,
            password=oc_password,
        )
        created_agent = True

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
                verify = run_pipeline_verification(
                    repo,
                    pipeline=config.pipeline,
                    tests_cmds=config.tests_cmds,
                    artifacts_dir=iter_artifacts_dir,
                    unattended=config.unattended,
                )
                state["last_test_rc"] = stage_rc(verify.tests)
                state["last_deploy_setup_rc"] = stage_rc(verify.deploy_setup)
                state["last_deploy_health_rc"] = stage_rc(verify.deploy_health)
                state["last_benchmark_rc"] = stage_rc(verify.benchmark)
                state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
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
                    + fmt_stage_tail("AUTH", verify.auth)
                    + fmt_stage_tail("TESTS", verify.tests)
                    + fmt_stage_tail("DEPLOY_SETUP", verify.deploy_setup)
                    + fmt_stage_tail("DEPLOY_HEALTH", verify.deploy_health)
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
            verify = run_pipeline_verification(
                repo,
                pipeline=config.pipeline,
                tests_cmds=config.tests_cmds,
                artifacts_dir=iter_artifacts_dir,
                unattended=config.unattended,
            )
            write_json(
                iter_artifacts_dir / "summary.json",
                {
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                    "auth_rc": stage_rc(verify.auth),
                    "test_rc": stage_rc(verify.tests),
                    "deploy_setup_rc": stage_rc(verify.deploy_setup),
                    "deploy_health_rc": stage_rc(verify.deploy_health),
                    "benchmark_rc": stage_rc(verify.benchmark),
                    "metrics_path": verify.metrics_path,
                    "metrics_errors": verify.metrics_errors or [],
                },
            )
            state["last_test_rc"] = stage_rc(verify.tests)
            state["last_deploy_setup_rc"] = stage_rc(verify.deploy_setup)
            state["last_deploy_health_rc"] = stage_rc(verify.deploy_health)
            state["last_benchmark_rc"] = stage_rc(verify.benchmark)
            state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
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
