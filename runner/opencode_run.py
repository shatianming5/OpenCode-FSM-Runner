from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .contract_hints import suggest_contract_hints
from .contract_repair import repair_contract
from .dotenv import load_dotenv
from .eval_audit import (
    audit_eval_script_for_hardcoded_nonzero_score,
    audit_eval_script_has_real_execution,
    audit_eval_script_mentions_any_anchor,
)
from .env_local import deploy, deploy_teardown, open_env, rollout_and_evaluate, with_runtime_env_path
from .subprocess_utils import tail


def _default_artifacts_dir(repo: Path, *, run_id: str) -> Path:
    """中文说明：
    - 含义：为本次 `opencode_run` 生成默认 artifacts 输出目录。
    - 内容：写入目标 repo 的 `.aider_fsm/artifacts/<run_id>/opencode_run/`，便于审计与复现。
    - 可简略：是（纯路径拼接；集中成函数便于测试与保持一致）。
    """
    return (repo / ".aider_fsm" / "artifacts" / run_id / "opencode_run").resolve()


def _parse_env_kv(raw: str) -> tuple[str, str]:
    s = str(raw or "")
    if "=" not in s:
        raise ValueError(f"invalid --env: {raw!r} (expected KEY=VALUE)")
    key, value = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"invalid --env: {raw!r} (empty key)")
    return key, value


def _read_json_object(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_json_object(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_runtime_env_model_dir(runtime_env_path: Path, trained_model_dir: str) -> None:
    """Best-effort: attach trained model dir into runtime_env.json for audit/debug."""
    p = Path(str(runtime_env_path)).expanduser().resolve()
    if not p.exists():
        return
    obj = _read_json_object(p)
    if obj is None:
        return
    inf = obj.get("inference")
    if not isinstance(inf, dict):
        inf = {}
        obj["inference"] = inf
    inf.setdefault("type", "local_hf")
    inf["model_dir"] = str(trained_model_dir)
    _write_json_object(p, obj)


def _snapshot_contract_files(repo: Path, out_dir: Path) -> None:
    """Snapshot contract files for audit/debug (best-effort)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = (repo / "pipeline.yml").resolve()
        if pipeline.exists():
            shutil.copy2(pipeline, out_dir / "pipeline.yml")
        stages = (repo / ".aider_fsm" / "stages").resolve()
        if stages.exists() and stages.is_dir():
            dst = (out_dir / "stages").resolve()
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(stages, dst)
    except Exception:
        return

def _repair_contract(
    *,
    repo: Path,
    model: str,
    opencode_url: str,
    unattended: str,
    artifacts_dir: Path,
    failed_stage: str,
    deploy_artifacts_dir: Path,
    rollout_eval_artifacts_dir: Path,
    command_hints: list[str] | None = None,
    extra_context: str = "",
) -> None:
    repair_contract(
        repo=repo,
        model=model,
        opencode_url=opencode_url,
        unattended=unattended,
        artifacts_dir=artifacts_dir,
        failed_stage=failed_stage,
        deploy_artifacts_dir=deploy_artifacts_dir,
        rollout_eval_artifacts_dir=rollout_eval_artifacts_dir,
        command_hints=command_hints,
        extra_context=extra_context,
    )
    return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    oc_username = None
    oc_password = None
    if str(opencode_url or "").strip():
        oc_username = str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode"
        oc_password = str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip() or None

    agent = OpenCodeClient(
        repo=repo,
        plan_rel="PLAN.md",
        pipeline_rel="pipeline.yml",
        model=str(model or "").strip(),
        base_url=(str(opencode_url or "").strip() or None),
        timeout_seconds=300,
        bash_mode="restricted",
        scaffold_bash_mode="full",
        unattended=str(unattended or "strict"),
        server_log_path=artifacts_dir / "opencode_server.log",
        session_title=f"{repo.name}:repair",
        username=oc_username,
        password=oc_password,
    )
    try:
        deploy_err = _read_text_tail(deploy_artifacts_dir / "deploy_setup_stderr.txt", n=4000)
        rollout_err = _read_text_tail(rollout_eval_artifacts_dir / "rollout_stderr.txt", n=4000)
        eval_err = _read_text_tail(rollout_eval_artifacts_dir / "evaluation_stderr.txt", n=4000)
        deploy_setup_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "deploy_setup.sh", n=4000)
        rollout_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "rollout.sh", n=4000)
        evaluation_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "evaluation.sh", n=4000)
        runtime_env_preview = _read_text_tail(repo / ".aider_fsm" / "runtime_env.json", n=2000)
        metrics_preview = _read_text_tail(repo / ".aider_fsm" / "metrics.json", n=2000)
        extra = str(extra_context or "").strip()
        extra_block = f"\n[EXTRA_CONTEXT]\n{extra}\n" if extra else ""
        hints = [str(s).strip() for s in (command_hints or []) if str(s).strip()]
        hints_block = ""
        if hints:
            shown = hints[:20]
            hints_block = (
                "\n"
                "[CANDIDATE_COMMAND_HINTS]\n"
                "These commands were extracted from the repo docs (README/docs). Prefer using them for real execution.\n"
                "If hints exist, evaluation.sh MUST run at least one hint (or a direct adaptation) and derive score from its outputs.\n"
                "Do NOT replace repo-provided evaluation with proxy scoring (micro-benchmarks). If hints cannot run, write ok=false + reason and EXIT NON-ZERO.\n"
                + "".join([f"- {line}\n" for line in shown])
                + ("\n" if len(hints) > len(shown) else "")
            )
        prompt = (
            "You are a contract repair agent.\n"
            "\n"
            "Goal: fix the repo-local contract so the runner can execute deploy -> rollout -> evaluation without errors.\n"
            "\n"
            "Hard constraints:\n"
            "1) You may ONLY write files under `.aider_fsm/`.\n"
            "2) Do NOT modify `pipeline.yml`.\n"
            "3) Keep everything non-interactive (assume unattended strict).\n"
            "4) Do NOT use `sed -i` (not portable).\n"
            "5) If you embed python via heredoc, do NOT rely on sys.argv and NEVER put shell args after the heredoc terminator.\n"
            "\n"
            "Contract requirements:\n"
            "- deploy must write `.aider_fsm/runtime_env.json` (JSON object)\n"
            "- rollout must write `.aider_fsm/rollout.json` (JSON object)\n"
            "- evaluation must write `.aider_fsm/metrics.json` (JSON object, must contain keys `ok` and `score`; require `ok=true` for success)\n"
            "- NEVER set `ok=true` unless the benchmark/evaluation command actually ran and succeeded.\n"
            "- NEVER hardcode a non-zero score. Derive the score from real benchmark execution and parse outputs.\n"
            "- deploy_teardown should stop any started services/containers (best-effort)\n"
            "\n"
            f"FAILED_STAGE: {failed_stage}\n"
            f"REPO_ROOT: {repo}\n"
            f"DEPLOY_ARTIFACTS_DIR: {deploy_artifacts_dir}\n"
            f"ROLLOUT_EVAL_ARTIFACTS_DIR: {rollout_eval_artifacts_dir}\n"
            f"{hints_block}"
            "\n"
            "[DEPLOY_SETUP_STDERR_TAIL]\n"
            f"{deploy_err}\n"
            "\n"
            "[ROLLOUT_STDERR_TAIL]\n"
            f"{rollout_err}\n"
            "\n"
            "[EVALUATION_STDERR_TAIL]\n"
            f"{eval_err}\n"
            "\n"
            "[DEPLOY_SETUP_SH_TAIL]\n"
            f"{deploy_setup_sh}\n"
            "\n"
            "[ROLLOUT_SH_TAIL]\n"
            f"{rollout_sh}\n"
            "\n"
            "[EVALUATION_SH_TAIL]\n"
            f"{evaluation_sh}\n"
            "\n"
            "[RUNTIME_ENV_JSON_TAIL]\n"
            f"{runtime_env_preview}\n"
            "\n"
            "[METRICS_JSON_TAIL]\n"
            f"{metrics_preview}\n"
            f"{extra_block}"
            "\n"
            "Now: inspect `.aider_fsm/stages/*.sh` and fix the scripts so the contract passes.\n"
        )
        try:
            res = agent.run(prompt, fsm_state="S_REPAIR", iter_idx=0, purpose="repair_contract")
            (artifacts_dir / "repair_agent_result.txt").write_text(tail(res.assistant_text or "", 20000) + "\n", encoding="utf-8")
        except Exception as e:
            # Same as scaffolding: a timeout/error might still have produced partial file writes.
            (artifacts_dir / "repair_agent_error.txt").write_text(tail(str(e), 4000) + "\n", encoding="utf-8")
    finally:
        try:
            agent.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    """中文说明：
    - 含义：一键闭环：OpenCode scaffold 合同（若缺）→ deploy（产出 runtime_env）→ rollout → evaluation（产出 metrics）。
    - 内容：不写任何 benchmark-specific 逻辑；差异全部由目标 repo 的 `pipeline.yml` + `.aider_fsm/**` 自描述。
    - 可简略：否（这是“只用 OpenCode 去 run”的最小入口；用于脚本/CI/外部系统调用）。
    """
    parser = argparse.ArgumentParser(description="OpenCode one-shot runner: deploy -> rollout -> evaluation")
    parser.add_argument("--repo", required=True, help="repo root path or git URL (required)")
    parser.add_argument("--model", default="", help="model name as provider/model (optional)")
    parser.add_argument("--opencode-url", default="", help="OpenCode server base URL (optional; default: auto-start)")
    parser.add_argument(
        "--unattended",
        choices=("strict", "guided"),
        default="strict",
        help="unattended mode: strict blocks likely-interactive commands; guided allows interactive auth steps",
    )
    parser.add_argument(
        "--require-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require evaluation metrics JSON with required keys (default: true)",
    )
    parser.add_argument(
        "--strict-opencode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "strict scaffold mode (compat flag). "
            "Runner no longer prewrites/fallback-writes scaffold files; contract files must be produced by OpenCode/repo."
        ),
    )
    parser.add_argument(
        "--teardown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run deploy teardown after rollout/evaluation (default: true)",
    )
    parser.add_argument("--repair-iters", type=int, default=3, help="auto-repair .aider_fsm contract on failure (default: 3)")
    parser.add_argument(
        "--trained-model-dir",
        default="",
        help="path to a trained HF model directory (exported by your training script); injected as AIDER_TRAINED_MODEL_DIR",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="extra env override in KEY=VALUE form (repeatable)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="artifacts output directory (default: <repo>/.aider_fsm/artifacts/<run_id>/opencode_run)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="dotenv file to load before running (default: .env; set to empty to disable)",
    )
    parser.add_argument(
        "--env-override",
        action="store_true",
        help="override existing environment variables with values from --env-file",
    )

    args = parser.parse_args(argv)

    env_file = str(args.env_file or "").strip()
    if env_file:
        load_dotenv(env_file, override=bool(args.env_override))

    user_overrides: dict[str, str] = {}
    try:
        for raw in (args.env or []):
            k, v = _parse_env_kv(raw)
            user_overrides[k] = v
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    trained_model_dir_raw = str(args.trained_model_dir or "").strip()
    trained_model_dir = ""
    if trained_model_dir_raw:
        trained_model_dir = str(Path(trained_model_dir_raw).expanduser().resolve())
        user_overrides.setdefault("AIDER_TRAINED_MODEL_DIR", trained_model_dir)

    # `open_env()` may clone/prepare the repo and (when pipeline.yml is missing) ask OpenCode
    # to scaffold the minimal contract under repo root.
    try:
        env = open_env(
            str(args.repo),
            require_pipeline=True,
            scaffold_contract="opencode",
            scaffold_require_metrics=bool(args.require_metrics),
            model=str(args.model or ""),
            opencode_url=str(args.opencode_url or ""),
            unattended=str(args.unattended or "strict"),
            seed_stage_skeleton=not bool(args.strict_opencode),
            write_fallback_pipeline_yml=not bool(args.strict_opencode),
        )
    except Exception as e:
        print(f"ERROR: failed to open env: {e}", file=sys.stderr)
        return 2

    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    artifacts_dir_raw = str(args.artifacts_dir or "").strip()
    artifacts_dir = Path(artifacts_dir_raw).expanduser().resolve() if artifacts_dir_raw else _default_artifacts_dir(env.repo, run_id=run_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    hints = suggest_contract_hints(env.repo)
    if hints.commands:
        try:
            (artifacts_dir / "command_hints.txt").write_text("\n".join(hints.commands) + "\n", encoding="utf-8")
            (artifacts_dir / "command_hint_anchors.txt").write_text("\n".join(hints.anchors) + "\n", encoding="utf-8")
        except Exception:
            pass
        # Enforce: evaluation must run at least one hinted/official command, otherwise ok=false and fail.
        user_overrides.setdefault("AIDER_FSM_REQUIRE_HINTS", "1")
        try:
            user_overrides.setdefault("AIDER_FSM_HINTS_JSON", json.dumps(list(hints.commands[:20]), ensure_ascii=False))
        except Exception:
            user_overrides.setdefault("AIDER_FSM_HINTS_JSON", "[]")
        try:
            user_overrides.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", json.dumps(list(hints.anchors[:20]), ensure_ascii=False))
        except Exception:
            user_overrides.setdefault("AIDER_FSM_HINT_ANCHORS_JSON", "[]")

    # Make run id available to stage scripts without requiring hardcoded derivations.
    user_overrides.setdefault("AIDER_FSM_RUN_ID", run_id)
    max_repairs = int(args.repair_iters or 0)
    last_failed_stage = ""
    for attempt in range(max_repairs + 1):
        deploy_dir = artifacts_dir / "deploy"
        roll_eval_dir = artifacts_dir / "rollout_evaluation"
        teardown_dir = artifacts_dir / f"deploy_teardown_attempt_{attempt+1:02d}"
        _snapshot_contract_files(env.repo, artifacts_dir / f"contract_snapshot_attempt_{attempt+1:02d}")

        deploy_res = deploy(
            env,
            artifacts_dir=deploy_dir,
            env_overrides=user_overrides,
            unattended=str(args.unattended or "strict"),
            run_bootstrap_first=True,
        )
        if not deploy_res.ok:
            last_failed_stage = "deploy"
            if attempt >= max_repairs:
                print("ERROR: deploy failed", file=sys.stderr)
                return 1
            _repair_contract(
                repo=env.repo,
                model=str(args.model or "").strip(),
                opencode_url=str(args.opencode_url or ""),
                unattended=str(args.unattended or "strict"),
                artifacts_dir=artifacts_dir / f"repair_{attempt+1}",
                failed_stage=last_failed_stage,
                deploy_artifacts_dir=deploy_dir,
                rollout_eval_artifacts_dir=roll_eval_dir,
                command_hints=hints.commands,
            )
            continue

        runtime_env_path = deploy_res.runtime_env_path or (env.repo / ".aider_fsm" / "runtime_env.json").resolve()
        if trained_model_dir:
            try:
                _patch_runtime_env_model_dir(runtime_env_path, trained_model_dir)
            except Exception:
                pass

        overrides = dict(user_overrides)
        overrides.update(with_runtime_env_path(runtime_env_path))

        rollout_res, eval_res = rollout_and_evaluate(
            env,
            artifacts_dir=roll_eval_dir,
            env_overrides=overrides,
            unattended=str(args.unattended or "strict"),
            run_bootstrap_first=True,
        )
        def _maybe_teardown() -> None:
            if not bool(args.teardown):
                return
            try:
                deploy_teardown(
                    env,
                    artifacts_dir=teardown_dir,
                    env_overrides=overrides,
                    unattended=str(args.unattended or "strict"),
                )
            except Exception:
                return

        # Extra guardrail: avoid treating placeholder metrics as success even if the stage exited 0.
        if bool(args.require_metrics) and rollout_res.ok and eval_res.ok:
            m = eval_res.metrics or {}
            if not isinstance(m, dict) or m.get("ok") is not True:
                rollout_res = rollout_res  # unchanged
                eval_res = eval_res  # unchanged
                last_failed_stage = "evaluation"
                if attempt >= max_repairs:
                    print("ERROR: evaluation produced non-ok metrics", file=sys.stderr)
                    return 1
                _maybe_teardown()
                _repair_contract(
                    repo=env.repo,
                    model=str(args.model or "").strip(),
                    opencode_url=str(args.opencode_url or ""),
                    unattended=str(args.unattended or "strict"),
                    artifacts_dir=artifacts_dir / f"repair_{attempt+1}",
                    failed_stage=last_failed_stage,
                    deploy_artifacts_dir=deploy_dir,
                    rollout_eval_artifacts_dir=roll_eval_dir,
                    command_hints=hints.commands,
                )
                continue

        if rollout_res.ok and eval_res.ok:
            last_failed_stage = ""
            # Extra audit: reject obvious hardcoded constant scores even if the run "passes".
            if bool(args.require_metrics):
                audit_issue = audit_eval_script_for_hardcoded_nonzero_score(env.repo)
                audit_issue2 = audit_eval_script_has_real_execution(env.repo, extra_markers=hints.anchors)
                audit_issue3 = audit_eval_script_mentions_any_anchor(env.repo, hints.anchors)
                combined = "\n\n".join([x for x in (audit_issue, audit_issue2, audit_issue3) if x])
                if combined:
                    last_failed_stage = "evaluation"
                    if attempt >= max_repairs:
                        print("ERROR: evaluation script audit failed (likely hardcoded/placeholder)", file=sys.stderr)
                        return 1
                    _maybe_teardown()
                    _repair_contract(
                        repo=env.repo,
                        model=str(args.model or "").strip(),
                        opencode_url=str(args.opencode_url or ""),
                        unattended=str(args.unattended or "strict"),
                        artifacts_dir=artifacts_dir / f"repair_{attempt+1}",
                        failed_stage="evaluation",
                        deploy_artifacts_dir=deploy_dir,
                        rollout_eval_artifacts_dir=roll_eval_dir,
                        command_hints=hints.commands,
                        extra_context=combined,
                    )
                    continue
            _maybe_teardown()
            break

        last_failed_stage = "rollout" if not rollout_res.ok else "evaluation"
        if attempt >= max_repairs:
            print(f"ERROR: {last_failed_stage} failed", file=sys.stderr)
            return 1
        _maybe_teardown()
        _repair_contract(
            repo=env.repo,
            model=str(args.model or "").strip(),
            opencode_url=str(args.opencode_url or ""),
            unattended=str(args.unattended or "strict"),
            artifacts_dir=artifacts_dir / f"repair_{attempt+1}",
            failed_stage=last_failed_stage,
            deploy_artifacts_dir=deploy_dir,
            rollout_eval_artifacts_dir=roll_eval_dir,
            command_hints=hints.commands,
            extra_context="",
        )

    if last_failed_stage:
        print(f"ERROR: {last_failed_stage} failed", file=sys.stderr)
        return 1

    if eval_res.metrics is not None:
        try:
            print(json.dumps(eval_res.metrics, ensure_ascii=False))
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
