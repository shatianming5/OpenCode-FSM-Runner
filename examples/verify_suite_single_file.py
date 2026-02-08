from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env
from runner.dotenv import load_dotenv


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_targets_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _parse_env_kv(raw: str) -> tuple[str, str]:
    s = str(raw or "")
    if "=" not in s:
        raise ValueError(f"invalid --env: {raw!r} (expected KEY=VALUE)")
    k, v = s.split("=", 1)
    k = k.strip()
    if not k:
        raise ValueError(f"invalid --env: {raw!r} (empty key)")
    return k, v


def _read_json_object(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _latest_path(root: Path, pattern: str) -> Path | None:
    try:
        cands = [p for p in root.glob(pattern) if p.is_file()]
    except Exception:
        return None
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return cands[0]


def _ok_from_summary(path: Path) -> bool | None:
    obj = _read_json_object(path)
    if obj is None:
        return None
    if "ok" not in obj:
        return None
    return bool(obj.get("ok") is True)


def _attempt_idx(roll_eval_dir: Path) -> int | None:
    m = re.search(r"(?:^|_)attempt_(\d+)$", roll_eval_dir.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generic env.setup/rollout/evaluation verification suite (single file).")
    ap.add_argument("--targets", action="append", default=[], help="Target repo/dataset URL or local path (repeatable).")
    ap.add_argument("--targets-file", default="", help="File with one target per line (comments with #).")
    ap.add_argument("--llm", required=True, help="Local model dir path OR remote model id/name (passed through).")
    ap.add_argument("--eval-mode", default="full", choices=["smoke", "full"], help="AIDER_EVAL_MODE for stage scripts.")
    ap.add_argument("--require-samples", action="store_true", help="Require rollout.json.paths.samples_jsonl contract to be valid.")
    ap.add_argument("--repair-iters", type=int, default=3, help="Max contract repair iterations per target.")
    ap.add_argument("--audit", default="on", choices=["on", "off", "warn-only"], help="Heuristic audit strictness.")
    ap.add_argument("--unattended", default="strict", choices=["strict", "guided"], help="Runner unattended mode.")
    ap.add_argument("--opencode-model", default="", help="OpenCode model id for scaffold/repair (provider/model).")
    ap.add_argument("--opencode-repair-model", default="", help="OpenCode model id for repair only (optional).")
    ap.add_argument("--opencode-url", default="", help="Existing OpenCode server URL (optional).")
    ap.add_argument("--opencode-timeout-seconds", type=int, default=600, help="OpenCode scaffold/repair timeout seconds.")
    ap.add_argument(
        "--opencode-retry-attempts",
        type=int,
        default=2,
        help="OpenCode message retry attempts for timeout/transient errors.",
    )
    ap.add_argument(
        "--opencode-retry-backoff-seconds",
        type=float,
        default=2.0,
        help="OpenCode retry backoff base seconds (exponential).",
    )
    ap.add_argument(
        "--opencode-session-recover-attempts",
        type=int,
        default=0,
        help="OpenCode local session/server auto-recovery attempts on transport errors (0 = use default/env).",
    )
    ap.add_argument(
        "--opencode-session-recover-backoff-seconds",
        type=float,
        default=0.0,
        help="OpenCode session recovery backoff base seconds (0 = use default/env).",
    )
    ap.add_argument(
        "--opencode-context-length",
        type=int,
        default=0,
        help="OpenCode context length hint for scaffold/repair (0 = unset).",
    )
    ap.add_argument(
        "--opencode-max-prompt-chars",
        type=int,
        default=0,
        help="Max prompt chars sent to OpenCode (0 = no clipping).",
    )
    ap.add_argument(
        "--opencode-repair-timeout-seconds",
        type=int,
        default=0,
        help="OpenCode repair timeout seconds (0 = use --opencode-timeout-seconds).",
    )
    ap.add_argument(
        "--strict-opencode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict scaffold mode (default: true). "
            "The runner will not seed/fallback-write contract files; contracts must be produced by OpenCode or preexisting in the repo. "
            "(`--no-strict-opencode` is deprecated and kept for compatibility only.)"
        ),
    )
    ap.add_argument("--clones-dir", default="", help="Where to clone/download targets (optional).")
    ap.add_argument("--artifacts-dir", default="", help="Artifacts output dir (optional; default under target repo).")
    ap.add_argument("--env", action="append", default=[], help="Extra env vars injected into stages (KEY=VALUE).")
    ap.add_argument("--env-file", default=".env", help="dotenv file to load before running (default: .env; set empty to disable).")
    ap.add_argument("--env-override", action="store_true", help="override existing environment variables with values from --env-file")
    args = ap.parse_args()

    env_file = str(args.env_file or "").strip()
    if env_file:
        load_dotenv(env_file, override=bool(args.env_override))

    targets: list[str] = []
    targets.extend([str(t).strip() for t in (args.targets or []) if str(t).strip()])
    if str(args.targets_file or "").strip():
        targets.extend(_read_targets_file(Path(str(args.targets_file)).expanduser().resolve()))
    targets = [t for t in targets if t]
    if not targets:
        raise SystemExit("no targets provided (use --targets or --targets-file)")

    env_overrides: dict[str, str] = {}
    for raw in args.env or []:
        k, v = _parse_env_kv(raw)
        env_overrides[k] = v

    clones_dir = Path(str(args.clones_dir)).expanduser().resolve() if str(args.clones_dir or "").strip() else None
    artifacts_dir = Path(str(args.artifacts_dir)).expanduser().resolve() if str(args.artifacts_dir or "").strip() else None

    summary: dict[str, object] = {"ts": _now_iso(), "ok": True, "results": []}

    all_ok = True
    for target in targets:
        target_ok = False
        err: str | None = None
        failed_stage: str | None = None
        deploy_setup_ok: bool | None = None
        deploy_health_ok: bool | None = None
        rollout_ok = False
        eval_ok = False
        metrics: dict | None = None
        artifacts: str | None = None
        scaffold_provenance: str | None = None
        repair_provenances: list[str] = []
        missing_pipeline_yml = False
        sess = None
        try:
            repair_timeout = int(args.opencode_repair_timeout_seconds or 0)
            opencode_model = str(
                args.opencode_model
                or os.environ.get("OPENCODE_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or os.environ.get("CHAT_MODEL")
                or ""
            ).strip()
            opencode_repair_model = str(args.opencode_repair_model or "").strip() or None
            if opencode_repair_model is None and opencode_model:
                opencode_repair_model = opencode_model
            sess = env.setup(
                target,
                clones_dir=clones_dir,
                require_metrics=True,
                audit=str(args.audit or "on"),
                opencode_model=opencode_model,
                opencode_repair_model=opencode_repair_model,
                opencode_url=str(args.opencode_url or ""),
                opencode_timeout_seconds=int(args.opencode_timeout_seconds or 600),
                opencode_repair_timeout_seconds=(repair_timeout if repair_timeout > 0 else None),
                opencode_retry_attempts=int(args.opencode_retry_attempts or 0),
                opencode_retry_backoff_seconds=float(args.opencode_retry_backoff_seconds or 0.0),
                opencode_session_recover_attempts=(
                    int(args.opencode_session_recover_attempts)
                    if int(args.opencode_session_recover_attempts or 0) > 0
                    else None
                ),
                opencode_session_recover_backoff_seconds=(
                    float(args.opencode_session_recover_backoff_seconds)
                    if float(args.opencode_session_recover_backoff_seconds or 0.0) > 0
                    else None
                ),
                opencode_context_length=(int(args.opencode_context_length) if int(args.opencode_context_length or 0) > 0 else None),
                opencode_max_prompt_chars=(int(args.opencode_max_prompt_chars) if int(args.opencode_max_prompt_chars or 0) > 0 else None),
                unattended=str(args.unattended or "strict"),
                artifacts_dir=artifacts_dir,
                strict_opencode=bool(args.strict_opencode),
            )

            # Best-effort: locate the latest scaffold provenance for evidence.
            repo_root = Path(sess.env.repo).resolve() if getattr(sess, "env", None) is not None else None
            if repo_root is not None:
                artifacts_root = (repo_root / ".aider_fsm" / "artifacts").resolve()
                prov = _latest_path(artifacts_root, "*/scaffold/scaffold_provenance.json")
                if prov is not None:
                    scaffold_provenance = str(prov)
                    try:
                        scaffold_err = (prov.parent / "scaffold_error.txt").read_text(encoding="utf-8", errors="replace")
                        if "missing_pipeline_yml" in scaffold_err:
                            missing_pipeline_yml = True
                    except Exception:
                        pass

            rollout_res, eval_res = env.rollout_and_evaluation(
                str(args.llm),
                session=sess,
                mode=str(args.eval_mode or "full"),
                require_samples=bool(args.require_samples),
                env_overrides=env_overrides,
                repair_iters=int(args.repair_iters or 0),
                artifacts_dir=artifacts_dir,
            )
            rollout_ok = bool(rollout_res.ok)
            eval_ok = bool(eval_res.ok)
            metrics = eval_res.metrics if isinstance(eval_res.metrics, dict) else None
            artifacts = str(eval_res.artifacts_dir) if getattr(eval_res, "artifacts_dir", None) is not None else None
            target_ok = rollout_ok and eval_ok

            try:
                failed_stage = str(getattr(eval_res.verify, "failed_stage", None) or "") or None
            except Exception:
                failed_stage = None

            # Pull deploy/rollout/evaluation stage summaries from artifacts for stable reporting.
            if artifacts:
                roll_eval_dir = Path(artifacts).resolve()
                run_root = roll_eval_dir.parent
                idx = _attempt_idx(roll_eval_dir)
                if idx is not None:
                    deploy_dir = (run_root / f"deploy_attempt_{idx:02d}").resolve()
                else:
                    deploy_dir = (run_root / "deploy_attempt_01").resolve()

                deploy_setup_ok = _ok_from_summary(deploy_dir / "deploy_setup_summary.json")
                deploy_health_ok = _ok_from_summary(deploy_dir / "deploy_health_summary.json")
                # Use the stage summary files for rollout/evaluation even if ok flags differ.
                rollout_ok = bool(_ok_from_summary(roll_eval_dir / "rollout_summary.json") is True) if (roll_eval_dir / "rollout_summary.json").exists() else rollout_ok
                eval_ok = bool(_ok_from_summary(roll_eval_dir / "evaluation_summary.json") is True) if (roll_eval_dir / "evaluation_summary.json").exists() else eval_ok
                target_ok = bool(rollout_ok and eval_ok)

                # Evidence: repair provenances (if any).
                try:
                    for p in sorted(run_root.glob("repair_*/repair_provenance.json")):
                        if p.is_file():
                            repair_provenances.append(str(p.resolve()))
                except Exception:
                    pass
        except Exception as e:
            err = str(e)
            target_ok = False
            if "missing_pipeline_yml" in err:
                missing_pipeline_yml = True
        finally:
            try:
                if sess is not None:
                    env.teardown(session=sess)
            except Exception:
                pass

        all_ok = all_ok and target_ok
        summary["results"].append(
            {
                "target": target,
                "ok": target_ok,
                "failed_stage": failed_stage,
                "deploy_setup_ok": deploy_setup_ok,
                "deploy_health_ok": deploy_health_ok,
                "rollout_ok": rollout_ok,
                "evaluation_ok": eval_ok,
                "metrics": metrics,
                "artifacts_dir": artifacts,
                "scaffold_provenance": scaffold_provenance,
                "repair_provenances": repair_provenances,
                "missing_pipeline_yml_seen": bool(missing_pipeline_yml),
                "error": err,
            }
        )

    summary["ok"] = bool(all_ok)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
