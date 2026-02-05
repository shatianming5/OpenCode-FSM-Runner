from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
        "--opencode-repair-timeout-seconds",
        type=int,
        default=0,
        help="OpenCode repair timeout seconds (0 = use --opencode-timeout-seconds).",
    )
    ap.add_argument(
        "--strict-opencode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail if OpenCode doesn't create pipeline.yml; if false, seed skeleton + fallback pipeline.yml.",
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
        rollout_ok = False
        eval_ok = False
        metrics: dict | None = None
        artifacts: str | None = None
        sess = None
        try:
            repair_timeout = int(args.opencode_repair_timeout_seconds or 0)
            sess = env.setup(
                target,
                clones_dir=clones_dir,
                require_metrics=True,
                audit=str(args.audit or "on"),
                opencode_model=str(args.opencode_model or ""),
                opencode_repair_model=(str(args.opencode_repair_model).strip() or None),
                opencode_url=str(args.opencode_url or ""),
                opencode_timeout_seconds=int(args.opencode_timeout_seconds or 600),
                opencode_repair_timeout_seconds=(repair_timeout if repair_timeout > 0 else None),
                unattended=str(args.unattended or "strict"),
                artifacts_dir=artifacts_dir,
                strict_opencode=bool(args.strict_opencode),
            )
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
        except Exception as e:
            err = str(e)
            target_ok = False
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
                "rollout_ok": rollout_ok,
                "evaluation_ok": eval_ok,
                "metrics": metrics,
                "artifacts_dir": artifacts,
                "error": err,
            }
        )

    summary["ok"] = bool(all_ok)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
