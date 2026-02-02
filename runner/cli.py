from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dotenv import load_dotenv
from .paths import relpath_or_none, resolve_config_path
from .pipeline_spec import load_pipeline_spec
from .repo_resolver import prepare_repo
from .runner import RunnerConfig, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aider-FSM runner (single-process closed-loop executor)")
    parser.add_argument("--repo", default=".", help="repo root path (default: .)")
    parser.add_argument("--goal", default="", help="goal for PLAN.md (used only when PLAN.md is missing)")
    parser.add_argument("--model", default="gpt-4o-mini", help="model name (default: gpt-4o-mini)")
    parser.add_argument("--test-cmd", default=None, help='acceptance command (default: pipeline/tests or "pytest -q")')
    parser.add_argument("--pipeline", default="", help="pipeline YAML path relative to repo (optional)")
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
    parser.add_argument(
        "--clone-dir",
        default="",
        help="when --repo is a git URL, clone into this directory (default: /tmp/aider_fsm_targets)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="artifacts base directory (default: pipeline.artifacts.out_dir or .aider_fsm/artifacts under repo)",
    )
    parser.add_argument("--seed", action="append", default=[], help="seed file path (repeatable)")
    parser.add_argument("--max-iters", type=int, default=200, help="max iterations (default: 200)")
    parser.add_argument("--max-fix", type=int, default=10, help="max fix attempts per step (default: 10)")
    parser.add_argument("--plan-path", default="PLAN.md", help="plan file path relative to repo (default: PLAN.md)")
    parser.add_argument(
        "--unattended",
        choices=("strict", "guided"),
        default="strict",
        help="unattended mode: strict blocks likely-interactive commands; guided allows interactive auth steps",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run verification once then exit (does not start aider FSM loop)",
    )

    # Deprecated flags (kept to avoid silent breakage; prefer actions.yml).
    parser.add_argument("--ensure-tools", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ensure-kind", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--kind-config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--full-quickstart", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    env_file = str(args.env_file or "").strip()
    if env_file:
        load_dotenv(env_file, override=bool(args.env_override))

    if args.ensure_tools or args.ensure_kind or args.kind_name or args.kind_config or args.full_quickstart:
        print(
            "ERROR: --ensure-tools/--ensure-kind/--kind-*/--full-quickstart are removed.\n"
            "Use `.aider_fsm/actions.yml` to perform environment bootstrap steps instead.",
            file=sys.stderr,
        )
        return 2

    clone_dir_raw = str(args.clone_dir or "").strip()
    clone_dir = Path(clone_dir_raw).expanduser().resolve() if clone_dir_raw else None
    try:
        prepared = prepare_repo(str(args.repo), clones_dir=clone_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: failed to prepare --repo: {e}", file=sys.stderr)
        return 2

    repo = prepared.repo
    if prepared.cloned_from:
        print(f"INFO: cloned {prepared.cloned_from} -> {repo}", file=sys.stderr)

    pipeline_abs = None
    pipeline_rel = None
    pipeline = None
    if str(args.pipeline or "").strip():
        pipeline_abs = resolve_config_path(repo, str(args.pipeline).strip())
        pipeline = load_pipeline_spec(pipeline_abs)
        pipeline_rel = relpath_or_none(pipeline_abs, repo)

    tests_cmds: list[str]
    if args.test_cmd and str(args.test_cmd).strip():
        tests_cmds = [str(args.test_cmd).strip()]
    elif pipeline and pipeline.tests_cmds:
        tests_cmds = list(pipeline.tests_cmds)
    else:
        tests_cmds = ["pytest -q"]

    effective_test_cmd = " && ".join(tests_cmds)

    artifacts_base_raw = str(args.artifacts_dir or "").strip()
    if not artifacts_base_raw and pipeline:
        artifacts_base_raw = (pipeline.artifacts_out_dir or "").strip()
    if artifacts_base_raw:
        artifacts_base = resolve_config_path(repo, artifacts_base_raw)
    else:
        artifacts_base = (repo / ".aider_fsm" / "artifacts").resolve()

    seed_files: list[str] = []
    for raw in list(args.seed or []):
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        seed_files.append(str(p))

    cfg = RunnerConfig(
        repo=repo,
        goal=str(args.goal or ""),
        model=str(args.model or "gpt-4o-mini"),
        plan_rel=str(args.plan_path or "PLAN.md"),
        pipeline_abs=pipeline_abs,
        pipeline_rel=pipeline_rel,
        pipeline=pipeline,
        tests_cmds=tests_cmds,
        effective_test_cmd=effective_test_cmd,
        artifacts_base=artifacts_base,
        seed_files=seed_files,
        max_iters=int(args.max_iters),
        max_fix=int(args.max_fix),
        unattended=str(args.unattended),
        preflight_only=bool(args.preflight_only),
    )
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
