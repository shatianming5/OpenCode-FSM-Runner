from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .dotenv import load_dotenv
from .paths import relpath_or_none, resolve_config_path
from .pipeline_spec import load_pipeline_spec
from .repo_resolver import prepare_repo
from .runner import RunnerConfig, run


def _list_opencode_models() -> list[str]:
    """中文说明：
    - 含义：调用 `opencode models` 获取本机可用模型列表（去掉 ANSI 色彩）。
    - 内容：用于默认模型选择与把裸模型名解析到正确 provider（例如 myproxy/...）。
    - 可简略：可能（与 env_local 有重复实现；可抽公共模块）。
    """
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
    """中文说明：
    - 含义：将用户传入的模型参数规范化为 `provider/model`。
    - 内容：空字符串时尽量从 `opencode models` 选择可用默认值；裸 model_id 会尝试在 `opencode models` 里匹配并优先 myproxy；最终回退到 `openai/<id>`。
    - 可简略：可能（与 env_local 重复；但保留集中解析可避免默认值漂移）。
    """
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

    # If the user passed a bare model ID (e.g. "deepseek-v3.2"), try to resolve it
    # via `opencode models` to pick the right provider (e.g. "myproxy/deepseek-v3.2").
    candidates = _list_opencode_models()
    matches = [m for m in candidates if m.split("/", 1)[1] == s]
    if matches:
        # Prefer the most common internal proxy provider name if present.
        for m in matches:
            if m.startswith("myproxy/"):
                return m
        return matches[0]

    # Backward-compatible fallback.
    return f"openai/{s}"


def main(argv: list[str] | None = None) -> int:
    """中文说明：
    - 含义：CLI 入口（`python -m runner ...` / `runner.__main__` / `fsm_runner.py`）。
    - 内容：解析参数→加载 dotenv→准备 repo（本地/clone/HF）→加载/必要时 scaffold pipeline 合同→构造 RunnerConfig→运行 preflight 或进入闭环 run。
    - 可简略：否（入口胶水层；但可考虑把重复逻辑拆到更小的 helper 以便复用/测试）。
    """
    parser = argparse.ArgumentParser(description="OpenCode-FSM runner (single-process closed-loop executor)")
    parser.add_argument("--repo", default=".", help="repo root path (default: .)")
    parser.add_argument("--goal", default="", help="goal for PLAN.md (used only when PLAN.md is missing)")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "model name as provider/model (recommended; run `opencode models` to list). "
            "If omitted, uses OPENAI_MODEL or LITELLM_CHAT_MODEL when present."
        ),
    )
    parser.add_argument("--test-cmd", default=None, help='acceptance command (default: pipeline/tests or "pytest -q")')
    parser.add_argument("--pipeline", default="", help="pipeline YAML path relative to repo (optional)")
    parser.add_argument(
        "--require-pipeline",
        action="store_true",
        help="fail if no pipeline is provided/found (recommended for one-command contract runs)",
    )
    parser.add_argument(
        "--scaffold-contract",
        choices=("off", "opencode"),
        default="off",
        help=(
            "when pipeline.yml is missing: generate a contract scaffold. "
            "`opencode` uses the agent to generate the contract. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--scaffold-require-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require `.aider_fsm/metrics.json` with required keys (default: true)",
    )
    parser.add_argument(
        "--scaffold-opencode-bash",
        choices=("restricted", "full"),
        default="restricted",
        help="OpenCode bash permission mode during `--scaffold-contract opencode` (default: restricted)",
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
        help="when --repo is a git URL, clone into this directory (default: /data/tiansha/aider_fsm_targets if writable, else /tmp/aider_fsm_targets)",
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
        help="run verification once then exit (does not start the agent loop)",
    )
    parser.add_argument("--opencode-url", default="", help="OpenCode server base URL (if set, do not auto-start)")
    parser.add_argument(
        "--opencode-timeout",
        type=int,
        default=300,
        help="OpenCode HTTP timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--opencode-retry-attempts",
        type=int,
        default=2,
        help="OpenCode message retry attempts for timeout/transient errors (default: 2)",
    )
    parser.add_argument(
        "--opencode-retry-backoff-seconds",
        type=float,
        default=2.0,
        help="OpenCode retry backoff base seconds (exponential; default: 2.0)",
    )
    parser.add_argument(
        "--opencode-context-length",
        type=int,
        default=0,
        help="OpenCode context length hint for scaffold/repair (0 = unset)",
    )
    parser.add_argument(
        "--opencode-max-prompt-chars",
        type=int,
        default=0,
        help="Max prompt chars sent to OpenCode (0 = no clipping)",
    )
    parser.add_argument(
        "--opencode-bash",
        choices=("restricted", "full"),
        default="restricted",
        help="OpenCode bash permission mode (default: restricted)",
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
        # OpenCode's OpenAI provider reads `OPENAI_BASE_URL` (not `OPENAI_API_BASE`).
        # Keep backward compatibility with existing `.env` files, but avoid breaking
        # when an internal proxy is down/unreachable.
        base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip()
        api_base = str(os.environ.get("OPENAI_API_BASE") or "").strip()
        should_set = (bool(args.env_override) or not base_url) and bool(api_base)
        if should_set:
            u = urlparse(api_base if "://" in api_base else f"http://{api_base}")
            host = u.hostname
            port = int(u.port or (443 if (u.scheme or "").lower() == "https" else 80))
            reachable = False
            if host:
                try:
                    with socket.create_connection((host, port), timeout=0.5):
                        reachable = True
                except OSError:
                    reachable = False
            if reachable:
                b = api_base.rstrip("/")
                if not b.endswith("/v1"):
                    b = b + "/v1"
                os.environ["OPENAI_BASE_URL"] = b
            else:
                print(
                    f"WARN: OPENAI_API_BASE is set but not reachable ({api_base}); "
                    "OpenCode will use its default base URL unless OPENAI_BASE_URL is set.",
                    file=sys.stderr,
                )

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
    pipeline_arg = str(args.pipeline or "").strip()
    if pipeline_arg:
        pipeline_abs = resolve_config_path(repo, pipeline_arg)
        pipeline = load_pipeline_spec(pipeline_abs)
        pipeline_rel = relpath_or_none(pipeline_abs, repo)
    else:
        # Contract mode convenience: repos can self-describe via a root `pipeline.yml`.
        default_pipeline = (repo / "pipeline.yml").resolve()
        if default_pipeline.exists():
            pipeline_abs = default_pipeline
            pipeline = load_pipeline_spec(pipeline_abs)
            pipeline_rel = relpath_or_none(pipeline_abs, repo)
    scaffold_contract = str(args.scaffold_contract or "off").strip().lower() or "off"
    if scaffold_contract not in ("off", "opencode"):
        scaffold_contract = "off"

    # Convenience: for remote repos without a contract, default to OpenCode scaffolding so
    # `--repo <url>` can run end-to-end without requiring repo-side YAML upfront.
    if scaffold_contract == "off" and prepared.cloned_from and pipeline_abs is None:
        scaffold_contract = "opencode"
        print(
            "INFO: pipeline.yml not found in remote repo; auto-scaffolding a minimal contract (opencode). "
            "Disable with --scaffold-contract off.",
            file=sys.stderr,
        )

    if args.require_pipeline and pipeline_abs is None and scaffold_contract == "off":
        print(
            "ERROR: pipeline.yml not found and --pipeline not provided. "
            "For one-command contract runs, add a pipeline.yml at the repo root (version: 1).",
            file=sys.stderr,
        )
        return 2

    tests_from_user = bool(args.test_cmd and str(args.test_cmd).strip())

    tests_cmds: list[str]
    if tests_from_user:
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

    model = str(args.model or "").strip()
    if not model:
        model = str(
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("CHAT_MODEL")
            or os.environ.get("LITELLM_CHAT_MODEL")
            or ""
        ).strip()
    model = _resolve_model(model)

    cfg = RunnerConfig(
        repo=repo,
        goal=str(args.goal or ""),
        model=model or "openai/gpt-4o-mini",
        plan_rel=str(args.plan_path or "PLAN.md"),
        pipeline_abs=pipeline_abs,
        pipeline_rel=pipeline_rel,
        pipeline=pipeline,
        tests_cmds=tests_cmds,
        effective_test_cmd=effective_test_cmd,
        tests_from_user=tests_from_user,
        require_pipeline=bool(args.require_pipeline),
        scaffold_contract=scaffold_contract,
        scaffold_require_metrics=bool(args.scaffold_require_metrics),
        strict_opencode=bool(args.strict_opencode),
        artifacts_base=artifacts_base,
        seed_files=seed_files,
        max_iters=int(args.max_iters),
        max_fix=int(args.max_fix),
        unattended=str(args.unattended),
        preflight_only=bool(args.preflight_only),
        opencode_url=str(args.opencode_url or ""),
        opencode_timeout_seconds=int(args.opencode_timeout or 300),
        opencode_retry_attempts=int(args.opencode_retry_attempts or 0),
        opencode_retry_backoff_seconds=float(args.opencode_retry_backoff_seconds or 0.0),
        opencode_context_length=(int(args.opencode_context_length) if int(args.opencode_context_length or 0) > 0 else None),
        opencode_max_prompt_chars=(int(args.opencode_max_prompt_chars) if int(args.opencode_max_prompt_chars or 0) > 0 else None),
        opencode_bash=str(args.opencode_bash or "restricted"),
        scaffold_opencode_bash=str(args.scaffold_opencode_bash or "restricted"),
    )
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
