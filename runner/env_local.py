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
from .contract_hints import suggest_contract_hints
from .opencode_client import OpenCodeClient
from .prompts import make_scaffold_contract_prompt
from .pipeline_spec import PipelineSpec, load_pipeline_spec
from .pipeline_verify import run_pipeline_verification
from .repo_resolver import prepare_repo
from .scaffold_validation import validate_scaffolded_files, validate_scaffolded_pipeline
from .subprocess_utils import tail, write_text
from .types import VerificationResult


@dataclass(frozen=True)
class EnvHandle:
    """中文说明：
    - 含义：同机调用（programmatic）模式下的“环境句柄”。
    - 内容：包含目标 repo 根目录、pipeline.yml 路径与解析后的 PipelineSpec；供 rollout/evaluate API 复用。
    - 可简略：可能（只是数据载体；但显式结构便于类型检查与扩展）。
    """

    repo: Path
    pipeline_path: Path
    pipeline: PipelineSpec


@dataclass(frozen=True)
class RolloutCallResult:
    """中文说明：
    - 含义：一次 rollout 调用的返回结果（programmatic API）。
    - 内容：包含 ok、artifacts_dir、可选 rollout.json 路径，以及完整 VerificationResult（便于读取 stage 结果与 stdout/stderr）。
    - 可简略：可能（对外 API 结果结构；可以裁字段但会影响调用方兼容性）。
    """

    ok: bool
    artifacts_dir: Path
    rollout_path: Path | None
    verify: VerificationResult


@dataclass(frozen=True)
class EvaluationCallResult:
    """中文说明：
    - 含义：一次 evaluation 调用的返回结果（programmatic API）。
    - 内容：包含 ok、artifacts_dir、metrics_path 与解析后的 metrics dict，以及完整 VerificationResult。
    - 可简略：可能（对外 API 结果结构；可以裁字段但会影响调用方兼容性）。
    """

    ok: bool
    artifacts_dir: Path
    metrics_path: Path | None
    metrics: dict | None
    verify: VerificationResult


@dataclass(frozen=True)
class DeployCallResult:
    """中文说明：
    - 含义：一次 deploy（setup+health）调用的返回结果（programmatic API）。
    - 内容：包含 ok、artifacts_dir、runtime_env.json 路径与解析后的 runtime_env dict，以及完整 VerificationResult。
    - 可简略：可能（对外 API 结果结构；字段可裁剪但会影响调用方兼容性）。
    """

    ok: bool
    artifacts_dir: Path
    runtime_env_path: Path | None
    runtime_env: dict | None
    verify: VerificationResult


def _default_artifacts_dir(repo: Path, *, prefix: str) -> Path:
    """中文说明：
    - 含义：生成本次 programmatic 调用的默认 artifacts 输出目录。
    - 内容：路径为 `.aider_fsm/artifacts/<run_id>/<prefix>`；run_id 使用当前时间戳。
    - 可简略：可能（小工具；但统一目录结构有利于审计与调试）。
    """
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return (repo / ".aider_fsm" / "artifacts" / run_id / prefix).resolve()


def _list_opencode_models() -> list[str]:
    """中文说明：
    - 含义：调用 `opencode models` 获取可用模型列表（去 ANSI）。
    - 内容：用于默认模型选择与把裸模型名解析到 provider/model。
    - 可简略：是（与 `runner/cli.py` 重复，可抽公共模块）。
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
    - 含义：规范化模型参数为 `provider/model`。
    - 内容：优先选择本机可用的默认模型；裸模型名会尝试在 `opencode models` 中匹配 provider（优先 myproxy）；最终回退到 openai/<id>。
    - 可简略：是（与 `runner/cli.py` 重复，可抽公共模块）。
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
    candidates = _list_opencode_models()
    matches = [m for m in candidates if m.split("/", 1)[1] == s]
    if matches:
        for m in matches:
            if m.startswith("myproxy/"):
                return m
        return matches[0]
    return f"openai/{s}"


def _ensure_contract_stage_skeleton(repo_root: Path) -> None:
    """Best-effort: create robust default stage scripts before OpenCode scaffolding.

    This reduces common scaffolding failures (e.g., broken heredocs) without introducing
    benchmark-specific logic. The scaffolder may still edit/override these scripts.
    """
    repo_root = Path(repo_root).resolve()
    stages = (repo_root / ".aider_fsm" / "stages").resolve()
    stages.mkdir(parents=True, exist_ok=True)

    def _write_if_missing(rel: str, content: str) -> None:
        p = (repo_root / rel).resolve()
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    _write_if_missing(
        ".aider_fsm/stages/tests.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo tests_skipped\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/deploy_setup.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "\n"
        "if [ \"${AIDER_LLM_KIND:-local_hf}\" = \"remote\" ]; then\n"
        "  if [ -z \"${AIDER_LLM_MODEL:-}\" ]; then\n"
        "    \"$AIDER_FSM_PYTHON\" - <<'PY' > \"$RUNTIME_ENV_JSON\"\n"
        "import json, os, time\n"
        "base = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL') or ''\n"
        "model = os.getenv('AIDER_LLM_MODEL') or os.getenv('OPENAI_MODEL') or ''\n"
        "obj = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'run_id': os.getenv('AIDER_FSM_RUN_ID',''),\n"
        "  'service': {},\n"
        "  'inference': {'type': 'openai_compat', 'openai_base_url': base, 'model': model},\n"
        "  'paths': {'rollout_path': '.aider_fsm/rollout.json', 'metrics_path': '.aider_fsm/metrics.json'},\n"
        "  'ok': False,\n"
        "  'reason': 'AIDER_LLM_MODEL is not set',\n"
        "}\n"
        "print(json.dumps(obj, ensure_ascii=False, indent=2))\n"
        "PY\n"
        "    exit 2\n"
        "  fi\n"
        "  \"$AIDER_FSM_PYTHON\" - <<'PY' > \"$RUNTIME_ENV_JSON\"\n"
        "import json, os, time\n"
        "base = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL') or ''\n"
        "model = os.getenv('AIDER_LLM_MODEL') or os.getenv('OPENAI_MODEL') or ''\n"
        "obj = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'run_id': os.getenv('AIDER_FSM_RUN_ID',''),\n"
        "  'service': {},\n"
        "  'inference': {'type': 'openai_compat', 'openai_base_url': base, 'model': model},\n"
        "  'paths': {'rollout_path': '.aider_fsm/rollout.json', 'metrics_path': '.aider_fsm/metrics.json'},\n"
        "  'ok': True,\n"
        "}\n"
        "print(json.dumps(obj, ensure_ascii=False, indent=2))\n"
        "PY\n"
        "  echo \"Wrote $RUNTIME_ENV_JSON\"\n"
        "  exit 0\n"
        "fi\n"
        "\n"
        "if [ -z \"${AIDER_TRAINED_MODEL_DIR:-}\" ]; then\n"
        "  \"$AIDER_FSM_PYTHON\" - <<'PY' > \"$RUNTIME_ENV_JSON\"\n"
        "import json, os, time\n"
        "obj = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'run_id': os.getenv('AIDER_FSM_RUN_ID',''),\n"
        "  'service': {},\n"
        "  'inference': {'type': 'local_hf', 'model_dir': ''},\n"
        "  'paths': {'rollout_path': '.aider_fsm/rollout.json', 'metrics_path': '.aider_fsm/metrics.json'},\n"
        "  'ok': False,\n"
        "  'reason': 'AIDER_TRAINED_MODEL_DIR is not set',\n"
        "}\n"
        "print(json.dumps(obj, ensure_ascii=False, indent=2))\n"
        "PY\n"
        "  exit 2\n"
        "fi\n"
        "\n"
        "# Recommended: use the runner's built-in OpenAI-compatible server helper.\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.ml.serve_openai_compat start \\\n"
        "  --backend hf \\\n"
        "  --model-dir \"$AIDER_TRAINED_MODEL_DIR\" \\\n"
        "  --host 127.0.0.1 \\\n"
        "  --port 0 \\\n"
        "  --runtime-env-out \"$RUNTIME_ENV_JSON\" \\\n"
        "  --pid-file .aider_fsm/server.pid \\\n"
        "  --log-file \"${AIDER_FSM_ARTIFACTS_DIR:-.aider_fsm/artifacts}/server.log\"\n"
        "echo \"Wrote $RUNTIME_ENV_JSON\"\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/deploy_health.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "test -f \"$RUNTIME_ENV_JSON\"\n"
        "# Best-effort: check health_url if present.\n"
        "\"$AIDER_FSM_PYTHON\" - <<'PY'\n"
        "import json, urllib.request, os\n"
        "p = os.environ.get('AIDER_RUNTIME_ENV_PATH') or '.aider_fsm/runtime_env.json'\n"
        "obj = json.load(open(p,'r',encoding='utf-8'))\n"
        "url = (obj.get('service') or {}).get('health_url')\n"
        "if not url:\n"
        "  raise SystemExit(0)\n"
        "with urllib.request.urlopen(url, timeout=3) as r:\n"
        "  _ = r.read()\n"
        "print('ok')\n"
        "PY\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/deploy_teardown.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.ml.serve_openai_compat stop --pid-file .aider_fsm/server.pid || true\n"
        "echo teardown_ok\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/rollout.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.generic_rollout\n"
        "echo \"Wrote .aider_fsm/rollout.json\"\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/evaluation.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.generic_evaluation\n",
    )
    _write_if_missing(
        ".aider_fsm/stages/benchmark.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo benchmark_skipped\n",
    )


def _write_fallback_pipeline_yml(repo_root: Path, *, pipeline_rel: str, require_metrics: bool) -> None:
    """Write a minimal v1 pipeline.yml referencing `.aider_fsm/stages/*.sh` (best-effort)."""
    repo_root = Path(repo_root).resolve()
    pipeline_rel = str(pipeline_rel or "pipeline.yml").strip() or "pipeline.yml"
    pipeline_path = (repo_root / pipeline_rel).resolve()
    if pipeline_path.exists():
        return
    required_keys = "[score, ok]" if require_metrics else "[]"
    pipeline_path.write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 7200",
                "  max_total_seconds: 86400",
                "tests:",
                "  cmds:",
                "    - bash .aider_fsm/stages/tests.sh",
                "deploy:",
                "  setup_cmds:",
                "    - bash .aider_fsm/stages/deploy_setup.sh",
                "  health_cmds:",
                "    - bash .aider_fsm/stages/deploy_health.sh",
                "  teardown_policy: on_failure",
                "  teardown_cmds:",
                "    - bash .aider_fsm/stages/deploy_teardown.sh",
                "rollout:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/rollout.sh",
                "evaluation:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/evaluation.sh",
                "  metrics_path: .aider_fsm/metrics.json",
                f"  required_keys: {required_keys}",
                "benchmark:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/benchmark.sh",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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
    seed_stage_skeleton: bool = True,
    write_fallback_pipeline_yml: bool = True,
    agent: AgentClient | None = None,
) -> EnvHandle:
    """中文说明：
    - 含义：打开一个“可同机调用”的目标 repo 环境（必要时自动下载/clone 并 scaffold 合同）。
    - 内容：
      1) `prepare_repo`：本地路径/远程 git/HF dataset → 得到 repo_root
      2) pipeline.yml 存在则加载；否则（scaffold_contract=opencode）启动/连接 OpenCode 生成 `pipeline.yml` + `.aider_fsm/**`
      3) scaffold 后校验 pipeline 满足最小要求（尤其 metrics 合同）
      4) 返回 EnvHandle（repo/pipeline_path/pipeline）
    - 可简略：否（这是 programmatic API 的入口胶水层）。
    """
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
            scaffold_err = ""
            try:
                if bool(seed_stage_skeleton):
                    # Provide a robust starting point to reduce common scaffolding failures.
                    _ensure_contract_stage_skeleton(repo_root)
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
                try:
                    hints = suggest_contract_hints(repo_root)
                    if hints.commands:
                        write_text(out_dir / "scaffold_command_hints.txt", "\n".join(hints.commands) + "\n")
                    res = agent.run(
                        make_scaffold_contract_prompt(
                            repo_root,
                            pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                            require_metrics=bool(scaffold_require_metrics),
                            command_hints=hints.commands,
                        ),
                        fsm_state="S0_SCAFFOLD",
                        iter_idx=0,
                        purpose="scaffold_contract",
                    )
                    write_text(out_dir / "scaffold_agent_result.txt", tail(res.assistant_text or "", 20000) + "\n")
                except Exception as e:
                    # Important: OpenCode timeouts can happen even when the agent has already written files.
                    # If the contract artifacts exist and validate, continue instead of failing fast.
                    scaffold_err = tail(str(e), 4000)
                    write_text(out_dir / "scaffold_agent_error.txt", scaffold_err + "\n")
            finally:
                if created_agent and agent is not None:
                    try:
                        agent.close()
                    except Exception:
                        pass

            if not pipeline_path.exists():
                if bool(write_fallback_pipeline_yml):
                    # Fallback: write a minimal pipeline.yml ourselves so downstream repair can proceed.
                    # This keeps the behavior generic and avoids hard-failing when a model forgets tool calls.
                    try:
                        _write_fallback_pipeline_yml(
                            repo_root,
                            pipeline_rel=str(pipeline_rel).strip() or "pipeline.yml",
                            require_metrics=bool(scaffold_require_metrics),
                        )
                        write_text(out_dir / "scaffold_fallback_used.txt", "wrote_fallback_pipeline_yml\n")
                    except Exception:
                        pass
            if not pipeline_path.exists():
                write_text(out_dir / "scaffold_error.txt", "scaffold_contract_failed: missing_pipeline_yml\n")
                raise RuntimeError(
                    f"scaffold_contract_failed: pipeline not created: {pipeline_path}"
                    + (f" (agent_error: {scaffold_err})" if scaffold_err else "")
                )

            parsed = load_pipeline_spec(pipeline_path)
            missing = validate_scaffolded_pipeline(parsed, require_metrics=bool(scaffold_require_metrics))
            missing_files = validate_scaffolded_files(repo_root)
            if missing or missing_files:
                write_text(
                    out_dir / "scaffold_error.txt",
                    "scaffold_contract_failed: incomplete_contract\n"
                    + "\n".join([f"- {x}" for x in missing])
                    + ("\n" if missing else "")
                    + "\n".join([f"- missing_file: {x}" for x in missing_files])
                    + ("\n" if missing_files else ""),
                )
                raise RuntimeError(
                    f"scaffold_contract_failed: incomplete contract: {missing + missing_files}"
                    + (f" (agent_error: {scaffold_err})" if scaffold_err else "")
                )
        elif require_pipeline:
            raise FileNotFoundError(f"pipeline not found: {pipeline_path}")
        else:
            pipeline = PipelineSpec()
            return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)
    pipeline = load_pipeline_spec(pipeline_path)
    return EnvHandle(repo=repo_root, pipeline_path=pipeline_path, pipeline=pipeline)


def _merge_env(base: dict[str, str], overrides: dict[str, str] | None) -> dict[str, str]:
    """中文说明：
    - 含义：合并环境变量映射（overrides 覆盖 base）。
    - 内容：用于把调用方传入的 `env_overrides` 合并到 pipeline 的 stage env 中。
    - 可简略：可能（简单 merge；但单独函数能让调用点更清晰）。
    """
    out = dict(base or {})
    if overrides:
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


def _stage_only_pipeline(
    pipeline: PipelineSpec,
    *,
    deploy: bool = False,
    rollout: bool,
    evaluation: bool,
    benchmark: bool,
    env_overrides: dict[str, str] | None,
) -> PipelineSpec:
    """中文说明：
    - 含义：构造一个“只跑指定 stages”的临时 PipelineSpec（用于 programmatic 调用）。
    - 内容：保留 security 限制；按需要保留 rollout/evaluation/benchmark 的 run_cmds 与 metrics 校验配置；并把 env_overrides 合并到各 stage 的 env。
    - 可简略：可能（programmatic 专用；也可直接在调用点构造，但会很冗长）。
    """
    env_overrides = env_overrides or {}
    return PipelineSpec(
        # security
        security_mode=str(pipeline.security_mode or "safe"),
        security_allowlist=list(pipeline.security_allowlist or []),
        security_denylist=list(pipeline.security_denylist or []),
        security_max_cmd_seconds=pipeline.security_max_cmd_seconds,
        security_max_total_seconds=pipeline.security_max_total_seconds,
        # deploy
        deploy_setup_cmds=list(pipeline.deploy_setup_cmds or []) if deploy else [],
        deploy_health_cmds=list(pipeline.deploy_health_cmds or []) if deploy else [],
        deploy_teardown_cmds=list(pipeline.deploy_teardown_cmds or []) if deploy else [],
        deploy_timeout_seconds=pipeline.deploy_timeout_seconds if deploy else None,
        deploy_retries=int(pipeline.deploy_retries or 0) if deploy else 0,
        deploy_env=_merge_env(pipeline.deploy_env, env_overrides) if deploy else {},
        deploy_workdir=pipeline.deploy_workdir if deploy else None,
        # Keep the environment running after `deploy()` so callers can run rollout/evaluation later.
        deploy_teardown_policy=("on_failure" if deploy else str(pipeline.deploy_teardown_policy or "always")),
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
    """中文说明：
    - 含义：在同机调用模式下执行 rollout（并记录 artifacts）。
    - 内容：构造 stage-only pipeline（rollout=true）；可选先执行 `.aider_fsm/bootstrap.yml`；然后调用 `run_pipeline_verification`（tests 用 echo 跳过）并返回 RolloutCallResult。
    - 可简略：可能（与 evaluate/rollout_and_evaluate 结构相似，可进一步抽公共逻辑）。
    """
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=True, evaluation=False, benchmark=False, env_overrides=env_overrides)
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
    """中文说明：
    - 含义：在同机调用模式下执行 evaluation（并进行 metrics 校验与 artifacts 记录）。
    - 内容：构造 stage-only pipeline（evaluation=true）；可选 bootstrap；调用 `run_pipeline_verification` 并从结果中提取 metrics_path/metrics。
    - 可简略：可能（与 rollout/rollout_and_evaluate 结构相似，可抽公共逻辑）。
    """
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=False, evaluation=True, benchmark=False, env_overrides=env_overrides)
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
    """中文说明：
    - 含义：同机调用模式下一次性执行 rollout + evaluation（单次 verification pass）。
    - 内容：构造 stage-only pipeline（rollout=true,evaluation=true）；可选 bootstrap；然后一次 `run_pipeline_verification` 同时跑两阶段并校验 evaluation metrics。
    - 可简略：可能（组合 API；保留能让调用方更方便）。
    """
    # One verification pass runs rollout then evaluation (and validates evaluation metrics).
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="rollout_evaluation")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=False, rollout=True, evaluation=True, benchmark=False, env_overrides=env_overrides)
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

    rollout_stage = getattr(verify, "rollout", None)
    rollout_stage_ok = bool(getattr(rollout_stage, "ok", False)) if rollout_stage is not None else bool(verify.ok)
    return (
        RolloutCallResult(ok=rollout_stage_ok, artifacts_dir=artifacts_dir, rollout_path=rollout_path, verify=verify),
        EvaluationCallResult(
            ok=bool(verify.ok),
            artifacts_dir=artifacts_dir,
            metrics_path=metrics_path,
            metrics=metrics,
            verify=verify,
        ),
    )


def with_env_vars(extra_env: dict[str, str]) -> dict[str, str]:
    """中文说明：
    - 含义：便捷函数：把一组 env 变量包装成可传入 `env_overrides` 的 dict。
    - 内容：当前只是调用 `_merge_env({}, extra_env)`；示例：传入 `OPENCODE_MODEL` 控制 stage 脚本使用的模型。
    - 可简略：是（非常薄的 helper；但对调用方可读性友好）。

    Example:
      rollout_and_evaluate(env, env_overrides=with_env_vars({"OPENCODE_MODEL": "opencode/gpt-5-nano"}))
    """

    return _merge_env({}, extra_env)


def with_runtime_env_path(runtime_env_path: str | Path) -> dict[str, str]:
    """中文说明：
    - 含义：把 deploy 产出的 runtime_env.json 路径注入为 stage 可读的环境变量（programmatic API 便捷函数）。
    - 内容：返回 `{"AIDER_RUNTIME_ENV_PATH": "<abs path>"}` 形式的 env_overrides，可直接传给 `rollout/evaluate/rollout_and_evaluate`。
    - 可简略：是（只是便捷 wrapper；你也可以手动传 env_overrides）。
    """
    p = Path(str(runtime_env_path)).expanduser().resolve()
    return _merge_env({}, {"AIDER_RUNTIME_ENV_PATH": str(p)})


def deploy(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
    run_bootstrap_first: bool = True,
    runtime_env_rel: str = ".aider_fsm/runtime_env.json",
) -> DeployCallResult:
    """中文说明：
    - 含义：在同机调用模式下执行 deploy（setup+health），并读取 deploy 产出的 runtime_env.json。
    - 内容：
      - 构造 stage-only pipeline（deploy=true），跳过 tests；
      - 可选先执行 `.aider_fsm/bootstrap.yml`；
      - 要求 deploy_setup 至少能写出 `runtime_env_rel`（否则 runtime_env_path=None）。
    - 可简略：否（这是“先启动服务，再调用 rollout/evaluation”的关键入口）。
    """
    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="deploy")).resolve()
    p = _stage_only_pipeline(env.pipeline, deploy=True, rollout=False, evaluation=False, benchmark=False, env_overrides=env_overrides)
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
            return DeployCallResult(
                ok=False,
                artifacts_dir=artifacts_dir,
                runtime_env_path=None,
                runtime_env=None,
                verify=verify,
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

    runtime_env_path = (env.repo / str(runtime_env_rel)).resolve()
    runtime_env = None
    if runtime_env_path.exists():
        try:
            import json

            runtime_env = json.loads(runtime_env_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(runtime_env, dict):
                runtime_env = None
        except Exception:
            runtime_env = None
    else:
        runtime_env_path = None

    return DeployCallResult(
        ok=bool(verify.ok),
        artifacts_dir=artifacts_dir,
        runtime_env_path=runtime_env_path,
        runtime_env=runtime_env,
        verify=verify,
    )


def deploy_teardown(
    env: EnvHandle,
    *,
    artifacts_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    unattended: str = "strict",
) -> bool:
    """Best-effort run of deploy teardown commands.

    中文说明：
    - 含义：在同机调用模式下显式执行 deploy teardown（用于 one-shot 场景，避免遗留后台服务/容器）。
    - 内容：复用 pipeline 的 deploy.teardown_cmds；在单独的 verification pass 中以 `teardown_policy=always` 触发 finally teardown 执行；
      通过读取 `deploy_teardown_summary.json` 判断是否成功。
    - 可简略：否（实际运行会产生副作用；显式 API 能让调用方可控地清理资源）。
    """
    if not list(env.pipeline.deploy_teardown_cmds or []):
        return True

    artifacts_dir = (artifacts_dir or _default_artifacts_dir(env.repo, prefix="deploy_teardown")).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    p = PipelineSpec(
        # security
        security_mode=str(env.pipeline.security_mode or "safe"),
        security_allowlist=list(env.pipeline.security_allowlist or []),
        security_denylist=list(env.pipeline.security_denylist or []),
        security_max_cmd_seconds=env.pipeline.security_max_cmd_seconds,
        security_max_total_seconds=env.pipeline.security_max_total_seconds,
        # deploy (teardown only)
        deploy_setup_cmds=[],
        deploy_health_cmds=[],
        deploy_teardown_cmds=list(env.pipeline.deploy_teardown_cmds or []),
        deploy_timeout_seconds=env.pipeline.deploy_timeout_seconds,
        deploy_retries=0,
        deploy_env=_merge_env(env.pipeline.deploy_env, env_overrides),
        deploy_workdir=env.pipeline.deploy_workdir,
        deploy_teardown_policy="always",
        # no rollout/eval/bench/auth
    )

    # run_pipeline_verification executes deploy_teardown in its `finally` block.
    run_pipeline_verification(
        env.repo,
        pipeline=p,
        tests_cmds=["echo tests_skipped"],
        artifacts_dir=artifacts_dir,
        unattended=str(unattended or "strict"),
    )

    summary_path = (artifacts_dir / "deploy_teardown_summary.json").resolve()
    if not summary_path.exists():
        return False
    try:
        import json

        obj = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
        return bool(isinstance(obj, dict) and obj.get("ok") is True)
    except Exception:
        return False
