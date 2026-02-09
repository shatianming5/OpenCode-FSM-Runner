from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .subprocess_utils import read_text_if_exists


@dataclass(frozen=True)
class PipelineSpec:
    """中文说明：
    - 含义：`pipeline.yml`（v1）的解析后结构：声明式验收合同（tests/deploy/rollout/evaluation/benchmark/metrics 等）。
    - 内容：保存各 stage 的命令、超时、重试、env/workdir、安全策略与 artifacts 输出目录；runner 会按既定顺序执行并记录结果。
    - 可简略：否（这是项目最核心的“契约层”；字段变化会影响兼容性与安全边界）。

    ---

    English (original intent):
    Declarative contract for verification stages.
    NOTE: The runner intentionally treats this file as *human-owned* and will revert
    any model edits during plan-update/execute steps.
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈112 行；引用次数≈48（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_spec.py:24；类型=class；引用≈48；规模≈112行

    version: int = 1

    # tests
    tests_cmds: list[str] = None  # type: ignore[assignment]
    tests_timeout_seconds: int | None = None
    tests_retries: int = 0
    tests_env: dict[str, str] = None  # type: ignore[assignment]
    tests_workdir: str | None = None

    # deploy
    deploy_setup_cmds: list[str] = None  # type: ignore[assignment]
    deploy_health_cmds: list[str] = None  # type: ignore[assignment]
    deploy_teardown_cmds: list[str] = None  # type: ignore[assignment]
    deploy_timeout_seconds: int | None = None
    deploy_retries: int = 0
    deploy_env: dict[str, str] = None  # type: ignore[assignment]
    deploy_workdir: str | None = None
    deploy_teardown_policy: str = "always"  # always|on_success|on_failure|never

    kubectl_dump_enabled: bool = False
    kubectl_dump_namespace: str | None = None
    kubectl_dump_label_selector: str | None = None
    kubectl_dump_include_logs: bool = False

    # rollout (optional)
    rollout_run_cmds: list[str] = None  # type: ignore[assignment]
    rollout_timeout_seconds: int | None = None
    rollout_retries: int = 0
    rollout_env: dict[str, str] = None  # type: ignore[assignment]
    rollout_workdir: str | None = None

    # evaluation (optional; preferred over benchmark for "evaluation metrics")
    evaluation_run_cmds: list[str] = None  # type: ignore[assignment]
    evaluation_timeout_seconds: int | None = None
    evaluation_retries: int = 0
    evaluation_env: dict[str, str] = None  # type: ignore[assignment]
    evaluation_workdir: str | None = None
    evaluation_metrics_path: str | None = None
    evaluation_required_keys: list[str] = None  # type: ignore[assignment]

    # benchmark
    benchmark_run_cmds: list[str] = None  # type: ignore[assignment]
    benchmark_timeout_seconds: int | None = None
    benchmark_retries: int = 0
    benchmark_env: dict[str, str] = None  # type: ignore[assignment]
    benchmark_workdir: str | None = None
    benchmark_metrics_path: str | None = None
    benchmark_required_keys: list[str] = None  # type: ignore[assignment]

    # auth (optional)
    auth_cmds: list[str] = None  # type: ignore[assignment]
    auth_timeout_seconds: int | None = None
    auth_retries: int = 0
    auth_env: dict[str, str] = None  # type: ignore[assignment]
    auth_workdir: str | None = None
    auth_interactive: bool = False

    # artifacts
    artifacts_out_dir: str | None = None

    # tooling (deprecated; prefer .aider_fsm/actions.yml)
    tooling_ensure_tools: bool = False
    tooling_ensure_kind_cluster: bool = False
    tooling_kind_cluster_name: str = "kind"
    tooling_kind_config: str | None = None

    # security
    security_mode: str = "safe"  # safe|system
    security_allowlist: list[str] = None  # type: ignore[assignment]
    security_denylist: list[str] = None  # type: ignore[assignment]
    security_max_cmd_seconds: int | None = None
    security_max_total_seconds: int | None = None

    def __post_init__(self) -> None:
        """中文说明：
        - 含义：把所有 list/dict 字段从 None 归一化为 `[]/{}`。
        - 内容：避免调用点写大量 `or []` / `or {}`；同时保持 dataclass frozen 的不变性。
        - 可简略：否
        - 原因：统一空值语义能显著减少 `None` 分支与调用点的防御性写法，避免隐性 bug。
        """
        # 作用：中文说明：
        # 能否简略：部分
        # 原因：规模≈25 行；引用次数≈0（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/pipeline_spec.py:104；类型=method；引用≈0；规模≈25行
        for attr in (
            "tests_cmds",
            "deploy_setup_cmds",
            "deploy_health_cmds",
            "deploy_teardown_cmds",
            "rollout_run_cmds",
            "evaluation_run_cmds",
            "benchmark_run_cmds",
            "evaluation_required_keys",
            "benchmark_required_keys",
            "auth_cmds",
            "security_allowlist",
            "security_denylist",
        ):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, [])
        for attr in ("tests_env", "deploy_env", "rollout_env", "evaluation_env", "benchmark_env", "auth_env"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, {})


def load_pipeline_spec(path: Path) -> PipelineSpec:
    """中文说明：
    - 含义：从 `pipeline.yml` 读取并解析 PipelineSpec（v1）。
    - 内容：使用 PyYAML 加载顶层 mapping；校验 version=1；把各 stage 的 cmd/cmds、env、workdir、metrics_path/required_keys、安全策略等字段解析为强类型对象。
    - 可简略：否（契约解析与校验决定 runner 能否安全、确定性地执行）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈187 行；引用次数≈13（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/pipeline_spec.py:131；类型=function；引用≈13；规模≈187行
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"pipeline file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("pipeline must be a YAML mapping (dict) at the top level")

    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported pipeline version: {version}")

    def _as_mapping(value: Any, name: str) -> dict[str, Any]:
        """中文说明：
        - 含义：把 YAML 字段规整为 dict（mapping），便于后续强类型解析。
        - 内容：允许 None → `{}`；否则要求是 dict，否则抛出带路径的 ValueError。
        - 可简略：是（小工具函数；也可用重复的 if/raise 替代）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈11 行；引用次数≈11（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/pipeline_spec.py:153；类型=function；引用≈11；规模≈11行
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"pipeline.{name} must be a mapping")
        return value

    def _as_env(value: Any, name: str) -> dict[str, str]:
        """中文说明：
        - 含义：把某个 `*.env` 字段解析为 `dict[str, str]`。
        - 内容：过滤空 key；把值统一转成字符串（None → `\"\"`）；类型不符时报错。
        - 可简略：可能（重复较多，但集中处理能保证一致性与可测试性）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈19 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/pipeline_spec.py:165；类型=function；引用≈7；规模≈19行
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"pipeline.{name}.env must be a mapping")
        out: dict[str, str] = {}
        for k, v in value.items():
            if k is None:
                continue
            ks = str(k).strip()
            if not ks:
                continue
            out[ks] = "" if v is None else str(v)
        return out

    def _as_cmds(m: dict[str, Any], *, cmd_key: str, cmds_key: str) -> list[str]:
        """中文说明：
        - 含义：把 `cmd`/`cmds` 两种写法统一成 `list[str]`。
        - 内容：优先读取 `cmds_key`（要求 list[str]）；否则读取 `cmd_key`（要求非空 str）；都缺失则返回空列表。
        - 可简略：是（本质是兼容层；若未来只保留一种字段，可删除）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈17 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/pipeline_spec.py:185；类型=function；引用≈9；规模≈17行
        if cmds_key in m and m.get(cmds_key) is not None:
            v = m.get(cmds_key)
            if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                raise ValueError(f"pipeline field {cmds_key} must be a list of non-empty strings")
            return [x.strip() for x in v if x.strip()]
        v = m.get(cmd_key)
        if v is None:
            return []
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"pipeline field {cmd_key} must be a non-empty string")
        return [v.strip()]

    tests = _as_mapping(data.get("tests"), "tests")
    deploy = _as_mapping(data.get("deploy"), "deploy")
    rollout = _as_mapping(data.get("rollout"), "rollout")
    evaluation = _as_mapping(data.get("evaluation"), "evaluation")
    bench = _as_mapping(data.get("benchmark"), "benchmark")
    artifacts = _as_mapping(data.get("artifacts"), "artifacts")
    tooling = _as_mapping(data.get("tooling"), "tooling")
    auth = _as_mapping(data.get("auth"), "auth")
    security = _as_mapping(data.get("security"), "security")

    kubectl_dump = _as_mapping(deploy.get("kubectl_dump"), "deploy.kubectl_dump")

    eval_required_keys = evaluation.get("required_keys") or []
    if eval_required_keys is None:
        eval_required_keys = []
    if not isinstance(eval_required_keys, list) or not all(isinstance(k, str) for k in eval_required_keys):
        raise ValueError("pipeline.evaluation.required_keys must be a list of strings")

    required_keys = bench.get("required_keys") or []
    if required_keys is None:
        required_keys = []
    if not isinstance(required_keys, list) or not all(isinstance(k, str) for k in required_keys):
        raise ValueError("pipeline.benchmark.required_keys must be a list of strings")

    teardown_policy = str(deploy.get("teardown_policy") or "always").strip().lower()
    if teardown_policy not in ("always", "on_success", "on_failure", "never"):
        raise ValueError("pipeline.deploy.teardown_policy must be one of: always, on_success, on_failure, never")

    kind_cluster_name = str(tooling.get("kind_cluster_name") or "kind").strip() or "kind"

    security_mode = str(security.get("mode") or "safe").strip().lower()
    if security_mode not in ("safe", "system"):
        raise ValueError("pipeline.security.mode must be one of: safe, system")

    allowlist = security.get("allowlist") or []
    if allowlist is None:
        allowlist = []
    if not isinstance(allowlist, list) or not all(isinstance(x, str) for x in allowlist):
        raise ValueError("pipeline.security.allowlist must be a list of strings")

    denylist = security.get("denylist") or []
    if denylist is None:
        denylist = []
    if not isinstance(denylist, list) or not all(isinstance(x, str) for x in denylist):
        raise ValueError("pipeline.security.denylist must be a list of strings")

    # auth: accept cmds or steps as alias
    auth_cmds = _as_cmds(auth, cmd_key="cmd", cmds_key="cmds")
    if not auth_cmds and auth.get("steps") is not None:
        steps = auth.get("steps")
        if not isinstance(steps, list) or not all(isinstance(x, str) and x.strip() for x in steps):
            raise ValueError("pipeline.auth.steps must be a list of non-empty strings")
        auth_cmds = [x.strip() for x in steps if x.strip()]

    return PipelineSpec(
        version=version,
        tests_cmds=_as_cmds(tests, cmd_key="cmd", cmds_key="cmds"),
        tests_timeout_seconds=(int(tests.get("timeout_seconds")) if tests.get("timeout_seconds") else None),
        tests_retries=int(tests.get("retries") or 0),
        tests_env=_as_env(tests.get("env"), "tests"),
        tests_workdir=(str(tests.get("workdir")).strip() if tests.get("workdir") else None),
        deploy_setup_cmds=_as_cmds(deploy, cmd_key="setup_cmd", cmds_key="setup_cmds"),
        deploy_health_cmds=_as_cmds(deploy, cmd_key="health_cmd", cmds_key="health_cmds"),
        deploy_teardown_cmds=_as_cmds(deploy, cmd_key="teardown_cmd", cmds_key="teardown_cmds"),
        deploy_timeout_seconds=(int(deploy.get("timeout_seconds")) if deploy.get("timeout_seconds") else None),
        deploy_retries=int(deploy.get("retries") or 0),
        deploy_env=_as_env(deploy.get("env"), "deploy"),
        deploy_workdir=(str(deploy.get("workdir")).strip() if deploy.get("workdir") else None),
        deploy_teardown_policy=teardown_policy,
        kubectl_dump_enabled=bool(kubectl_dump.get("enabled") or False),
        kubectl_dump_namespace=(str(kubectl_dump.get("namespace")).strip() if kubectl_dump.get("namespace") else None),
        kubectl_dump_label_selector=(
            str(kubectl_dump.get("label_selector")).strip() if kubectl_dump.get("label_selector") else None
        ),
        kubectl_dump_include_logs=bool(kubectl_dump.get("include_logs") or False),
        rollout_run_cmds=_as_cmds(rollout, cmd_key="run_cmd", cmds_key="run_cmds"),
        rollout_timeout_seconds=(int(rollout.get("timeout_seconds")) if rollout.get("timeout_seconds") else None),
        rollout_retries=int(rollout.get("retries") or 0),
        rollout_env=_as_env(rollout.get("env"), "rollout"),
        rollout_workdir=(str(rollout.get("workdir")).strip() if rollout.get("workdir") else None),
        evaluation_run_cmds=_as_cmds(evaluation, cmd_key="run_cmd", cmds_key="run_cmds"),
        evaluation_timeout_seconds=(
            int(evaluation.get("timeout_seconds")) if evaluation.get("timeout_seconds") else None
        ),
        evaluation_retries=int(evaluation.get("retries") or 0),
        evaluation_env=_as_env(evaluation.get("env"), "evaluation"),
        evaluation_workdir=(str(evaluation.get("workdir")).strip() if evaluation.get("workdir") else None),
        evaluation_metrics_path=(
            str(evaluation.get("metrics_path")).strip() if evaluation.get("metrics_path") else None
        ),
        evaluation_required_keys=[str(k).strip() for k in eval_required_keys if str(k).strip()],
        benchmark_run_cmds=_as_cmds(bench, cmd_key="run_cmd", cmds_key="run_cmds"),
        benchmark_timeout_seconds=(int(bench.get("timeout_seconds")) if bench.get("timeout_seconds") else None),
        benchmark_retries=int(bench.get("retries") or 0),
        benchmark_env=_as_env(bench.get("env"), "benchmark"),
        benchmark_workdir=(str(bench.get("workdir")).strip() if bench.get("workdir") else None),
        benchmark_metrics_path=(str(bench.get("metrics_path")).strip() if bench.get("metrics_path") else None),
        benchmark_required_keys=[str(k).strip() for k in required_keys if str(k).strip()],
        auth_cmds=auth_cmds,
        auth_timeout_seconds=(int(auth.get("timeout_seconds")) if auth.get("timeout_seconds") else None),
        auth_retries=int(auth.get("retries") or 0),
        auth_env=_as_env(auth.get("env"), "auth"),
        auth_workdir=(str(auth.get("workdir")).strip() if auth.get("workdir") else None),
        auth_interactive=bool(auth.get("interactive") or False),
        artifacts_out_dir=(str(artifacts.get("out_dir")).strip() if artifacts.get("out_dir") else None),
        tooling_ensure_tools=bool(tooling.get("ensure_tools") or False),
        tooling_ensure_kind_cluster=bool(tooling.get("ensure_kind_cluster") or False),
        tooling_kind_cluster_name=kind_cluster_name,
        tooling_kind_config=(str(tooling.get("kind_config")).strip() if tooling.get("kind_config") else None),
        security_mode=security_mode,
        security_allowlist=[str(x).strip() for x in allowlist if str(x).strip()],
        security_denylist=[str(x).strip() for x in denylist if str(x).strip()],
        security_max_cmd_seconds=(int(security.get("max_cmd_seconds")) if security.get("max_cmd_seconds") else None),
        security_max_total_seconds=(int(security.get("max_total_seconds")) if security.get("max_total_seconds") else None),
    )
