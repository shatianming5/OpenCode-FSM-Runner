from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .subprocess_utils import read_text_if_exists


@dataclass(frozen=True)
class PipelineSpec:
    """Declarative contract for verification stages.

    NOTE: The runner intentionally treats this file as *human-owned* and will revert
    any model edits during plan-update/execute steps.
    """

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
        for attr in (
            "tests_cmds",
            "deploy_setup_cmds",
            "deploy_health_cmds",
            "deploy_teardown_cmds",
            "benchmark_run_cmds",
            "benchmark_required_keys",
            "auth_cmds",
            "security_allowlist",
            "security_denylist",
        ):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, [])
        for attr in ("tests_env", "deploy_env", "benchmark_env", "auth_env"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, {})


def load_pipeline_spec(path: Path) -> PipelineSpec:
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
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"pipeline.{name} must be a mapping")
        return value

    def _as_env(value: Any, name: str) -> dict[str, str]:
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
    bench = _as_mapping(data.get("benchmark"), "benchmark")
    artifacts = _as_mapping(data.get("artifacts"), "artifacts")
    tooling = _as_mapping(data.get("tooling"), "tooling")
    auth = _as_mapping(data.get("auth"), "auth")
    security = _as_mapping(data.get("security"), "security")

    kubectl_dump = _as_mapping(deploy.get("kubectl_dump"), "deploy.kubectl_dump")

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
