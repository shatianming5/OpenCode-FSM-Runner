import shlex
import sys
from pathlib import Path

import pytest

from runner.pipeline_spec import PipelineSpec, load_pipeline_spec
from runner.pipeline_verify import run_pipeline_verification


def _py_cmd(exit_code: int) -> str:
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit({exit_code})"'


def test_load_pipeline_spec_ok(tmp_path: Path):
    p = tmp_path / "pipeline.yml"
    p.write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
                "  max_total_seconds: 600",
                "tests:",
                "  cmds: [pytest -q]",
                "  retries: 2",
                "  env: {FOO: bar}",
                "  workdir: .",
                "auth:",
                "  steps: [echo auth]",
                "  interactive: false",
                "deploy:",
                "  setup_cmds: [echo deploy]",
                "  health_cmds: [echo health]",
                "  teardown_cmds: [echo teardown]",
                "  timeout_seconds: 120",
                "  teardown_policy: on_failure",
                "  kubectl_dump:",
                "    enabled: true",
                "    namespace: default",
                "    label_selector: app=myapp",
                "    include_logs: true",
                "benchmark:",
                "  run_cmds: [echo bench]",
                "  timeout_seconds: 300",
                "  metrics_path: metrics.json",
                "  required_keys: [score]",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "tooling:",
                "  ensure_tools: true",
                "  ensure_kind_cluster: true",
                "  kind_cluster_name: kind",
                "  kind_config: kind/kind-config.yaml",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec = load_pipeline_spec(p)
    assert spec.version == 1
    assert spec.security_mode == "safe"
    assert spec.security_max_cmd_seconds == 60
    assert spec.security_max_total_seconds == 600
    assert spec.tests_cmds == ["pytest -q"]
    assert spec.tests_retries == 2
    assert spec.tests_env["FOO"] == "bar"
    assert spec.tests_workdir == "."
    assert spec.auth_cmds == ["echo auth"]
    assert spec.deploy_setup_cmds == ["echo deploy"]
    assert spec.deploy_teardown_policy == "on_failure"
    assert spec.kubectl_dump_enabled is True
    assert spec.kubectl_dump_label_selector == "app=myapp"
    assert spec.benchmark_metrics_path == "metrics.json"
    assert spec.benchmark_required_keys == ["score"]
    assert spec.artifacts_out_dir == ".aider_fsm/artifacts"
    assert spec.tooling_ensure_tools is True
    assert spec.tooling_ensure_kind_cluster is True
    assert spec.tooling_kind_config == "kind/kind-config.yaml"


def test_load_pipeline_spec_invalid_version(tmp_path: Path):
    p = tmp_path / "pipeline.yml"
    p.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_spec(p)


def test_run_pipeline_verification_metrics_ok(tmp_path: Path):
    (tmp_path / "metrics.json").write_text('{"score": 1}\n', encoding="utf-8")
    pipeline = PipelineSpec(
        benchmark_run_cmds=[_py_cmd(0)],
        benchmark_metrics_path="metrics.json",
        benchmark_required_keys=["score"],
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is True
    assert verify.failed_stage is None
    assert verify.metrics_path is not None
    assert verify.metrics == {"score": 1}


def test_run_pipeline_verification_metrics_missing_key(tmp_path: Path):
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    pipeline = PipelineSpec(
        benchmark_run_cmds=[_py_cmd(0)],
        benchmark_metrics_path="metrics.json",
        benchmark_required_keys=["score"],
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is False
    assert verify.failed_stage == "metrics"
    assert any("missing_keys" in e for e in (verify.metrics_errors or []))


def test_run_pipeline_verification_safe_mode_blocks_sudo(tmp_path: Path):
    pipeline = PipelineSpec(security_mode="safe")
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=["sudo echo hi"],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is False
    assert verify.failed_stage == "tests"
    assert verify.tests is not None
    assert verify.tests.results
    assert verify.tests.results[-1].rc == 126
