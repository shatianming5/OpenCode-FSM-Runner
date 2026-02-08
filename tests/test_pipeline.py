import shlex
import sys
from pathlib import Path

import pytest

from runner.pipeline_spec import PipelineSpec, load_pipeline_spec
from runner.pipeline_verify import run_pipeline_verification


def _py_cmd(exit_code: int) -> str:
    """中文说明：
    - 含义：构造一个会以指定 exit code 退出的 Python 命令（`python -c ...`）。
    - 内容：用于在 tests_cmds / rollout/evaluation/benchmark cmd 中制造可控的成功/失败。
    - 可简略：是（测试 helper；可直接内联拼接）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈8 行；引用次数≈22（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:17；类型=function；引用≈22；规模≈8行
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit({exit_code})"'


def _py_inline(code: str) -> str:
    # 作用：内部符号：_py_inline
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:22；类型=function；引用≈6；规模≈3行
    py = shlex.quote(sys.executable)
    return f"{py} -c {shlex.quote(code)}"


def test_load_pipeline_spec_ok(tmp_path: Path):
    """中文说明：
    - 含义：验证 `load_pipeline_spec` 能解析较完整的 v1 pipeline.yml。
    - 内容：写入覆盖 security/tests/auth/deploy/rollout/evaluation/benchmark/artifacts/tooling 的 YAML，并断言关键字段映射正确。
    - 可简略：可能（可拆成更细粒度的 schema 单测；但当前是高覆盖回归测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈91 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:32；类型=function；引用≈1；规模≈91行
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
                "rollout:",
                "  run_cmds: [echo rollout]",
                "  timeout_seconds: 200",
                "  retries: 1",
                "  env: {ROLLOUT_MODE: quick}",
                "  workdir: .",
                "evaluation:",
                "  run_cmds: [echo eval]",
                "  timeout_seconds: 250",
                "  metrics_path: eval_metrics.json",
                "  required_keys: [score]",
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
    assert spec.rollout_run_cmds == ["echo rollout"]
    assert spec.rollout_timeout_seconds == 200
    assert spec.rollout_retries == 1
    assert spec.rollout_env["ROLLOUT_MODE"] == "quick"
    assert spec.rollout_workdir == "."
    assert spec.evaluation_run_cmds == ["echo eval"]
    assert spec.evaluation_timeout_seconds == 250
    assert spec.evaluation_metrics_path == "eval_metrics.json"
    assert spec.evaluation_required_keys == ["score"]
    assert spec.benchmark_metrics_path == "metrics.json"
    assert spec.benchmark_required_keys == ["score"]
    assert spec.artifacts_out_dir == ".aider_fsm/artifacts"
    assert spec.tooling_ensure_tools is True
    assert spec.tooling_ensure_kind_cluster is True
    assert spec.tooling_kind_config == "kind/kind-config.yaml"


def test_load_pipeline_spec_invalid_version(tmp_path: Path):
    """中文说明：
    - 含义：验证 pipeline.yml 的 version 不支持时会报错。
    - 内容：写入 version=2 并断言抛出 ValueError。
    - 可简略：是（典型负例测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈10 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:125；类型=function；引用≈1；规模≈10行
    p = tmp_path / "pipeline.yml"
    p.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_spec(p)


def test_run_pipeline_verification_metrics_ok(tmp_path: Path):
    """中文说明：
    - 含义：验证 benchmark metrics 的读取与 required_keys 校验通过时验收成功。
    - 内容：预写 metrics.json（含 score），构造 PipelineSpec 的 benchmark 配置并运行 verification，断言 ok 与 metrics 正确。
    - 可简略：否（metrics 契约是 benchmark/evaluation 的核心；建议保留覆盖）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈22 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:137；类型=function；引用≈1；规模≈22行
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


def test_run_pipeline_verification_evaluation_metrics_ok(tmp_path: Path):
    """中文说明：
    - 含义：验证 evaluation metrics 的读取与 required_keys 校验通过时验收成功。
    - 内容：预写 metrics.json（含 score），构造 PipelineSpec 的 evaluation 配置并运行 verification，断言 evaluation stage ok 且 metrics 正确。
    - 可简略：否（evaluation 的 metrics 契约属于关键路径覆盖）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈24 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:161；类型=function；引用≈1；规模≈24行
    (tmp_path / "metrics.json").write_text('{"score": 1}\n', encoding="utf-8")
    pipeline = PipelineSpec(
        evaluation_run_cmds=[_py_cmd(0)],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score"],
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is True
    assert verify.failed_stage is None
    assert verify.evaluation is not None
    assert verify.evaluation.ok is True
    assert verify.metrics_path is not None
    assert verify.metrics == {"score": 1}


def test_run_pipeline_verification_metrics_missing_key(tmp_path: Path):
    """中文说明：
    - 含义：验证 benchmark metrics 缺少 required_keys 时会失败并给出错误原因。
    - 内容：metrics.json 写 `{}`，required_keys=[score]，断言 failed_stage=metrics 且 errors 包含 missing_keys。
    - 可简略：否（关键负例覆盖，防止 silently pass）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈21 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:187；类型=function；引用≈1；规模≈21行
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


def test_run_pipeline_verification_evaluation_metrics_missing_key(tmp_path: Path):
    """中文说明：
    - 含义：验证 evaluation metrics 缺少 required_keys 时会失败并给出错误原因。
    - 内容：metrics.json 写 `{}`，evaluation_required_keys=[score]，断言 failed_stage=metrics 且 errors 包含 missing_keys。
    - 可简略：否（关键负例覆盖）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈21 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:210；类型=function；引用≈1；规模≈21行
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    pipeline = PipelineSpec(
        evaluation_run_cmds=[_py_cmd(0)],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score"],
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


def test_run_pipeline_verification_required_ok_enforces_true(tmp_path: Path):
    """中文说明：
    - 含义：当 required_keys 包含 `ok` 时，runner 需要 `metrics.ok === true`，避免 placeholder success。
    - 内容：写入 `ok:false` 的 metrics.json，required_keys=[score,ok]，断言 failed_stage=metrics 且 errors 包含 ok_not_true。
    - 可简略：否（这是 contract 防呆关键路径，避免 silently pass）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈21 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:233；类型=function；引用≈1；规模≈21行
    (tmp_path / "metrics.json").write_text('{"ok": false, "score": 0}\n', encoding="utf-8")
    pipeline = PipelineSpec(
        evaluation_run_cmds=[_py_cmd(0)],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score", "ok"],
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is False
    assert verify.failed_stage == "metrics"
    assert any("ok_not_true" in e for e in (verify.metrics_errors or []))


def test_run_pipeline_verification_require_hints_used_missing_fails(tmp_path: Path):
    """When hint execution is required, missing hints_used.json must fail verification."""
    # 作用：When hint execution is required, missing hints_used.json must fail verification.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈27 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:252；类型=function；引用≈1；规模≈27行
    eval_cmd = _py_inline(
        "import json, pathlib; "
        "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
        "pathlib.Path('metrics.json').write_text(json.dumps({'ok': True, 'score': 1}) + '\\n')"
    )
    pipeline = PipelineSpec(
        evaluation_run_cmds=[eval_cmd],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score", "ok"],
        evaluation_env={
            "AIDER_FSM_REQUIRE_HINTS": "1",
            "AIDER_FSM_HINT_ANCHORS_JSON": '["pytest"]',
        },
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is False
    assert verify.failed_stage == "evaluation"
    assert verify.evaluation is not None
    assert verify.evaluation.ok is True
    assert any("evaluation.hints_requirement_failed" in e for e in (verify.metrics_errors or []))


def test_run_pipeline_verification_require_hints_used_ok_passes(tmp_path: Path):
    """When hint execution is required, hints_used.json + hints_run.json must validate."""
    # 作用：When hint execution is required, a valid hints_used.json should allow verification to pass.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈28 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:281；类型=function；引用≈1；规模≈28行
    eval_cmd = _py_inline(
        "import json, pathlib; "
        "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
        "pathlib.Path('.aider_fsm/hints_used.json').write_text(json.dumps({"
        "'ok': True, 'used_anchors': ['pytest'], 'commands': ['pytest -q']"
        "}) + '\\n'); "
        "pathlib.Path('.aider_fsm/hints_run.json').write_text(json.dumps({"
        "'ok': True, 'score': 1.0, 'chosen_command': 'pytest -q', 'executed_attempts': 1"
        "}) + '\\n'); "
        "pathlib.Path('metrics.json').write_text(json.dumps({'ok': True, 'score': 1}) + '\\n')"
    )
    pipeline = PipelineSpec(
        evaluation_run_cmds=[eval_cmd],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score", "ok"],
        evaluation_env={
            "AIDER_FSM_REQUIRE_HINTS": "1",
            "AIDER_FSM_HINT_ANCHORS_JSON": '["pytest"]',
        },
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is True
    assert verify.failed_stage is None
    assert verify.metrics == {"ok": True, "score": 1}


def test_run_pipeline_verification_require_real_score_requires_hints_run_ok(tmp_path: Path):
    """When real score is required, we must also have a successful hints_run.json."""
    # 作用：When real score is required, we must also have a successful hints_run.json.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈29 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:311；类型=function；引用≈1；规模≈29行
    eval_cmd = _py_inline(
        "import json, pathlib; "
        "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
        "pathlib.Path('.aider_fsm/hints_used.json').write_text(json.dumps({"
        "'ok': True, 'used_anchors': ['pytest'], 'commands': ['pytest -q']"
        "}) + '\\n'); "
        "pathlib.Path('metrics.json').write_text(json.dumps({'ok': True, 'score': 1}) + '\\n')"
    )
    pipeline = PipelineSpec(
        evaluation_run_cmds=[eval_cmd],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score", "ok"],
        evaluation_env={
            "AIDER_FSM_REQUIRE_HINTS": "1",
            "AIDER_FSM_HINT_ANCHORS_JSON": '["pytest"]',
            "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
        },
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is False
    assert verify.failed_stage == "evaluation"
    assert any("evaluation.hints_run_requirement_failed" in e for e in (verify.metrics_errors or []))


def test_run_pipeline_verification_require_real_score_hints_run_ok_passes(tmp_path: Path):
    """With require_real_score, a valid hints_run.json should allow verification to pass."""
    # 作用：With require_real_score, a valid hints_run.json should allow verification to pass.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈32 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:342；类型=function；引用≈1；规模≈32行
    eval_cmd = _py_inline(
        "import json, pathlib; "
        "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
        "pathlib.Path('.aider_fsm/hints_used.json').write_text(json.dumps({"
        "'ok': True, 'used_anchors': ['pytest'], 'commands': ['pytest -q']"
        "}) + '\\n'); "
        "pathlib.Path('.aider_fsm/hints_run.json').write_text(json.dumps({"
        "'ok': True, 'score': 0.25, 'chosen_command': 'pytest -q', 'executed_attempts': 1"
        "}) + '\\n'); "
        "pathlib.Path('metrics.json').write_text(json.dumps({'ok': True, 'score': 1}) + '\\n')"
    )
    pipeline = PipelineSpec(
        evaluation_run_cmds=[eval_cmd],
        evaluation_metrics_path="metrics.json",
        evaluation_required_keys=["score", "ok"],
        evaluation_env={
            "AIDER_FSM_REQUIRE_HINTS": "1",
            "AIDER_FSM_HINT_ANCHORS_JSON": '["pytest"]',
            "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
        },
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is True
    assert verify.failed_stage is None
    assert verify.metrics == {"ok": True, "score": 1}


def test_run_pipeline_verification_safe_mode_blocks_sudo(tmp_path: Path):
    """中文说明：
    - 含义：验证 safe security mode 会拦截 `sudo` 等危险命令。
    - 内容：tests_cmds 传入 `sudo echo hi`，断言 tests stage 失败且 rc=126（被策略拒绝）。
    - 可简略：否（安全边界测试，建议保留）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈18 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:380；类型=function；引用≈1；规模≈18行
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


def test_run_pipeline_verification_rollout_runs(tmp_path: Path):
    """中文说明：
    - 含义：验证 rollout stage 在配置存在时会被执行（并影响 overall ok）。
    - 内容：构造包含 rollout_run_cmds 与 benchmark_run_cmds 的 PipelineSpec，运行 verification 并断言 rollout.ok 为 True。
    - 可简略：可能（可补充更多 rollout/eval/bench 组合；当前覆盖“rollout 触发”主路径）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈19 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:400；类型=function；引用≈1；规模≈19行
    pipeline = PipelineSpec(
        rollout_run_cmds=[_py_cmd(0)],
        benchmark_run_cmds=[_py_cmd(0)],
    )
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts",
    )
    assert verify.ok is True
    assert verify.rollout is not None
    assert verify.rollout.ok is True


def test_run_pipeline_verification_env_override_extends_max_cmd_seconds(tmp_path: Path):
    """Env var override should allow extending security.max_cmd_seconds without editing pipeline.yml."""
    # 作用：Env var override should allow extending security.max_cmd_seconds without editing pipeline.yml.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈33 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_pipeline.py:417；类型=function；引用≈1；规模≈33行
    sleep_cmd = _py_inline("import time; time.sleep(2)")

    # Baseline: 1s cap should time out a 2s command.
    pipeline = PipelineSpec(security_max_cmd_seconds=1, evaluation_run_cmds=[sleep_cmd])
    verify = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts_baseline",
    )
    assert verify.ok is False
    assert verify.failed_stage == "evaluation"
    assert verify.evaluation is not None
    assert verify.evaluation.ok is False

    # Override: extend cap to 5s via stage env injection (common in programmatic calls).
    pipeline2 = PipelineSpec(
        security_max_cmd_seconds=1,
        evaluation_run_cmds=[sleep_cmd],
        evaluation_env={"AIDER_FSM_MAX_CMD_SECONDS": "5"},
    )
    verify2 = run_pipeline_verification(
        tmp_path,
        pipeline=pipeline2,
        tests_cmds=[_py_cmd(0)],
        artifacts_dir=tmp_path / "artifacts_override",
    )
    assert verify2.ok is True
    assert verify2.failed_stage is None
    assert verify2.evaluation is not None
    assert verify2.evaluation.ok is True
