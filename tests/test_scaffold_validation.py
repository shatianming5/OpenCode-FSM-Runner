from __future__ import annotations

from pathlib import Path

from runner.pipeline_spec import PipelineSpec
from runner.scaffold_validation import validate_scaffold_contract


def _write(path: Path, content: str) -> None:
    # 作用：内部符号：_write
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈24（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_scaffold_validation.py:10；类型=function；引用≈24；规模≈3行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_pipeline() -> PipelineSpec:
    # 作用：内部符号：_minimal_pipeline
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_scaffold_validation.py:15；类型=function；引用≈3；规模≈9行
    return PipelineSpec(
        security_max_cmd_seconds=60,
        deploy_setup_cmds=["echo setup"],
        rollout_run_cmds=["echo rollout"],
        evaluation_run_cmds=["echo eval"],
        evaluation_metrics_path=".aider_fsm/metrics.json",
        evaluation_required_keys=["score", "ok"],
    )


def _write_required_stage_scripts(repo: Path) -> None:
    # 作用：内部符号：_write_required_stage_scripts
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_scaffold_validation.py:26；类型=function；引用≈3；规模≈5行
    _write(repo / ".aider_fsm" / "stages" / "deploy_setup.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "deploy_health.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "rollout.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "evaluation.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")


def test_validate_scaffold_contract_reports_stage_script_syntax_errors(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈12 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_scaffold_validation.py:33；类型=function；引用≈1；规模≈12行
    repo = tmp_path / "repo"
    _write_required_stage_scripts(repo)
    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nif [ \"x\" = \"x\" ]; then\n  echo broken\n",
    )

    report = validate_scaffold_contract(repo, pipeline=_minimal_pipeline(), require_metrics=True)
    assert report.errors
    assert report.stage_script_errors
    assert any("evaluation.sh" in x for x in report.stage_script_errors)


def test_validate_scaffold_contract_reports_bootstrap_parse_error(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈8 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_scaffold_validation.py:47；类型=function；引用≈1；规模≈8行
    repo = tmp_path / "repo"
    _write_required_stage_scripts(repo)
    _write(repo / ".aider_fsm" / "bootstrap.yml", "version: [\n")

    report = validate_scaffold_contract(repo, pipeline=_minimal_pipeline(), require_metrics=True)
    assert report.errors
    assert report.bootstrap_errors
