from __future__ import annotations

from pathlib import Path

from runner.pipeline_spec import PipelineSpec
from runner.scaffold_validation import validate_scaffold_contract


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_pipeline() -> PipelineSpec:
    return PipelineSpec(
        security_max_cmd_seconds=60,
        deploy_setup_cmds=["echo setup"],
        rollout_run_cmds=["echo rollout"],
        evaluation_run_cmds=["echo eval"],
        evaluation_metrics_path=".aider_fsm/metrics.json",
        evaluation_required_keys=["score", "ok"],
    )


def _write_required_stage_scripts(repo: Path) -> None:
    _write(repo / ".aider_fsm" / "stages" / "deploy_setup.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "deploy_health.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "rollout.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    _write(repo / ".aider_fsm" / "stages" / "evaluation.sh", "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")


def test_validate_scaffold_contract_reports_stage_script_syntax_errors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_required_stage_scripts(repo)
    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nif [ \"x\" = \"x\" ]; then\n  echo broken\n",
    )

    report = validate_scaffold_contract(repo, pipeline=_minimal_pipeline(), require_metrics=True)
    assert report.ok is False
    assert report.stage_script_errors
    assert any("evaluation.sh" in x for x in report.stage_script_errors)


def test_validate_scaffold_contract_reports_bootstrap_parse_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_required_stage_scripts(repo)
    _write(repo / ".aider_fsm" / "bootstrap.yml", "version: [\n")

    report = validate_scaffold_contract(repo, pipeline=_minimal_pipeline(), require_metrics=True)
    assert report.ok is False
    assert report.bootstrap_errors
