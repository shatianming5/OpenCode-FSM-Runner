from pathlib import Path

from runner.actions import run_pending_actions


def _write_actions(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_actions_write_file_protected_path_is_blocked(tmp_path: Path):
    repo = tmp_path
    plan = repo / "PLAN.md"
    plan.write_text("# PLAN\n", encoding="utf-8")

    actions_path = repo / ".aider_fsm" / "actions.yml"
    _write_actions(
        actions_path,
        "\n".join(
            [
                "version: 1",
                "actions:",
                "  - id: p0",
                "    kind: write_file",
                "    path: PLAN.md",
                "    content: hacked",
                "",
            ]
        ),
    )

    stage = run_pending_actions(
        repo,
        pipeline=None,
        unattended="strict",
        actions_path=actions_path,
        artifacts_dir=repo / ".aider_fsm" / "artifacts",
        protected_paths=[plan],
    )
    assert stage is not None
    assert stage.ok is False
    assert stage.results[-1].rc == 2
    assert "path_is_protected" in stage.results[-1].stderr


def test_actions_run_cmd_default_safe_blocks_sudo(tmp_path: Path):
    repo = tmp_path
    actions_path = repo / ".aider_fsm" / "actions.yml"
    _write_actions(
        actions_path,
        "\n".join(
            [
                "version: 1",
                "actions:",
                "  - id: a0",
                "    kind: run_cmd",
                "    cmd: sudo echo hi",
                "",
            ]
        ),
    )

    stage = run_pending_actions(
        repo,
        pipeline=None,
        unattended="strict",
        actions_path=actions_path,
        artifacts_dir=repo / ".aider_fsm" / "artifacts",
        protected_paths=[],
    )
    assert stage is not None
    assert stage.ok is False
    assert stage.results[-1].rc == 126
    assert "blocked_by_default_safe_deny" in stage.results[-1].stderr
