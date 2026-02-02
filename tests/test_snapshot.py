from pathlib import Path

from runner.snapshot import build_snapshot, get_git_changed_files


def test_build_snapshot_non_git(tmp_path: Path):
    plan = tmp_path / "PLAN.md"
    plan.write_text("# PLAN\n", encoding="utf-8")
    snapshot, text = build_snapshot(tmp_path, plan)
    assert snapshot["repo"] == str(tmp_path)
    assert "[SNAPSHOT]" in text
    assert "plan_md:" in text
    assert "git_status_porcelain:" in text


def test_get_git_changed_files_non_git(tmp_path: Path):
    assert get_git_changed_files(tmp_path) is None
