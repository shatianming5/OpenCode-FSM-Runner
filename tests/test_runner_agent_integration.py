import subprocess
import shutil
import shlex
import sys
from pathlib import Path

import pytest

from runner.agent_client import AgentResult
from runner.runner import RunnerConfig, run


class _FakeAgent:
    def __init__(self, repo: Path):
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        # Plan update must not touch code/pipeline; runner should revert these.
        if purpose == "plan_update_attempt_1":
            (self._repo / "foo.txt").write_text(f"hacked at {fsm_state}/{iter_idx}\n", encoding="utf-8")
            (self._repo / "pipeline.yml").write_text("hacked pipeline\n", encoding="utf-8")

        # Execute must not touch PLAN/pipeline; runner should revert these by content.
        if purpose == "execute_step":
            (self._repo / "PLAN.md").write_text("hacked plan\n", encoding="utf-8")
            (self._repo / "pipeline.yml").write_text("hacked pipeline 2\n", encoding="utf-8")

        return AgentResult(assistant_text="ok")

    def close(self) -> None:
        return


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)


def test_runner_reverts_illegal_agent_edits(tmp_path: Path):
    if not shutil.which("git"):  # pragma: no cover
        pytest.skip("git not available")

    repo = tmp_path
    _init_git_repo(repo)

    # Track a file so `git diff --name-only` detects illegal edits.
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "foo.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

    # Pipeline exists to exercise content-based revert guards.
    (repo / "pipeline.yml").write_text("pipeline: original\n", encoding="utf-8")

    py = shlex.quote(sys.executable)
    ok_cmd = f'{py} -c "import sys; sys.exit(0)"'

    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=(repo / "pipeline.yml"),
        pipeline_rel="pipeline.yml",
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=(repo / ".aider_fsm" / "artifacts"),
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=False,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
    )

    rc = run(cfg, agent=_FakeAgent(repo))
    assert rc == 1  # MAX_ITERS

    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"
    assert (repo / "pipeline.yml").read_text(encoding="utf-8") == "pipeline: original\n"
    assert "hacked plan" not in (repo / "PLAN.md").read_text(encoding="utf-8")
