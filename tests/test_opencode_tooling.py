from __future__ import annotations

from pathlib import Path

from runner.opencode_tooling import ToolPolicy, parse_tool_calls


def test_parse_tool_calls_detects_file_write_json_fence():
    text = "hi\n```json\n{\"filePath\":\"PLAN.md\",\"content\":\"# PLAN\\n\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "PLAN.md"
    assert payload["content"].startswith("# PLAN")


def test_parse_tool_calls_detects_bash_call():
    text = "```bash\nbash\n{\"command\":\"git status --porcelain\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "git status --porcelain"


def test_parse_tool_calls_detects_self_closing_write_tag():
    text = '<write filePath=\"hello.txt\" content=\"hello\\n\" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["content"] == "hello\n"


def test_tool_policy_plan_update_only_allows_plan_md(tmp_path: Path):
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="plan_update_attempt_1",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / "foo.txt")
    assert not ok
    assert reason == "plan_update_allows_only_plan_md"


def test_tool_policy_execute_step_denies_plan_and_pipeline(tmp_path: Path):
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert not ok and reason == "execute_step_disallows_plan_md"

    ok, reason = policy.allow_file_write(pipeline)
    assert not ok and reason == "execute_step_disallows_pipeline_yml"

    ok, reason = policy.allow_file_write(repo / "src" / "x.py")
    assert ok and reason is None


def test_tool_policy_restricted_bash_blocks_shell_metacharacters(tmp_path: Path):
    repo = tmp_path.resolve()
    policy = ToolPolicy(
        repo=repo,
        plan_path=repo / "PLAN.md",
        pipeline_path=None,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_bash('echo \"hi\" > hello.txt')
    assert not ok
    assert reason in ("blocked_shell_metacharacters", "blocked_by_restricted_bash_mode")
