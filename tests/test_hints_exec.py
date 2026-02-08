from __future__ import annotations

from pathlib import Path

import pytest

from runner.hints_exec import normalize_hint_command, run_hints


def test_normalize_hint_command_rewrites_dotted_entrypoint_to_python_module() -> None:
    cmd, reason = normalize_hint_command("foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd.startswith("python3 -m foo.bar --x 1")


def test_normalize_hint_command_keeps_existing_python_module_invocations() -> None:
    cmd, reason = normalize_hint_command("python3 -m foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd == "python3 -m foo.bar --x 1"


def test_normalize_hint_command_absolutizes_repo_relative_aider_fsm_python(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".aider_fsm" / "venv" / "bin").mkdir(parents=True, exist_ok=True)
    py = repo / ".aider_fsm" / "venv" / "bin" / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    cmd, reason = normalize_hint_command(
        "foo.bar --x 1",
        env={
            "AIDER_FSM_REPO_ROOT": str(repo),
            "AIDER_FSM_PYTHON": ".aider_fsm/venv/bin/python",
        },
    )
    assert reason is None
    assert cmd.startswith(f"{py} -m foo.bar --x 1")


def test_run_hints_runs_docker_hints_in_isolated_artifacts_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "AIDER_FSM_HINTS_JSON": '["docker run --rm hello-world"]',
        "AIDER_FSM_ARTIFACTS_DIR": str(artifacts_dir),
    }

    monkeypatch.setattr("runner.hints_exec.shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else "/bin/bash")

    seen = {}

    class _R:
        def __init__(self, rc: int, out: str = "", err: str = "") -> None:
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        if cmd == ["docker", "info"]:
            return _R(0)
        if isinstance(cmd, (list, tuple)) and cmd[:2] == ["bash", "-c"]:
            seen["cwd"] = kwargs.get("cwd")
            return _R(0, out="ok")
        raise AssertionError(f"unexpected subprocess.run: {cmd!r}")

    monkeypatch.setattr("runner.hints_exec.subprocess.run", fake_run)

    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert seen.get("cwd") == str(artifacts_dir / "hints_workdir" / "attempt_01")


def test_run_hints_runs_openai_codegen_hints_in_isolated_workdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "mytool").mkdir(parents=True, exist_ok=True)
    (repo / "mytool" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "mytool" / "evaluate.py").write_text(
        "\n".join(
            [
                "import os",
                "os.makedirs('results/humaneval', exist_ok=True)",
                "print('pass@1: 0.5')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # If this hint ran from repo cwd it would fail to create the subdir.
    (repo / "results").mkdir(parents=True, exist_ok=True)
    (repo / "results").chmod(0o555)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -m mytool.evaluate --backend openai --model x --dataset y"]',
        "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
        "AIDER_FSM_ARTIFACTS_DIR": str(artifacts_dir),
        "AIDER_FSM_REPO_ROOT": str(repo),
    }
    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert (artifacts_dir / "hints_workdir" / "attempt_01" / "results" / "humaneval").exists()


def test_run_hints_skips_docker_commands_when_docker_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    # One docker-based hint; if Docker isn't reachable we should skip it without attempting execution.
    env = {"AIDER_FSM_HINTS_JSON": '["docker run --rm hello-world"]'}

    monkeypatch.setattr("runner.hints_exec.shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_run(cmd, *args, **kwargs):
        if cmd == ["docker", "info"]:
            class R:
                returncode = 1
                stdout = ""
                stderr = "daemon not running"

            return R()
        raise AssertionError(f"unexpected subprocess.run: {cmd!r}")

    monkeypatch.setattr("runner.hints_exec.subprocess.run", fake_run)

    res = run_hints(repo=repo, max_attempts=3, timeout_seconds=5, env=env)
    assert res.get("ok") is False
    attempts = res.get("attempts") or []
    assert attempts and attempts[0].get("skip_reason", "").startswith("docker_unavailable:")


def test_run_hints_marks_unrunnable_when_binary_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {"AIDER_FSM_HINTS_JSON": '["definitely_missing_binary_for_runner --version"]'}

    res = run_hints(repo=repo, max_attempts=2, timeout_seconds=5, env=env)
    assert res.get("ok") is False
    assert res.get("reason") == "all_hints_unrunnable"
    attempts = res.get("attempts") or []
    assert any(str(a.get("skip_reason") or "").startswith("probe_failed: binary_not_found:") for a in attempts if isinstance(a, dict))


def test_run_hints_used_anchors_only_when_matched(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -c \\"print(1)\\""]',
        "AIDER_FSM_HINT_ANCHORS_JSON": '["pytest"]',
    }

    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert res.get("used_anchors") == []


def test_run_hints_deduplicates_sanitized_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -c \\"print(1)\\"", "python3 -c \\"print(1)\\""]',
    }

    res = run_hints(repo=repo, max_attempts=2, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    attempts = res.get("attempts") or []
    assert any((a.get("skip_reason") == "duplicate_sanitized_hint") for a in attempts if isinstance(a, dict))


def test_run_hints_strict_compat_skips_backend_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": (
            '["evalplus.evaluate --model deepseek-v3.2 --dataset humaneval --backend vllm --greedy", '
            '"python3 -c \\"print(1)\\""]'
        ),
        "AIDER_FSM_HINT_STRICT_COMPAT": "1",
        "AIDER_LLM_KIND": "remote",
        "OPENAI_BASE_URL": "https://api.vveai.com/v1",
    }

    res = run_hints(repo=repo, max_attempts=2, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    attempts = res.get("attempts") or []
    assert any(
        str(a.get("skip_reason") or "").startswith("incompatible_hint: backend_mismatch:")
        for a in attempts
        if isinstance(a, dict)
    )


def test_run_hints_skips_followup_openai_after_auth_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": (
            '["evalplus.evaluate --model deepseek-v3.2 --dataset humaneval --backend openai --greedy", '
            '"evalplus.evaluate --model deepseek-v3.2 --dataset humaneval --base-url https://api.vveai.com/v1 --backend openai --greedy", '
            '"python3 -c \\"print(1)\\""]'
        ),
        "AIDER_LLM_KIND": "remote",
        "OPENAI_BASE_URL": "https://api.vveai.com/v1",
    }

    monkeypatch.setattr("runner.hints_exec._probe_hint_command", lambda **_kwargs: (True, "ok"))

    class _R:
        def __init__(self, rc: int, out: str = "", err: str = "") -> None:
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "evalplus.evaluate" in text:
            return _R(1, out="Error code: 401 - {'error': {'code': 'invalid_api_key'}}")
        return _R(0, out="ok")

    monkeypatch.setattr("runner.hints_exec.subprocess.run", fake_run)

    res = run_hints(repo=repo, max_attempts=3, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    attempts = res.get("attempts") or []
    assert any(
        a.get("skip_reason") == "skipped_after_openai_auth_failure"
        for a in attempts
        if isinstance(a, dict)
    )


def test_run_hints_require_real_score_fails_when_unparseable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -c \\"print(1)\\""]',
        "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
    }

    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is False
    assert res.get("reason") == "all_hints_no_real_score"


def test_run_hints_require_real_score_parses_pass_at_1(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -c \\"print(\\\\\\\"pass@1: 0.25\\\\\\\")\\""]',
        "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
    }

    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert abs(float(res.get("score") or 0.0) - 0.25) < 1e-9
