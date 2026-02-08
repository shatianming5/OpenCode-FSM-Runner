from __future__ import annotations

from pathlib import Path

import pytest

from runner.hints_exec import normalize_hint_command, run_hints


def test_normalize_hint_command_rewrites_dotted_entrypoint_to_python_module() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:11；类型=function；引用≈1；规模≈4行
    cmd, reason = normalize_hint_command("foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd.startswith("python3 -m foo.bar --x 1")


def test_normalize_hint_command_keeps_existing_python_module_invocations() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:17；类型=function；引用≈1；规模≈4行
    cmd, reason = normalize_hint_command("python3 -m foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd == "python3 -m foo.bar --x 1"


def test_normalize_hint_command_strips_common_shell_prompts() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈6 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:23；类型=function；引用≈1；规模≈6行
    cmd, reason = normalize_hint_command("$ python3 -m foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd == "python3 -m foo.bar --x 1"


def test_normalize_hint_command_strips_pytest_xdist_flags_by_default() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：防止缺少 pytest-xdist 时 `pytest -n ...` 直接 usage error，导致严格评估卡死在 hints
    # 证据：位置=tests/test_hints_exec.py；类型=function；引用≈1；规模≈10行
    cmd, reason = normalize_hint_command("pytest -n 5 --dist loadscope -q", env={})
    assert reason is None
    assert " -n " not in f" {cmd} "
    assert " --dist " not in f" {cmd} "
    assert cmd.startswith("pytest ")


def test_normalize_hint_command_strips_pytest_xdist_flags_under_uv_run() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：BrowserGym 类 repo 常用 `uv run pytest -n ...`，需要同样做兼容性去旗标处理
    # 证据：位置=tests/test_hints_exec.py；类型=function；引用≈1；规模≈12行
    cmd, reason = normalize_hint_command(
        "uv run pytest -n 5 --durations=10 -m 'not pricy' -v tests/core",
        env={},
    )
    assert reason is None
    assert " -n " not in f" {cmd} "
    assert cmd.startswith("uv run pytest ")


def test_normalize_hint_command_absolutizes_repo_relative_aider_fsm_python(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈14 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:23；类型=function；引用≈1；规模≈14行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈36 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:41；类型=function；引用≈1；规模≈36行
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
        # 作用：内部符号：test_run_hints_runs_docker_hints_in_isolated_artifacts_workdir._R
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:56；类型=class；引用≈6；规模≈5行
        def __init__(self, rc: int, out: str = "", err: str = "") -> None:
            # 作用：内部符号：test_run_hints_runs_docker_hints_in_isolated_artifacts_workdir._R.__init__
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_hints_exec.py:57；类型=method；引用≈1；规模≈4行
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        # 作用：内部符号：test_run_hints_runs_docker_hints_in_isolated_artifacts_workdir.fake_run
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:62；类型=function；引用≈12；规模≈7行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈33 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:77；类型=function；引用≈1；规模≈33行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈25 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:112；类型=function；引用≈1；规模≈25行
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    # One docker-based hint; if Docker isn't reachable we should skip it without attempting execution.
    env = {"AIDER_FSM_HINTS_JSON": '["docker run --rm hello-world"]'}

    monkeypatch.setattr("runner.hints_exec.shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_run(cmd, *args, **kwargs):
        # 作用：内部符号：test_run_hints_skips_docker_commands_when_docker_unavailable.fake_run
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈9 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:121；类型=function；引用≈12；规模≈9行
        if cmd == ["docker", "info"]:
            class R:
                # 作用：内部符号：test_run_hints_skips_docker_commands_when_docker_unavailable.fake_run.R
                # 能否简略：是
                # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
                # 证据：位置=tests/test_hints_exec.py:123；类型=class；引用≈2；规模≈4行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈10 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:139；类型=function；引用≈1；规模≈10行
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {"AIDER_FSM_HINTS_JSON": '["definitely_missing_binary_for_runner --version"]'}

    res = run_hints(repo=repo, max_attempts=2, timeout_seconds=5, env=env)
    assert res.get("ok") is False
    assert res.get("reason") == "all_hints_unrunnable"
    attempts = res.get("attempts") or []
    assert any(str(a.get("skip_reason") or "").startswith("probe_failed: binary_not_found:") for a in attempts if isinstance(a, dict))


def test_run_hints_used_anchors_only_when_matched(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:151；类型=function；引用≈1；规模≈11行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:164；类型=function；引用≈1；规模≈11行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈21 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:177；类型=function；引用≈1；规模≈21行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈40 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:203；类型=function；引用≈1；规模≈40行
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
        # 作用：内部符号：test_run_hints_skips_followup_openai_after_auth_failure._R
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:218；类型=class；引用≈6；规模≈5行
        def __init__(self, rc: int, out: str = "", err: str = "") -> None:
            # 作用：内部符号：test_run_hints_skips_followup_openai_after_auth_failure._R.__init__
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_hints_exec.py:219；类型=method；引用≈1；规模≈4行
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        # 作用：内部符号：test_run_hints_skips_followup_openai_after_auth_failure.fake_run
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:224；类型=function；引用≈12；规模≈5行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:242；类型=function；引用≈1；规模≈11行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:255；类型=function；引用≈1；规模≈11行
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        "AIDER_FSM_HINTS_JSON": '["python3 -c \\"print(\\\\\\\"pass@1: 0.25\\\\\\\")\\""]',
        "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
    }

    res = run_hints(repo=repo, max_attempts=1, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert abs(float(res.get("score") or 0.0) - 0.25) < 1e-9


def test_run_hints_tries_install_hint_when_pytest_hints_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈46 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_hints_exec.py:??；类型=function；引用≈1；规模≈46行
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    env = {
        "AIDER_FSM_HINTS_JSON": '["pytest -q tests/a","pytest -q tests/b","pip install foo"]',
        "AIDER_FSM_HINT_ANCHORS_JSON": '["foo"]',
    }

    # Treat all hints as runnable for probe phase to focus this test on execution ordering.
    monkeypatch.setattr("runner.hints_exec._probe_hint_command", lambda **_kwargs: (True, "ok"))

    class _R:
        # 作用：内部符号：test_run_hints_tries_install_hint_when_pytest_hints_fail._R
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:??；类型=class；引用≈6；规模≈5行
        def __init__(self, rc: int, out: str = "", err: str = "") -> None:
            # 作用：内部符号：test_run_hints_tries_install_hint_when_pytest_hints_fail._R.__init__
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_hints_exec.py:??；类型=method；引用≈1；规模≈4行
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        # 作用：内部符号：test_run_hints_tries_install_hint_when_pytest_hints_fail.fake_run
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_hints_exec.py:??；类型=function；引用≈12；规模≈11行
        if isinstance(cmd, (list, tuple)) and cmd[:2] == ["bash", "-c"]:
            s = str(cmd[2])
            if s.startswith("pytest "):
                return _R(1, out="", err="failed")
            if s == "pip install foo":
                return _R(0, out="ok", err="")
            raise AssertionError(f"unexpected hint cmd: {s!r}")
        raise AssertionError(f"unexpected subprocess.run: {cmd!r}")

    monkeypatch.setattr("runner.hints_exec.subprocess.run", fake_run)

    res = run_hints(repo=repo, max_attempts=2, timeout_seconds=5, env=env)
    assert res.get("ok") is True
    assert res.get("chosen_command") == "pip install foo"
    assert res.get("used_anchors") == ["foo"]
