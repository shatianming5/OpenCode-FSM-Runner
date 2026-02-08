from __future__ import annotations

import importlib.util
from pathlib import Path


def test_no_repo_level_env_module() -> None:
    """The repo should not ship a top-level `env.py` compatibility module."""
    # 作用：The repo should not ship a top-level `env.py` compatibility module.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈10 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_public_api_surface.py:9；类型=function；引用≈1；规模≈10行
    root = Path(__file__).resolve().parents[1]
    assert not (root / "env.py").exists()

    spec = importlib.util.find_spec("env")
    if spec is None or spec.origin is None:
        return
    # If an unrelated third-party `env` module exists, ensure it's not from this repo.
    assert not Path(spec.origin).resolve().is_relative_to(root)


def test_runner_env_public_surface_is_minimal() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈22 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_public_api_surface.py:20；类型=function；引用≈1；规模≈22行
    from runner import env as runner_env

    assert hasattr(runner_env, "setup")
    assert not hasattr(runner_env, "deploy")
    assert not hasattr(runner_env, "rollout")
    assert not hasattr(runner_env, "evaluation")
    assert not hasattr(runner_env, "rollout_and_evaluation")
    assert not hasattr(runner_env, "teardown")
    assert not hasattr(runner_env, "current_session")

    sess = runner_env.setup  # smoke import path
    assert callable(sess)

    assert hasattr(runner_env, "EnvSession")
    cls = runner_env.EnvSession
    assert hasattr(cls, "rollout")
    assert hasattr(cls, "evaluate")
    assert not hasattr(cls, "deploy")
    assert not hasattr(cls, "evaluation")
    assert not hasattr(cls, "rollout_and_evaluation")
    assert not hasattr(cls, "teardown")

    # Also support the convenient top-level import alias (`import runner_env`).
    import runner_env as runner_env_alias

    assert hasattr(runner_env_alias, "setup")
    assert not hasattr(runner_env_alias, "deploy")
    assert not hasattr(runner_env_alias, "rollout")
    assert not hasattr(runner_env_alias, "evaluation")
    assert not hasattr(runner_env_alias, "rollout_and_evaluation")
    assert not hasattr(runner_env_alias, "teardown")
    assert not hasattr(runner_env_alias, "current_session")

    assert hasattr(runner_env_alias, "EnvSession")
