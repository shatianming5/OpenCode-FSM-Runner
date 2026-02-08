from __future__ import annotations

from pathlib import Path

from runner.contract_hints import suggest_contract_hints


def test_suggest_contract_hints_from_readme_code_fence(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈23 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_contract_hints.py:9；类型=function；引用≈1；规模≈23行
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "Run evaluation:",
                "```bash",
                "python -m benchtool.evaluate --dataset demo --model openai",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hints = suggest_contract_hints(repo)
    assert "python -m benchtool.evaluate --dataset demo --model openai" in hints.commands
    # Anchors should contain the module and its top-level package to make audits robust.
    assert "benchtool.evaluate" in hints.anchors
    assert "benchtool" in hints.anchors


def test_suggest_contract_hints_strips_prompts_and_anchors_cd(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈33 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_contract_hints.py:33；类型=function；引用≈1；规模≈33行
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "Run evaluation:",
                "```bash",
                "Below are the commands to run the evaluation:",
                "git clone https://github.com/acme/evaluation-harness.git",
                "cd evaluation-harness",
                "$ docker pull ghcr.io/acme/evaluation-harness",
                "title = {not a command}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hints = suggest_contract_hints(repo)
    assert "cd evaluation-harness" in hints.commands
    # `$` prompt prefix should be stripped.
    assert "docker pull ghcr.io/acme/evaluation-harness" in hints.commands
    # Prose / key-value lines should not be treated as runnable hints.
    assert not any(x.startswith("Below ") for x in hints.commands)
    assert not any(x.startswith("title =") for x in hints.commands)
    # Anchors should include the `cd` target so audits can match it.
    assert "evaluation-harness" in hints.anchors
