from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from runner import env as runner_env
from runner import env_local as runner_env_local
from runner.agent_client import AgentResult
from runner.env_local import EnvHandle
from runner.pipeline_spec import PipelineSpec


def test_env_setup_strict_disables_seed_and_fallback(tmp_path: Path, monkeypatch) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈27 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_strict_opencode_mode.py:16；类型=function；引用≈1；规模≈27行
    repo = tmp_path / "repo"
    repo.mkdir()
    pipeline_path = repo / "pipeline.yml"
    pipeline_path.write_text("version: 1\n", encoding="utf-8")
    handle = EnvHandle(repo=repo, pipeline_path=pipeline_path, pipeline=PipelineSpec())

    calls: list[dict] = []

    def _fake_open_env(_target: str, **kwargs) -> EnvHandle:
        # 作用：内部符号：test_env_setup_strict_disables_seed_and_fallback._fake_open_env
        # 能否简略：部分
        # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=tests/test_strict_opencode_mode.py:25；类型=function；引用≈4；规模≈3行
        calls.append(dict(kwargs))
        return handle

    monkeypatch.setattr(runner_env, "open_env", _fake_open_env)
    monkeypatch.setattr(
        runner_env,
        "suggest_contract_hints",
        lambda _repo: SimpleNamespace(commands=[], anchors=[]),
    )

    runner_env.setup("dummy-target", strict_opencode=True)
    assert calls[-1]["seed_stage_skeleton"] is False
    assert calls[-1]["write_fallback_pipeline_yml"] is False

    runner_env.setup("dummy-target", strict_opencode=False)
    assert calls[-1]["seed_stage_skeleton"] is True
    assert calls[-1]["write_fallback_pipeline_yml"] is True

def _write_valid_contract(repo: Path) -> None:
    # 作用：内部符号：_write_valid_contract
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈71 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_strict_opencode_mode.py:44；类型=function；引用≈2；规模≈71行
    stages = repo / ".aider_fsm" / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    scripts = {
        "tests.sh": "#!/usr/bin/env bash\nset -euo pipefail\necho tests_ok\n",
        "deploy_setup.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p .aider_fsm\n"
            f"{shlex.quote(sys.executable)} - <<'PY'\n"
            "import json\n"
            "open('.aider_fsm/runtime_env.json','w',encoding='utf-8').write(json.dumps({'ok': True})+'\\n')\n"
            "PY\n"
        ),
        "deploy_health.sh": "#!/usr/bin/env bash\nset -euo pipefail\ntest -f .aider_fsm/runtime_env.json\n",
        "deploy_teardown.sh": "#!/usr/bin/env bash\nset -euo pipefail\necho teardown\n",
        "rollout.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p .aider_fsm\n"
            f"{shlex.quote(sys.executable)} - <<'PY'\n"
            "import json\n"
            "open('.aider_fsm/samples.jsonl','w',encoding='utf-8').write(json.dumps({'prompt':'p','completion':'c','reward':0.0})+'\\n')\n"
            "open('.aider_fsm/rollout.json','w',encoding='utf-8').write(json.dumps({'paths': {'samples_jsonl': '.aider_fsm/samples.jsonl'}})+'\\n')\n"
            "PY\n"
        ),
        "evaluation.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p .aider_fsm\n"
            f"{shlex.quote(sys.executable)} - <<'PY'\n"
            "import json\n"
            "open('.aider_fsm/metrics.json','w',encoding='utf-8').write(json.dumps({'ok': True, 'score': 0.0})+'\\n')\n"
            "open('.aider_fsm/hints_used.json','w',encoding='utf-8').write(json.dumps({'ok': True, 'used_anchors': ['pytest'], 'commands': ['pytest -q']})+'\\n')\n"
            "PY\n"
        ),
        "benchmark.sh": "#!/usr/bin/env bash\nset -euo pipefail\necho benchmark\n",
    }
    for name, body in scripts.items():
        (stages / name).write_text(body, encoding="utf-8")

    (repo / "pipeline.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 3600",
                "tests:",
                "  cmds:",
                "    - bash .aider_fsm/stages/tests.sh",
                "deploy:",
                "  setup_cmds:",
                "    - bash .aider_fsm/stages/deploy_setup.sh",
                "  health_cmds:",
                "    - bash .aider_fsm/stages/deploy_health.sh",
                "  teardown_policy: on_failure",
                "  teardown_cmds:",
                "    - bash .aider_fsm/stages/deploy_teardown.sh",
                "rollout:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/rollout.sh",
                "evaluation:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/evaluation.sh",
                "  metrics_path: .aider_fsm/metrics.json",
                "  required_keys: [score, ok]",
                "benchmark:",
                "  run_cmds:",
                "    - bash .aider_fsm/stages/benchmark.sh",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        ),
        encoding="utf-8",
    )


class _SecondAttemptWritesContractAgent:
    # 作用：内部符号：_SecondAttemptWritesContractAgent
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈14 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_strict_opencode_mode.py:117；类型=class；引用≈2；规模≈14行
    def __init__(self, repo: Path) -> None:
        """中文说明：
        - 含义：测试用 agent：记录 repo 与调用次数，用于模拟“第一次 scaffold 不产出合同，第二次才产出”的场景。
        - 内容：保存 `repo`，并初始化 `calls=0`。
        - 可简略：是
        - 原因：纯测试替身；只要能提供最小状态即可。
        """
        # 作用：内部符号：_SecondAttemptWritesContractAgent.__init__
        # 能否简略：是
        # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=tests/test_strict_opencode_mode.py:118；类型=method；引用≈1；规模≈3行
        self.repo = repo
        self.calls = 0

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：测试用的 `run()`：在 scaffold_contract 场景下第 2 次调用时写入一份有效合同。
        - 内容：当 `purpose == \"scaffold_contract\"` 时递增 `calls`；若 `calls >= 2` 则调用 `_write_valid_contract(repo)`；返回带计数的 assistant_text。
        - 可简略：是
        - 原因：纯测试逻辑，用于稳定触发 retry 分支与成功分支。
        """
        # 作用：内部符号：_SecondAttemptWritesContractAgent.run
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈6 行；引用次数≈29（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_strict_opencode_mode.py:122；类型=method；引用≈29；规模≈6行
        if purpose == "scaffold_contract":
            self.calls += 1
            if self.calls >= 2:
                _write_valid_contract(self.repo)
        return AgentResult(assistant_text=f"attempt={self.calls}")

    def close(self) -> None:
        """中文说明：
        - 含义：测试用的 close（无资源需要释放）。
        - 内容：no-op。
        - 可简略：是
        - 原因：测试替身不持有外部资源（进程/句柄），保留方法只是为了满足 AgentClient 协议形状。
        """
        # 作用：内部符号：_SecondAttemptWritesContractAgent.close
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈2 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_strict_opencode_mode.py:129；类型=method；引用≈10；规模≈2行
        return


def test_open_env_scaffold_retries_and_succeeds_without_runner_fallback(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈16 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_strict_opencode_mode.py:133；类型=function；引用≈1；规模≈16行
    repo = tmp_path / "repo_retry"
    repo.mkdir()
    agent = _SecondAttemptWritesContractAgent(repo)
    handle = runner_env_local.open_env(
        repo,
        require_pipeline=True,
        scaffold_contract="opencode",
        scaffold_require_metrics=True,
        opencode_retry_attempts=2,
        seed_stage_skeleton=True,
        write_fallback_pipeline_yml=True,
        agent=agent,
    )
    assert handle.pipeline_path.exists()
    assert agent.calls >= 2
