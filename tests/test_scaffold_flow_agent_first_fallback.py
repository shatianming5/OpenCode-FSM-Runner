from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.agent_client import AgentResult
from runner.runner import RunnerConfig, run


def _ok_cmd() -> str:
    """中文说明：
    - 含义：构造一个稳定返回 0 的 Python 命令（用于 tests_cmds）。
    - 内容：`python -c "sys.exit(0)"`，并对解释器路径做 shell quote。
    - 可简略：是（测试 helper；可直接内联）。
    """
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit(0)"'


def _latest_run_dir(artifacts_base: Path) -> Path:
    """中文说明：
    - 含义：找到 runner 写入的最新一次 run 目录（按目录名排序取最后）。
    - 内容：列出 artifacts_base 下的目录并断言非空，用于定位本次运行的 artifacts 输出。
    - 可简略：可能（也可通过读取 summary.json/固定 run_id；当前按排序足够稳定）。
    """
    runs = sorted([p for p in artifacts_base.iterdir() if p.is_dir()])
    assert runs, f"no run dirs under {artifacts_base}"
    return runs[-1]


class _AgentWritesValidPipeline:
    """中文说明：
    - 含义：测试用 agent：在 scaffold_contract 阶段写出一个“可验证通过”的 pipeline.yml。
    - 内容：写入最小安全配置 + tests cmd + benchmark 写 metrics.json，并满足 required_keys，模拟 scaffold 成功路径。
    - 可简略：可能（可用固定 fixture pipeline 代替；但 agent 写入更贴近真实流程）。
    """

    def __init__(self, repo: Path):
        """中文说明：
        - 含义：绑定一个可写 repo，用于写入 pipeline.yml。
        - 内容：保存 repo 路径到实例字段。
        - 可简略：是（测试样板）。
        """
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：模拟 agent 运行：仅在 scaffold_contract 时写出有效 pipeline.yml。
        - 内容：生成 ok_cmd 与写 metrics 的命令，落盘 pipeline.yml；返回 assistant_text 供 runner 记录。
        - 可简略：否（该写入内容决定 scaffold 是否能通过，是测试核心）。
        """
        if purpose == "scaffold_contract":
            ok_cmd = _ok_cmd()
            # Create required stage scripts under `.aider_fsm/stages/`.
            stages = self._repo / ".aider_fsm" / "stages"
            stages.mkdir(parents=True, exist_ok=True)
            (stages / "tests.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n", encoding="utf-8")
            (stages / "deploy_setup.sh").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        "mkdir -p .aider_fsm",
                        f"{shlex.quote(sys.executable)} - <<'PY'",
                        "import json, os, pathlib",
                        "p = pathlib.Path('.aider_fsm/runtime_env.json')",
                        "p.write_text(json.dumps({",
                        "  'ts': '',",
                        "  'run_id': os.getenv('AIDER_FSM_RUN_ID',''),",
                        "  'service': {},",
                        "  'paths': {'rollout_path': '.aider_fsm/rollout.json', 'metrics_path': '.aider_fsm/metrics.json'},",
                        "}, ensure_ascii=False) + '\\n', encoding='utf-8')",
                        "PY",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (stages / "deploy_health.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\ntest -f .aider_fsm/runtime_env.json\n",
                encoding="utf-8",
            )
            (stages / "deploy_teardown.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\necho teardown_skipped\n",
                encoding="utf-8",
            )
            (stages / "rollout.sh").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        "mkdir -p .aider_fsm",
                        f"{shlex.quote(sys.executable)} - <<'PY'",
                        "import json, os, pathlib",
                        "pathlib.Path('.aider_fsm/rollout.json').write_text(",
                        "  json.dumps({'rollout': {'ok': True}, 'run_id': os.getenv('AIDER_FSM_RUN_ID','')}, ensure_ascii=False) + '\\n'",
                        ")",
                        "PY",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (stages / "evaluation.sh").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        "mkdir -p .aider_fsm",
                        f"{shlex.quote(sys.executable)} - <<'PY'",
                        "import json, os, pathlib",
                        "pathlib.Path('.aider_fsm/metrics.json').write_text(",
                        "  json.dumps({'ok': True, 'score': 0.0, 'run_id': os.getenv('AIDER_FSM_RUN_ID','')}, ensure_ascii=False) + '\\n'",
                        ")",
                        "PY",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (stages / "benchmark.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\necho benchmark_skipped\n",
                encoding="utf-8",
            )

            (self._repo / "pipeline.yml").write_text(
                "\n".join(
                    [
                        "version: 1",
                        "security:",
                        "  mode: safe",
                        "  max_cmd_seconds: 60",
                        "tests:",
                        "  cmds:",
                        f"    - {json.dumps(ok_cmd)}",
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
                        "artifacts:",
                        "  out_dir: .aider_fsm/artifacts",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        """中文说明：
        - 含义：释放资源（测试中无实际资源）。
        - 内容：空实现满足 Agent 协议。
        - 可简略：是（测试 stub）。
        """
        return


class _AgentWritesInvalidPipeline:
    """中文说明：
    - 含义：测试用 agent：在 scaffold_contract 阶段写出一个“语法可读但版本不支持”的 pipeline.yml。
    - 内容：写入 version: 2，用于触发 pipeline 解析失败并验证 runner fast-fail 行为。
    - 可简略：可能（用函数桩也可以；保留类是为了复用 agent 协议）。
    """

    def __init__(self, repo: Path):
        """中文说明：
        - 含义：绑定 repo 以写入 pipeline.yml。
        - 内容：保存 repo 路径。
        - 可简略：是。
        """
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：模拟 agent 运行：仅在 scaffold_contract 时写出无效版本的 pipeline.yml。
        - 内容：落盘 `version: 2`，然后返回 assistant_text。
        - 可简略：是（测试桩）。
        """
        if purpose == "scaffold_contract":
            (self._repo / "pipeline.yml").write_text("version: 2\n", encoding="utf-8")
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        """中文说明：
        - 含义：释放资源（测试中无实际资源）。
        - 内容：空实现满足协议。
        - 可简略：是。
        """
        return


class _AgentWritesParseableButMissingMetricsContract:
    """中文说明：
    - 含义：测试用 agent：写出 YAML 可解析但缺少 metrics 契约的 pipeline.yml。
    - 内容：仅包含 artifacts.out_dir，缺少 benchmark/evaluation 配置，用于触发 pipeline 语义校验失败路径。
    - 可简略：可能（可直接写 fixture 文件；保留 agent 形式更贴近真实 scaffold 产出）。
    """

    def __init__(self, repo: Path):
        """中文说明：
        - 含义：绑定 repo 以写入 pipeline.yml。
        - 内容：保存 repo 路径。
        - 可简略：是。
        """
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：模拟 agent 运行：仅在 scaffold_contract 时写出缺少 metrics 契约的 pipeline.yml。
        - 内容：落盘一个可解析但不满足 runner 要求的 pipeline.yml，然后返回 assistant_text。
        - 可简略：是（测试桩）。
        """
        if purpose == "scaffold_contract":
            (self._repo / "pipeline.yml").write_text(
                "\n".join(
                    [
                        "version: 1",
                        "artifacts:",
                        "  out_dir: .aider_fsm/artifacts",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        """中文说明：
        - 含义：释放资源（测试中无实际资源）。
        - 内容：空实现满足协议。
        - 可简略：是。
        """
        return


def test_opencode_scaffold_agent_first_success(tmp_path: Path):
    """中文说明：
    - 含义：验证 scaffold_contract=opencode 时，runner 会先让 agent 生成 pipeline，并在成功时完成 preflight。
    - 内容：用 _AgentWritesValidPipeline 写出满足 metrics 契约的 pipeline；运行 runner preflight-only；断言 pipeline/metrics 与 artifacts 产物存在。
    - 可简略：否（scaffold 合同是核心能力；建议保留端到端覆盖）。
    """
    repo = tmp_path / "repo_ok"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesValidPipeline(repo))
    assert rc == 0

    assert (repo / "pipeline.yml").exists()
    assert (repo / ".aider_fsm" / "runtime_env.json").exists()
    assert (repo / ".aider_fsm" / "rollout.json").exists()
    assert (repo / ".aider_fsm" / "metrics.json").exists()

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_result.txt").exists()
    assert (run_dir / "preflight" / "summary.json").exists()


def test_opencode_scaffold_invalid_pipeline_fails_fast(tmp_path: Path):
    """中文说明：
    - 含义：验证 scaffold 生成的 pipeline 解析失败时，runner 会快速失败并写出错误 artifacts。
    - 内容：agent 写 version=2；运行 runner preflight-only；断言返回码=2 且 scaffold_error/parse_error 文件存在。
    - 可简略：否（关键负例覆盖，防止 silent fallback）。
    """
    repo = tmp_path / "repo_bad"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesInvalidPipeline(repo))
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_pipeline_parse_error.txt").exists()
    assert (run_dir / "scaffold_error.txt").exists()
    assert not (run_dir / "preflight").exists()


def test_opencode_scaffold_incomplete_pipeline_fails_fast(tmp_path: Path):
    """中文说明：
    - 含义：验证 scaffold 生成的 pipeline 虽可解析但缺少 metrics 合同时会失败。
    - 内容：agent 写仅含 artifacts 的 pipeline；运行 runner preflight-only；断言返回码=2 且 validation_error/scaffold_error 文件存在。
    - 可简略：否（关键负例覆盖：语义校验必须严格）。
    """
    repo = tmp_path / "repo_incomplete"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesParseableButMissingMetricsContract(repo))
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_pipeline_validation_error.txt").exists()
    assert (run_dir / "scaffold_error.txt").exists()
    assert not (run_dir / "preflight").exists()
