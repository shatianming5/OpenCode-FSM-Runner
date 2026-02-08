from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from runner import cli as runner_cli
from runner import env as runner_env
from runner import env_local as runner_env_local
from runner.agent_client import AgentResult
from runner.env_local import EnvHandle
from runner.pipeline_spec import PipelineSpec
from runner.runner import RunnerConfig, run


def _ok_cmd() -> str:
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit(0)"'


def _latest_run_dir(artifacts_base: Path) -> Path:
    runs = sorted([p for p in artifacts_base.iterdir() if p.is_dir()])
    assert runs
    return runs[-1]


class _NoWriteAgent:
    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        return AgentResult(assistant_text=f"noop ({purpose})")

    def close(self) -> None:
        return


def test_env_setup_strict_disables_seed_and_fallback(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pipeline_path = repo / "pipeline.yml"
    pipeline_path.write_text("version: 1\n", encoding="utf-8")
    handle = EnvHandle(repo=repo, pipeline_path=pipeline_path, pipeline=PipelineSpec())

    calls: list[dict] = []

    def _fake_open_env(_target: str, **kwargs) -> EnvHandle:
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


def test_runner_strict_opencode_never_writes_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "repo_strict"
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
        strict_opencode=True,
    )

    rc = run(cfg, agent=_NoWriteAgent())
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert not (run_dir / "scaffold_fallback_used.txt").exists()
    prov = json.loads((run_dir / "scaffold_provenance.json").read_text(encoding="utf-8"))
    assert prov.get("strict_opencode") is True
    assert prov.get("runner_written_paths") == []


def test_runner_non_strict_opencode_never_writes_runner_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "repo_non_strict"
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
        strict_opencode=False,
    )

    rc = run(cfg, agent=_NoWriteAgent())
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert not (run_dir / "scaffold_fallback_used.txt").exists()
    prov = json.loads((run_dir / "scaffold_provenance.json").read_text(encoding="utf-8"))
    assert prov.get("strict_opencode") is False
    assert list(prov.get("runner_written_paths") or []) == []


def _write_valid_contract(repo: Path) -> None:
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
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls = 0

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        if purpose == "scaffold_contract":
            self.calls += 1
            if self.calls >= 2:
                _write_valid_contract(self.repo)
        return AgentResult(assistant_text=f"attempt={self.calls}")

    def close(self) -> None:
        return


def test_open_env_scaffold_retries_and_succeeds_without_runner_fallback(tmp_path: Path) -> None:
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


def test_cli_passes_strict_opencode_flag(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo_cli"
    repo.mkdir()
    (repo / "pipeline.yml").write_text("version: 1\n", encoding="utf-8")

    monkeypatch.setattr(runner_cli, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_cli, "_resolve_model", lambda _raw: "openai/gpt-4o-mini")
    monkeypatch.setattr(
        runner_cli,
        "prepare_repo",
        lambda _repo, clones_dir=None: SimpleNamespace(repo=repo, cloned_from=""),
    )

    captured: dict[str, object] = {}

    def _fake_run(cfg) -> int:
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(runner_cli, "run", _fake_run)
    rc = runner_cli.main(["--repo", str(repo), "--no-strict-opencode", "--preflight-only"])
    assert rc == 0
    cfg = captured.get("cfg")
    assert cfg is not None
    assert getattr(cfg, "strict_opencode") is False
