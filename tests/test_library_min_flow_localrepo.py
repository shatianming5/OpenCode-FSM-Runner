from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import runner_env


def _write_json(path: Path, obj: dict) -> None:
    # 作用：内部符号：_write_json
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈18（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_library_min_flow_localrepo.py:12；类型=function；引用≈18；规模≈3行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")


def test_library_setup_rollout_evaluate_min_repo(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：部分
    # 原因：测试代码（优先可读性）；规模≈143 行；引用次数≈1（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=tests/test_library_min_flow_localrepo.py:17；类型=function；引用≈1；规模≈143行
    repo = tmp_path / "repo"
    repo.mkdir()

    stages = repo / ".aider_fsm" / "stages"
    stages.mkdir(parents=True, exist_ok=True)

    py = shlex.quote(sys.executable)

    (stages / "tests.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\necho tests_ok\n", encoding="utf-8")
    (stages / "deploy_setup.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "mkdir -p .aider_fsm",
                f"{py} - <<'PY'",
                "import json",
                "open('.aider_fsm/runtime_env.json','w',encoding='utf-8').write(json.dumps({'ok': True})+'\\n')",
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
        "#!/usr/bin/env bash\nset -euo pipefail\necho teardown > .aider_fsm/teardown_ran.txt\n",
        encoding="utf-8",
    )

    (stages / "rollout.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "mkdir -p .aider_fsm",
                f"{py} - <<'PY'",
                "import json, pathlib",
                "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True)",
                "pathlib.Path('.aider_fsm/samples.jsonl').write_text(json.dumps({'prompt':'p','completion':'c','reward':0.0})+'\\n', encoding='utf-8')",
                "pathlib.Path('.aider_fsm/rollout.json').write_text(json.dumps({'paths': {'samples_jsonl': '.aider_fsm/samples.jsonl'}})+'\\n', encoding='utf-8')",
                "PY",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Use the generic evaluation helper so we also validate the hints/metrics contracts.
    (stages / "evaluation.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "\"$AIDER_FSM_PYTHON\" \"$AIDER_FSM_RUNNER_ROOT/runner/generic_evaluation.py\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (repo / "pipeline.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
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
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sess = runner_env.setup(repo, strict_opencode=True, require_metrics=True, unattended="strict")
    hint_cmd = f"{py} -c \"print('score 0.2')\""
    common_overrides = {
        "AIDER_FSM_REQUIRE_HINTS": "1",
        "AIDER_FSM_REQUIRE_REAL_SCORE": "1",
        "AIDER_FSM_HINTS_JSON": json.dumps([hint_cmd]),
        "AIDER_FSM_HINT_ANCHORS_JSON": json.dumps(["score"]),
        "AIDER_FSM_HINT_MAX_ATTEMPTS": "1",
        "AIDER_FSM_HINT_TIMEOUT_SECONDS": "30",
    }

    r = sess.rollout(
        llm="deepseek-v3.2",
        mode="smoke",
        require_samples=True,
        repair_iters=0,
        env_overrides=dict(common_overrides),
    )
    assert r.ok is True

    e = sess.evaluate(
        mode="smoke",
        repair_iters=0,
        env_overrides=dict(common_overrides),
    )
    assert e.ok is True
    assert isinstance(e.metrics, dict)
    assert e.metrics.get("ok") is True
    assert abs(float(e.metrics.get("score")) - 0.2) < 1e-6

    # Evidence files produced by the contract + generic evaluation.
    assert (repo / ".aider_fsm" / "rollout.json").exists()
    assert (repo / ".aider_fsm" / "metrics.json").exists()
    assert (repo / ".aider_fsm" / "hints_used.json").exists()
    assert (repo / ".aider_fsm" / "hints_run.json").exists()

    # `sess.evaluate()` must auto-teardown best-effort (deploy_teardown.sh writes a marker).
    assert (repo / ".aider_fsm" / "teardown_ran.txt").exists()

    # `sess.evaluate(llm=...)` should work too (even if called on a fresh session).
    sess2 = runner_env.setup(repo, strict_opencode=True, require_metrics=True, unattended="strict")
    e2 = sess2.evaluate(
        llm="deepseek-v3.2",
        mode="smoke",
        repair_iters=0,
        env_overrides=dict(common_overrides),
    )
    assert e2.ok is True
