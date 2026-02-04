from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.env_local import open_env, rollout_and_evaluate


def _py() -> str:
    return shlex.quote(sys.executable)


def test_env_local_rollout_and_evaluate(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".aider_fsm").mkdir()

    rollout_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('.aider_fsm/rollout.json').write_text("
            "json.dumps({'ok': True, 'model': os.getenv('OPENCODE_MODEL','')})+'\\n'"
            ")"
        )
    )
    eval_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "rollout=json.loads(pathlib.Path('.aider_fsm/rollout.json').read_text()); "
            "score=1 if rollout.get('ok') else 0; "
            "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('.aider_fsm/metrics.json').write_text("
            "json.dumps({'score': score, 'model': os.getenv('OPENCODE_MODEL','')})+'\\n'"
            ")"
        )
    )

    (repo / "pipeline.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
                "tests:",
                "  cmds: [echo ok]",
                "rollout:",
                "  run_cmds:",
                f"    - {json.dumps(rollout_cmd)}",
                "evaluation:",
                "  run_cmds:",
                f"    - {json.dumps(eval_cmd)}",
                "  metrics_path: .aider_fsm/metrics.json",
                "  required_keys: [score]",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = open_env(repo)
    out_dir = tmp_path / "artifacts_out"
    rollout_res, eval_res = rollout_and_evaluate(
        env,
        artifacts_dir=out_dir,
        env_overrides={"OPENCODE_MODEL": "opencode/gpt-5-nano"},
    )

    assert rollout_res.ok is True
    assert eval_res.ok is True

    assert rollout_res.rollout_path is not None
    assert rollout_res.rollout_path.exists()

    assert eval_res.metrics_path is not None
    assert eval_res.metrics_path.exists()

    assert eval_res.metrics is not None
    assert eval_res.metrics.get("score") == 1
    assert eval_res.metrics.get("model") == "opencode/gpt-5-nano"

    # Runner writes stage artifacts into the specified artifacts dir.
    assert (out_dir / "evaluation_summary.json").exists()
