from __future__ import annotations

import json
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_env_module_deploy_rollout_evaluation_smoke(tmp_path: Path):
    import env

    repo = tmp_path / "repo"
    (repo / ".aider_fsm" / "stages").mkdir(parents=True)

    _write(
        repo / ".aider_fsm" / "stages" / "tests.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo ok\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_setup.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "test -n \"${AIDER_TRAINED_MODEL_DIR:-}\" || (echo \"missing AIDER_TRAINED_MODEL_DIR\" >&2; exit 2)\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.ml.serve_openai_compat start \\\n"
        "  --backend echo \\\n"
        "  --model-dir \"$AIDER_TRAINED_MODEL_DIR\" \\\n"
        "  --host 127.0.0.1 \\\n"
        "  --port 0 \\\n"
        "  --runtime-env-out \"$RUNTIME_ENV_JSON\" \\\n"
        "  --pid-file .aider_fsm/server.pid \\\n"
        "  --log-file \"$AIDER_FSM_ARTIFACTS_DIR/server.log\"\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_health.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "test -f \"$RUNTIME_ENV_JSON\"\n"
        "\"$AIDER_FSM_PYTHON\" - <<'PY'\n"
        "import json, os, urllib.request\n"
        "p = os.environ.get('AIDER_RUNTIME_ENV_PATH') or '.aider_fsm/runtime_env.json'\n"
        "obj = json.load(open(p,'r',encoding='utf-8'))\n"
        "url = (obj.get('service') or {}).get('health_url')\n"
        "assert url\n"
        "with urllib.request.urlopen(url, timeout=3) as r:\n"
        "  data = r.read().decode('utf-8', errors='replace')\n"
        "assert 'ok' in data\n"
        "print('ok')\n"
        "PY\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_teardown.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "\"$AIDER_FSM_PYTHON\" -m runner.ml.serve_openai_compat stop --pid-file .aider_fsm/server.pid || true\n"
        "echo teardown_ok\n",
    )

    _write(
        repo / ".aider_fsm" / "stages" / "rollout.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "OUT_DIR=\"${AIDER_FSM_ARTIFACTS_DIR:-.aider_fsm/artifacts}\"\n"
        "mkdir -p \"$OUT_DIR\"\n"
        "\"$AIDER_FSM_PYTHON\" - <<'PY'\n"
        "import json, os, pathlib, time, urllib.request\n"
        "runtime = os.environ.get('AIDER_RUNTIME_ENV_PATH') or '.aider_fsm/runtime_env.json'\n"
        "obj = json.load(open(runtime,'r',encoding='utf-8'))\n"
        "base = (obj.get('service') or {}).get('base_url')\n"
        "assert base\n"
        "url = base.rstrip('/') + '/v1/chat/completions'\n"
        "payload = {\n"
        "  'model': (obj.get('inference') or {}).get('model') or 'echo',\n"
        "  'messages': [{'role':'user','content':'hello'}],\n"
        "  'max_tokens': 32,\n"
        "}\n"
        "req = urllib.request.Request(url, method='POST', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'})\n"
        "with urllib.request.urlopen(req, timeout=3) as r:\n"
        "  data = json.loads(r.read().decode('utf-8', errors='replace'))\n"
        "text = (((data.get('choices') or [{}])[0]).get('message') or {}).get('content')\n"
        "assert isinstance(text, str) and text\n"
        "out_dir = pathlib.Path(os.environ.get('AIDER_FSM_ARTIFACTS_DIR') or '.aider_fsm/artifacts')\n"
        "traj = out_dir / 'rollout_samples.jsonl'\n"
        "traj.write_text(json.dumps({'prompt': 'hello', 'completion': text, 'reward': 1.0}, ensure_ascii=False)+'\\n', encoding='utf-8')\n"
        "rollout = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'ok': True,\n"
        "  'paths': {'samples_jsonl': str(traj)},\n"
        "  'n': 1,\n"
        "}\n"
        "pathlib.Path('.aider_fsm/rollout.json').write_text(json.dumps(rollout, ensure_ascii=False, indent=2)+'\\n', encoding='utf-8')\n"
        "print('ok')\n"
        "PY\n",
    )

    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "\"$AIDER_FSM_PYTHON\" .aider_fsm/eval.py\n",
    )
    _write(
        repo / ".aider_fsm" / "eval.py",
        "\n".join(
            [
                "import json",
                "import os",
                "import pathlib",
                "import time",
                "import urllib.request",
                "",
                "runtime = os.environ.get('AIDER_RUNTIME_ENV_PATH') or '.aider_fsm/runtime_env.json'",
                "obj = json.load(open(runtime,'r',encoding='utf-8'))",
                "health = (obj.get('service') or {}).get('health_url')",
                "assert health",
                "with urllib.request.urlopen(health, timeout=3) as r:",
                "  data = json.loads(r.read().decode('utf-8', errors='replace'))",
                "assert data.get('ok') is True",
                "score = float(data.get('ok') is True)",
                "metrics = {",
                "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),",
                "  'ok': True,",
                "  'score': score,",
                "}",
                "pathlib.Path('.aider_fsm/metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\\n', encoding='utf-8')",
                "print('ok')",
                "",
            ]
        )
        + "\n",
    )

    _write(
        repo / "pipeline.yml",
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
                "  max_total_seconds: 300",
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
        )
        + "\n",
    )

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    sess = env.setup(repo)
    rollout_res = env.rollout(model_dir, require_samples=True)
    assert rollout_res.ok is True
    assert rollout_res.rollout_path is not None and rollout_res.rollout_path.exists()

    eval_res = env.evaluate()
    assert eval_res.ok is True
    assert eval_res.metrics is not None
    assert eval_res.metrics.get("ok") is True
    assert eval_res.metrics.get("score") == 1.0

    eval_res2 = sess.evaluate()
    assert eval_res2.ok is True

    assert env.teardown() is True
    assert not (repo / ".aider_fsm" / "server.pid").exists()

    rollout_obj = json.loads(rollout_res.rollout_path.read_text(encoding="utf-8"))
    assert isinstance(rollout_obj, dict)
    assert "paths" in rollout_obj


def test_env_module_rollout_evaluation_remote_llm_smoke(tmp_path: Path, monkeypatch):
    import env

    monkeypatch.setenv("OPENAI_API_BASE", "https://api.example.test/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    repo = tmp_path / "repo"
    (repo / ".aider_fsm" / "stages").mkdir(parents=True)

    _write(
        repo / ".aider_fsm" / "stages" / "tests.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo ok\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_setup.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "test \"${AIDER_LLM_KIND:-}\" = \"remote\" || (echo \"expected remote\" >&2; exit 2)\n"
        "test -n \"${AIDER_LLM_MODEL:-}\" || (echo \"missing AIDER_LLM_MODEL\" >&2; exit 2)\n"
        "test -n \"${OPENAI_BASE_URL:-}\" || (echo \"missing OPENAI_BASE_URL\" >&2; exit 2)\n"
        "test \"${OPENAI_BASE_URL:-}\" = \"${OPENAI_API_BASE:-}\" || (echo \"base url mismatch\" >&2; exit 2)\n"
        "\"$AIDER_FSM_PYTHON\" - <<'PY' > \"$RUNTIME_ENV_JSON\"\n"
        "import json, os, time\n"
        "obj = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'run_id': os.getenv('AIDER_FSM_RUN_ID',''),\n"
        "  'service': {},\n"
        "  'inference': {'type': 'openai_compat', 'model': os.getenv('AIDER_LLM_MODEL','')},\n"
        "  'paths': {'rollout_path': '.aider_fsm/rollout.json', 'metrics_path': '.aider_fsm/metrics.json'},\n"
        "}\n"
        "print(json.dumps(obj, ensure_ascii=False, indent=2))\n"
        "PY\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_health.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "RUNTIME_ENV_JSON=\"${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}\"\n"
        "test -f \"$RUNTIME_ENV_JSON\"\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "deploy_teardown.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo teardown_ok\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "rollout.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "OUT_DIR=\"${AIDER_FSM_ARTIFACTS_DIR:-.aider_fsm/artifacts}\"\n"
        "mkdir -p \"$OUT_DIR\"\n"
        "\"$AIDER_FSM_PYTHON\" - <<'PY'\n"
        "import json, os, pathlib, time\n"
        "out_dir = pathlib.Path(os.environ.get('AIDER_FSM_ARTIFACTS_DIR') or '.aider_fsm/artifacts')\n"
        "traj = out_dir / 'rollout_samples.jsonl'\n"
        "traj.write_text(json.dumps({'prompt': 'p', 'completion': 'c', 'reward': 1.0}, ensure_ascii=False)+'\\n', encoding='utf-8')\n"
        "rollout = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'ok': True,\n"
        "  'paths': {'samples_jsonl': str(traj)},\n"
        "  'n': 1,\n"
        "}\n"
        "pathlib.Path('.aider_fsm/rollout.json').write_text(json.dumps(rollout, ensure_ascii=False, indent=2)+'\\n', encoding='utf-8')\n"
        "print('ok')\n"
        "PY\n",
    )
    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p .aider_fsm\n"
        "test \"${AIDER_LLM_KIND:-}\" = \"remote\" || (echo \"expected remote\" >&2; exit 2)\n"
        "test -n \"${AIDER_LLM_MODEL:-}\" || (echo \"missing AIDER_LLM_MODEL\" >&2; exit 2)\n"
        "\"$AIDER_FSM_PYTHON\" -c \"import json, os, pathlib, time; m=os.environ.get('AIDER_LLM_MODEL',''); metrics={'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()), 'ok': True, 'score': float(bool(m))}; pathlib.Path('.aider_fsm/metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\\\\n', encoding='utf-8'); print('ok')\"\n",
    )

    _write(
        repo / "pipeline.yml",
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
                "  max_total_seconds: 300",
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
        )
        + "\n",
    )

    env.setup(repo)
    rollout_res, eval_res = env.rollout_and_evaluation("openai/gpt-4o-mini", require_samples=True)
    assert rollout_res.ok is True
    assert rollout_res.rollout_path is not None and rollout_res.rollout_path.exists()
    assert eval_res.ok is True
    assert eval_res.metrics is not None
    assert eval_res.metrics.get("ok") is True
    assert eval_res.metrics.get("score") == 1.0
