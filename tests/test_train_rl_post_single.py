from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_single_file_rl_script_help(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = (root / "examples" / "train_rl_post_single.py").resolve()
    res = subprocess.run([sys.executable, str(script), "--help"], check=False, capture_output=True, text=True, cwd=str(root))
    assert res.returncode == 0
    assert "env.setup" in (res.stdout + res.stderr)


def test_single_file_rl_script_dry_run_executes_env_commands(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = (root / "examples" / "train_rl_post_single.py").resolve()

    repo = tmp_path / "bench_repo"
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
        "  'max_tokens': 16,\n"
        "}\n"
        "req = urllib.request.Request(url, method='POST', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'})\n"
        "with urllib.request.urlopen(req, timeout=3) as r:\n"
        "  data = json.loads(r.read().decode('utf-8', errors='replace'))\n"
        "text = (((data.get('choices') or [{}])[0]).get('message') or {}).get('content')\n"
        "assert isinstance(text, str) and text\n"
        "out_dir = pathlib.Path(os.environ.get('AIDER_FSM_ARTIFACTS_DIR') or '.aider_fsm/artifacts')\n"
        "samples = out_dir / 'samples.jsonl'\n"
        "samples.write_text(json.dumps({'prompt': 'hello', 'completion': text, 'reward': 1.0}, ensure_ascii=False)+'\\n', encoding='utf-8')\n"
        "rollout = {\n"
        "  'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),\n"
        "  'ok': True,\n"
        "  'paths': {'samples_jsonl': str(samples)},\n"
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

    out_root = tmp_path / "out"
    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--bench",
            str(repo),
            "--dry-run",
            "--eval-mode",
            "smoke",
            "--eval-limit",
            "5",
            "--opencode-model",
            "",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr

    summary_path = out_root / "rl_post_train_summary.json"
    assert summary_path.exists()
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(obj, dict)
    runs = obj.get("runs")
    assert isinstance(runs, list) and len(runs) == 1
    run0 = runs[0]
    assert run0.get("rollout_ok") is True
    assert run0.get("eval_ok") is True
    metrics = run0.get("metrics") or {}
    assert metrics.get("ok") is True
    assert metrics.get("score") == 1.0

    assert not (repo / ".aider_fsm" / "server.pid").exists()
