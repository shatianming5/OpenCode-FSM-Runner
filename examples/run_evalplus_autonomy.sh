#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m runner \
  --repo "https://github.com/evalplus/evalplus" \
  --goal "Run a Docker-safe EvalPlus smoke benchmark (humaneval; <=3 problems; 1 sample; greedy). Write summary to .aider_fsm/evalplus_smoke_summary.json with ok=true only after a real run." \
  --test-cmd "bash .aider_fsm/evalplus_smoke_run.sh && python3 -c \"import json,pathlib; p=pathlib.Path('.aider_fsm/evalplus_smoke_summary.json'); d=json.loads(p.read_text()); assert d.get('ok') is True; assert d.get('dataset')=='humaneval'; assert int(d.get('n_problems', 999))<=3; assert int(d.get('n_samples_per_problem', 999))==1; assert isinstance(d.get('commands'), list) and d['commands']\"" \
  --seed "$ROOT/examples/seed_evalplus_autonomy.md" \
  --model "myproxy/gpt-5.2" \
  --max-iters 80 \
  --max-fix 10 \
  --unattended strict
