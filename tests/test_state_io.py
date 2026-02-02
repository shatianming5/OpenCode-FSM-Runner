import json
from pathlib import Path

from runner.state import append_jsonl, default_state, load_state, save_state


def test_state_roundtrip(tmp_path: Path):
    state_path = tmp_path / "state.json"
    defaults = default_state(
        repo=tmp_path, plan_rel="PLAN.md", model="gpt-4o-mini", test_cmd="pytest -q"
    )
    save_state(state_path, defaults)
    loaded = load_state(state_path, defaults)
    assert loaded["version"] == defaults["version"]
    assert loaded["plan_path"] == "PLAN.md"


def test_state_merge_defaults(tmp_path: Path):
    state_path = tmp_path / "state.json"
    defaults = default_state(
        repo=tmp_path, plan_rel="PLAN.md", model="gpt-4o-mini", test_cmd="pytest -q"
    )
    state_path.write_text(json.dumps({"iter_idx": 7}) + "\n", encoding="utf-8")
    loaded = load_state(state_path, defaults)
    assert loaded["iter_idx"] == 7
    assert loaded["plan_path"] == "PLAN.md"


def test_append_jsonl(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    append_jsonl(log, {"a": 1})
    append_jsonl(log, {"b": 2})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
