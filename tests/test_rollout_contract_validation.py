from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from runner.env import _validate_rollout_samples


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_rollout_samples_enforces_hf_qa_min_samples(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo / "data" / "hf_manifest.json", "{}\n")

    table = pa.table({"question": ["q1", "q2", "q3"], "answer": ["a1", "a2", "a3"]})
    parquet_path = repo / "main" / "test-00000-of-00001.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    samples_path = repo / "samples.jsonl"
    _write(samples_path, json.dumps({"prompt": "q1", "completion": "c1", "reward": 1.0}) + "\n")

    rollout = {"paths": {"samples_jsonl": str(samples_path.resolve())}}
    _write(repo / ".aider_fsm" / "rollout.json", json.dumps(rollout) + "\n")

    ok, reason = _validate_rollout_samples(repo, None, mode="full", eval_limit=3)
    assert ok is False
    assert "hf_qa_samples_too_few" in reason

    _write(
        samples_path,
        "\n".join(
            [
                json.dumps({"prompt": "q1", "completion": "c1", "reward": 1.0}),
                json.dumps({"prompt": "q2", "completion": "c2", "reward": 0.0}),
                json.dumps({"prompt": "q3", "completion": "c3", "reward": 0.0}),
                "",
            ]
        ),
    )
    ok2, reason2 = _validate_rollout_samples(repo, None, mode="full", eval_limit=3)
    assert ok2 is True, reason2


def test_validate_rollout_samples_enforces_hf_qa_prompt_diversity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo / "data" / "hf_manifest.json", "{}\n")

    table = pa.table({"question": ["q1", "q2", "q3"], "answer": ["a1", "a2", "a3"]})
    parquet_path = repo / "main" / "test-00000-of-00001.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    samples_path = repo / "samples.jsonl"
    _write(
        samples_path,
        "\n".join(
            [
                json.dumps({"prompt": "same", "completion": "c1", "reward": 1.0}),
                json.dumps({"prompt": "same", "completion": "c2", "reward": 0.0}),
                json.dumps({"prompt": "same", "completion": "c3", "reward": 0.0}),
                "",
            ]
        ),
    )

    rollout = {"paths": {"samples_jsonl": str(samples_path.resolve())}}
    _write(repo / ".aider_fsm" / "rollout.json", json.dumps(rollout) + "\n")

    ok, reason = _validate_rollout_samples(repo, None, mode="full", eval_limit=3)
    assert ok is False
    assert "hf_qa_prompts_not_diverse" in reason


def test_validate_rollout_samples_enforces_hf_qa_prompt_anchoring(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo / "data" / "hf_manifest.json", "{}\n")

    table = pa.table(
        {
            "question": [
                "A long and unique question one?",
                "A long and unique question two?",
                "A long and unique question three?",
            ],
            "answer": ["a1", "a2", "a3"],
        }
    )
    parquet_path = repo / "main" / "test-00000-of-00001.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    samples_path = repo / "samples.jsonl"
    _write(
        samples_path,
        "\n".join(
            [
                json.dumps({"prompt": "p1 unrelated", "completion": "c1", "reward": 1.0}),
                json.dumps({"prompt": "p2 unrelated", "completion": "c2", "reward": 0.0}),
                json.dumps({"prompt": "p3 unrelated", "completion": "c3", "reward": 0.0}),
                "",
            ]
        ),
    )

    rollout = {"paths": {"samples_jsonl": str(samples_path.resolve())}}
    _write(repo / ".aider_fsm" / "rollout.json", json.dumps(rollout) + "\n")

    ok, reason = _validate_rollout_samples(repo, None, mode="full", eval_limit=3)
    assert ok is False
    assert "hf_qa_prompts_not_anchored" in reason


def test_validate_rollout_samples_rejects_all_empty_completions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    samples_path = repo / "samples.jsonl"
    _write(
        samples_path,
        "\n".join(
            [
                json.dumps({"prompt": "p1", "completion": "", "reward": 0.0}),
                json.dumps({"prompt": "p2", "completion": "  ", "reward": 0.0}),
                "",
            ]
        ),
    )
    rollout = {"paths": {"samples_jsonl": str(samples_path.resolve())}}
    _write(repo / ".aider_fsm" / "rollout.json", json.dumps(rollout) + "\n")

    ok, reason = _validate_rollout_samples(repo, None, mode="smoke", eval_limit=2)
    assert ok is False
    assert "samples_jsonl_all_empty_completions" in reason

    _write(samples_path, json.dumps({"prompt": "p1", "completion": "c", "reward": 0.0}) + "\n")
    ok2, reason2 = _validate_rollout_samples(repo, None, mode="smoke", eval_limit=1)
    assert ok2 is True, reason2
