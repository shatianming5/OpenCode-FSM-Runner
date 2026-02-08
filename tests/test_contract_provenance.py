from __future__ import annotations

from pathlib import Path

from runner.contract_provenance import (
    build_contract_provenance_report,
    changed_paths,
    snapshot_contract_files,
)


def test_contract_provenance_classifies_runner_and_tool_writes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pipeline.yml").write_text("version: 1\n", encoding="utf-8")

    before = snapshot_contract_files(repo)
    (repo / "pipeline.yml").write_text("version: 1\nsecurity:\n  mode: safe\n", encoding="utf-8")
    (repo / ".aider_fsm" / "stages").mkdir(parents=True, exist_ok=True)
    (repo / ".aider_fsm" / "stages" / "rollout.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    after = snapshot_contract_files(repo)

    report = build_contract_provenance_report(
        repo=repo,
        purpose="scaffold_contract",
        strict_opencode=False,
        before=before,
        after=after,
        tool_trace=[
            {
                "turn": 1,
                "results": [
                    {
                        "tool": "write",
                        "ok": True,
                        "filePath": str((repo / "pipeline.yml").resolve()),
                    }
                ],
            }
        ],
        runner_written_paths={".aider_fsm/stages/rollout.sh"},
    )

    changed = {x["path"]: x for x in report["files"] if x["status"] != "unchanged"}
    assert changed["pipeline.yml"]["source"] == "opencode_tool_write"
    assert changed[".aider_fsm/stages/rollout.sh"]["source"] == "runner_prewrite_or_fallback"
    assert ".aider_fsm/stages/rollout.sh" in changed_paths(before, after)
