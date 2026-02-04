from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .pipeline_spec import PipelineSpec


_STEP_RE = re.compile(r"^\s*-\s*\[\s*([xX ])\s*\]\s*\(STEP_ID=([0-9]+)\)\s*(.*?)\s*$")


def plan_template(goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> str:
    goal = goal.strip() or "<fill goal>"
    acceptance: list[str] = [f"- [ ] TEST_CMD passes: `{test_cmd}`"]
    if pipeline:
        if pipeline.deploy_setup_cmds or pipeline.deploy_health_cmds:
            acceptance.append("- [ ] Deploy succeeds (see pipeline.yml)")
        if getattr(pipeline, "rollout_run_cmds", None):
            acceptance.append("- [ ] Rollout succeeds (see pipeline.yml)")
        if getattr(pipeline, "evaluation_run_cmds", None):
            acceptance.append("- [ ] Evaluation succeeds (see pipeline.yml)")
        if pipeline.benchmark_run_cmds:
            acceptance.append("- [ ] Benchmark succeeds (see pipeline.yml)")
        if (
            getattr(pipeline, "evaluation_metrics_path", None)
            or getattr(pipeline, "evaluation_required_keys", None)
            or pipeline.benchmark_metrics_path
            or pipeline.benchmark_required_keys
        ):
            acceptance.append("- [ ] Metrics file/keys present (see pipeline.yml)")
    return (
        "# PLAN\n"
        "\n"
        "## Goal\n"
        f"- {goal}\n"
        "\n"
        "## Acceptance\n"
        + "\n".join(acceptance)
        + "\n"
        "\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) Build Backlog: break the goal into smallest steps (each step = one edit + one verify)\n"
        "\n"
        "## Backlog\n"
        "\n"
        "## Done\n"
        "- [x] (STEP_ID=000) Initialized plan file\n"
        "\n"
        "## Notes\n"
        "- \n"
    )


def ensure_plan_file(plan_abs: Path, goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> None:
    if plan_abs.exists():
        return
    plan_abs.parent.mkdir(parents=True, exist_ok=True)
    plan_abs.write_text(plan_template(goal, test_cmd, pipeline=pipeline), encoding="utf-8")


def _extract_section_lines(lines: list[str], heading_prefix: str) -> list[str] | None:
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return lines[start:end]


def _parse_step_lines(section_lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    steps: list[dict[str, Any]] = []
    bad: list[str] = []
    for line in section_lines:
        if not re.match(r"^\s*-\s*\[", line):
            continue
        m = _STEP_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        checked = m.group(1).lower() == "x"
        steps.append({"id": m.group(2), "text": m.group(3).strip(), "checked": checked})
    return steps, bad


def find_duplicate_step_ids(plan_text: str) -> list[str]:
    lines = plan_text.splitlines()
    ids: list[str] = []
    for heading in ("## Next", "## Backlog", "## Done"):
        section = _extract_section_lines(lines, heading)
        if section is None:
            continue
        steps, _bad = _parse_step_lines(section)
        ids.extend([s["id"] for s in steps])
    counts: dict[str, int] = {}
    for sid in ids:
        counts[sid] = counts.get(sid, 0) + 1
    return [sid for sid, c in counts.items() if c > 1]


def parse_next_step(plan_text: str) -> tuple[dict[str, str] | None, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Next")
    if section is None:
        return None, "missing_next_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return None, "bad_next_line"
    if len(steps) == 0:
        return None, None
    if len(steps) != 1:
        return None, "next_count_not_one"
    if steps[0]["checked"]:
        return None, "next_is_checked"
    dups = find_duplicate_step_ids(plan_text)
    if dups:
        return None, "duplicate_step_id"
    return {"id": steps[0]["id"], "text": steps[0]["text"]}, None


def parse_backlog_open_count(plan_text: str) -> tuple[int, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Backlog")
    if section is None:
        return 0, "missing_backlog_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return 0, "bad_backlog_line"
    return sum(1 for s in steps if not s["checked"]), None


def parse_plan(plan_text: str) -> dict[str, Any]:
    next_step, next_err = parse_next_step(plan_text)
    backlog_open, backlog_err = parse_backlog_open_count(plan_text)
    errors: list[str] = []
    if next_err:
        errors.append(next_err)
    if backlog_err:
        errors.append(backlog_err)
    return {
        "next_step": next_step,
        "backlog_open_count": backlog_open,
        "errors": errors,
    }
