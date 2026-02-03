from __future__ import annotations

import json
from pathlib import Path

from .pipeline_verify import fmt_stage_tail
from .subprocess_utils import STDIO_TAIL_CHARS, tail
from .types import VerificationResult


def make_plan_update_prompt(snapshot_text: str, test_cmd: str, *, extra: str = "") -> str:
    extra = extra.strip()
    extra_block = f"\n[EXTRA]\n{extra}\n" if extra else ""
    return (
        "You are a strict plan editor. Your job: ONLY edit PLAN.md so it becomes machine-parseable and executable.\n"
        "\n"
        "Tooling:\n"
        "- To READ a file, emit:\n"
        "  ```json\n"
        "  {\"filePath\": \"PATH\"}\n"
        "  ```\n"
        "- To WRITE a file, emit:\n"
        "  ```json\n"
        "  {\"filePath\": \"PATH\", \"content\": \"...\"}\n"
        "  ```\n"
        "- To run a command, emit:\n"
        "  ```bash\n"
        "  bash\n"
        "  {\"command\":\"...\",\"description\":\"...\"}\n"
        "  ```\n"
        "Prefer file WRITE tool calls for any edits/creates (do NOT use bash redirections like `>` to write files).\n"
        "The runner executes tool calls and replies with a ```tool_result``` block.\n"
        "\n"
        "Hard constraints:\n"
        "1) You may ONLY modify PLAN.md. Do NOT touch any other file.\n"
        "2) Under `## Next (exactly ONE item)` there must be exactly ONE unchecked item:\n"
        "   `- [ ] (STEP_ID=NNN) ...`\n"
        "3) Each step must be atomic: one edit + one verification.\n"
        "4) Verification is ALWAYS the runner's TEST_CMD. Write steps so that after completing the step, running TEST_CMD will pass.\n"
        f"5) `## Acceptance` MUST include the exact line (verbatim): - [ ] TEST_CMD passes: `{test_cmd}`\n"
        "6) Do NOT put placeholders like `{TEST_CMD}` into PLAN.md; always use the exact TEST_CMD string.\n"
        "7) The `Next` item must be an IMPLEMENTATION step (edits code/scripts/config), not a meta planning step about editing PLAN.md.\n"
        "8) It is OK if the Next step creates new files; do not mark Blocked just to ask to 'add a file to chat'.\n"
        "9) Put uncertainty into `## Notes`. If you need human input, mark the step Blocked and specify what is needed.\n"
        "\n"
        f"TEST_CMD: {test_cmd}\n"
        "\n"
        f"{snapshot_text}"
        f"{extra_block}"
        "\n"
        "Now: edit PLAN.md only.\n"
    )


def make_execute_prompt(snapshot_text: str, step: dict[str, str]) -> str:
    return (
        "You are a strict executor. Your job: implement ONLY the single `Next` step.\n"
        "\n"
        "Tooling:\n"
        "- To READ a file, emit:\n"
        "  ```json\n"
        "  {\"filePath\": \"PATH\"}\n"
        "  ```\n"
        "- To WRITE a file, emit:\n"
        "  ```json\n"
        "  {\"filePath\": \"PATH\", \"content\": \"...\"}\n"
        "  ```\n"
        "- To run a command, emit:\n"
        "  ```bash\n"
        "  bash\n"
        "  {\"command\":\"...\",\"description\":\"...\"}\n"
        "  ```\n"
        "Prefer file WRITE tool calls for any edits/creates (do NOT use bash redirections like `>` to write files).\n"
        "The runner executes tool calls and replies with a ```tool_result``` block.\n"
        "\n"
        "Hard constraints:\n"
        "1) Do ONLY this one thing. No refactors, no extra features.\n"
        "2) Keep changes as small as possible.\n"
        "3) Do NOT modify PLAN.md.\n"
        "4) Do NOT modify the pipeline YAML (human-owned contract; runner will revert edits).\n"
        "5) You MAY create new files if needed; this run is unattended and file-create/add-to-chat prompts are auto-approved.\n"
        "\n"
        f"NEXT_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"{snapshot_text}\n"
    )


def make_fix_or_replan_prompt(
    step: dict[str, str],
    verify: VerificationResult,
    *,
    tests_cmds: list[str],
    artifacts_dir: Path,
) -> str:
    metrics_errors = verify.metrics_errors or []
    metrics_block = ""
    if verify.metrics_path or metrics_errors:
        metrics_block = (
            "[METRICS]\n"
            f"metrics_path: {verify.metrics_path}\n"
            f"errors: {metrics_errors}\n"
            f"metrics_preview: {tail(json.dumps(verify.metrics or {}, ensure_ascii=False), 2000)}\n"
            "\n"
        )
    return (
        "Verification failed. You must choose exactly ONE:\n"
        "A) Fix code/scripts/manifests until verification passes (preferred).\n"
        "B) If it truly cannot be closed without missing info: ONLY edit PLAN.md to split the step or mark it Blocked.\n"
        "\n"
        f"FAILED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"FAILED_STAGE: {verify.failed_stage}\n"
        f"TEST_CMDS: {' && '.join(tests_cmds)}\n"
        f"ARTIFACTS_DIR: {artifacts_dir}\n"
        "\n"
        "You MAY create new files if needed; this run is unattended and file-create/add-to-chat prompts are auto-approved.\n"
        "\n"
        "If this is an environment/tooling/auth issue, you may write `.aider_fsm/actions.yml` for the runner to execute.\n"
        "actions.yml format (YAML):\n"
        "version: 1\n"
        "actions:\n"
        "- id: fix-001\n"
        "  kind: run_cmd\n"
        "  cmd: <shell command>\n"
        "  timeout_seconds: 300\n"
        "  retries: 0\n"
        "  risk_level: low|medium|high\n"
        "  rationale: <why>\n"
        "Notes:\n"
        "- In strict unattended mode, avoid interactive login commands (e.g. `docker login` without non-interactive flags).\n"
        "- Runner records artifacts and then deletes actions.yml.\n"
        "\n"
        f"{fmt_stage_tail('AUTH', verify.auth)}"
        f"{fmt_stage_tail('TESTS', verify.tests)}"
        f"{fmt_stage_tail('DEPLOY_SETUP', verify.deploy_setup)}"
        f"{fmt_stage_tail('DEPLOY_HEALTH', verify.deploy_health)}"
        f"{fmt_stage_tail('BENCHMARK', verify.benchmark)}"
        f"{metrics_block}"
    )


def make_mark_done_prompt(step: dict[str, str]) -> str:
    return (
        "This step passed verification. ONLY edit PLAN.md:\n"
        f"1) Move `- [ ] (STEP_ID={step['id']}) ...` from Next to Done, and change it to `- [x]`.\n"
        "2) Pick ONE smallest atomic unchecked item from Backlog into Next (keep Next to exactly one item).\n"
        "3) If Backlog is empty, leave Next empty (keep the heading, no items).\n"
    )


def make_block_step_prompt(step: dict[str, str], last_failure: str) -> str:
    return (
        "Fix attempts exceeded the limit. ONLY edit PLAN.md:\n"
        "1) Remove the step from Next; in Notes, explain why it's Blocked and what human input is needed.\n"
        "2) Pick one item from Backlog into Next (or leave Next empty).\n"
        "\n"
        f"BLOCKED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        "[LAST_FAILURE]\n"
        f"{tail(last_failure, STDIO_TAIL_CHARS)}\n"
    )
