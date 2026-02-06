from __future__ import annotations

import os
from pathlib import Path

from .opencode_client import OpenCodeClient
from .subprocess_utils import tail


def _read_text_tail(path: Path, *, n: int) -> str:
    try:
        return tail(path.read_text(encoding="utf-8", errors="replace"), n)
    except Exception:
        return ""


def repair_contract(
    *,
    repo: Path,
    model: str,
    opencode_url: str,
    unattended: str,
    artifacts_dir: Path,
    failed_stage: str,
    deploy_artifacts_dir: Path,
    rollout_eval_artifacts_dir: Path,
    command_hints: list[str] | None = None,
    extra_context: str = "",
    timeout_seconds: int = 300,
) -> None:
    """Ask OpenCode to repair the repo-local contract under `.aider_fsm/` (best-effort).

    Hard constraints are enforced in the prompt:
    - may ONLY write `.aider_fsm/**`
    - may NOT modify `pipeline.yml`
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    oc_username = None
    oc_password = None
    if str(opencode_url or "").strip():
        oc_username = str(os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip() or "opencode"
        oc_password = str(os.environ.get("OPENCODE_SERVER_PASSWORD") or "").strip() or None

    agent = OpenCodeClient(
        repo=repo,
        plan_rel="PLAN.md",
        pipeline_rel="pipeline.yml",
        model=str(model or "").strip(),
        base_url=(str(opencode_url or "").strip() or None),
        timeout_seconds=int(timeout_seconds or 300),
        bash_mode="restricted",
        scaffold_bash_mode="full",
        unattended=str(unattended or "strict"),
        server_log_path=artifacts_dir / "opencode_server.log",
        session_title=f"{repo.name}:repair",
        username=oc_username,
        password=oc_password,
    )
    try:
        deploy_err = _read_text_tail(deploy_artifacts_dir / "deploy_setup_stderr.txt", n=4000)
        rollout_err = _read_text_tail(rollout_eval_artifacts_dir / "rollout_stderr.txt", n=4000)
        eval_err = _read_text_tail(rollout_eval_artifacts_dir / "evaluation_stderr.txt", n=4000)
        # When evaluation uses `runner.generic_evaluation`, the real failure reason is typically
        # recorded in `.aider_fsm/hints_run.json` rather than evaluation_stderr.txt.
        hints_run_preview = _read_text_tail(repo / ".aider_fsm" / "hints_run.json", n=6000)
        hints_used_preview = _read_text_tail(repo / ".aider_fsm" / "hints_used.json", n=3000)
        bootstrap_preview = _read_text_tail(repo / ".aider_fsm" / "bootstrap.yml", n=3000)
        deploy_setup_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "deploy_setup.sh", n=4000)
        rollout_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "rollout.sh", n=4000)
        evaluation_sh = _read_text_tail(repo / ".aider_fsm" / "stages" / "evaluation.sh", n=4000)
        runtime_env_preview = _read_text_tail(repo / ".aider_fsm" / "runtime_env.json", n=2000)
        metrics_preview = _read_text_tail(repo / ".aider_fsm" / "metrics.json", n=2000)
        extra = str(extra_context or "").strip()
        extra_block = f"\n[EXTRA_CONTEXT]\n{extra}\n" if extra else ""
        hints = [str(s).strip() for s in (command_hints or []) if str(s).strip()]
        hints_block = ""
        if hints:
            shown = hints[:20]
            hints_block = (
                "\n"
                "[CANDIDATE_COMMAND_HINTS]\n"
                "These commands were extracted from the repo docs (README/docs). Prefer using them for real execution.\n"
                "If hints exist, evaluation.sh MUST run at least one hint (or a direct adaptation) and derive score from its outputs.\n"
                "Do NOT replace repo-provided evaluation with proxy scoring (micro-benchmarks). If hints cannot run, write ok=false + reason and EXIT NON-ZERO.\n"
                + "".join([f"- {line}\n" for line in shown])
                + ("\n" if len(hints) > len(shown) else "")
            )
        prompt = (
            "You are a contract repair agent.\n"
            "\n"
            "Goal: fix the repo-local contract so the runner can execute deploy -> rollout -> evaluation without errors.\n"
            "\n"
            "Hard constraints:\n"
            "1) You may ONLY write files under `.aider_fsm/`.\n"
            "2) Do NOT modify `pipeline.yml`.\n"
            "3) Keep everything non-interactive (assume unattended strict).\n"
            "4) Do NOT use `sed -i` (not portable).\n"
            "5) If you embed python via heredoc, do NOT rely on sys.argv and NEVER put shell args after the heredoc terminator.\n"
            "6) If you need Python, you MUST use `$AIDER_FSM_PYTHON` (preferred) or `python3`. Do NOT call `python`.\n"
            "\n"
            "Contract requirements:\n"
            "- deploy must write `.aider_fsm/runtime_env.json` (JSON object)\n"
            "- rollout must write `.aider_fsm/rollout.json` (JSON object)\n"
            "- evaluation must write `.aider_fsm/metrics.json` (JSON object, must contain keys `ok` and `score`; require `ok=true` for success)\n"
            "  `metrics.json.ok` MUST be a JSON boolean true/false. Do NOT write 1/0 and do NOT write a string.\n"
            "- If command hints exist (see [CANDIDATE_COMMAND_HINTS]), evaluation MUST run at least one hinted/official command.\n"
            "  Additionally, when `AIDER_FSM_REQUIRE_HINTS=1`, evaluation MUST write `.aider_fsm/hints_used.json` with:\n"
            "    - `ok`: boolean (true only if a hinted/official command succeeded)\n"
            "    - `used_anchors`: list[string] (must include at least one token from `AIDER_FSM_HINT_ANCHORS_JSON`)\n"
            "    - `commands`: list[string] (recommended; attempted commands)\n"
            "    - `reason`: string (required when ok=false)\n"
            "  If no hinted/official command can run, set ok=false with a clear reason and EXIT NON-ZERO.\n"
            "- Preferred benchmark-agnostic evaluation implementation: call the built-in helper in evaluation.sh:\n"
            "    `$AIDER_FSM_PYTHON -m runner.generic_evaluation`\n"
            "  It already executes `AIDER_FSM_HINTS_JSON` safely, writes `hints_used.json` + `metrics.json`, and exits non-zero when required hints fail.\n"
            "  Do NOT hand-roll JSON parsing of hint env vars (easy to get wrong).\n"
            "- If hinted/official commands fail due to missing deps, import errors, or version mismatches, DO NOT fake success.\n"
            "  Instead, add `.aider_fsm/bootstrap.yml` to set up an isolated environment (e.g., venv) and install repo deps deterministically,\n"
            "  then ensure evaluation runs the hinted command within that environment.\n"
            "  IMPORTANT: bootstrap.yml is for **environment preparation only**. Do NOT run evaluation/test/benchmark commands in bootstrap\n"
            "  (e.g. do NOT run `pytest`, `evalplus.evaluate`, `make test`, etc). Those belong in the stage scripts (especially evaluation.sh).\n"
            "  IMPORTANT: if you create a venv, all installs MUST use the venv interpreter (`.aider_fsm/venv/bin/python -m pip ...`),\n"
            "  not the system `python3 -m pip ...` (do not pollute global/user site-packages).\n"
            "  Example bootstrap.yml pattern:\n"
            "    version: 1\n"
            "    cmds:\n"
            "      - python3 -m venv .aider_fsm/venv\n"
            "      - .aider_fsm/venv/bin/python -m pip install -U pip\n"
            "      - .aider_fsm/venv/bin/pip install -e .\n"
            "    env:\n"
            "      PATH: \".aider_fsm/venv/bin:$PATH\"\n"
            "  If you see `ValueError: Incompatible Language version ...` coming from tree-sitter,\n"
            "  it is usually a dependency mismatch between `tree_sitter` and `tree-sitter-python`.\n"
            "  Fix it in bootstrap.yml by pinning compatible versions in the venv before running the hinted command.\n"
            "  Known-good pair (example): `tree_sitter==0.22.0` and `tree-sitter-python==0.21.0`.\n"
            "- NEVER set `ok=true` unless the evaluation actually ran and succeeded.\n"
            "- NEVER hardcode a non-zero score. Derive the score from real execution outputs.\n"
            "- NOTE: the runner audits `.aider_fsm/stages/evaluation.sh` and will reject hardcoded non-zero scores and proxy/no-op evaluations.\n"
            "- rollout MUST also write a samples JSONL file under `$AIDER_FSM_ARTIFACTS_DIR` and include its path in `rollout.json.paths.samples_jsonl`.\n"
            "  Each JSONL line must be an object with keys: `prompt` (string), `completion` (string), `reward` (number).\n"
            "- IMPORTANT: if [EXTRA_CONTEXT] mentions `hf_qa_samples_too_few` OR `hf_qa_prompts_not_anchored`, this target is an HF QA snapshot.\n"
            "  Rollout MUST:\n"
            "  - generate at least N valid samples if N is specified (and have prompt diversity)\n"
            "  - anchor prompts to the HF test parquet questions (i.e., include the real question text; do NOT synthesize unrelated tasks)\n"
            "  In this case, do NOT hand-roll placeholder sample generators.\n"
            "  Preferred benchmark-agnostic fix: call the built-in helper (recommended): `$AIDER_FSM_PYTHON -m runner.generic_rollout`.\n"
            "  It already handles HF QA snapshots and respects `AIDER_EVAL_MODE` / `AIDER_EVAL_LIMIT`.\n"
            "- If a hinted/official command fails due to permission errors writing outputs (e.g., cannot create a results directory),\n"
            "  redirect its outputs to a writable location under `$AIDER_FSM_ARTIFACTS_DIR` (or `.aider_fsm/`).\n"
            "  Prefer passing an explicit output flag (common names: `--output`, `--output_file`, `--out`, `--out_dir`, `--root`) rather than relying on tool defaults.\n"
            "- scripts MUST support `AIDER_RUNTIME_ENV_PATH` and local/remote inference inputs (`AIDER_TRAINED_MODEL_DIR` OR `AIDER_LLM_KIND=remote` + `AIDER_LLM_MODEL`, with endpoint/auth from env like `OPENAI_API_KEY`).\n"
            "- If you use an OpenAI-compatible client library that requires an API key, DO NOT embed secrets in files. Instead, read it from env (e.g. OPENAI_API_KEY) and fail clearly if missing.\n"
            "- deploy_teardown should stop any started services/containers (best-effort)\n"
            "\n"
            f"FAILED_STAGE: {failed_stage}\n"
            f"REPO_ROOT: {repo}\n"
            f"DEPLOY_ARTIFACTS_DIR: {deploy_artifacts_dir}\n"
            f"ROLLOUT_EVAL_ARTIFACTS_DIR: {rollout_eval_artifacts_dir}\n"
            f"{hints_block}"
            "\n"
            "[DEPLOY_SETUP_STDERR_TAIL]\n"
            f"{deploy_err}\n"
            "\n"
            "[ROLLOUT_STDERR_TAIL]\n"
            f"{rollout_err}\n"
            "\n"
            "[EVALUATION_STDERR_TAIL]\n"
            f"{eval_err}\n"
            "\n"
            "[HINTS_RUN_JSON_TAIL]\n"
            f"{hints_run_preview}\n"
            "\n"
            "[HINTS_USED_JSON_TAIL]\n"
            f"{hints_used_preview}\n"
            "\n"
            "[BOOTSTRAP_YML_TAIL]\n"
            f"{bootstrap_preview}\n"
            "\n"
            "[DEPLOY_SETUP_SH_TAIL]\n"
            f"{deploy_setup_sh}\n"
            "\n"
            "[ROLLOUT_SH_TAIL]\n"
            f"{rollout_sh}\n"
            "\n"
            "[EVALUATION_SH_TAIL]\n"
            f"{evaluation_sh}\n"
            "\n"
            "[RUNTIME_ENV_JSON_TAIL]\n"
            f"{runtime_env_preview}\n"
            "\n"
            "[METRICS_JSON_TAIL]\n"
            f"{metrics_preview}\n"
            f"{extra_block}"
            "\n"
            "Now: inspect `.aider_fsm/stages/*.sh` and fix the scripts so the contract passes.\n"
        )
        try:
            res = agent.run(prompt, fsm_state="S_REPAIR", iter_idx=0, purpose="repair_contract")
            (artifacts_dir / "repair_agent_result.txt").write_text(tail(res.assistant_text or "", 20000) + "\n", encoding="utf-8")
        except Exception as e:
            # A timeout/error might still have produced partial file writes.
            (artifacts_dir / "repair_agent_error.txt").write_text(tail(str(e), 4000) + "\n", encoding="utf-8")
            return
    finally:
        try:
            agent.close()
        except Exception:
            pass
