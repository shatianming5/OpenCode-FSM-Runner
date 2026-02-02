# EvalPlus autonomy (smoke run)

Goal: run a **small**, **safe** EvalPlus benchmark run against an OpenAI-compatible endpoint and produce a machine-checkable summary file.

## Hard constraints

- Do **not** execute untrusted generated code on the host. Prefer EvalPlus **Docker safe execution**.
- Keep cost/time bounded: run **Humaneval** only and limit to a **very small subset** (≤ 3 problems, 1 sample/problem, greedy decoding).
- Use env vars (already provided by the runner):
  - `OPENAI_API_KEY` (required)
  - `OPENAI_API_BASE` (may be missing `/v1`; add it if needed)
  - `OPENAI_MODEL` (preferred model name; fallback to `gpt-4o-mini`)
- Do not use `sudo`.

## Required artifact (for acceptance)

Write a non-empty JSON file at:

`./.aider_fsm/evalplus_smoke_summary.json`

It must include at least:

- `dataset` (e.g. `"humaneval"`)
- `model`
- `base_url`
- `n_problems`
- `n_samples_per_problem`
- `commands` (array of shell commands executed)
- `outputs` (paths to any generated files)
- `ok` (boolean; must be `true` only after a real bounded run)

## Suggested approach (recommended)

Important: the runner verifies ONLY `TEST_CMD` each iteration. In this scenario, `TEST_CMD` is expected to run the smoke script and then validate the produced summary. Do **not** satisfy it with placeholders; make it reflect a real bounded run (`ok: true` only after running).

1) Create a script `./.aider_fsm/evalplus_smoke_run.sh` that:
   - uses `set -euo pipefail`
   - uses/creates a working dir at runtime (eg `/tmp/evalplus_smoke_*` or a subdir under `.aider_fsm/` created via shell)
   - computes `BASE_URL`:
     - if `$OPENAI_API_BASE` ends with `/v1`, use it
     - else use `$OPENAI_API_BASE/v1`
   - pulls and runs `ganler/evalplus:latest` to:
     - inspect help (`evalplus.evaluate --help` / `evalplus.codegen --help`) if you need flags for limiting problems
     - generate samples (if required) and evaluate them
2) Run the script once (bounded humaneval only; ≤3 problems; 1 sample; greedy).
3) Write `.aider_fsm/evalplus_smoke_summary.json` with the required fields and point to the created result files.

If you cannot find a CLI flag to limit to ≤3 problems, **do not** run the full benchmark; instead, mark the plan step Blocked and explain exactly what is missing.
