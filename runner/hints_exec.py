from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import cmd_allowed


def _parse_json_str_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s:
            out.append(s)
    return out


def _read_hints_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _find_latest_scaffold_hints_file(repo: Path) -> Path | None:
    repo = Path(repo).resolve()
    root = (repo / ".aider_fsm" / "artifacts").resolve()
    if not root.exists():
        return None
    candidates = list(root.glob("*/scaffold/scaffold_command_hints.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]


_FLAG_VALUE_RE = re.compile(r"(?P<flag>--[A-Za-z0-9_.-]+)\s+(?P<val>(?:\"[^\"]*\"|'[^']*'|\S+))")


def _replace_flag_value(cmd: str, *, flag: str, new_value: str) -> str:
    flag = flag.strip()
    if not flag:
        return cmd
    if not new_value:
        return cmd

    def repl(m: re.Match[str]) -> str:
        if m.group("flag") != flag:
            return m.group(0)
        v = new_value
        # Quote the value if it has whitespace.
        if any(ch.isspace() for ch in v):
            v = json.dumps(v)
        return f"{flag} {v}"

    return _FLAG_VALUE_RE.sub(repl, cmd)


_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_ANGLE_GROUP_RE = re.compile(r"<[^>]+>")
_GHA_EXPR_RE = re.compile(r"\$\{\{\s*([^}]+)\s*\}\}")
_PIPE_TO_BASH_RE = re.compile(r"(?i)\b(?:curl|wget)\b[^\n]*\|[^\n]*\bbash\b")


def normalize_hint_command(cmd: str, *, env: dict[str, str]) -> tuple[str, str | None]:
    """Normalize a doc-derived command hint into something runnable.

    Returns (sanitized_cmd, skip_reason). If skip_reason is not None, callers should skip it.
    """
    s = str(cmd or "").strip()
    if not s:
        return "", "empty"

    # Replace bracketed option groups like [a|b|c] -> a (first option).
    def bracket_repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        if "|" in inner:
            return inner.split("|", 1)[0].strip()
        # Keep unresolved placeholders as-is; we'll skip later if still bracketed.
        return m.group(0)

    s2 = _BRACKET_GROUP_RE.sub(bracket_repl, s)
    # Remove angle placeholders like <TENSOR_PARALLEL_SIZE>
    s2 = _ANGLE_GROUP_RE.sub("", s2)

    # Replace common GitHub Actions expressions (e.g., matrix python versions) with local defaults.
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    def gha_repl(m: re.Match[str]) -> str:
        inner = str(m.group(1) or "").strip().lower()
        if "matrix.python-version" in inner or "matrix.python_version" in inner or "python-version" in inner:
            return py_ver
        return ""

    s2 = _GHA_EXPR_RE.sub(gha_repl, s2)

    # Replace model/base-url flags with env-provided values when available.
    model = (env.get("AIDER_LLM_MODEL") or env.get("OPENAI_MODEL") or "").strip()
    base_url = (env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL") or "").strip()
    if model:
        s2 = _replace_flag_value(s2, flag="--model", new_value=model)
    if base_url:
        s2 = _replace_flag_value(s2, flag="--base-url", new_value=base_url)

    # Normalize whitespace but preserve newlines for multi-command scripts.
    s2 = re.sub(r"[ \t]+", " ", s2)
    lines: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    s2 = "\n".join(lines).strip()
    if not s2:
        return "", "empty_after_sanitize"

    # If the command still contains bracket placeholders, it's likely not directly runnable.
    if "[" in s2 and "]" in s2:
        return s2, "unresolved_brackets"
    if "<" in s2 and ">" in s2:
        return s2, "unresolved_angle_placeholders"

    # Block common unsafe installer patterns from docs.
    if _PIPE_TO_BASH_RE.search(s2):
        return s2, "blocked_pipe_to_bash"

    # Apply the runner's safe-mode denylist as a generic guardrail.
    allowed, reason = cmd_allowed(s2, pipeline=None)
    if not allowed:
        return s2, reason or "blocked_by_policy"
    return s2, None


@dataclass(frozen=True)
class HintAttempt:
    raw: str
    sanitized: str
    rc: int
    seconds: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    skip_reason: str | None = None


def _tail(text: str, n: int) -> str:
    t = str(text or "")
    if len(t) <= n:
        return t
    return t[-n:]


def _is_truthy(value: str | None) -> bool:
    v = str(value or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def run_hints(
    *,
    repo: Path,
    max_attempts: int = 3,
    timeout_seconds: int = 600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    env2 = dict(env or os.environ)

    raw_hints = _parse_json_str_list(env2.get("AIDER_FSM_HINTS_JSON"))
    if not raw_hints:
        hints_file = _find_latest_scaffold_hints_file(repo)
        if hints_file is not None:
            raw_hints = _read_hints_file(hints_file)

    anchors = _parse_json_str_list(env2.get("AIDER_FSM_HINT_ANCHORS_JSON"))
    used_anchors = [anchors[0]] if anchors else []

    kind = (env2.get("AIDER_LLM_KIND") or "").strip().lower()
    prefer_offline = _is_truthy(env2.get("AIDER_FSM_PREFER_OFFLINE_HINTS"))

    def _priority(raw: str) -> int:
        s = str(raw or "").lower()
        p = 0
        # Prefer commands that actually *run* evaluations/tests over setup/build steps.
        if "pytest" in s:
            p += 50
        if "evalplus.evaluate" in s or " evalplus " in s:
            p += 30
        if "evaluate" in s or "evaluation" in s:
            p += 20
        if "benchmark" in s:
            p += 10
        has_eval = ("pytest" in s) or ("evaluate" in s) or ("evaluation" in s) or ("benchmark" in s)
        if "docker build" in s and not has_eval:
            p -= 5
        if ("pip install" in s or "poetry install" in s or "conda install" in s) and not has_eval:
            p -= 10
        if prefer_offline:
            if " --samples" in f" {s} ":
                p += 15
            if " --backend openai" in s or " backend openai" in s:
                p -= 30
        else:
            if " --backend openai" in s or " backend openai" in s:
                p += 20
            if "openai" in s and kind == "remote":
                p += 5
        if s.startswith("docker "):
            p += 3
        if "vllm" in s and kind == "remote":
            p -= 5
        if "anthropic" in s or "bedrock" in s or "google" in s:
            p -= 5
        if "ollama" in s:
            p -= 3
        return p

    raw_hints.sort(key=_priority, reverse=True)

    attempts: list[HintAttempt] = []
    chosen: str | None = None
    ok = False
    score = 0.0
    reason = ""

    def _parse_pytest_counts(text: str) -> tuple[int, int, int] | None:
        """Parse (passed, failed, errors) from pytest output (best-effort)."""
        t = str(text or "")
        # Strip ANSI color codes so regexes can match reliably.
        t = re.sub(r"\x1b\[[0-9;]*m", "", t)
        passed_ms = list(re.finditer(r"(?i)\b(\d+)\s+passed\b", t))
        failed_ms = list(re.finditer(r"(?i)\b(\d+)\s+failed\b", t))
        errors_ms = list(re.finditer(r"(?i)\b(\d+)\s+error(?:s)?\b", t))
        try:
            passed = int(passed_ms[-1].group(1)) if passed_ms else 0
        except Exception:
            passed = 0
        try:
            failed = int(failed_ms[-1].group(1)) if failed_ms else 0
        except Exception:
            failed = 0
        try:
            errors = int(errors_ms[-1].group(1)) if errors_ms else 0
        except Exception:
            errors = 0
        total = passed + failed + errors
        if total <= 0:
            return None
        return passed, failed, errors

    def _maybe_generate_evalplus_samples(cmd: str) -> None:
        """Best-effort helper: create a minimal `samples.jsonl` for evalplus `--samples` runs.

        This makes the doc-hinted command `evalplus.evaluate --dataset X --samples samples.jsonl`
        runnable in offline settings (no remote inference), without benchmark-specific hardcoding.
        """
        low = str(cmd or "").lower()
        if "evalplus.evaluate" not in low or " --samples" not in f" {low} ":
            return
        try:
            parts = shlex.split(str(cmd), posix=True)
        except Exception:
            return
        if "--samples" not in parts:
            return
        try:
            samples_token = parts[parts.index("--samples") + 1]
        except Exception:
            return
        samples_path = Path(str(samples_token or "").strip())
        if not samples_path:
            return
        if not samples_path.is_absolute():
            samples_path = (repo / samples_path).resolve()
        if samples_path.exists():
            return

        dataset = ""
        if "--dataset" in parts:
            try:
                dataset = str(parts[parts.index("--dataset") + 1]).strip().lower()
            except Exception:
                dataset = ""
        dataset = dataset or "humaneval"

        try:
            if dataset == "humaneval":
                from evalplus.data import get_human_eval_plus  # type: ignore

                tasks = get_human_eval_plus()
            elif dataset == "mbpp":
                from evalplus.data import get_mbpp_plus  # type: ignore

                tasks = get_mbpp_plus()
            else:
                return
        except Exception:
            return

        if not isinstance(tasks, dict) or not tasks:
            return

        try:
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            with samples_path.open("w", encoding="utf-8") as f:
                for task_id in tasks.keys():
                    tid = str(task_id or "").strip()
                    if not tid:
                        continue
                    f.write(json.dumps({"task_id": tid, "completion": "    pass"}, ensure_ascii=False) + "\n")
        except Exception:
            return

    for raw in raw_hints:
        if len(attempts) >= int(max(0, max_attempts)):
            break

        sanitized, skip_reason = normalize_hint_command(raw, env=env2)
        if skip_reason is not None:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=skip_reason,
                )
            )
            continue

        # Best-effort: if the hinted command expects a local samples file, create it.
        _maybe_generate_evalplus_samples(sanitized)

        t0 = time.monotonic()
        timed_out = False
        try:
            res = subprocess.run(
                ["bash", "-lc", sanitized],
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                cwd=str(repo),
                env=env2,
            )
            rc = int(res.returncode)
            out = res.stdout or ""
            err = res.stderr or ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = 124
            out = getattr(e, "stdout", "") or ""
            err = getattr(e, "stderr", "") or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")

        dt = time.monotonic() - t0
        attempts.append(
            HintAttempt(
                raw=raw,
                sanitized=sanitized,
                rc=rc,
                seconds=float(dt),
                timed_out=timed_out,
                stdout_tail=_tail(out, 4000),
                stderr_tail=_tail(err, 4000),
                skip_reason=None,
            )
        )

        if rc == 0:
            chosen = sanitized
            ok = True
            score = 1.0
            reason = ""
            break

        # Some evaluation commands (notably `pytest`) intentionally return non-zero
        # when checks fail, but still produce useful numeric metrics. Treat these as
        # a "successful run" and derive score from the outputs.
        low_cmd = sanitized.lower()
        if "pytest" in low_cmd:
            # Parse from the tail: pytest summaries are printed at the end, and
            # this avoids scanning extremely large outputs.
            tail_text = _tail(out, 20000) + "\n" + _tail(err, 20000)
            counts = _parse_pytest_counts(tail_text)
            if counts is not None:
                passed, failed, errors = counts
                total = max(1, passed + failed + errors)
                chosen = sanitized
                ok = True
                score = float(passed) / float(total)
                reason = f"pytest_nonzero_exit: passed={passed} failed={failed} errors={errors}"
                break

    if not ok:
        if not raw_hints:
            reason = "no_hints"
        elif any(a.skip_reason == "unresolved_brackets" for a in attempts):
            reason = "all_hints_unresolved_or_failed"
        else:
            reason = "all_hints_failed"

    return {
        "ok": bool(ok),
        "score": float(score) if ok else 0.0,
        "chosen_command": chosen,
        "used_anchors": used_anchors,
        "attempts": [
            {
                "raw": a.raw,
                "sanitized": a.sanitized,
                "rc": a.rc,
                "seconds": a.seconds,
                "timed_out": a.timed_out,
                "stdout_tail": a.stdout_tail,
                "stderr_tail": a.stderr_tail,
                "skip_reason": a.skip_reason,
            }
            for a in attempts
        ],
        "reason": reason,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run doc-derived hint commands with best-effort sanitization.")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    args = ap.parse_args()

    repo_root = Path(os.environ.get("AIDER_FSM_REPO_ROOT") or ".").resolve()
    res = run_hints(repo=repo_root, max_attempts=int(args.max_attempts), timeout_seconds=int(args.timeout_seconds))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
