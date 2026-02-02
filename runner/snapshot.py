from __future__ import annotations

from pathlib import Path
from typing import Any

from .subprocess_utils import run_cmd, tail


PLAN_TAIL_CHARS = 20000


def build_snapshot(repo: Path, plan_abs: Path, pipeline_abs: Path | None = None) -> tuple[dict[str, Any], str]:
    plan_text = ""
    if plan_abs.exists():
        plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
    plan_text = tail(plan_text, PLAN_TAIL_CHARS)

    pipeline_text = ""
    if pipeline_abs and pipeline_abs.exists():
        pipeline_text = pipeline_abs.read_text(encoding="utf-8", errors="replace")
    pipeline_text = tail(pipeline_text, PLAN_TAIL_CHARS)

    actions_text = ""
    actions_path = repo / ".aider_fsm" / "actions.yml"
    if actions_path.exists():
        actions_text = actions_path.read_text(encoding="utf-8", errors="replace")
    actions_text = tail(actions_text, PLAN_TAIL_CHARS)

    state_text = ""
    state_path = repo / ".aider_fsm" / "state.json"
    if state_path.exists():
        state_text = state_path.read_text(encoding="utf-8", errors="replace")
    state_text = tail(state_text, PLAN_TAIL_CHARS)

    rc1, gs, ge = run_cmd("git status --porcelain", repo)
    rc2, diff, de = run_cmd("git diff --stat", repo)
    rc3, names, ne = run_cmd("git diff --name-only", repo)

    env_probe = {
        "pyproject.toml": (repo / "pyproject.toml").exists(),
        "package.json": (repo / "package.json").exists(),
        "pytest.ini": (repo / "pytest.ini").exists(),
        "requirements.txt": (repo / "requirements.txt").exists(),
        "setup.cfg": (repo / "setup.cfg").exists(),
    }

    snapshot = {
        "repo": str(repo),
        "plan_path": str(plan_abs),
        "pipeline_path": str(pipeline_abs) if pipeline_abs else "",
        "env_probe": env_probe,
        "git": {
            "status_rc": rc1,
            "status": gs,
            "status_err": ge,
            "diff_stat_rc": rc2,
            "diff_stat": diff,
            "diff_stat_err": de,
            "diff_names_rc": rc3,
            "diff_names": names,
            "diff_names_err": ne,
        },
        "plan_md": plan_text,
        "pipeline_yml": pipeline_text,
        "actions_yml": actions_text,
        "state_json": state_text,
    }

    env_lines = "\n".join([f"- {k}: {'yes' if v else 'no'}" for k, v in env_probe.items()])
    snapshot_text = (
        "[SNAPSHOT]\n"
        f"repo: {repo}\n"
        f"plan_path: {plan_abs}\n"
        f"pipeline_path: {pipeline_abs}\n"
        "env_probe:\n"
        f"{env_lines}\n"
        "git_status_porcelain:\n"
        f"{gs}\n"
        "git_diff_stat:\n"
        f"{diff}\n"
        "changed_files:\n"
        f"{names}\n"
        "state_json:\n"
        f"{state_text}\n"
        "pipeline_yml:\n"
        f"{pipeline_text}\n"
        "actions_yml:\n"
        f"{actions_text}\n"
        "plan_md:\n"
        f"{plan_text}\n"
    )
    return snapshot, snapshot_text


def get_git_changed_files(repo: Path) -> list[str] | None:
    rc, out, _err = run_cmd("git diff --name-only", repo)
    if rc != 0:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_checkout(repo: Path, paths: list[str]) -> tuple[int, str, str]:
    if not paths:
        return 0, "", ""
    # Keep shell=True quoting simple; callers control inputs.
    import shlex

    cmd = "git checkout -- " + " ".join(shlex.quote(p) for p in paths)
    return run_cmd(cmd, repo)


def non_plan_changes(changed_files: list[str], plan_rel: str) -> list[str]:
    return [p for p in changed_files if p != plan_rel]

