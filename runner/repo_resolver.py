from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def is_probably_repo_url(repo: str) -> bool:
    s = str(repo or "").strip()
    if not s:
        return False
    if s.startswith(("http://", "https://", "ssh://", "git@")):
        return True
    if s.endswith(".git") and ("/" in s or ":" in s):
        return True
    if _OWNER_REPO_RE.match(s):
        return True
    return False


def normalize_repo_url(repo: str) -> str:
    """Normalize shorthand forms into a git-cloneable URL."""
    s = str(repo or "").strip()
    if _OWNER_REPO_RE.match(s):
        # Default to GitHub for shorthand `owner/repo`.
        return f"https://github.com/{s}.git"
    return s


def _repo_slug(repo_url: str) -> str:
    s = repo_url.strip().rstrip("/")
    if s.startswith("git@") and ":" in s:
        s = s.split(":", 1)[1]
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.rstrip(".git")
    parts = [p for p in re.split(r"[/:]", s) if p]
    name = parts[-1] if parts else "repo"
    owner = parts[-2] if len(parts) >= 2 else "remote"
    slug = f"{owner}_{name}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug[:80]


@dataclass(frozen=True)
class PreparedRepo:
    repo: Path
    cloned_from: str | None = None


def prepare_repo(repo_arg: str, *, clones_dir: Path | None = None) -> PreparedRepo:
    """Return a usable local repo path.

    - If repo_arg is an existing local path -> use it.
    - If repo_arg looks like a git URL -> clone to clones_dir (or /tmp).
    """
    raw = str(repo_arg or "").strip()
    if not raw:
        raise ValueError("--repo is required")

    p = Path(raw).expanduser()
    if p.exists():
        return PreparedRepo(repo=p.resolve(), cloned_from=None)

    if not is_probably_repo_url(raw):
        raise FileNotFoundError(f"repo path not found: {raw}")

    url = normalize_repo_url(raw)
    base = clones_dir or (Path(tempfile.gettempdir()) / "aider_fsm_targets")
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dest = base / f"{_repo_slug(url)}_{ts}"
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    # Depth=1 keeps it fast; users can re-run with a local clone if needed.
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "git clone failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {proc.returncode}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}\n"
        )

    # Make sure local commits (if any) won't fail due to missing identity.
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "aider-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "aider-fsm@example.com"], env=env, check=False)

    return PreparedRepo(repo=dest.resolve(), cloned_from=url)
