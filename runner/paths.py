from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def relpath_or_none(path: Path, base: Path) -> str | None:
    if not is_relative_to(path, base):
        return None
    return str(path.relative_to(base))


def resolve_config_path(repo: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def resolve_workdir(repo: Path, workdir: str | None) -> Path:
    if not workdir or not str(workdir).strip():
        return repo
    p = Path(workdir).expanduser()
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    if not is_relative_to(p, repo):
        raise ValueError(f"workdir must be within repo: {p} (repo={repo})")
    return p

