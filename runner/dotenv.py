from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE pairs into os.environ.

    - Lines starting with `#` or blank lines are ignored.
    - Supports optional `export KEY=VALUE`.
    - If override=False, existing env vars are preserved.
    - Returns the list of keys written (values are intentionally not returned).
    """
    if path is None:
        return []

    raw = str(path).strip()
    if not raw:
        return []

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if not p.exists() or not p.is_file():
        return []

    written: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].lstrip()
        if "=" not in s:
            continue

        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
            written.append(key)

    return written

