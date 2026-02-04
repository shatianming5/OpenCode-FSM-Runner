from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_ARCHIVE_HOSTS = {"github.com", "www.github.com"}
_HF_HOSTS = {"huggingface.co", "www.huggingface.co"}


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


def _parse_github_owner_repo(url: str) -> tuple[str, str] | None:
    s = str(url or "").strip()
    if not s:
        return None

    # https://github.com/<owner>/<repo>(.git)?
    m = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    # git@github.com:<owner>/<repo>(.git)?
    m = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    # ssh://git@github.com/<owner>/<repo>(.git)?
    m = re.match(r"^ssh://git@([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    return None


def _download_file(
    url: str,
    *,
    out_path: Path,
    timeout_seconds: int = 60,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> tuple[bool, str]:
    try:
        h = {
            "User-Agent": "opencode-fsm/1.0",
            "Accept": "application/octet-stream",
        }
        if headers:
            h.update({str(k): str(v) for k, v in headers.items() if str(k).strip()})
        req = Request(
            url,
            headers=h,
            method="GET",
        )
        with urlopen(req, timeout=timeout_seconds) as resp:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                total = 0
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > int(max_bytes):
                        raise OSError(f"max_bytes_exceeded: {max_bytes}")
                    f.write(chunk)
        return True, ""
    except HTTPError as e:
        return False, f"HTTPError {getattr(e, 'code', '')}: {str(e)}"
    except URLError as e:
        return False, f"URLError: {str(e)}"
    except OSError as e:
        return False, f"OSError: {str(e)}"


def _extract_github_zip(zip_path: Path, *, extract_dir: Path, repo_name: str) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"invalid_zip: {e}") from e

    dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]

    prefix = f"{repo_name}-"
    candidates = [d for d in dirs if d.name.startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(f"unexpected_zip_layout: dirs={[d.name for d in dirs]}")


def _parse_hf_dataset(url: str) -> tuple[str, str] | None:
    u = str(url or "").strip()
    if not u.startswith(("http://", "https://")):
        return None
    parsed = urlparse(u)
    host = (parsed.hostname or "").strip().lower()
    if host not in _HF_HOSTS:
        return None
    parts = [p for p in (parsed.path or "").split("/") if p]
    if len(parts) < 3:
        return None
    if parts[0] != "datasets":
        return None
    namespace, name = parts[1].strip(), parts[2].strip()
    if not namespace or not name:
        return None
    return namespace, name


def _hf_dataset_api_info(
    namespace: str,
    name: str,
    *,
    token: str | None,
    timeout_seconds: int = 20,
) -> dict[str, object]:
    url = f"https://huggingface.co/api/datasets/{namespace}/{name}"
    headers: dict[str, str] = {"Accept": "application/json", "User-Agent": "opencode-fsm/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise RuntimeError("hf_api_invalid_json")
    return data


def _download_hf_dataset_snapshot(
    *,
    namespace: str,
    name: str,
    dest: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Download a Hugging Face dataset snapshot via the HF REST API (no git required)."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    token = (
        str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "")
        .strip()
        or None
    )
    try:
        info = _hf_dataset_api_info(namespace, name, token=token)
    except HTTPError as e:
        return False, f"hf_api_http_error {getattr(e, 'code', '')}: {e}"
    except URLError as e:
        return False, f"hf_api_url_error: {e}"
    except Exception as e:
        return False, f"hf_api_error: {e}"

    gated = bool(info.get("gated") or False)
    private = bool(info.get("private") or False)
    if (gated or private) and not token:
        return False, "hf_dataset_requires_token (set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)"

    revision = str(info.get("sha") or "main").strip() or "main"
    siblings = info.get("siblings") or []
    if not isinstance(siblings, list):
        siblings = []

    max_total_bytes = int(os.environ.get("AIDER_FSM_HF_MAX_TOTAL_BYTES") or 512 * 1024 * 1024)
    max_file_bytes = int(os.environ.get("AIDER_FSM_HF_MAX_FILE_BYTES") or 256 * 1024 * 1024)
    total_bytes = 0
    downloaded: list[dict[str, object]] = []
    errors: list[str] = []

    extra_headers: dict[str, str] = {}
    if token:
        extra_headers["Authorization"] = f"Bearer {token}"

    for s in siblings:
        if not isinstance(s, dict):
            continue
        rfilename = str(s.get("rfilename") or "").strip()
        if not rfilename:
            continue
        # Skip very unlikely problematic paths.
        if rfilename.startswith(("/", "\\")) or ".." in rfilename.split("/"):
            errors.append(f"skip_unsafe_path: {rfilename}")
            continue

        if total_bytes >= max_total_bytes:
            errors.append(f"max_total_bytes_exceeded: {max_total_bytes}")
            break

        encoded = quote(rfilename, safe="/")
        url = f"https://huggingface.co/datasets/{namespace}/{name}/resolve/{revision}/{encoded}"
        out_path = dest / rfilename
        ok, err = _download_file(
            url,
            out_path=out_path,
            timeout_seconds=120,
            headers=extra_headers,
            max_bytes=min(max_file_bytes, max_total_bytes - total_bytes),
        )
        if not ok:
            errors.append(f"{rfilename}: {err}")
            continue
        try:
            size = out_path.stat().st_size
        except OSError:
            size = None
        if isinstance(size, int):
            total_bytes += int(size)
        downloaded.append({"path": rfilename, "bytes": size})

    manifest = {
        "source": "huggingface_dataset",
        "dataset_id": f"{namespace}/{name}",
        "revision": revision,
        "gated": gated,
        "private": private,
        "downloaded_files": downloaded,
        "total_bytes": total_bytes,
        "errors": errors,
    }
    (dest / ".aider_fsm").mkdir(exist_ok=True)
    (dest / "data").mkdir(exist_ok=True)
    (dest / "data" / "hf_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Initialize git for revert guards (best effort).
    subprocess.run(["git", "-C", str(dest), "init"], env=env, check=False, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "add", "-A"], env=env, check=False, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(dest), "commit", "-m", "init hf snapshot", "--no-gpg-sign"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if errors and not downloaded:
        return False, "hf_download_failed: " + "; ".join(errors[-5:])
    return True, ""


def _archive_clone_github(
    *,
    owner: str,
    repo: str,
    dest: Path,
    env: dict[str, str],
    timeout_seconds: int = 60,
) -> tuple[bool, str]:
    """Best-effort fallback clone via GitHub archive zip.

    Returns (ok, detail). On success, the repo is extracted into dest.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    zip_path = dest.parent / f"{dest.name}.zip"
    extract_dir = dest.parent / f"{dest.name}_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        for branch in ("main", "master"):
            url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            ok, err = _download_file(url, out_path=zip_path, timeout_seconds=timeout_seconds)
            if not ok:
                errors.append(f"{url}: {err}")
                continue

            try:
                root = _extract_github_zip(zip_path, extract_dir=extract_dir, repo_name=repo)
            except Exception as e:
                errors.append(f"{url}: extract_failed: {e}")
                continue

            try:
                shutil.move(str(root), str(dest))
            except Exception as e:
                errors.append(f"{url}: move_failed: {e}")
                continue

            # Initialize git for revert guards (best effort).
            subprocess.run(["git", "-C", str(dest), "init"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "add", "-A"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(
                ["git", "-C", str(dest), "commit", "-m", "init", "--no-gpg-sign"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            return True, f"archive_branch={branch}"
    finally:
        try:
            zip_path.unlink()
        except Exception:
            pass
        shutil.rmtree(extract_dir, ignore_errors=True)

    return False, "; ".join(errors[-5:])  # tail


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

    hf = _parse_hf_dataset(raw)
    if hf:
        namespace, name = hf
        base = clones_dir or (Path(tempfile.gettempdir()) / "aider_fsm_targets")
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dest = base / f"hf_{namespace}_{name}_{ts}"
        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        ok, detail = _download_hf_dataset_snapshot(namespace=namespace, name=name, dest=dest, env=env)
        if not ok:
            raise RuntimeError(f"hf dataset download failed: {detail}")
        return PreparedRepo(repo=dest.resolve(), cloned_from=raw)

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
        # Clean up partial clones to avoid confusing fallbacks.
        shutil.rmtree(dest, ignore_errors=True)

        owner_repo = _parse_github_owner_repo(url)
        if owner_repo:
            owner, name = owner_repo
            ok, detail = _archive_clone_github(owner=owner, repo=name, dest=dest, env=env)
            if ok:
                return PreparedRepo(repo=dest.resolve(), cloned_from=url)

            raise RuntimeError(
                "git clone failed; GitHub archive fallback also failed\n"
                f"git_cmd: {' '.join(cmd)}\n"
                f"git_rc: {proc.returncode}\n"
                f"git_stdout: {proc.stdout[-2000:]}\n"
                f"git_stderr: {proc.stderr[-2000:]}\n"
                f"archive_detail: {detail}\n"
            )

        raise RuntimeError(
            "git clone failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {proc.returncode}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}\n"
        )

    # Make sure local commits (if any) won't fail due to missing identity.
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)

    return PreparedRepo(repo=dest.resolve(), cloned_from=url)
