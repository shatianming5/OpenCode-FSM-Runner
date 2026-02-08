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
_PREFERRED_CLONES_BASE = Path("/data/tiansha/aider_fsm_targets")


def _default_clones_base() -> Path:
    """Prefer /data for large clone/snapshot caches, fallback to system temp."""
    candidates = [
        _PREFERRED_CLONES_BASE,
        Path(tempfile.gettempdir()) / "aider_fsm_targets",
    ]
    for raw in candidates:
        base = raw.expanduser().resolve()
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        probe = base / ".aider_fsm_write_probe"
        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            return base
        except Exception:
            continue
    # Last resort: temp dir path even if writability probe failed above.
    return (Path(tempfile.gettempdir()) / "aider_fsm_targets").expanduser().resolve()


def is_probably_repo_url(repo: str) -> bool:
    """中文说明：
    - 含义：判断字符串是否“看起来像”一个可获取的远程 repo（URL/SSH/owner/repo）。
    - 内容：用于区分本地路径 vs 远程地址；匹配 http(s)/ssh/git@、以 .git 结尾的形式、以及 `owner/repo` 简写。
    - 可简略：可能（启发式；但集中实现便于统一行为）。
    """
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
    # 中文说明：
    # - 含义：把 `owner/repo` 这种简写规范化为可 git clone 的 URL。
    # - 内容：目前默认映射到 GitHub HTTPS：`https://github.com/{owner/repo}.git`；其余输入保持不变。
    # - 可简略：可能（如果不想默认 GitHub，可移除此简写规则）。
    s = str(repo or "").strip()
    if _OWNER_REPO_RE.match(s):
        # Default to GitHub for shorthand `owner/repo`.
        return f"https://github.com/{s}.git"
    return s


def _repo_slug(repo_url: str) -> str:
    """中文说明：
    - 含义：从 repo URL 生成一个适合作为本地目录名的 slug。
    - 内容：提取 owner/repo（尽力），替换非法字符为 `_`，并限制长度；用于 `<clones_base>/<slug>_<ts>`。
    - 可简略：可能（命名策略可调整；但需要保持稳定与避免路径注入）。
    """
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
    """中文说明：
    - 含义：从 GitHub URL/SSH 地址解析出 (owner, repo)。
    - 内容：支持 https://github.com/owner/repo(.git)、git@github.com:owner/repo(.git)、ssh://git@github.com/owner/repo(.git)。
    - 可简略：可能（只为 GitHub ZIP fallback 服务；若去掉 fallback 可删除）。
    """
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
    """中文说明：
    - 含义：下载一个 URL 到本地文件（带超时/可选 header/可选大小上限）。
    - 内容：流式读取并写入；若超过 max_bytes 则中止；返回 (ok, err_str) 而不是抛错，便于上层聚合错误信息。
    - 可简略：否（repo 获取与 HF 下载都依赖；需要稳定的错误语义与限速/限量能力）。
    """
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
    """中文说明：
    - 含义：解压 GitHub archive zip，并返回解压出的 repo 根目录路径。
    - 内容：兼容 `repo-main/`、`repo-master/` 等目录结构；若结构不符合预期则抛错。
    - 可简略：是（仅 GitHub ZIP fallback 需要；若去掉 fallback 可删除）。
    """
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
    """中文说明：
    - 含义：识别 Hugging Face dataset URL，并解析出 (namespace, name)。
    - 内容：匹配 `https://huggingface.co/datasets/<namespace>/<name>` 路径；用于触发“HF 快照下载”逻辑。
    - 可简略：是（只在需要支持 HF dataset URL 时才需要）。
    """
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
    """中文说明：
    - 含义：调用 Hugging Face datasets API 获取数据集元信息（sha/siblings/private/gated 等）。
    - 内容：请求 `https://huggingface.co/api/datasets/{namespace}/{name}`；若提供 token 则加 Bearer；返回解析后的 dict。
    - 可简略：是（只在 HF dataset 支持场景需要）。
    """
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
    """中文说明：
    - 含义：通过 HF REST API 下载 dataset 的一个“快照副本”（无需 git）。
    - 内容：读取 dataset API 的 siblings 列表并逐文件下载到 `dest/`；写入 `data/hf_manifest.json`；并 best-effort 初始化 git（用于后续 revert guards）。
    - 可简略：是（可选能力；若你只处理 git repo，可移除整个 HF 分支）。

    ---

    English (original intent):
    Download a Hugging Face dataset snapshot via the HF REST API (no git required).
    """
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
    # 中文说明：
    # - 含义：当 `git clone` 失败时，尝试通过 GitHub 的 archive zip 下载并解压作为 fallback。
    # - 内容：尝试 main/master 分支 zip；成功后初始化本地 git（用于 revert guards）。
    # - 可简略：是（增强鲁棒性；若运行环境保证 git clone 可用，可移除）。
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
    """中文说明：
    - 含义：`prepare_repo` 的返回结构：一个可用的本地 repo 路径，以及可选的来源 URL。
    - 内容：当输入是远程 URL 时 cloned_from 会记录来源；当输入是本地路径时为 None。
    - 可简略：可能（字段很少；但作为清晰返回类型更利于测试/日志）。
    """

    repo: Path
    cloned_from: str | None = None


def _find_reusable_clone(base: Path, *, prefix: str) -> Path | None:
    """Best-effort: reuse an existing clone/snapshot under `base` when available.

    Motivation: when callers pass an explicit `clones_dir`, they usually want stable
    caching to avoid repeated network downloads and repeated OpenCode scaffolding.
    We keep the behavior generic: no repo-specific knowledge, only directory naming
    conventions produced by this file.
    """
    base = Path(base).expanduser().resolve()
    pref = str(prefix or "").strip()
    if not pref:
        return None

    candidates: list[Path] = []
    stable = (base / pref).resolve()
    if stable.exists():
        candidates.append(stable)
    candidates.extend(sorted(base.glob(f"{pref}_*")))

    reusable: list[Path] = []
    for p in candidates:
        try:
            if not p.exists() or not p.is_dir():
                continue
        except Exception:
            continue
        # Marker(s) that strongly suggest this directory is a complete fetched snapshot.
        try:
            if (p / ".git").exists() or (p / "data" / "hf_manifest.json").exists():
                reusable.append(p)
        except Exception:
            continue

    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    reusable.sort(key=_mtime, reverse=True)
    return reusable[0].resolve() if reusable else None


def prepare_repo(repo_arg: str, *, clones_dir: Path | None = None) -> PreparedRepo:
    """中文说明：
    - 含义：把用户输入的 `--repo` 参数解析成可用的本地目录。
    - 内容：
      - 若是已存在的本地路径：直接返回
      - 若是远程 git URL/owner/repo：clone 到临时目录（失败时可走 GitHub zip fallback）
      - 若是 HF dataset URL：走 HF API 下载快照
    - 可简略：否（runner 对“只给 URL”场景的核心入口；简化会大幅降低适配范围）。
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
        base = clones_dir or _default_clones_base()
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)

        if clones_dir is not None:
            prefix = f"hf_{namespace}_{name}"
            reused = _find_reusable_clone(base, prefix=prefix)
            if reused is not None:
                return PreparedRepo(repo=reused.resolve(), cloned_from=raw)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dest = base / f"hf_{namespace}_{name}_{ts}"
        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        ok, detail = _download_hf_dataset_snapshot(namespace=namespace, name=name, dest=dest, env=env)
        if not ok:
            raise RuntimeError(f"hf dataset download failed: {detail}")
        return PreparedRepo(repo=dest.resolve(), cloned_from=raw)

    url = normalize_repo_url(raw)
    base = clones_dir or _default_clones_base()
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    slug = _repo_slug(url)
    if clones_dir is not None:
        reused = _find_reusable_clone(base, prefix=slug)
        if reused is not None:
            return PreparedRepo(repo=reused.resolve(), cloned_from=url)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dest = base / f"{slug}_{ts}"
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    # Depth=1 keeps it fast; users can re-run with a local clone if needed.
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    try:
        clone_timeout = float(os.environ.get("AIDER_FSM_GIT_CLONE_TIMEOUT_SECONDS") or 90)
    except Exception:
        clone_timeout = 90.0
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=clone_timeout)
        rc = int(proc.returncode)
        out = proc.stdout or ""
        err = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        rc = 124
        out_raw = getattr(e, "stdout", "") or ""
        err_raw = getattr(e, "stderr", "") or ""
        if isinstance(out_raw, bytes):
            out = out_raw.decode("utf-8", errors="replace")
        else:
            out = str(out_raw)
        if isinstance(err_raw, bytes):
            err = err_raw.decode("utf-8", errors="replace")
        else:
            err = str(err_raw)
        err = (err + "\n" if err else "") + f"git_clone_timeout_exceeded: {clone_timeout}s"

    if rc != 0:
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
                f"git_rc: {rc}\n"
                f"git_stdout: {out[-2000:]}\n"
                f"git_stderr: {err[-2000:]}\n"
                f"archive_detail: {detail}\n"
            )

        raise RuntimeError(
            "git clone failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {rc}\n"
            f"stdout: {out[-2000:]}\n"
            f"stderr: {err[-2000:]}\n"
        )

    # Make sure local commits (if any) won't fail due to missing identity.
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)

    return PreparedRepo(repo=dest.resolve(), cloned_from=url)
