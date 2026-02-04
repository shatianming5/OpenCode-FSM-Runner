from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pytest

from runner.repo_resolver import is_probably_repo_url, prepare_repo


@pytest.mark.parametrize(
    ("s", "ok"),
    [
        ("https://github.com/evalplus/evalplus", True),
        ("https://github.com/evalplus/evalplus.git", True),
        ("git@github.com:evalplus/evalplus.git", True),
        ("ssh://git@github.com/evalplus/evalplus.git", True),
        ("evalplus/evalplus", True),
        (".", False),
        ("", False),
        (" /tmp/repo ", False),
    ],
)
def test_is_probably_repo_url(s: str, ok: bool):
    assert is_probably_repo_url(s) is ok


@dataclass(frozen=True)
class _FakeCompletedProcess:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _FakeHTTPResponse:
    def __init__(self, data: bytes):
        self._bio = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._bio.read(n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_zip_bytes(root_dir: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, content in files.items():
            zf.writestr(f"{root_dir}/{rel}", content)
    return buf.getvalue()


def test_prepare_repo_github_archive_fallback_on_git_clone_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    url = "https://github.com/foo/bar.git"
    zip_bytes = _make_zip_bytes("bar-main", {"README.md": "hello\n"})

    def fake_run(cmd, *args, **kwargs):
        # Fail `git clone`, succeed everything else.
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "clone":
            return _FakeCompletedProcess(returncode=1, stdout="", stderr="blocked")
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    def fake_urlopen(req, *args, **kwargs):
        target = str(getattr(req, "full_url", req))
        assert target.endswith("/archive/refs/heads/main.zip")
        return _FakeHTTPResponse(zip_bytes)

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", fake_run)
    monkeypatch.setattr("runner.repo_resolver.urlopen", fake_urlopen)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert (prepared.repo / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_prepare_repo_hf_dataset_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    url = "https://huggingface.co/datasets/openai/gsm8k"
    api_json = {
        "id": "openai/gsm8k",
        "sha": "deadbeef",
        "private": False,
        "gated": False,
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "main/train-00000-of-00001.parquet"},
        ],
    }
    files = {
        "README.md": b"hello\n",
        "main/train-00000-of-00001.parquet": b"parquet\n",
    }

    def fake_run(cmd, *args, **kwargs):
        # Always succeed (git init/add/commit best-effort).
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    def fake_urlopen(req, *args, **kwargs):
        target = str(getattr(req, "full_url", req))
        if target == "https://huggingface.co/api/datasets/openai/gsm8k":
            return _FakeHTTPResponse(json.dumps(api_json).encode("utf-8"))
        if "/resolve/deadbeef/" in target:
            rel = target.split("/resolve/deadbeef/", 1)[1]
            rel = unquote(rel)
            return _FakeHTTPResponse(files[rel])
        raise AssertionError(f"unexpected url: {target}")

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", fake_run)
    monkeypatch.setattr("runner.repo_resolver.urlopen", fake_urlopen)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert prepared.repo.name.startswith("hf_openai_gsm8k_")
    assert (prepared.repo / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert (prepared.repo / "main" / "train-00000-of-00001.parquet").read_text(encoding="utf-8") == "parquet\n"

    manifest = (prepared.repo / "data" / "hf_manifest.json").read_text(encoding="utf-8")
    assert "openai/gsm8k" in manifest
