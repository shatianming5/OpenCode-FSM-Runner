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
        ("https://github.com/example-org/example-repo", True),
        ("https://github.com/example-org/example-repo.git", True),
        ("git@github.com:example-org/example-repo.git", True),
        ("ssh://git@github.com/example-org/example-repo.git", True),
        ("example-org/example-repo", True),
        (".", False),
        ("", False),
        (" /tmp/repo ", False),
    ],
)
def test_is_probably_repo_url(s: str, ok: bool):
    """中文说明：
    - 含义：验证 `is_probably_repo_url` 对常见 git/GitHub 形式与明显非 repo 字符串的判定。
    - 内容：使用参数化样例覆盖 https/ssh/scp-like/owner-repo 简写等，并断言返回值与期望一致。
    - 可简略：可能（可增加更多样例；但当前已覆盖主分支）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_repo_resolver.py:34；类型=function；引用≈1；规模≈7行
    assert is_probably_repo_url(s) is ok


@dataclass(frozen=True)
class _FakeCompletedProcess:
    """中文说明：
    - 含义：测试用的 `subprocess.CompletedProcess` 替身（只提供本测试需要的字段）。
    - 内容：包含 returncode/stdout/stderr，用于 monkeypatch `subprocess.run` 的返回值。
    - 可简略：是（也可直接返回简单对象或 namedtuple）。
    """
    # 作用：中文说明：
    # 能否简略：部分
    # 原因：测试代码（优先可读性）；规模≈10 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=tests/test_repo_resolver.py:44；类型=class；引用≈4；规模≈10行

    returncode: int
    stdout: str = ""
    stderr: str = ""


class _FakeHTTPResponse:
    """中文说明：
    - 含义：测试用的 HTTP 响应对象（模拟 urlopen 返回值）。
    - 内容：用 BytesIO 持有数据，实现 `read` 以及上下文管理协议（with）。
    - 可简略：是（最小替身；也可用更完整的 mock）。
    """
    # 作用：中文说明：
    # 能否简略：部分
    # 原因：测试代码（优先可读性）；规模≈38 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=tests/test_repo_resolver.py:56；类型=class；引用≈4；规模≈38行

    def __init__(self, data: bytes):
        """中文说明：
        - 含义：用给定 bytes 初始化一个可读的响应体。
        - 内容：将 data 包装到 BytesIO，供 `read()` 消费。
        - 可简略：是
        - 原因：纯测试替身样板，只为模拟 `urlopen()` 返回值的最小行为。
        """
        # 作用：中文说明：
        # 能否简略：是
        # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=tests/test_repo_resolver.py:63；类型=method；引用≈1；规模≈7行
        self._bio = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        """中文说明：
        - 含义：读取响应体字节。
        - 内容：透传到内部 BytesIO.read；n=-1 表示读完。
        - 可简略：是
        - 原因：纯测试替身样板；直接转发 BytesIO.read 即可满足本测试需求。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:71；类型=method；引用≈7；规模≈7行
        return self._bio.read(n)

    def __enter__(self):
        """中文说明：
        - 含义：支持 `with urlopen(...) as resp:` 语法。
        - 内容：返回 self。
        - 可简略：是
        - 原因：纯测试替身样板；只需满足上下文管理协议即可。
        """
        # 作用：中文说明：
        # 能否简略：是
        # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=tests/test_repo_resolver.py:79；类型=method；引用≈0；规模≈7行
        return self

    def __exit__(self, exc_type, exc, tb):
        """中文说明：
        - 含义：上下文管理退出钩子。
        - 内容：返回 False 表示不吞异常。
        - 可简略：是
        - 原因：纯测试替身样板；返回 False 保持异常传播语义即可。
        """
        # 作用：中文说明：
        # 能否简略：是
        # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=tests/test_repo_resolver.py:87；类型=method；引用≈0；规模≈7行
        return False


def _make_zip_bytes(root_dir: str, files: dict[str, str]) -> bytes:
    """中文说明：
    - 含义：构造一个内存中的 zip 包（用于模拟 GitHub archive 下载）。
    - 内容：以 `root_dir/relpath` 的形式写入多个文件内容并返回 zip 的 bytes。
    - 可简略：可能（可以用固定 zip fixture；但动态生成更灵活）。
    """
    # 作用：中文说明：
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_repo_resolver.py:96；类型=function；引用≈2；规模≈11行
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, content in files.items():
            zf.writestr(f"{root_dir}/{rel}", content)
    return buf.getvalue()


def test_prepare_repo_github_archive_fallback_on_git_clone_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """中文说明：
    - 含义：验证 `prepare_repo` 在 `git clone` 失败时会退回 GitHub archive zip 下载。
    - 内容：fake_run 让 git clone 失败；fake_urlopen 返回 main.zip；断言最终 repo 中 README 内容正确且 cloned_from 被记录。
    - 可简略：否（是“只给 URL 也能拉取”的关键保障路径之一；建议保留）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈36 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_repo_resolver.py:109；类型=function；引用≈1；规模≈36行
    url = "https://github.com/foo/bar.git"
    zip_bytes = _make_zip_bytes("bar-main", {"README.md": "hello\n"})

    def fake_run(cmd, *args, **kwargs):
        """中文说明：
        - 含义：替换 `subprocess.run` 的测试桩。
        - 内容：仅对 `git clone` 返回失败，其它命令返回成功，触发 archive fallback 分支。
        - 可简略：是（测试桩；可用 mock 框架替代）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈10 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:118；类型=function；引用≈12；规模≈10行
        # Fail `git clone`, succeed everything else.
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "clone":
            return _FakeCompletedProcess(returncode=1, stdout="", stderr="blocked")
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    def fake_urlopen(req, *args, **kwargs):
        """中文说明：
        - 含义：替换 `urlopen` 的测试桩（只处理 archive zip 下载）。
        - 内容：断言请求目标是 `/archive/refs/heads/main.zip`，并返回包含 README 的 zip。
        - 可简略：是（测试桩；可用更通用的 stub 服务器替代）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈9 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:129；类型=function；引用≈6；规模≈9行
        target = str(getattr(req, "full_url", req))
        assert target.endswith("/archive/refs/heads/main.zip")
        return _FakeHTTPResponse(zip_bytes)

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", fake_run)
    monkeypatch.setattr("runner.repo_resolver.urlopen", fake_urlopen)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert (prepared.repo / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_prepare_repo_hf_dataset_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """中文说明：
    - 含义：验证 `prepare_repo` 能把 HuggingFace dataset URL 拉取为本地可用的 repo 目录。
    - 内容：fake_urlopen 模拟 HF API 返回 siblings 清单 + 文件下载；fake_run 让 git init/add/commit 成功；断言文件与 manifest 落盘。
    - 可简略：否（是“只给 URL”能力的另一关键分支：HF dataset）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈57 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_repo_resolver.py:147；类型=function；引用≈1；规模≈57行
    url = "https://huggingface.co/datasets/example/dataset"
    api_json = {
        "id": "example/dataset",
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
        """中文说明：
        - 含义：替换 `subprocess.run` 的测试桩（HF dataset 分支）。
        - 内容：假装所有命令都成功（git init/add/commit best-effort），以便测试聚焦在下载逻辑。
        - 可简略：是（测试桩）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈8 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:169；类型=function；引用≈12；规模≈8行
        # Always succeed (git init/add/commit best-effort).
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    def fake_urlopen(req, *args, **kwargs):
        """中文说明：
        - 含义：替换 `urlopen` 的测试桩，模拟 HF dataset API 与文件下载端点。
        - 内容：对 `/api/datasets/...` 返回 JSON；对 `/resolve/<sha>/...` 返回对应文件 bytes；其它 url 直接断言失败。
        - 可简略：可能（逻辑稍多；可抽象成路由表以减少 if/else）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈14 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:178；类型=function；引用≈6；规模≈14行
        target = str(getattr(req, "full_url", req))
        if target == "https://huggingface.co/api/datasets/example/dataset":
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
    assert prepared.repo.name.startswith("hf_example_dataset_")
    assert (prepared.repo / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert (prepared.repo / "main" / "train-00000-of-00001.parquet").read_text(encoding="utf-8") == "parquet\n"

    manifest = (prepared.repo / "data" / "hf_manifest.json").read_text(encoding="utf-8")
    assert "example/dataset" in manifest


def test_prepare_repo_reuses_existing_git_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When clones_dir is provided, prefer reusing the newest matching clone to avoid re-downloads."""
    # 作用：When clones_dir is provided, prefer reusing the newest matching clone to avoid re-downloads.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈15 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_repo_resolver.py:202；类型=function；引用≈1；规模≈15行
    url = "https://github.com/foo/bar.git"
    existing = tmp_path / "foo_bar_20000101_000000"
    (existing / ".git").mkdir(parents=True)

    def boom(*args, **kwargs):
        # 作用：内部符号：test_prepare_repo_reuses_existing_git_clone.boom
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈2 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:207；类型=function；引用≈6；规模≈2行
        raise AssertionError("unexpected external fetch; expected reuse")

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", boom)
    monkeypatch.setattr("runner.repo_resolver.urlopen", boom)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert prepared.repo == existing.resolve()


def test_prepare_repo_reuses_existing_hf_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When clones_dir is provided, prefer reusing the newest matching HF snapshot."""
    # 作用：When clones_dir is provided, prefer reusing the newest matching HF snapshot.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈17 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_repo_resolver.py:219；类型=function；引用≈1；规模≈17行
    url = "https://huggingface.co/datasets/example/dataset"
    existing = tmp_path / "hf_example_dataset_20000101_000000"
    (existing / ".git").mkdir(parents=True)
    (existing / "data").mkdir(parents=True)
    (existing / "data" / "hf_manifest.json").write_text("{}", encoding="utf-8")

    def boom(*args, **kwargs):
        # 作用：内部符号：test_prepare_repo_reuses_existing_hf_snapshot.boom
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈2 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_repo_resolver.py:226；类型=function；引用≈6；规模≈2行
        raise AssertionError("unexpected external fetch; expected reuse")

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", boom)
    monkeypatch.setattr("runner.repo_resolver.urlopen", boom)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert prepared.repo == existing.resolve()
