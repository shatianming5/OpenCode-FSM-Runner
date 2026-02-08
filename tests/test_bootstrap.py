import shlex
import sys
from pathlib import Path

import pytest

from runner.bootstrap import load_bootstrap_spec, load_bootstrap_spec_with_diagnostics, run_bootstrap


def _py_cmd(code: str) -> str:
    """中文说明：
    - 含义：把一段 Python 代码包装成可执行的命令行（`python -c ...`）。
    - 内容：对解释器路径做 shell quote，用于构造跨平台相对稳定的测试命令。
    - 可简略：是（测试里可直接拼接；保留主要为复用）。
    """
    py = shlex.quote(sys.executable)
    return f'{py} -c "{code}"'


def test_load_bootstrap_spec_ok(tmp_path: Path):
    """中文说明：
    - 含义：验证 `load_bootstrap_spec` 能正确解析 v1 bootstrap.yml。
    - 内容：写入包含 env/cmds/workdir/timeout/retries 的 YAML，断言解析后的字段与 raw 文本符合预期。
    - 可简略：可能（可拆成更细的字段校验参数化；但当前覆盖面已足够）。
    """
    p = tmp_path / "bootstrap.yml"
    p.write_text(
        "\n".join(
            [
                "version: 1",
                "env: {FOO: bar}",
                "cmds: [echo ok]",
                "workdir: .",
                "timeout_seconds: 10",
                "retries: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec, raw = load_bootstrap_spec(p)
    assert raw.strip()
    assert spec.version == 1
    assert spec.env["FOO"] == "bar"
    assert spec.cmds == ["echo ok"]
    assert spec.workdir == "."
    assert spec.timeout_seconds == 10
    assert spec.retries == 2


def test_load_bootstrap_spec_invalid_version(tmp_path: Path):
    """中文说明：
    - 含义：验证 bootstrap.yml 的 version 不支持时会报错。
    - 内容：写入 version=2，期望抛出 ValueError。
    - 可简略：是（典型负例测试）。
    """
    p = tmp_path / "bootstrap.yml"
    p.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bootstrap_spec(p)


def test_load_bootstrap_spec_accepts_boot_wrapper_and_step_mappings(tmp_path: Path) -> None:
    p = tmp_path / "bootstrap.yml"
    p.write_text(
        "\n".join(
            [
                "boot:",
                "  version: 1",
                "  steps:",
                "    - run: echo one",
                "    - cmd: echo two",
                "  cwd: .",
                "  timeout: 12",
                "  retry: 1",
                "  env:",
                "    FOO: bar",
                "",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_bootstrap_spec_with_diagnostics(p)
    assert loaded.spec.cmds == ["echo one", "echo two"]
    assert loaded.spec.workdir == "."
    assert loaded.spec.timeout_seconds == 12
    assert loaded.spec.retries == 1
    assert loaded.spec.env.get("FOO") == "bar"
    assert any("boot_mapping_unwrapped" in w for w in loaded.warnings)
    assert any("steps_alias_used" in w for w in loaded.warnings)


def test_run_bootstrap_applies_env_and_runs_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """中文说明：
    - 含义：验证 `run_bootstrap` 会应用 env（含变量展开）并执行 cmds。
    - 内容：bootstrap.yml 设置 `FOO=bar`、`BAR=${FOO}-baz`，命令检查 BAR 是否为 `bar-baz`，并断言 applied_env 正确。
    - 可简略：否（变量展开与落盘 artifacts 是 bootstrap 的关键契约；建议保留该覆盖）。
    """
    repo = tmp_path
    (repo / ".aider_fsm").mkdir(parents=True, exist_ok=True)
    bootstrap_path = repo / ".aider_fsm" / "bootstrap.yml"
    code = "import os,sys; sys.exit(0 if os.environ.get('BAR')=='bar-baz' else 3)"
    bootstrap_path.write_text(
        "\n".join(
            [
                "version: 1",
                "env:",
                "  FOO: bar",
                "  BAR: ${FOO}-baz",
                "cmds:",
                f"  - {_py_cmd(code)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Ensure a stable PATH for expansion tests (even though this spec doesn't change it).
    monkeypatch.setenv("PATH", "/usr/bin")

    stage, applied_env = run_bootstrap(
        repo,
        bootstrap_path=bootstrap_path,
        pipeline=None,
        unattended="strict",
        artifacts_dir=repo / "artifacts",
    )
    assert stage.ok is True
    assert applied_env["FOO"] == "bar"
    assert applied_env["BAR"] == "bar-baz"


def test_run_bootstrap_writes_parse_warnings_artifact(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / ".aider_fsm").mkdir(parents=True, exist_ok=True)
    bootstrap_path = repo / ".aider_fsm" / "bootstrap.yml"
    bootstrap_path.write_text(
        "\n".join(
            [
                "boot:",
                "  version: 1",
                "  steps:",
                "    - echo ok",
                "",
            ]
        ),
        encoding="utf-8",
    )

    stage, _env = run_bootstrap(
        repo,
        bootstrap_path=bootstrap_path,
        pipeline=None,
        unattended="strict",
        artifacts_dir=repo / "artifacts",
    )
    assert stage.ok is True
    warns = repo / "artifacts" / "bootstrap_parse_warnings.json"
    assert warns.exists()
