from __future__ import annotations

import os
from pathlib import Path

from runner.dotenv import load_dotenv


def test_load_dotenv_sets_missing_vars(tmp_path: Path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# comment",
                "OPENAI_API_KEY=abc",
                'export FOO="bar baz"',
                "EMPTY=",
                "BADLINE",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    written = load_dotenv(p)
    assert set(written) == {"OPENAI_API_KEY", "FOO", "EMPTY"}
    assert os.environ["OPENAI_API_KEY"] == "abc"
    assert os.environ["FOO"] == "bar baz"
    assert os.environ["EMPTY"] == ""


def test_load_dotenv_does_not_override_by_default(tmp_path: Path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "preexisting")
    written = load_dotenv(p, override=False)
    assert written == []
    assert os.environ["FOO"] == "preexisting"


def test_load_dotenv_override_true(tmp_path: Path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "preexisting")
    written = load_dotenv(p, override=True)
    assert written == ["FOO"]
    assert os.environ["FOO"] == "from_file"

