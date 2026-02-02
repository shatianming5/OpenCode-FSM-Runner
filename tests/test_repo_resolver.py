from __future__ import annotations

import pytest

from runner.repo_resolver import is_probably_repo_url


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

