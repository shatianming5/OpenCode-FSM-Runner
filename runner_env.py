"""Thin import alias for the library entrypoint.

This repo intentionally exposes a minimal public API:

- `sess = runner_env.setup(target)`
- `sess.rollout(llm=..., ...)`
- `sess.evaluate(...)`
"""

from runner.env import EnvSession, setup

__all__ = ["EnvSession", "setup"]

