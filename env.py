"""Top-level programmatic env API for single-file training scripts.

Example:

  import env

  sess = env.setup("https://github.com/<owner>/<repo>")
  sess.rollout("/abs/path/to/model_dir")
  sess.evaluate()
  sess.teardown()

This module primarily re-exports `runner.env`, but also provides a small
compatibility layer:

- `env.setup({...})` accepts a dict config with `repo=...`.
- `EnvSession.evaluate()` is an alias for `EnvSession.evaluation()`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runner.env import (
    EnvSession,
    current_session,
    deploy,
    evaluate,
    evaluation,
    rollout,
    rollout_and_evaluation,
    setup as _setup,
    teardown,
)


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s).expanduser().resolve()


def setup(target: Any, **kwargs: Any) -> EnvSession:
    """Open an `EnvSession`.

    - `env.setup("https://...")` behaves exactly like `runner.env.setup(...)`.
    - `env.setup({"repo": "...", ...})` is a convenience form for config-driven code.
    """
    if isinstance(target, Mapping):
        cfg = dict(target)
        repo = cfg.get("repo") or cfg.get("target")
        if repo is None or not str(repo).strip():
            raise ValueError("env.setup(config): missing required field `repo`")

        allowed = {
            "clones_dir",
            "pipeline_rel",
            "require_metrics",
            "audit",
            "opencode_model",
            "opencode_repair_model",
            "opencode_url",
            "unattended",
            "opencode_timeout_seconds",
            "opencode_repair_timeout_seconds",
            "opencode_retry_attempts",
            "opencode_retry_backoff_seconds",
            "opencode_context_length",
            "opencode_max_prompt_chars",
            "opencode_bash",
            "scaffold_opencode_bash",
            "strict_opencode",
            "artifacts_dir",
        }
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        for k in allowed:
            if k in cfg and cfg.get(k) is not None:
                call_kwargs[k] = cfg.get(k)

        # `runner.env.setup` expects Path objects for these fields. The mapping-form API
        # commonly passes strings, so coerce here to keep `env.setup({...})` robust.
        if "clones_dir" in call_kwargs:
            call_kwargs["clones_dir"] = _as_path(call_kwargs.get("clones_dir"))
        if "artifacts_dir" in call_kwargs:
            call_kwargs["artifacts_dir"] = _as_path(call_kwargs.get("artifacts_dir"))

        sess = _setup(str(repo), **call_kwargs)

        runtime_env_path = _as_path(cfg.get("runtime_env_path"))
        if runtime_env_path is not None:
            sess.runtime_env_path = runtime_env_path

        return sess

    return _setup(target, **kwargs)


__all__ = [
    "EnvSession",
    "setup",
    "deploy",
    "rollout",
    "evaluation",
    "evaluate",
    "rollout_and_evaluation",
    "teardown",
    "current_session",
]
