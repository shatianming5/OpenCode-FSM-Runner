"""Top-level programmatic env API for single-file training scripts.

Example:

  import env
  env.setup("https://github.com/<owner>/<repo>")
  env.rollout("/abs/path/to/model_dir")
  env.evaluation()
  env.teardown()
"""

from runner.env import (
    EnvSession,
    current_session,
    deploy,
    evaluation,
    rollout,
    rollout_and_evaluation,
    setup,
    teardown,
)

__all__ = [
    "EnvSession",
    "setup",
    "deploy",
    "rollout",
    "evaluation",
    "rollout_and_evaluation",
    "teardown",
    "current_session",
]

