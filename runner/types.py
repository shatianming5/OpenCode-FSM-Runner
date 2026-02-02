from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CmdResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class StageResult:
    ok: bool
    results: list[CmdResult]
    failed_index: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failed_stage: str | None
    auth: StageResult | None = None
    tests: StageResult | None = None
    deploy_setup: StageResult | None = None
    deploy_health: StageResult | None = None
    benchmark: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] | None = None

