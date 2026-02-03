from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    assistant_text: str
    raw: Any | None = None


class AgentClient(Protocol):
    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult: ...

    def close(self) -> None: ...

