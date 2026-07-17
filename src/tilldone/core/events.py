from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tilldone.core.enums import EventKind


@dataclass(frozen=True)
class AgentEvent:
    kind: EventKind
    seq: int
    data: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None
    call_id: str | None = None
    raw: Any = None
