from dataclasses import dataclass, field
from typing import Any, Literal

LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@dataclass(frozen=True)
class LogEntry:
    level: LogLevel
    service: str
    event: str
    environment: str
    message: str | None = None
    trace_id: str | None = None
    user_id: int | None = None
    voice_session_id: int | None = None
    error_type: str | None = None
    error_code: str | None = None
    elapsed_ms: int | None = None
    context: dict[str, Any] = field(default_factory=dict)
