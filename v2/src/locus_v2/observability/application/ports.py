from datetime import datetime
from typing import Protocol

from locus_v2.observability.domain import LogEntry


class EventLogRepository(Protocol):
    async def append(self, entry: LogEntry) -> None: ...

    async def delete_before(self, cutoff: datetime) -> int: ...
