from typing import Protocol

from locus_v2.admin.application.dto import AdminOverview


class OverviewReader(Protocol):
    async def read(self, *, environment: str, registered_adapters: list[str]) -> AdminOverview:
        """Build the read model used by the operations dashboard."""
