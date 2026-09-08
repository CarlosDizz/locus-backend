from typing import Protocol

from locus_v2.admin.application.dto import AdminOverview, BuildInfo


class OverviewReader(Protocol):
    async def read(
        self, *, environment: str, build: BuildInfo, registered_adapters: list[str]
    ) -> AdminOverview:
        """Build the read model used by the operations dashboard."""
