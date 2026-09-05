from locus_v2.admin.application.dto import AdminOverview
from locus_v2.admin.application.ports import OverviewReader


class AdminOverviewService:
    def __init__(self, reader: OverviewReader) -> None:
        self._reader = reader

    async def execute(
        self, *, environment: str, registered_adapters: list[str]
    ) -> AdminOverview:
        return await self._reader.read(
            environment=environment,
            registered_adapters=registered_adapters,
        )
