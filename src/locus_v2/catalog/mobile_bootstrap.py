"""Mobile seed-once policy around the existing catalog importer.

The administrative importer remains unchanged. Mobile requests must never
trigger paid candidates, localization, or the importer's implicit AI fallback.
"""

from sqlalchemy import select
from sqlalchemy.orm import raiseload

from locus_v2.catalog.bootstrap.dto import BootstrapPoi
from locus_v2.catalog.bootstrap.service import CatalogBootstrapService
from locus_v2.catalog.models import City, Poi


class MobileCatalogBootstrap(CatalogBootstrapService):
    existing_poi_count: int = 0

    def _openai_api_key(self) -> str:
        return ""

    async def _import_pois(
        self, city: City, *, radius_km: float, limit: int, use_ai_candidates: bool = True
    ) -> tuple[int, int, str, list[BootstrapPoi]]:
        # Serialize imports for a city across API workers until the base service
        # commits. Locking reads see the latest committed seed even under MySQL
        # REPEATABLE READ, rather than the earlier city lookup's snapshot.
        await self.session.execute(
            select(City.id).where(City.id == city.id).with_for_update()
        )
        active_flags = list(
            (
                await self.session.scalars(
                    select(Poi.is_active)
                    .options(raiseload("*"))
                    .where(Poi.city_id == city.id)
                    .with_for_update()
                )
            ).all()
        )
        self.existing_poi_count = sum(active_flags)
        # Even a small seed or an intentionally deactivated catalog is seeded.
        # A response limit must not decide whether another import is necessary.
        if active_flags:
            return 0, 0, "existing_catalog", []
        return await super()._import_pois(
            city, radius_km=radius_km, limit=limit, use_ai_candidates=False
        )
