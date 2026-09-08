from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.admin.application.dto import (
    AdminOverview,
    CalendarActivity,
    DailyUsageSummary,
    ModelSummary,
    OverviewMetric,
    PoiMapPoint,
)
from locus_v2.ai.models import AIModel, AIProvider, PromptVersion, RoutingProfile
from locus_v2.billing.models import UsageEvent
from locus_v2.catalog.models import City, Poi
from locus_v2.identity.models import User
from locus_v2.shared.clock import utc_now
from locus_v2.voice.models import VoiceSession, VoiceSessionStatus


class SqlAlchemyOverviewReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read(self, *, environment: str, registered_adapters: list[str]) -> AdminOverview:
        model_rows = (
            await self._session.execute(
                select(AIModel, AIProvider.code)
                .join(AIProvider, AIProvider.id == AIModel.provider_id)
                .order_by(AIProvider.code, AIModel.display_name)
            )
        ).all()
        counts = await self._read_counts()

        return AdminOverview(
            environment=environment,
            metrics=[
                OverviewMetric(label="Modelos activos", value=counts["models"], tone="blue"),
                OverviewMetric(
                    label="Prompts versionados",
                    value=counts["prompts"],
                    tone="terracotta",
                ),
                OverviewMetric(label="Usuarios", value=counts["users"], tone="green"),
                OverviewMetric(label="POIs", value=counts["pois"], tone="ink"),
            ],
            registered_adapters=registered_adapters,
            models=[
                ModelSummary(
                    id=model.id,
                    provider=provider_code,
                    external_id=model.external_id,
                    display_name=model.display_name,
                    service_kind=model.service_kind,
                    adapter_code=model.adapter_code,
                    lifecycle=model.lifecycle,
                    enabled=model.enabled,
                    selectable=model.selectable,
                    capabilities=model.capabilities_json,
                )
                for model, provider_code in model_rows
            ],
            usage=await self._read_usage(),
            activities=await self._read_activities(),
            poi_map=await self._read_poi_map(),
        )

    async def _read_counts(self) -> dict[str, int]:
        queries = {
            "models": select(func.count()).select_from(AIModel).where(AIModel.enabled.is_(True)),
            "prompts": select(func.count()).select_from(PromptVersion),
            "profiles": select(func.count()).select_from(RoutingProfile),
            "users": select(func.count()).select_from(User),
            "pois": select(func.count()).select_from(Poi).where(Poi.is_active.is_(True)),
            "live": select(func.count())
            .select_from(VoiceSession)
            .where(VoiceSession.status == VoiceSessionStatus.ACTIVE),
        }
        return {
            key: int(await self._session.scalar(statement) or 0)
            for key, statement in queries.items()
        }

    async def _read_usage(self) -> list[DailyUsageSummary]:
        first_day = utc_now().date() - timedelta(days=13)
        rows = (
            await self._session.execute(
                select(
                    func.date(UsageEvent.created_at),
                    func.count(UsageEvent.id),
                    func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
                    func.coalesce(func.sum(UsageEvent.provider_cost_eur_cents), 0),
                )
                .where(UsageEvent.created_at >= datetime.combine(first_day, datetime.min.time()))
                .group_by(func.date(UsageEvent.created_at))
                .order_by(func.date(UsageEvent.created_at))
            )
        ).all()
        by_day = {
            self._as_date(day): (int(interactions), int(charged), int(cost))
            for day, interactions, charged, cost in rows
        }
        return [
            DailyUsageSummary(
                day=first_day + timedelta(days=offset),
                interactions=by_day.get(first_day + timedelta(days=offset), (0, 0, 0))[0],
                charged_cents=by_day.get(first_day + timedelta(days=offset), (0, 0, 0))[1],
                provider_cost_eur_cents=by_day.get(
                    first_day + timedelta(days=offset), (0, 0, 0)
                )[2],
            )
            for offset in range(14)
        ]

    async def _read_activities(self) -> list[CalendarActivity]:
        sessions = (
            await self._session.scalars(
                select(VoiceSession).order_by(VoiceSession.created_at.desc()).limit(40)
            )
        ).all()
        return [
            CalendarActivity(
                id=session.public_id,
                title=f"Voz · {session.locale}",
                start=session.started_at or session.created_at,
                kind=session.status,
            )
            for session in sessions
        ]

    async def _read_poi_map(self) -> list[PoiMapPoint]:
        rows = (
            await self._session.execute(
                select(Poi, City.name)
                .outerjoin(City, City.id == Poi.city_id)
                .where(Poi.is_active.is_(True), Poi.lat.is_not(None), Poi.lng.is_not(None))
                .order_by(Poi.updated_at.desc())
                .limit(120)
            )
        ).all()
        return [
            PoiMapPoint(
                id=poi.public_id,
                name=poi.name,
                city=city_name or "Sin ciudad",
                lat=float(poi.lat),
                lng=float(poi.lng),
            )
            for poi, city_name in rows
            if poi.lat is not None and poi.lng is not None
        ]

    @staticmethod
    def _as_date(value: date | str) -> date:
        return value if isinstance(value, date) else date.fromisoformat(value)
