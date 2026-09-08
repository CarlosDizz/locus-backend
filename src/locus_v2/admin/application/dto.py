from datetime import date

from pydantic import BaseModel

from locus_v2.shared.clock import UtcDatetime


class OverviewMetric(BaseModel):
    label: str
    value: int
    tone: str


class ModelSummary(BaseModel):
    id: int
    provider: str
    external_id: str
    display_name: str
    service_kind: str
    adapter_code: str
    lifecycle: str
    enabled: bool
    selectable: bool
    capabilities: dict


class DailyUsageSummary(BaseModel):
    day: date
    interactions: int
    charged_cents: int
    provider_cost_eur_cents: int


class CalendarActivity(BaseModel):
    id: str
    title: str
    start: UtcDatetime
    kind: str


class PoiMapPoint(BaseModel):
    id: str
    name: str
    city: str
    lat: float
    lng: float


class BuildInfo(BaseModel):
    """Which build the panel is looking at.

    `version` is maintained by hand and `commit` is stamped by the deploy, so
    the pair stays truthful even when someone forgets to bump the version.
    """

    version: str
    commit: str
    deployed_at: str


class AdminOverview(BaseModel):
    environment: str
    build: BuildInfo
    metrics: list[OverviewMetric]
    registered_adapters: list[str]
    models: list[ModelSummary]
    usage: list[DailyUsageSummary]
    activities: list[CalendarActivity]
    poi_map: list[PoiMapPoint]
