from datetime import date, datetime
from typing import Protocol

from pydantic import BaseModel


class BillingTotals(BaseModel):
    usage_events: int
    voice_sessions: int
    active_users: int
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int
    pending_events: int
    failed_events: int


class BillingDailyPoint(BaseModel):
    day: date
    usage_events: int
    provider_cost_eur_cents: int
    charged_amount_cents: int


class BillingModelBreakdown(BaseModel):
    provider: str
    model: str
    usage_events: int
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int


class BillingUsageItem(BaseModel):
    id: str
    user_email: str | None
    provider: str
    model: str
    interaction_type: str
    status: str
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int
    text_tokens: int
    audio_tokens: int
    created_at: datetime


class BillingLedgerItem(BaseModel):
    id: str
    user_email: str
    kind: str
    description: str
    amount_cents: int
    balance_after_cents: int
    created_at: datetime


class AdminBillingDashboard(BaseModel):
    period_days: int
    totals: BillingTotals
    daily: list[BillingDailyPoint]
    by_model: list[BillingModelBreakdown]
    recent_usage: list[BillingUsageItem]
    recent_ledger: list[BillingLedgerItem]


class BillingDashboardReader(Protocol):
    async def read(self, *, days: int) -> AdminBillingDashboard: ...


class AdminBillingDashboardService:
    def __init__(self, reader: BillingDashboardReader) -> None:
        self._reader = reader

    async def execute(self, *, days: int) -> AdminBillingDashboard:
        return await self._reader.read(days=max(0, days))
