"""Las recargas vistas desde el panel: quién ha pagado, cuánto y cuándo.

Es el reverso del panel de consumos. Aquel cuenta el dinero que sale hacia los
proveedores de IA; éste cuenta el que entra, lo que se queda la pasarela por el
camino y cuánto saldo queda vivo sin gastar — que no es caja, es una deuda en
servicio.
"""

from datetime import date
from typing import Protocol

from pydantic import BaseModel

from locus_v2.shared.clock import UtcDatetime


class TopUpTotals(BaseModel):
    top_ups: int
    paying_users: int
    gross_cents: int
    """Lo que pagaron los clientes, antes de comisiones."""
    estimated_fees_cents: int
    net_cents: int
    """Bruto menos comisiones estimadas. Lo que debería acabar en la cuenta."""
    average_cents: int
    bonus_cents: int
    pending_top_ups: int
    pending_cents: int
    """Cobrado en la pasarela y sin abonar todavía. Cualquier cosa que no sea
    cero y reciente merece una mirada: es dinero de alguien sin su saldo."""
    outstanding_balance_cents: int
    """Saldo sin gastar de todos los monederos. No depende del periodo: es una
    foto de hoy, y es un pasivo, no un ingreso."""


class TopUpDailyPoint(BaseModel):
    day: date
    top_ups: int
    gross_cents: int


class TopUpProviderBreakdown(BaseModel):
    provider: str
    top_ups: int
    gross_cents: int
    estimated_fees_cents: int


class TopUpItem(BaseModel):
    id: str
    user_email: str
    user_name: str
    amount_cents: int
    bonus_cents: int
    estimated_fee_cents: int
    provider: str
    provider_reference: str
    status: str
    created_at: UtcDatetime
    completed_at: UtcDatetime | None


class AdminTopUpsDashboard(BaseModel):
    period_days: int
    totals: TopUpTotals
    daily: list[TopUpDailyPoint]
    by_provider: list[TopUpProviderBreakdown]
    recent: list[TopUpItem]


class TopUpsReader(Protocol):
    async def read(self, *, days: int, limit: int) -> AdminTopUpsDashboard: ...


class AdminTopUpsService:
    def __init__(self, reader: TopUpsReader) -> None:
        self._reader = reader

    async def execute(self, *, days: int, limit: int = 100) -> AdminTopUpsDashboard:
        return await self._reader.read(days=max(0, days), limit=max(1, min(limit, 500)))
