from datetime import datetime

from pydantic import BaseModel, Field


class WalletResponse(BaseModel):
    user_id: int
    currency: str
    balance_cents: int


class LedgerEntryResponse(BaseModel):
    id: int
    entry_type: str
    amount_cents: int
    balance_after_cents: int
    description: str
    reference_type: str
    reference_id: str
    created_at: datetime


class TopUpRequest(BaseModel):
    amount_cents: int = Field(gt=0)
    provider: str = Field(default="manual")
    provider_reference: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class TopUpResponse(BaseModel):
    id: int
    amount_cents: int
    bonus_cents: int
    provider: str
    provider_reference: str
    status: str
    created_at: datetime


class UsageRecordRequest(BaseModel):
    session_id: str | None = None
    provider: str = "openai"
    endpoint: str
    model: str
    response_id: str = ""
    usage: dict
    metadata: dict = Field(default_factory=dict)


class UsageRecordResponse(BaseModel):
    id: int
    charged_amount_cents: int
    balance_cents: int
