import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from locus_v2.ai.enums import Lifecycle
from locus_v2.ai.models import AIModel, AIProvider, ProviderPriceSnapshot
from locus_v2.billing.models import LedgerEntry, TopUp, UsageEvent, Wallet
from locus_v2.catalog.models import City, Poi, PoiType
from locus_v2.identity.models import Role, User, UserRole, UserStatus
from locus_v2.infrastructure.database import models as database_models  # noqa: F401
from locus_v2.migrations.models import DataImportRun, LegacyAppSession
from locus_v2.shared.ids import new_public_id

logger = structlog.get_logger()

TABLES = (
    "users",
    "wallets",
    "price_snapshots",
    "usage_events",
    "ledger_entries",
    "top_ups",
    "cities",
    "poi_types",
    "pois",
    "app_sessions",
)


def async_mysql_url(url: str) -> str:
    for prefix in ("mysql+pymysql://", "mysql://"):
        if url.startswith(prefix):
            return "mysql+asyncmy://" + url[len(prefix) :]
    return url


def json_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def copied_at(row: Mapping[str, Any], field: str, fallback: datetime | None = None) -> datetime:
    value = row.get(field)
    if isinstance(value, datetime):
        return value
    return fallback or datetime.now()


class LegacyV1Importer:
    """Read-only V1 adapter and idempotent V2 import use case."""

    def __init__(self, source_url: str, target: AsyncSession, admin_email: str) -> None:
        self.source: AsyncEngine = create_async_engine(async_mysql_url(source_url), pool_pre_ping=True)
        self.target = target
        self.admin_email = admin_email.lower()
        self.counts: dict[str, int] = {}
        self.skipped = 0

    async def inspect(self) -> dict[str, int]:
        async with self.source.connect() as connection:
            existing = {
                row[0]
                for row in (await connection.execute(text("SHOW TABLES"))).all()
            }
            counts: dict[str, int] = {}
            for table in TABLES:
                if table in existing:
                    counts[table] = int(
                        (await connection.execute(text(f"SELECT COUNT(*) FROM `{table}`"))).scalar_one()
                    )
            return counts

    async def rows(self, table: str) -> list[Mapping[str, Any]]:
        async with self.source.connect() as connection:
            result = await connection.execute(text(f"SELECT * FROM `{table}` ORDER BY 1"))
            return list(result.mappings().all())

    async def _existing(self, model: type[Any]) -> dict[int, Any]:
        result = await self.target.scalars(select(model).where(model.legacy_v1_id.is_not(None)))
        return {int(item.legacy_v1_id): item for item in result.all()}

    async def run(self) -> DataImportRun:
        run = DataImportRun(source="locus-v1-production", status="running")
        self.target.add(run)
        await self.target.flush()
        try:
            await self._users()
            await self._catalog()
            await self._wallets()
            await self._prices()
            await self._usage()
            await self._ledger()
            await self._top_ups()
            await self._sessions()
            run.status = "completed"
            run.imported_rows = sum(self.counts.values())
            run.skipped_rows = self.skipped
            run.table_counts_json = self.counts
            await self.target.commit()
            logger.info("legacy_v1_import_completed", counts=self.counts, skipped=self.skipped)
            return run
        except Exception as exc:
            await self.target.rollback()
            failed = DataImportRun(
                source="locus-v1-production",
                status="failed",
                imported_rows=0,
                skipped_rows=self.skipped,
                failed_rows=1,
                table_counts_json=self.counts,
                error_log=str(exc),
            )
            self.target.add(failed)
            await self.target.commit()
            raise
        finally:
            await self.source.dispose()

    async def _users(self) -> None:
        existing = await self._existing(User)
        target_users = (await self.target.scalars(select(User))).all()
        users_by_email = {user.email.strip().lower(): user for user in target_users}
        user_role = await self.target.scalar(select(Role).where(Role.code == "user"))
        admin_role = await self.target.scalar(select(Role).where(Role.code == "admin"))
        imported = merged = 0
        for row in await self.rows("users"):
            legacy_id = int(row["id"])
            email = str(row["email"]).strip().lower()
            user = existing.get(legacy_id)
            if user is None:
                user = users_by_email.get(email)
                if user is not None:
                    if user.legacy_v1_id not in (None, legacy_id):
                        raise ValueError(
                            f"User {email} is already linked to legacy id {user.legacy_v1_id}"
                        )
                    user.legacy_v1_id = legacy_id
                    user.display_name = user.display_name or row.get("display_name") or email.split("@", 1)[0]
                    user.avatar_url = user.avatar_url or row.get("avatar_url") or None
                    user.provider_subject = user.provider_subject or row.get("google_sub") or None
                    merged += 1
                else:
                    user = User(
                        legacy_v1_id=legacy_id,
                        email=email,
                        display_name=row.get("display_name") or email.split("@", 1)[0],
                        avatar_url=row.get("avatar_url") or None,
                        auth_provider=row.get("auth_provider") or "google",
                        provider_subject=row.get("google_sub") or None,
                        locale="es-ES",
                        status=UserStatus.ACTIVE if row.get("is_active", True) else UserStatus.BLOCKED,
                        created_at=copied_at(row, "created_at"),
                        updated_at=copied_at(row, "updated_at", copied_at(row, "created_at")),
                    )
                    self.target.add(user)
                    users_by_email[email] = user
                    imported += 1
                await self.target.flush()
                existing[legacy_id] = user
            legacy_created_at = copied_at(row, "created_at")
            if user.created_at > legacy_created_at:
                user.created_at = legacy_created_at
            if user_role is not None:
                await self._grant_role(user.id, user_role.id)
            if admin_role is not None and user.email.lower() == self.admin_email:
                await self._grant_role(user.id, admin_role.id)
        self.counts["users"] = imported + merged

    async def _grant_role(self, user_id: int, role_id: int) -> None:
        exists = await self.target.scalar(
            select(UserRole).where(UserRole.user_id == user_id, UserRole.role_id == role_id)
        )
        if exists is None:
            self.target.add(UserRole(user_id=user_id, role_id=role_id))

    async def _catalog(self) -> None:
        city_map = await self._existing(City)
        type_map = await self._existing(PoiType)
        city_count = type_count = poi_count = 0
        for row in await self.rows("cities"):
            legacy_id = int(row["id"])
            if legacy_id not in city_map:
                city = City(
                    legacy_v1_id=legacy_id,
                    slug=row["slug"],
                    name=row["name"],
                    names_json=json_value(row.get("names_json"), {}),
                    country_code=row.get("country_code") or "",
                    lat=row.get("lat"),
                    lng=row.get("lng"),
                    source=row.get("source") or "legacy_v1",
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "created_at"),
                )
                self.target.add(city)
                await self.target.flush()
                city_map[legacy_id] = city
                city_count += 1
        for row in await self.rows("poi_types"):
            legacy_id = int(row["id"])
            if legacy_id not in type_map:
                poi_type = PoiType(
                    legacy_v1_id=legacy_id,
                    code=row["code"],
                    name=row["name"],
                    description=row.get("description") or "",
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "created_at"),
                )
                self.target.add(poi_type)
                await self.target.flush()
                type_map[legacy_id] = poi_type
                type_count += 1
        existing_pois = await self._existing(Poi)
        for row in await self.rows("pois"):
            legacy_id = int(row["id"])
            if legacy_id in existing_pois:
                continue
            city = city_map.get(int(row["city_id"])) if row.get("city_id") is not None else None
            poi_type = (
                type_map.get(int(row["poi_type_id"]))
                if row.get("poi_type_id") is not None
                else None
            )
            poi = Poi(
                legacy_v1_id=legacy_id,
                city_id=city.id if city else None,
                poi_type_id=poi_type.id if poi_type else None,
                slug=row["slug"],
                name=row["name"],
                names_json=json_value(row.get("names_json"), {}),
                lat=row.get("lat"),
                lng=row.get("lng"),
                short_description=row.get("short_description") or "",
                short_descriptions_json=json_value(row.get("short_descriptions_json"), {}),
                long_description=row.get("long_description") or "",
                source_of_truth=row.get("source_of_truth") or "legacy_v1",
                wikidata_id=row.get("wikidata_id") or "",
                wikipedia_title=row.get("wikipedia_title") or "",
                google_place_id=row.get("google_place_id") or "",
                is_active=bool(row.get("is_active", True)),
                metadata_json=json_value(row.get("metadata_json"), {}),
                created_at=copied_at(row, "created_at"),
                updated_at=copied_at(row, "updated_at", copied_at(row, "created_at")),
            )
            self.target.add(poi)
            poi_count += 1
        await self.target.flush()
        self.counts.update(cities=city_count, poi_types=type_count, pois=poi_count)

    async def _wallets(self) -> None:
        users = await self._existing(User)
        existing = await self._existing(Wallet)
        imported = 0
        for row in await self.rows("wallets"):
            legacy_id = int(row["id"])
            user = users.get(int(row["user_id"]))
            if legacy_id in existing or user is None:
                continue
            self.target.add(
                Wallet(
                    legacy_v1_id=legacy_id,
                    user_id=user.id,
                    currency=row.get("currency") or "EUR",
                    balance_cents=int(row.get("balance_cents") or 0),
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "updated_at", copied_at(row, "created_at")),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["wallets"] = imported

    async def _provider_model(self, provider_code: str, model_code: str) -> tuple[AIProvider, AIModel]:
        provider = await self.target.scalar(select(AIProvider).where(AIProvider.code == provider_code))
        if provider is None:
            provider = AIProvider(code=provider_code, name=provider_code.title(), config_json={})
            self.target.add(provider)
            await self.target.flush()
        model = await self.target.scalar(
            select(AIModel).where(
                AIModel.provider_id == provider.id,
                AIModel.external_id == model_code,
            )
        )
        if model is None:
            model = AIModel(
                provider_id=provider.id,
                external_id=model_code,
                display_name=model_code,
                service_kind="voice" if "realtime" in model_code else "chat",
                adapter_code="legacy_v1",
                lifecycle=Lifecycle.RETIRED,
                enabled=False,
                selectable=False,
                capabilities_json={"historical_only": True},
            )
            self.target.add(model)
            await self.target.flush()
        return provider, model

    async def _prices(self) -> None:
        existing = await self._existing(ProviderPriceSnapshot)
        imported = 0
        for row in await self.rows("price_snapshots"):
            legacy_id = int(row["id"])
            if legacy_id in existing:
                continue
            provider, model = await self._provider_model(row["provider"], row["model"])
            excluded = {"id", "provider", "model", "created_at", "active_from", "fetched_at"}
            pricing = {key: str(value) for key, value in row.items() if key not in excluded}
            self.target.add(
                ProviderPriceSnapshot(
                    legacy_v1_id=legacy_id,
                    provider_id=provider.id,
                    model_id=model.id,
                    currency=row.get("currency") or "USD",
                    pricing_json=pricing,
                    source_url=row.get("source_url") or "legacy:v1",
                    effective_from=copied_at(row, "active_from"),
                    active=False,
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "created_at"),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["price_snapshots"] = imported

    async def _usage(self) -> None:
        existing = await self._existing(UsageEvent)
        users = await self._existing(User)
        prices = await self._existing(ProviderPriceSnapshot)
        imported = 0
        for row in await self.rows("usage_events"):
            legacy_id = int(row["id"])
            if legacy_id in existing:
                continue
            provider, model = await self._provider_model(row["provider"], row["model"])
            user = users.get(int(row["user_id"])) if row.get("user_id") is not None else None
            price = (
                prices.get(int(row["price_snapshot_id"]))
                if row.get("price_snapshot_id") is not None
                else None
            )
            raw = dict(row)
            for key, value in list(raw.items()):
                if isinstance(value, (datetime, Decimal)):
                    raw[key] = str(value)
            raw["metadata_json"] = json_value(row.get("metadata_json"), {})
            self.target.add(
                UsageEvent(
                    legacy_v1_id=legacy_id,
                    user_id=user.id if user else None,
                    provider_id=provider.id,
                    model_id=model.id,
                    price_snapshot_id=price.id if price else None,
                    dedupe_key=row.get("dedupe_key") or f"legacy-v1-{legacy_id}",
                    request_id=row.get("response_id") or row.get("call_id"),
                    interaction_type=row.get("interaction_type") or row.get("endpoint") or "legacy",
                    text_input_tokens=int(row.get("input_tokens") or 0),
                    cached_text_input_tokens=int(row.get("cached_input_tokens") or 0),
                    text_output_tokens=int(row.get("output_tokens") or 0),
                    audio_input_tokens=int(row.get("audio_input_tokens") or 0),
                    audio_output_tokens=int(row.get("audio_output_tokens") or 0),
                    provider_cost_microusd=int(row.get("provider_cost_microusd") or 0),
                    provider_cost_eur_cents=int(row.get("provider_cost_eur_cents") or 0),
                    charged_amount_cents=int(row.get("charged_amount_cents") or 0),
                    gross_margin_cents=int(row.get("gross_margin_cents") or 0),
                    currency=row.get("currency") or "EUR",
                    margin_multiplier=row.get("margin_multiplier") or Decimal("1"),
                    raw_usage_json=raw,
                    status=row.get("status") or "charged",
                    trace_id=f"legacy-v1-usage-{legacy_id}",
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "created_at"),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["usage_events"] = imported

    async def _ledger(self) -> None:
        existing = await self._existing(LedgerEntry)
        users = await self._existing(User)
        wallets = await self._existing(Wallet)
        usage = await self._existing(UsageEvent)
        imported = 0
        for row in await self.rows("ledger_entries"):
            legacy_id = int(row["id"])
            if legacy_id in existing:
                continue
            user = users.get(int(row["user_id"]))
            wallet = wallets.get(int(row["wallet_id"]))
            if user is None or wallet is None:
                self.skipped += 1
                continue
            ref_id = str(row.get("reference_id") or "")
            reference_type = str(row.get("reference_type") or "")
            usage_event = (
                usage.get(int(ref_id))
                if reference_type == "usage_event" and ref_id.isdigit()
                else None
            )
            amount = int(row.get("amount_cents") or 0)
            entry_type = row.get("entry_type") or "legacy"
            if "refund" in entry_type:
                kind = "refund"
            elif amount < 0 or "charge" in entry_type:
                kind = "charge"
            elif "adjust" in entry_type:
                kind = "adjustment"
            else:
                kind = "credit"
            metadata = json_value(row.get("metadata_json"), {})
            metadata["legacy_entry_type"] = entry_type
            self.target.add(
                LedgerEntry(
                    legacy_v1_id=legacy_id,
                    user_id=user.id,
                    wallet_id=wallet.id,
                    usage_event_id=usage_event.id if usage_event else None,
                    kind=kind,
                    amount_cents=amount,
                    currency=wallet.currency,
                    balance_after_cents=int(row.get("balance_after_cents") or 0),
                    description=row.get("description") or entry_type,
                    reference_type=reference_type,
                    reference_id=ref_id,
                    metadata_json=metadata,
                    trace_id=f"legacy-v1-ledger-{legacy_id}",
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "created_at"),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["ledger_entries"] = imported

    async def _top_ups(self) -> None:
        existing = await self._existing(TopUp)
        users = await self._existing(User)
        wallets = await self._existing(Wallet)
        imported = 0
        for row in await self.rows("top_ups"):
            legacy_id = int(row["id"])
            user = users.get(int(row["user_id"]))
            wallet = wallets.get(int(row["wallet_id"]))
            if legacy_id in existing or user is None or wallet is None:
                continue
            self.target.add(
                TopUp(
                    legacy_v1_id=legacy_id,
                    user_id=user.id,
                    wallet_id=wallet.id,
                    amount_cents=int(row.get("amount_cents") or 0),
                    bonus_cents=int(row.get("bonus_cents") or 0),
                    provider=row.get("provider") or "manual",
                    provider_reference=row.get("provider_reference") or "",
                    status=row.get("status") or "completed",
                    metadata_json=json_value(row.get("metadata_json"), {}),
                    completed_at=row.get("completed_at"),
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "completed_at", copied_at(row, "created_at")),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["top_ups"] = imported

    async def _sessions(self) -> None:
        existing = {
            item.legacy_session_id
            for item in (await self.target.scalars(select(LegacyAppSession))).all()
        }
        imported = 0
        for row in await self.rows("app_sessions"):
            session_id = str(row["session_id"])
            if session_id in existing:
                continue
            snapshot = dict(row)
            for key, value in list(snapshot.items()):
                if isinstance(value, (datetime, Decimal)):
                    snapshot[key] = str(value)
                elif key.endswith("_json"):
                    snapshot[key] = json_value(value, None)
            self.target.add(
                LegacyAppSession(
                    legacy_session_id=session_id,
                    legacy_user_id=row.get("user_id"),
                    profile_context=row.get("profile_context") or "",
                    profile_language=row.get("profile_language") or "es",
                    snapshot_json=snapshot,
                    created_at=copied_at(row, "created_at"),
                    updated_at=copied_at(row, "updated_at", copied_at(row, "created_at")),
                )
            )
            imported += 1
        await self.target.flush()
        self.counts["app_sessions"] = imported
