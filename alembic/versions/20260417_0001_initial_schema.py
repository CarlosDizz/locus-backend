"""initial schema

Revision ID: 20260417_0001
Revises: None
Create Date: 2026-04-17 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260417_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=False)

    op.create_table(
        "auth_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("token_hash", name="uq_auth_tokens_token_hash"),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"], unique=False)
    op.create_index("ix_auth_tokens_token_hash", "auth_tokens", ["token_hash"], unique=False)

    op.create_table(
        "wallets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="EUR"),
        sa.Column("balance_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_wallets_user_id"),
    )
    op.create_index("ix_wallets_user_id", "wallets", ["user_id"], unique=False)

    op.create_table(
        "ledger_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("wallets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entry_type", sa.String(length=64), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("balance_after_cents", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("reference_type", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("reference_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ledger_entries_wallet_id", "ledger_entries", ["wallet_id"], unique=False)
    op.create_index("ix_ledger_entries_user_id", "ledger_entries", ["user_id"], unique=False)
    op.create_index("ix_ledger_entries_entry_type", "ledger_entries", ["entry_type"], unique=False)

    op.create_table(
        "top_ups",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("wallets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("bonus_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column("provider_reference", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_top_ups_user_id", "top_ups", ["user_id"], unique=False)
    op.create_index("ix_top_ups_wallet_id", "top_ups", ["wallet_id"], unique=False)
    op.create_index("ix_top_ups_status", "top_ups", ["status"], unique=False)

    op.create_table(
        "price_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="USD"),
        sa.Column("input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("cached_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("output_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("active_from", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_price_snapshots_provider", "price_snapshots", ["provider"], unique=False)
    op.create_index("ix_price_snapshots_endpoint", "price_snapshots", ["endpoint"], unique=False)
    op.create_index("ix_price_snapshots_model", "price_snapshots", ["model"], unique=False)

    op.create_table(
        "cities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("country_code", sa.String(length=8), nullable=False, server_default=""),
        sa.Column("lat", sa.Numeric(10, 7), nullable=True),
        sa.Column("lng", sa.Numeric(10, 7), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("slug", name="uq_cities_slug"),
    )
    op.create_index("ix_cities_slug", "cities", ["slug"], unique=False)
    op.create_index("ix_cities_name", "cities", ["name"], unique=False)

    op.create_table(
        "poi_types",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("code", name="uq_poi_types_code"),
        sa.UniqueConstraint("name", name="uq_poi_types_name"),
    )
    op.create_index("ix_poi_types_code", "poi_types", ["code"], unique=False)

    op.create_table(
        "pois",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("city_id", sa.Integer(), sa.ForeignKey("cities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("poi_type_id", sa.Integer(), sa.ForeignKey("poi_types.id", ondelete="SET NULL"), nullable=True),
        sa.Column("slug", sa.String(length=160), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("lat", sa.Numeric(10, 7), nullable=True),
        sa.Column("lng", sa.Numeric(10, 7), nullable=True),
        sa.Column("short_description", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("long_description", sa.Text(), nullable=False),
        sa.Column("source_of_truth", sa.String(length=64), nullable=False, server_default="wikidata"),
        sa.Column("wikidata_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("wikipedia_title", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("google_place_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("city_id", "slug", name="uq_pois_city_slug"),
    )
    op.create_index("ix_pois_city_id", "pois", ["city_id"], unique=False)
    op.create_index("ix_pois_poi_type_id", "pois", ["poi_type_id"], unique=False)
    op.create_index("ix_pois_slug", "pois", ["slug"], unique=False)
    op.create_index("ix_pois_name", "pois", ["name"], unique=False)
    op.create_index("ix_pois_wikidata_id", "pois", ["wikidata_id"], unique=False)
    op.create_index("ix_pois_google_place_id", "pois", ["google_place_id"], unique=False)

    op.create_table(
        "app_sessions",
        sa.Column("session_id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("profile_context", sa.Text(), nullable=False),
        sa.Column("profile_language", sa.String(length=16), nullable=False, server_default="es"),
        sa.Column("profile_preferences_json", sa.JSON(), nullable=False),
        sa.Column("lat", sa.Numeric(10, 7), nullable=True),
        sa.Column("lng", sa.Numeric(10, 7), nullable=True),
        sa.Column("active_poi_json", sa.JSON(), nullable=True),
        sa.Column("nearby_pois_json", sa.JSON(), nullable=False),
        sa.Column("memory_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_app_sessions_user_id", "app_sessions", ["user_id"], unique=False)

    op.create_table(
        "usage_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("response_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("image_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_cost_microusd", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("charged_amount_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="EUR"),
        sa.Column("margin_multiplier", sa.Numeric(6, 3), nullable=False, server_default="1.150"),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_usage_events_user_id", "usage_events", ["user_id"], unique=False)
    op.create_index("ix_usage_events_session_id", "usage_events", ["session_id"], unique=False)
    op.create_index("ix_usage_events_provider", "usage_events", ["provider"], unique=False)
    op.create_index("ix_usage_events_endpoint", "usage_events", ["endpoint"], unique=False)
    op.create_index("ix_usage_events_model", "usage_events", ["model"], unique=False)

    poi_types_table = sa.table(
        "poi_types",
        sa.column("code", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.String()),
    )
    op.bulk_insert(
        poi_types_table,
        [
            {"code": "monument", "name": "Monumento", "description": "Monumentos y memoriales"},
            {"code": "museum", "name": "Museo", "description": "Museos y pinacotecas"},
            {"code": "church", "name": "Iglesia", "description": "Iglesias, catedrales y basílicas"},
            {"code": "square", "name": "Plaza", "description": "Plazas y espacios urbanos"},
            {"code": "building", "name": "Edificio", "description": "Edificios históricos o emblemáticos"},
            {"code": "archaeological_site", "name": "Yacimiento", "description": "Ruinas y yacimientos arqueológicos"},
        ],
    )

    prices_table = sa.table(
        "price_snapshots",
        sa.column("provider", sa.String()),
        sa.column("endpoint", sa.String()),
        sa.column("model", sa.String()),
        sa.column("currency", sa.String()),
        sa.column("input_per_million", sa.Numeric(12, 6)),
        sa.column("cached_input_per_million", sa.Numeric(12, 6)),
        sa.column("output_per_million", sa.Numeric(12, 6)),
    )
    op.bulk_insert(
        prices_table,
        [
            {
                "provider": "openai",
                "endpoint": "responses",
                "model": "gpt-5.2",
                "currency": "USD",
                "input_per_million": 1.75,
                "cached_input_per_million": 0.175,
                "output_per_million": 14.0,
            },
            {
                "provider": "openai",
                "endpoint": "responses",
                "model": "gpt-5-mini",
                "currency": "USD",
                "input_per_million": 0.25,
                "cached_input_per_million": 0.025,
                "output_per_million": 2.0,
            },
            {
                "provider": "openai",
                "endpoint": "realtime",
                "model": "gpt-realtime",
                "currency": "USD",
                "input_per_million": 4.0,
                "cached_input_per_million": 0.4,
                "output_per_million": 16.0,
            },
            {
                "provider": "openai",
                "endpoint": "realtime",
                "model": "gpt-realtime-mini",
                "currency": "USD",
                "input_per_million": 0.6,
                "cached_input_per_million": 0.06,
                "output_per_million": 2.4,
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_usage_events_model", table_name="usage_events")
    op.drop_index("ix_usage_events_endpoint", table_name="usage_events")
    op.drop_index("ix_usage_events_provider", table_name="usage_events")
    op.drop_index("ix_usage_events_session_id", table_name="usage_events")
    op.drop_index("ix_usage_events_user_id", table_name="usage_events")
    op.drop_table("usage_events")

    op.drop_index("ix_app_sessions_user_id", table_name="app_sessions")
    op.drop_table("app_sessions")

    op.drop_index("ix_pois_google_place_id", table_name="pois")
    op.drop_index("ix_pois_wikidata_id", table_name="pois")
    op.drop_index("ix_pois_name", table_name="pois")
    op.drop_index("ix_pois_slug", table_name="pois")
    op.drop_index("ix_pois_poi_type_id", table_name="pois")
    op.drop_index("ix_pois_city_id", table_name="pois")
    op.drop_table("pois")

    op.drop_index("ix_poi_types_code", table_name="poi_types")
    op.drop_table("poi_types")

    op.drop_index("ix_cities_name", table_name="cities")
    op.drop_index("ix_cities_slug", table_name="cities")
    op.drop_table("cities")

    op.drop_index("ix_price_snapshots_model", table_name="price_snapshots")
    op.drop_index("ix_price_snapshots_endpoint", table_name="price_snapshots")
    op.drop_index("ix_price_snapshots_provider", table_name="price_snapshots")
    op.drop_table("price_snapshots")

    op.drop_index("ix_top_ups_status", table_name="top_ups")
    op.drop_index("ix_top_ups_wallet_id", table_name="top_ups")
    op.drop_index("ix_top_ups_user_id", table_name="top_ups")
    op.drop_table("top_ups")

    op.drop_index("ix_ledger_entries_entry_type", table_name="ledger_entries")
    op.drop_index("ix_ledger_entries_user_id", table_name="ledger_entries")
    op.drop_index("ix_ledger_entries_wallet_id", table_name="ledger_entries")
    op.drop_table("ledger_entries")

    op.drop_index("ix_wallets_user_id", table_name="wallets")
    op.drop_table("wallets")

    op.drop_index("ix_auth_tokens_token_hash", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_user_id", table_name="auth_tokens")
    op.drop_table("auth_tokens")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
