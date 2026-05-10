"""billing hardening and price refresh support

Revision ID: 20260510_0003
Revises: 20260509_0002
Create Date: 2026-05-10 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260510_0003"
down_revision: Union[str, Sequence[str], None] = "20260509_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("price_snapshots", sa.Column("text_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("text_cached_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("text_output_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("audio_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("audio_cached_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("audio_output_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("image_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("image_cached_input_per_million", sa.Numeric(12, 6), nullable=False, server_default="0"))
    op.add_column("price_snapshots", sa.Column("source_url", sa.String(length=2048), nullable=False, server_default=""))
    op.add_column("price_snapshots", sa.Column("source_label", sa.String(length=128), nullable=False, server_default=""))
    op.add_column("price_snapshots", sa.Column("raw_source_hash", sa.String(length=128), nullable=False, server_default=""))
    op.add_column("price_snapshots", sa.Column("fetched_at", sa.DateTime(), nullable=True))

    op.add_column("usage_events", sa.Column("bill_to_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True))
    op.add_column("usage_events", sa.Column("interaction_type", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("usage_events", sa.Column("source", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("usage_events", sa.Column("dedupe_key", sa.String(length=128), nullable=True))
    op.add_column("usage_events", sa.Column("price_snapshot_id", sa.Integer(), sa.ForeignKey("price_snapshots.id", ondelete="SET NULL"), nullable=True))
    op.add_column("usage_events", sa.Column("provider_cost_eur_cents", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("usage_events", sa.Column("gross_margin_cents", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("usage_events", sa.Column("status", sa.String(length=32), nullable=False, server_default="charged"))

    op.create_index("ix_usage_events_bill_to_user_id", "usage_events", ["bill_to_user_id"], unique=False)
    op.create_index("ix_usage_events_interaction_type", "usage_events", ["interaction_type"], unique=False)
    op.create_index("ix_usage_events_source", "usage_events", ["source"], unique=False)
    op.create_index("ix_usage_events_dedupe_key", "usage_events", ["dedupe_key"], unique=True)
    op.create_index("ix_usage_events_price_snapshot_id", "usage_events", ["price_snapshot_id"], unique=False)
    op.create_index("ix_usage_events_status", "usage_events", ["status"], unique=False)

    op.execute(
        """
        UPDATE price_snapshots
        SET
            text_input_per_million = input_per_million,
            text_cached_input_per_million = cached_input_per_million,
            text_output_per_million = output_per_million,
            source_label = 'seed:initial_schema',
            fetched_at = COALESCE(fetched_at, CURRENT_TIMESTAMP)
        """
    )
    op.execute(
        """
        UPDATE price_snapshots
        SET
            audio_input_per_million = 32.000000,
            audio_cached_input_per_million = 0.400000,
            audio_output_per_million = 64.000000,
            image_input_per_million = 5.000000,
            image_cached_input_per_million = 0.500000
        WHERE model = 'gpt-realtime'
        """
    )

    op.execute(
        """
        INSERT INTO price_snapshots (
            provider, endpoint, model, currency,
            input_per_million, cached_input_per_million, output_per_million,
            text_input_per_million, text_cached_input_per_million, text_output_per_million,
            audio_input_per_million, audio_cached_input_per_million, audio_output_per_million,
            image_input_per_million, image_cached_input_per_million,
            source_url, source_label, raw_source_hash, fetched_at
        )
        SELECT
            'openai', 'responses', 'gpt-5.4-mini', 'USD',
            0.750000, 0.075000, 4.500000,
            0.750000, 0.075000, 4.500000,
            0.000000, 0.000000, 0.000000,
            0.000000, 0.000000,
            'https://developers.openai.com/api/docs/models/gpt-5.4-mini',
            'openai_model_doc',
            '',
            CURRENT_TIMESTAMP
        WHERE NOT EXISTS (
            SELECT 1 FROM price_snapshots WHERE provider = 'openai' AND endpoint = 'responses' AND model = 'gpt-5.4-mini'
        )
        """
    )
    op.execute(
        """
        INSERT INTO price_snapshots (
            provider, endpoint, model, currency,
            input_per_million, cached_input_per_million, output_per_million,
            text_input_per_million, text_cached_input_per_million, text_output_per_million,
            audio_input_per_million, audio_cached_input_per_million, audio_output_per_million,
            image_input_per_million, image_cached_input_per_million,
            source_url, source_label, raw_source_hash, fetched_at
        )
        SELECT
            'openai', 'realtime', 'gpt-realtime-2', 'USD',
            4.000000, 0.400000, 24.000000,
            4.000000, 0.400000, 24.000000,
            32.000000, 0.400000, 64.000000,
            5.000000, 0.500000,
            'https://developers.openai.com/api/docs/models/gpt-realtime-2',
            'openai_model_doc',
            '',
            CURRENT_TIMESTAMP
        WHERE NOT EXISTS (
            SELECT 1 FROM price_snapshots WHERE provider = 'openai' AND endpoint = 'realtime' AND model = 'gpt-realtime-2'
        )
        """
    )

    op.execute("UPDATE usage_events SET bill_to_user_id = user_id WHERE bill_to_user_id IS NULL")
    op.execute("UPDATE usage_events SET interaction_type = endpoint WHERE interaction_type = ''")
    op.execute(
        """
        UPDATE usage_events
        SET source = COALESCE(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.source')), ''), endpoint)
        WHERE source = ''
        """
    )
    op.execute(
        """
        UPDATE usage_events
        SET
            provider_cost_eur_cents = ROUND(provider_cost_microusd / 10000),
            gross_margin_cents = charged_amount_cents - ROUND(provider_cost_microusd / 10000),
            status = 'charged'
        """
    )
    op.alter_column(
        "usage_events",
        "margin_multiplier",
        existing_type=sa.Numeric(6, 3),
        server_default="1.800",
    )


def downgrade() -> None:
    op.alter_column(
        "usage_events",
        "margin_multiplier",
        existing_type=sa.Numeric(6, 3),
        server_default="1.150",
    )
    op.drop_index("ix_usage_events_status", table_name="usage_events")
    op.drop_index("ix_usage_events_price_snapshot_id", table_name="usage_events")
    op.drop_index("ix_usage_events_dedupe_key", table_name="usage_events")
    op.drop_index("ix_usage_events_source", table_name="usage_events")
    op.drop_index("ix_usage_events_interaction_type", table_name="usage_events")
    op.drop_index("ix_usage_events_bill_to_user_id", table_name="usage_events")

    op.drop_column("usage_events", "status")
    op.drop_column("usage_events", "gross_margin_cents")
    op.drop_column("usage_events", "provider_cost_eur_cents")
    op.drop_column("usage_events", "price_snapshot_id")
    op.drop_column("usage_events", "dedupe_key")
    op.drop_column("usage_events", "source")
    op.drop_column("usage_events", "interaction_type")
    op.drop_column("usage_events", "bill_to_user_id")

    op.drop_column("price_snapshots", "fetched_at")
    op.drop_column("price_snapshots", "raw_source_hash")
    op.drop_column("price_snapshots", "source_label")
    op.drop_column("price_snapshots", "source_url")
    op.drop_column("price_snapshots", "image_cached_input_per_million")
    op.drop_column("price_snapshots", "image_input_per_million")
    op.drop_column("price_snapshots", "audio_output_per_million")
    op.drop_column("price_snapshots", "audio_cached_input_per_million")
    op.drop_column("price_snapshots", "audio_input_per_million")
    op.drop_column("price_snapshots", "text_output_per_million")
    op.drop_column("price_snapshots", "text_cached_input_per_million")
    op.drop_column("price_snapshots", "text_input_per_million")
