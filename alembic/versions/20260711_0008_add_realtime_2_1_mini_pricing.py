"""add GPT-Realtime-2.1 mini pricing

Revision ID: 20260711_0008
Revises: 20260524_0007
Create Date: 2026-07-11 00:00:00
"""

from alembic import op


revision = "20260711_0008"
down_revision = "20260524_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO price_snapshots (
            provider, endpoint, model, currency,
            input_per_million, cached_input_per_million, output_per_million,
            text_input_per_million, text_cached_input_per_million, text_output_per_million,
            audio_input_per_million, audio_cached_input_per_million, audio_output_per_million,
            image_input_per_million, image_cached_input_per_million,
            source_url, source_label, raw_source_hash, fetched_at, active_from
        )
        SELECT
            'openai', 'realtime', 'gpt-realtime-2.1-mini', 'USD',
            0.600000, 0.060000, 2.400000,
            0.600000, 0.060000, 2.400000,
            10.000000, 0.300000, 20.000000,
            0.600000, 0.060000,
            'https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini',
            'seed:official_model_doc_2026-07-11',
            'seed:gpt-realtime-2.1-mini:2026-07-11',
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        WHERE NOT EXISTS (
            SELECT 1
            FROM price_snapshots
            WHERE provider = 'openai'
              AND endpoint = 'realtime'
              AND model = 'gpt-realtime-2.1-mini'
        )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM price_snapshots
        WHERE provider = 'openai'
          AND endpoint = 'realtime'
          AND model = 'gpt-realtime-2.1-mini'
          AND raw_source_hash = 'seed:gpt-realtime-2.1-mini:2026-07-11'
        """
    )
