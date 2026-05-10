"""fix realtime mini pricing buckets

Revision ID: 20260510_0004
Revises: 20260510_0003
Create Date: 2026-05-10 00:30:00
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260510_0004"
down_revision: Union[str, Sequence[str], None] = "20260510_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE price_snapshots
        SET
            text_input_per_million = CASE WHEN text_input_per_million = 0 THEN input_per_million ELSE text_input_per_million END,
            text_cached_input_per_million = CASE WHEN text_cached_input_per_million = 0 THEN cached_input_per_million ELSE text_cached_input_per_million END,
            text_output_per_million = CASE WHEN text_output_per_million = 0 THEN output_per_million ELSE text_output_per_million END,
            audio_input_per_million = CASE WHEN audio_input_per_million = 0 THEN input_per_million ELSE audio_input_per_million END,
            audio_cached_input_per_million = CASE WHEN audio_cached_input_per_million = 0 THEN cached_input_per_million ELSE audio_cached_input_per_million END,
            audio_output_per_million = CASE WHEN audio_output_per_million = 0 THEN output_per_million ELSE audio_output_per_million END,
            source_label = CASE
                WHEN source_label = '' OR source_label = 'seed:initial_schema' THEN 'seed:realtime_mini_normalized'
                ELSE source_label
            END
        WHERE provider = 'openai'
          AND endpoint = 'realtime'
          AND model = 'gpt-realtime-mini'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE price_snapshots
        SET
            audio_input_per_million = 0.000000,
            audio_cached_input_per_million = 0.000000,
            audio_output_per_million = 0.000000,
            source_label = CASE
                WHEN source_label = 'seed:realtime_mini_normalized' THEN 'seed:initial_schema'
                ELSE source_label
            END
        WHERE provider = 'openai'
          AND endpoint = 'realtime'
          AND model = 'gpt-realtime-mini'
        """
    )
