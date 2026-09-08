"""add cached audio and image usage modalities

Revision ID: a91c2e4b7f30
Revises: d62f87e10a43
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a91c2e4b7f30"
down_revision: str | None = "d62f87e10a43"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "usage_events",
        sa.Column(
            "cached_audio_input_tokens",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "usage_events",
        sa.Column("image_input_tokens", sa.BigInteger(), server_default="0", nullable=False),
    )
    op.add_column(
        "usage_events",
        sa.Column(
            "cached_image_input_tokens",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("usage_events", "cached_image_input_tokens")
    op.drop_column("usage_events", "image_input_tokens")
    op.drop_column("usage_events", "cached_audio_input_tokens")
