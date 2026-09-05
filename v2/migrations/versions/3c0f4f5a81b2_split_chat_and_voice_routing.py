"""split chat and voice routing

Revision ID: 3c0f4f5a81b2
Revises: 8c31d4f9a621
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3c0f4f5a81b2"
down_revision: str | None = "8c31d4f9a621"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "prompt_definitions",
        sa.Column("service_kind", sa.String(length=20), server_default="voice", nullable=False),
    )
    op.add_column(
        "ai_routing_profiles",
        sa.Column("service_kind", sa.String(length=20), server_default="voice", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("ai_routing_profiles", "service_kind")
    op.drop_column("prompt_definitions", "service_kind")
