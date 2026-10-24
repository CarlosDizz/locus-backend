"""trip item lodging: qué parada es el alojamiento

Revision ID: d9b2e7c4a1f5
Revises: c4e8f2a6d1b3
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9b2e7c4a1f5"
down_revision: str | None = "c4e8f2a6d1b3"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "trip_items",
        sa.Column("lodging", sa.Boolean(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("trip_items", "lodging")
