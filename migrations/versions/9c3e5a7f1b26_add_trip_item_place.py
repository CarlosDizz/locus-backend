"""add trip item place

Revision ID: 9c3e5a7f1b26
Revises: 7d41a9e2c5b8
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9c3e5a7f1b26"
down_revision: str | None = "7d41a9e2c5b8"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "trip_items",
        sa.Column("place", sa.String(length=255), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("trip_items", "place")
