"""add trip item map

Revision ID: a4d8e2c6f913
Revises: 9c3e5a7f1b26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d8e2c6f913"
down_revision: str | None = "9c3e5a7f1b26"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trip_items", sa.Column("map_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("trip_items", "map_json")
