"""add trip routes

Revision ID: b7e1f4a9c2d5
Revises: a4d8e2c6f913
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e1f4a9c2d5"
down_revision: str | None = "a4d8e2c6f913"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("routes_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("trips", "routes_json")
