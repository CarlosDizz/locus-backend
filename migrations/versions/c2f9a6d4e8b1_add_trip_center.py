"""add trip center

Revision ID: c2f9a6d4e8b1
Revises: b7e1f4a9c2d5
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2f9a6d4e8b1"
down_revision: str | None = "b7e1f4a9c2d5"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("center_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("trips", "center_json")
