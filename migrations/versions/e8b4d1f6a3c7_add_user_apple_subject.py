"""add user apple subject

Revision ID: e8b4d1f6a3c7
Revises: d5a3c8e7b2f4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8b4d1f6a3c7"
down_revision: str | None = "d5a3c8e7b2f4"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("apple_subject", sa.String(length=255), nullable=True))
    op.create_index(op.f("ix_users_apple_subject"), "users", ["apple_subject"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_users_apple_subject"), table_name="users")
    op.drop_column("users", "apple_subject")
