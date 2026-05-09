"""google auth support

Revision ID: 20260509_0002
Revises: 20260417_0001
Create Date: 2026-05-09 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260509_0002"
down_revision: Union[str, Sequence[str], None] = "20260417_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("auth_provider", sa.String(length=32), nullable=False, server_default="local"))
    op.add_column("users", sa.Column("google_sub", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("avatar_url", sa.String(length=1024), nullable=False, server_default=""))
    op.alter_column("users", "password_hash", existing_type=sa.String(length=512), server_default="")

    op.create_index("ix_users_auth_provider", "users", ["auth_provider"], unique=False)
    op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_index("ix_users_auth_provider", table_name="users")

    op.alter_column("users", "password_hash", existing_type=sa.String(length=512), server_default=None)
    op.drop_column("users", "avatar_url")
    op.drop_column("users", "google_sub")
    op.drop_column("users", "auth_provider")
