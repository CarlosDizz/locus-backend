"""session app version: con qué versión de la app entra cada sesión

Revision ID: b5e1c8d3f7a2
Revises: d9b2e7c4a1f5
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b5e1c8d3f7a2"
down_revision: str | None = "d9b2e7c4a1f5"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user_sessions", sa.Column("app_version", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("user_sessions", "app_version")
