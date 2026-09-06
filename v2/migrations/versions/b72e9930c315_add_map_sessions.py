"""add map sessions

Revision ID: b72e9930c315
Revises: f3a6c9d21b74
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b72e9930c315"
down_revision: str | None = "f3a6c9d21b74"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "map_sessions",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("profile_context", sa.Text(), nullable=False),
        sa.Column("profile_language", sa.String(length=16), nullable=False),
        sa.Column("profile_preferences_json", sa.JSON(), nullable=False),
        sa.Column("lat", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("lng", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("active_poi_json", sa.JSON(), nullable=True),
        sa.Column("nearby_pois_json", sa.JSON(), nullable=False),
        sa.Column("memory_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_map_sessions_user_id", "map_sessions", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_map_sessions_user_id", table_name="map_sessions")
    op.drop_table("map_sessions")
