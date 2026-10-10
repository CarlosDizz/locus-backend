"""add trips

Revision ID: 7d41a9e2c5b8
Revises: 5b2e8f1c4d70
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7d41a9e2c5b8"
down_revision: str | None = "5b2e8f1c4d70"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trips",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("city", sa.String(length=120), nullable=False),
        sa.Column("country", sa.String(length=120), nullable=False),
        sa.Column("starts_at", sa.DateTime(), nullable=True),
        sa.Column("ends_at", sa.DateTime(), nullable=True),
        sa.Column("travelers", sa.String(length=255), nullable=False),
        sa.Column("pace", sa.String(length=255), nullable=False),
        sa.Column("conversation_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index("ix_trips_user_id", "trips", ["user_id"])
    op.create_table(
        "trip_items",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("trip_id", sa.BigInteger(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("time", sa.String(length=5), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("travel", sa.String(length=120), nullable=False),
        sa.Column("price", sa.String(length=40), nullable=False),
        sa.Column("poi_id", sa.BigInteger(), nullable=True),
        sa.Column("booking_json", sa.JSON(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["poi_id"], ["pois.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index("ix_trip_items_trip_id", "trip_items", ["trip_id"])


def downgrade() -> None:
    op.drop_index("ix_trip_items_trip_id", table_name="trip_items")
    op.drop_table("trip_items")
    op.drop_index("ix_trips_user_id", table_name="trips")
    op.drop_table("trips")
