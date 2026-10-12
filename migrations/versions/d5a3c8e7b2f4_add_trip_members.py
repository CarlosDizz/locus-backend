"""add trip members

Revision ID: d5a3c8e7b2f4
Revises: c2f9a6d4e8b1
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5a3c8e7b2f4"
down_revision: str | None = "c2f9a6d4e8b1"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("invite_code", sa.String(length=16), nullable=True))
    op.create_unique_constraint(op.f("uq_trips_invite_code"), "trips", ["invite_code"])
    op.create_table(
        "trip_members",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trip_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["trip_id"], ["trips.id"], name=op.f("fk_trip_members_trip_id_trips"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_trip_members_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_trip_members")),
        sa.UniqueConstraint("trip_id", "user_id", name=op.f("uq_trip_members_trip_id")),
    )
    op.create_index(op.f("ix_trip_members_trip_id"), "trip_members", ["trip_id"])
    op.create_index(op.f("ix_trip_members_user_id"), "trip_members", ["user_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_trip_members_user_id"), table_name="trip_members")
    op.drop_index(op.f("ix_trip_members_trip_id"), table_name="trip_members")
    op.drop_table("trip_members")
    op.drop_constraint(op.f("uq_trips_invite_code"), "trips", type_="unique")
    op.drop_column("trips", "invite_code")
