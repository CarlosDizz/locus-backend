"""borea: apps per trip, signup app, usage app, cent carry, climate, recommendations, referral clicks

Revision ID: a7d3e5f1b9c2
Revises: f1c7a5e9d2b8
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d3e5f1b9c2"
down_revision: str | None = "f1c7a5e9d2b8"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("apps", sa.Integer(), server_default="1", nullable=False))
    op.add_column("trips", sa.Column("climate_json", sa.JSON(), nullable=True))
    op.add_column("trips", sa.Column("recommendations_json", sa.JSON(), nullable=True))
    op.add_column(
        "users", sa.Column("signup_app", sa.String(length=20), server_default="guide", nullable=False)
    )
    op.add_column(
        "usage_events", sa.Column("app", sa.String(length=20), server_default="guide", nullable=False)
    )
    op.add_column(
        "wallets", sa.Column("carry_microeur", sa.BigInteger(), server_default="0", nullable=False)
    )
    op.create_table(
        "referral_clicks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("app", sa.String(length=20), nullable=False),
        sa.Column("program", sa.String(length=40), nullable=False),
        sa.Column("trip_public_id", sa.String(length=36), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_referral_clicks_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_referral_clicks")),
    )
    op.create_index(op.f("ix_referral_clicks_user_id"), "referral_clicks", ["user_id"])
    op.create_index(op.f("ix_referral_clicks_program"), "referral_clicks", ["program"])


def downgrade() -> None:
    op.drop_index(op.f("ix_referral_clicks_program"), table_name="referral_clicks")
    op.drop_index(op.f("ix_referral_clicks_user_id"), table_name="referral_clicks")
    op.drop_table("referral_clicks")
    op.drop_column("wallets", "carry_microeur")
    op.drop_column("usage_events", "app")
    op.drop_column("users", "signup_app")
    op.drop_column("trips", "recommendations_json")
    op.drop_column("trips", "climate_json")
    op.drop_column("trips", "apps")
