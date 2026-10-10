"""add call reviews

Revision ID: 5b2e8f1c4d70
Revises: e4d7b1c93a58
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5b2e8f1c4d70"
down_revision: str | None = "e4d7b1c93a58"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "call_reviews",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("call_id", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("poi_id", sa.BigInteger(), nullable=False),
        sa.Column("place_vote", sa.Boolean(), nullable=True),
        sa.Column("place_comment", sa.Text(), nullable=True),
        sa.Column("place_comment_status", sa.String(length=16), nullable=True),
        sa.Column("guide_stars", sa.SmallInteger(), nullable=True),
        sa.Column("guide_tags", sa.JSON(), nullable=False),
        sa.Column("guide_comment", sa.Text(), nullable=True),
        sa.Column("moderated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["poi_id"], ["pois.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "user_id"),
    )
    op.create_index("ix_call_reviews_user_id", "call_reviews", ["user_id"])
    op.create_index("ix_call_reviews_poi_created", "call_reviews", ["poi_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_call_reviews_poi_created", table_name="call_reviews")
    op.drop_index("ix_call_reviews_user_id", table_name="call_reviews")
    op.drop_table("call_reviews")
