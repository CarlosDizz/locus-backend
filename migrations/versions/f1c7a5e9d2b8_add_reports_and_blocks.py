"""add content reports and user blocks

Revision ID: f1c7a5e9d2b8
Revises: e8b4d1f6a3c7
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f1c7a5e9d2b8"
down_revision: str | None = "e8b4d1f6a3c7"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "content_reports",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("reporter_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("target", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["reporter_id"], ["users.id"], name=op.f("fk_content_reports_reporter_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_content_reports")),
    )
    op.create_index(op.f("ix_content_reports_reporter_id"), "content_reports", ["reporter_id"])
    op.create_table(
        "user_blocks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("blocked_user_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_user_blocks_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["blocked_user_id"], ["users.id"], name=op.f("fk_user_blocks_blocked_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_blocks")),
        sa.UniqueConstraint("user_id", "blocked_user_id", name=op.f("uq_user_blocks_user_id")),
    )
    op.create_index(op.f("ix_user_blocks_user_id"), "user_blocks", ["user_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_user_blocks_user_id"), table_name="user_blocks")
    op.drop_table("user_blocks")
    op.drop_index(op.f("ix_content_reports_reporter_id"), table_name="content_reports")
    op.drop_table("content_reports")
