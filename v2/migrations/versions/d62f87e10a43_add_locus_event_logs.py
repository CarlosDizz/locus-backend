"""add locus event logs

Revision ID: d62f87e10a43
Revises: 7af58e4c90d3
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d62f87e10a43"
down_revision: str | None = "7af58e4c90d3"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "locus_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("level", sa.String(length=10), nullable=False),
        sa.Column("service", sa.String(length=40), nullable=False),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("event", sa.String(length=120), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("voice_session_id", sa.BigInteger(), nullable=True),
        sa.Column("error_type", sa.String(length=160), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["voice_session_id"], ["voice_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_locus_logs_service_created", "locus_logs", ["service", "created_at"]
    )
    op.create_index(
        "ix_locus_logs_level_created", "locus_logs", ["level", "created_at"]
    )
    op.create_index(
        "ix_locus_logs_trace_created", "locus_logs", ["trace_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_locus_logs_trace_created", table_name="locus_logs")
    op.drop_index("ix_locus_logs_level_created", table_name="locus_logs")
    op.drop_index("ix_locus_logs_service_created", table_name="locus_logs")
    op.drop_table("locus_logs")
