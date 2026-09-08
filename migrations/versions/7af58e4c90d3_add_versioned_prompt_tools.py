"""add versioned prompt tools

Revision ID: 7af58e4c90d3
Revises: 3c0f4f5a81b2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7af58e4c90d3"
down_revision: str | None = "3c0f4f5a81b2"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_tools",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=False),
        sa.Column("handler_code", sa.String(length=100), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("requires_approval", sa.Boolean(), nullable=False),
        sa.Column("service_kinds_json", sa.JSON(), nullable=False),
        sa.Column("schema_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.add_column(
        "prompt_versions",
        sa.Column("tools_json", sa.JSON(), server_default=sa.text("('[]')"), nullable=False),
    )
    op.add_column(
        "prompt_versions",
        sa.Column(
            "runtime_config_json", sa.JSON(), server_default=sa.text("('{}')"), nullable=False
        ),
    )
    op.add_column(
        "ai_models",
        sa.Column(
            "runtime_defaults_json", sa.JSON(), server_default=sa.text("('{}')"), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_column("ai_models", "runtime_defaults_json")
    op.drop_column("prompt_versions", "runtime_config_json")
    op.drop_column("prompt_versions", "tools_json")
    op.drop_table("ai_tools")
