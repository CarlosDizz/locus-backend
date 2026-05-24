"""add multilingual short descriptions to pois

Revision ID: 20260524_0007
Revises: 20260524_0006
Create Date: 2026-05-24 18:45:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260524_0007"
down_revision = "20260524_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pois", sa.Column("short_descriptions_json", sa.JSON(), nullable=True))
    op.execute(
        """
        UPDATE pois
        SET short_descriptions_json = JSON_OBJECT('local', short_description)
        WHERE short_descriptions_json IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("pois", "short_descriptions_json")
