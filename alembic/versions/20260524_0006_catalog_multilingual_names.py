"""add multilingual names to catalog entities

Revision ID: 20260524_0006
Revises: 20260510_0005
Create Date: 2026-05-24 18:25:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260524_0006"
down_revision = "20260510_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cities", sa.Column("names_json", sa.JSON(), nullable=True))
    op.add_column("pois", sa.Column("names_json", sa.JSON(), nullable=True))

    op.execute(
        """
        UPDATE cities
        SET names_json = JSON_OBJECT('local', name)
        WHERE names_json IS NULL
        """
    )
    op.execute(
        """
        UPDATE pois
        SET names_json = JSON_OBJECT('local', name)
        WHERE names_json IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("pois", "names_json")
    op.drop_column("cities", "names_json")
