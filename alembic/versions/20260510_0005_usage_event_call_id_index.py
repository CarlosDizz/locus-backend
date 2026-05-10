"""add indexed call_id to usage_events

Revision ID: 20260510_0005
Revises: 20260510_0004
Create Date: 2026-05-10 17:05:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260510_0005"
down_revision = "20260510_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("usage_events", sa.Column("call_id", sa.String(length=32), nullable=True))
    op.create_index("ix_usage_events_call_id", "usage_events", ["call_id"], unique=False)
    op.create_index(
        "ix_usage_events_call_lookup",
        "usage_events",
        ["bill_to_user_id", "source", "interaction_type", "call_id", "id"],
        unique=False,
    )

    op.execute(
        """
        UPDATE usage_events
        SET call_id = JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.call_id'))
        WHERE call_id IS NULL
          AND JSON_EXTRACT(metadata_json, '$.call_id') IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index("ix_usage_events_call_lookup", table_name="usage_events")
    op.drop_index("ix_usage_events_call_id", table_name="usage_events")
    op.drop_column("usage_events", "call_id")
