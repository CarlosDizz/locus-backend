"""Serialize Google Play purchases across users, workers and retries."""

import sqlalchemy as sa
from alembic import op

revision = "c81a0042d610"
down_revision = "b72e9930c315"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("top_ups", "provider_reference", existing_type=sa.String(128),
                    type_=sa.String(2048), existing_nullable=False)
    op.add_column("top_ups", sa.Column("purchase_dedupe_key", sa.String(64), nullable=True))
    op.execute("UPDATE top_ups SET purchase_dedupe_key = SHA2(TRIM(provider_reference), 256) "
               "WHERE provider = 'google_play' AND TRIM(provider_reference) <> ''")
    # Existing duplicates deliberately fail the migration rather than deleting financial rows.
    op.create_unique_constraint("uq_top_ups_purchase_dedupe_key", "top_ups", ["purchase_dedupe_key"])


def downgrade() -> None:
    op.drop_constraint("uq_top_ups_purchase_dedupe_key", "top_ups", type_="unique")
    op.drop_column("top_ups", "purchase_dedupe_key")
    # Keep the wider reference column: truncating purchase tokens would destroy audit data.
