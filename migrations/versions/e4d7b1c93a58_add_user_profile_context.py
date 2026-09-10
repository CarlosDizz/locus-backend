"""Move the traveller profile from the session to the user.

`profile_context` used to live on `map_sessions`, keyed by a session id the app
generates at random per device (`LOCUS-` + eight characters). A new phone or a
cleared storage started a new session, so whatever the traveller had written
about themselves was gone — every row in production had the column empty.

Revision ID: e4d7b1c93a58
Revises: c81a0042d610
"""

import sqlalchemy as sa
from alembic import op

revision: str = "e4d7b1c93a58"
down_revision: str | None = "c81a0042d610"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # VARCHAR y no TEXT: MySQL rechaza un DEFAULT en columnas TEXT
    # ("BLOB, TEXT, GEOMETRY or JSON column can't have a default value"), y sin
    # default habría que rellenar cada fila existente antes de poner NOT NULL.
    op.add_column(
        "users",
        sa.Column("profile_context", sa.String(length=1000), nullable=False, server_default=""),
    )
    op.add_column(
        "users",
        sa.Column("preferred_name", sa.String(length=160), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("users", "preferred_name")
    op.drop_column("users", "profile_context")
