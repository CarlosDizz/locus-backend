"""Stable numeric IDs for the mobile facade, separate from internal primary keys."""

from typing import Any

from sqlalchemy import and_
from sqlalchemy.sql.elements import ColumnElement

MOBILE_ID_OFFSET = 1_000_000_000


def mobile_id(row: Any) -> int:
    legacy_id = getattr(row, "legacy_v1_id", None)
    return int(legacy_id) if legacy_id is not None else MOBILE_ID_OFFSET + int(row.id)


def mobile_id_clause(model: Any, value: int) -> ColumnElement[bool]:
    if value >= MOBILE_ID_OFFSET:
        return and_(model.legacy_v1_id.is_(None), model.id == value - MOBILE_ID_OFFSET)
    return model.legacy_v1_id == value
