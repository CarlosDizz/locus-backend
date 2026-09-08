from datetime import UTC, datetime
from typing import Annotated

from pydantic import PlainSerializer


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def as_utc(value: datetime) -> datetime:
    """Tag a datetime as UTC without shifting its wall-clock value.

    Every timestamp in this codebase is naive-but-UTC: `utc_now()` above
    strips tzinfo on write, and MySQL's `NOW()` (used by `TimestampMixin`'s
    server defaults) is confirmed UTC in this stack. A naive value has no
    timezone marker, so when it reaches the panel as JSON without an offset,
    the browser parses it as local time instead of converting UTC to local
    — every timestamp in the control panel then reads hours off. Attaching
    tzinfo here at the response boundary (never at write time, so the
    database keeps its plain UTC convention) is what lets a JSON serializer
    emit an explicit offset and the browser convert it correctly.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _serialize_utc(value: datetime) -> str:
    return as_utc(value).isoformat()


UtcDatetime = Annotated[datetime, PlainSerializer(_serialize_utc, return_type=str)]
"""Use for any Pydantic response field holding a timestamp from this database.

Only changes serialization (an explicit UTC offset instead of none) — assign
plain naive datetimes to the field as usual, `as_utc` is applied on the way
out. For datetimes placed in a plain `dict` response (no Pydantic model, so
this annotation cannot apply), wrap the value with `as_utc(...)` directly
before it goes in the dict.
"""
