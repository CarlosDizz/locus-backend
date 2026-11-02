"""Recordatorios del viaje: dos días antes y cada mañana (notifications/reminders.py)."""

from datetime import date, datetime

from locus_v2.notifications import reminders
from locus_v2.trips.models import Trip, TripItem


def _tokio() -> Trip:
    return Trip(
        public_id="t", city="Tokio", starts_at=datetime(2030, 9, 1, 12),
        ends_at=datetime(2030, 9, 5, 12), center_json={"lat": 35.68, "lng": 139.69},
        items=[
            TripItem(day=date(2030, 9, 2), time="10:00", title="Shibuya", kind="walk",
                     position=0, lodging=False),
            TripItem(day=date(2030, 9, 2), time="09:00", title="Meiji-jingū", kind="visit",
                     position=1, lodging=False),
            TripItem(day=date(2030, 9, 2), time="15:00", title="Hotel", kind="logistics",
                     position=2, lodging=True),
        ],
    )


def test_two_days_before_at_six_in_the_evening_local_time() -> None:
    # Tokio va 9 horas por delante: las 18:00 allí son las 09:00 UTC.
    assert reminders.due(_tokio(), datetime(2030, 8, 30, 9, 5)) == ("pre_trip", date(2030, 8, 30))
    assert reminders.due(_tokio(), datetime(2030, 8, 30, 12)) is None


def test_every_morning_with_the_stops_of_the_day() -> None:
    trip = _tokio()
    assert reminders.due(trip, datetime(2030, 9, 1, 23, 10)) == ("today", date(2030, 9, 2))
    assert reminders.items_of(trip, date(2030, 9, 2)) == ["Meiji-jingū", "Shibuya"]
    # Un día sin paradas no tiene aviso.
    assert reminders.due(trip, datetime(2030, 9, 2, 23, 10)) is None


def test_the_stops_are_joined_in_the_language_of_each_person() -> None:
    assert reminders.join_items(["A", "B", "C", "D"], "es") == "A, B y C"
    assert reminders.join_items(["A", "B"], "fr") == "A et B"
