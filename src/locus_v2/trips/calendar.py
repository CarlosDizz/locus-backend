"""El plan del viaje como calendario (.ics), para llevarlo al calendario del iPhone.

Es lo que Borea tiene y Locus Guide no (decidido el 2026-10-24, en lugar del
enlace público, que necesita web). Cada franja es un evento de una hora en la
hora local del sitio, sin zona horaria («hora flotante»): las 11:00 del Coliseo
son las 11:00 en Roma, viaje donde viaje el móvil.
"""

from datetime import datetime, timedelta

from locus_v2.trips.models import Trip

_DURATIONS = {"meal": 90, "visit": 120, "walk": 60, "transport": 60, "logistics": 30}


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
        .replace("\r\n", "\\n").replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """Las líneas de un .ics no pasan de 75 octetos; se parten con un espacio."""
    data = line.encode()
    if len(data) <= 75:
        return line
    parts, current = [], b""
    for char in line:
        encoded = char.encode()
        if len(current) + len(encoded) > (75 if not parts else 74):
            parts.append(current.decode())
            current = b""
        current += encoded
    parts.append(current.decode())
    return "\r\n ".join(parts)


def trip_ics(trip: Trip, stamp: datetime) -> str:
    title = f"{trip.city or 'Viaje'}"
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Locus//Borea//ES", "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_escape(title)}",
    ]
    for item in sorted(trip.items, key=lambda i: (i.day, i.time, i.position)):
        hour, minute = (int(x) for x in item.time.split(":"))
        start = datetime(item.day.year, item.day.month, item.day.day, hour, minute)
        end = start + timedelta(minutes=_DURATIONS.get(item.kind, 60))
        description = "\n".join(x for x in (item.note, item.travel) if x)
        lines += [
            "BEGIN:VEVENT",
            f"UID:{item.public_id}@borea.locusguide.es",
            f"DTSTAMP:{stamp.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
            f"SUMMARY:{_escape(item.title)}",
        ]
        if description:
            lines.append(f"DESCRIPTION:{_escape(description)}")
        if item.place and item.place != "-":
            lines.append(f"LOCATION:{_escape(item.place)}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
