"""Llamada en itinerario: qué tenéis cerca, a qué lado, y cuándo contarlo.

El guía no ve por dónde vais. La app le manda la ubicación cada pocos segundos y
esto decide, con funciones puras (sin red ni base de datos, para poder probarlo
todo), qué sitio del catálogo está pasando ahora, a qué lado queda, qué viene
después y si toca que el guía lo cuente.

El lado sale del rumbo del movimiento: lo da el GPS si el móvil se mueve, y si
no, se calcula entre dos posiciones. Parado no hay rumbo, y entonces todo es
«cerca», sin lado.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

Side = Literal["left", "right", "ahead", "behind", "near"]
TravelMode = Literal["walk", "bus", "car"]


@dataclass(frozen=True)
class ModeSettings:
    # A partir de qué distancia un sitio es «ahora».
    radius_m: float
    # Pausa mínima entre dos sitios contados, para no encadenarlos.
    cooldown_s: float
    # Velocidad supuesta si el GPS no la da, para estimar cuándo se llega.
    speed_ms: float
    # Hasta dónde se mira por delante para «próximamente».
    horizon_m: float


MODES: dict[str, ModeSettings] = {
    "walk": ModeSettings(radius_m=90, cooldown_s=40, speed_ms=1.3, horizon_m=600),
    "bus": ModeSettings(radius_m=200, cooldown_s=20, speed_ms=5.0, horizon_m=1500),
    "car": ModeSettings(radius_m=300, cooldown_s=15, speed_ms=9.0, horizon_m=2500),
}

# Por debajo de esta velocidad (m/s) el rumbo del GPS no es fiable.
MOVING_MS = 0.6
# Un desplazamiento menor no sirve para calcular rumbo: es ruido del GPS.
MIN_MOVE_M = 8.0
EARTH_RADIUS_M = 6_371_000


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    lat: float
    lng: float
    # Popularidad (0-3): ante dos sitios igual de cerca, se cuenta antes el que gusta más.
    tier: int = 1


@dataclass(frozen=True)
class Fix:
    lat: float
    lng: float
    t: float
    heading: float | None = None
    speed: float | None = None


@dataclass(frozen=True)
class Sighting:
    place: Place
    distance_m: float
    side: Side
    eta_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.place.id,
            "name": self.place.name,
            "distance_m": round(self.distance_m),
            "side": self.side,
            "eta_s": round(self.eta_s) if self.eta_s is not None else None,
        }


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lng2 - lng1)
    x = math.sin(dlmb) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def relative_angle(heading: float, bearing: float) -> float:
    """De -180 a 180: positivo a la derecha de la marcha, negativo a la izquierda."""
    return (bearing - heading + 540) % 360 - 180


def side_of(heading: float | None, bearing: float) -> Side:
    if heading is None:
        return "near"
    rel = relative_angle(heading, bearing)
    if abs(rel) <= 25:
        return "ahead"
    if abs(rel) >= 135:
        return "behind"
    return "right" if rel > 0 else "left"


@dataclass
class ItineraryTracker:
    mode: str
    places: list[Place]
    narrated: set[str] = field(default_factory=set)
    last_fix: Fix | None = None
    heading: float | None = None
    last_narration_t: float | None = None

    @property
    def settings(self) -> ModeSettings:
        return MODES.get(self.mode, MODES["walk"])

    def update(self, fix: Fix) -> None:
        """Toma una posición nueva y actualiza el rumbo."""
        moving = fix.speed is not None and fix.speed >= MOVING_MS
        if fix.heading is not None and moving:
            self.heading = fix.heading % 360
        elif self.last_fix is not None:
            moved = distance_m(self.last_fix.lat, self.last_fix.lng, fix.lat, fix.lng)
            if moved >= MIN_MOVE_M:
                self.heading = bearing_deg(self.last_fix.lat, self.last_fix.lng, fix.lat, fix.lng)
        if fix.speed is not None and fix.speed < MOVING_MS and fix.heading is None:
            # Parado: el último rumbo deja de valer (el bus ha girado en la parada).
            self.heading = None
        self.last_fix = fix

    def _sightings(self) -> list[Sighting]:
        fix = self.last_fix
        if fix is None:
            return []
        speed = fix.speed if fix.speed and fix.speed >= MOVING_MS else self.settings.speed_ms
        result = []
        for place in self.places:
            d = distance_m(fix.lat, fix.lng, place.lat, place.lng)
            if d > self.settings.horizon_m:
                continue
            bearing = bearing_deg(fix.lat, fix.lng, place.lat, place.lng)
            side = side_of(self.heading, bearing)
            eta = None
            if self.heading is not None:
                along = d * math.cos(math.radians(relative_angle(self.heading, bearing)))
                eta = max(0.0, along / speed) if along > 0 else None
            result.append(Sighting(place=place, distance_m=d, side=side, eta_s=eta))
        return result

    def now(self) -> Sighting | None:
        """El sitio que estáis pasando: el más cercano dentro del radio, si no queda detrás."""
        cerca = [
            s for s in self._sightings()
            if s.distance_m <= self.settings.radius_m and s.side != "behind"
        ]
        cerca.sort(key=lambda s: (s.distance_m - 40 * s.place.tier))
        return cerca[0] if cerca else None

    def upcoming(self, limit: int = 3) -> list[Sighting]:
        """Lo que viene por delante, aún sin contar, por orden de llegada."""
        if self.heading is None:
            return []
        delante = [
            s for s in self._sightings()
            if s.place.id not in self.narrated
            and s.eta_s is not None
            and s.distance_m > self.settings.radius_m
            and s.side in ("ahead", "left", "right")
        ]
        delante.sort(key=lambda s: s.eta_s or 0)
        return delante[:limit]

    def due(self, t: float) -> Sighting | None:
        """El sitio que toca contar ahora, o nada. Lo marca como contado.

        Solo uno por vez, nunca dos veces el mismo y con una pausa mínima
        desde el anterior, que depende de lo rápido que se va.
        """
        ultimo = self.last_narration_t
        if ultimo is not None and t - ultimo < self.settings.cooldown_s:
            return None
        candidato = next(
            (
                s for s in sorted(
                    (
                        s for s in self._sightings()
                        if s.distance_m <= self.settings.radius_m and s.side != "behind"
                    ),
                    key=lambda s: (s.distance_m - 40 * s.place.tier),
                )
                if s.place.id not in self.narrated
            ),
            None,
        )
        if candidato is None:
            return None
        self.narrated.add(candidato.place.id)
        self.last_narration_t = t
        return candidato


SIDE_TEXT = {
    "left": "a vuestra izquierda",
    "right": "a vuestra derecha",
    "ahead": "justo delante",
    "near": "muy cerca",
    "behind": "detrás",
}


def narration_prompt(sighting: Sighting, mode: str) -> str:
    """La instrucción silenciosa para que el guía cuente lo que aparece.

    Le da el tiempo que tiene antes de que el sitio quede atrás, para que ajuste
    lo que cuenta: en bus no hay tiempo para la historia entera de un edificio.
    """
    settings = MODES.get(mode, MODES["walk"])
    window = max(20, round((settings.radius_m + sighting.distance_m) / settings.speed_ms))
    window = min(window, 180)
    return (
        "[nota interna, no la leas en voz alta ni digas que te han avisado: ahora mismo, "
        f"{SIDE_TEXT[sighting.side]}, a unos {round(sighting.distance_m, -1):.0f} m, está "
        f"{sighting.place.name}. Cuéntalo ya, como un guía que lo señala mientras pasáis: "
        f"tenéis unos {window} segundos antes de que quede atrás, así que ve a lo que de "
        "verdad merece la pena de este sitio. Di hacia dónde mirar.]"
    )
