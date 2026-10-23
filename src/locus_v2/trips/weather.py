"""El tiempo de cada día del viaje: previsión real cuando queda cerca.

Dentro de los diez días que da WeatherKit, cada día lleva su icono y su máxima
(«Martes 6 · sol · 21°»). Más lejos no hay previsión: la app enseña en gris el
clima habitual que el planificador busca en la web (Trip.climate_json), marcado
como habitual, para no confundir una media con una previsión.

WeatherKit va incluido en la cuenta de desarrollador de Apple. Apple exige
mostrar su marca y un enlace a sus fuentes de datos junto a la previsión
(ATTRIBUTION_URL). Sin claves configuradas, no hay previsión y la app sigue.
"""

import time
from datetime import date
from typing import Any

import httpx
import jwt
import structlog
from pydantic import BaseModel

from locus_v2.config import Settings

logger = structlog.get_logger()

ATTRIBUTION_URL = "https://weatherkit.apple.com/legal-attribution.html"
FORECAST_DAYS = 10
_URL = "https://weatherkit.apple.com/api/v1/weather"

# Los códigos de condición de WeatherKit, reducidos a los iconos de la app.
_ICONS = {
    "Clear": "sun", "MostlyClear": "sun", "Hot": "sun",
    "PartlyCloudy": "cloud-sun", "MostlyCloudy": "cloud", "Cloudy": "cloud",
    "Haze": "cloud", "Foggy": "fog", "Smoky": "fog", "Dust": "fog", "Windy": "wind",
    "Breezy": "wind", "Drizzle": "rain", "Rain": "rain", "HeavyRain": "rain",
    "SunShowers": "rain", "IsolatedThunderstorms": "storm", "ScatteredThunderstorms": "storm",
    "Thunderstorms": "storm", "StrongStorms": "storm", "Flurries": "snow", "Snow": "snow",
    "HeavySnow": "snow", "Sleet": "snow", "FreezingRain": "snow", "Blizzard": "snow",
}


class DayWeather(BaseModel):
    date: date
    icon: str
    max_c: int
    min_c: int
    rain_chance: int = 0


class TripWeather(BaseModel):
    # forecast: hay previsión real para los días que aparecen. none: el viaje queda
    # lejos o no hay WeatherKit configurado; la app enseña el clima habitual.
    source: str
    days: list[DayWeather] = []
    attribution_url: str = ATTRIBUTION_URL


def _enabled(settings: Settings) -> bool:
    return bool(
        settings.weatherkit_team_id and settings.weatherkit_key_id
        and settings.weatherkit_service_id and settings.weatherkit_private_key
    )


def _token(settings: Settings) -> str:
    now = int(time.time())
    assert settings.weatherkit_private_key is not None
    return jwt.encode(
        {"iss": settings.weatherkit_team_id, "iat": now, "exp": now + 1800,
         "sub": settings.weatherkit_service_id},
        settings.weatherkit_private_key.get_secret_value(),
        algorithm="ES256",
        headers={"kid": settings.weatherkit_key_id,
                 "id": f"{settings.weatherkit_team_id}.{settings.weatherkit_service_id}"},
    )


def parse_daily(data: dict[str, Any], start: date, end: date) -> list[DayWeather]:
    days: list[DayWeather] = []
    for raw in (data.get("forecastDaily") or {}).get("days") or []:
        try:
            day = date.fromisoformat(str(raw.get("forecastStart", ""))[:10])
        except ValueError:
            continue
        if not start <= day <= end:
            continue
        days.append(DayWeather(
            date=day,
            icon=_ICONS.get(str(raw.get("conditionCode") or ""), "cloud"),
            max_c=round(float(raw.get("temperatureMax") or 0)),
            min_c=round(float(raw.get("temperatureMin") or 0)),
            rain_chance=round(float(raw.get("precipitationChance") or 0) * 100),
        ))
    return days


async def trip_weather(
    settings: Settings, lat: float | None, lng: float | None, start: date | None,
    end: date | None, today: date,
) -> TripWeather:
    if (
        not _enabled(settings) or lat is None or lng is None or start is None
        or (start - today).days > FORECAST_DAYS or (end or start) < today
    ):
        return TripWeather(source="none")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{_URL}/es/{lat:.4f}/{lng:.4f}",
                params={"dataSets": "forecastDaily"},
                headers={"Authorization": f"Bearer {_token(settings)}"},
            )
        response.raise_for_status()
        days = parse_daily(response.json(), max(start, today), end or start)
    except (httpx.HTTPError, ValueError, jwt.PyJWTError) as error:
        logger.warning("borea_weather_failed", error=str(error))
        return TripWeather(source="none")
    return TripWeather(source="forecast" if days else "none", days=days)
