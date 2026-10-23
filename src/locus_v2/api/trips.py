"""Mis viajes y el planificador."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from locus_v2.api.auth import ClientAppDep, CurrentUserDep, SessionDep, SettingsDep
from locus_v2.billing.models import Wallet
from locus_v2.chat.configuration import ChatConfigurationError
from locus_v2.shared.apps import ClientApp
from locus_v2.shared.clock import utc_now
from locus_v2.trips.calendar import trip_ics
from locus_v2.trips.routes import DayRoute, day_route
from locus_v2.trips.service import TripError, TripPlannerService, TripView
from locus_v2.trips.voice import MAX_VOICE_BYTES, VoiceError, transcribe
from locus_v2.trips.weather import TripWeather, trip_weather

router = APIRouter(prefix="/api/trips", tags=["trips"])


class PlannerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    locale: str = "es"


def _http(error: TripError) -> HTTPException:
    return HTTPException(error.status, str(error))


def get_planner(
    session: SessionDep, settings: SettingsDep, app: ClientAppDep
) -> TripPlannerService:
    """El planificador de la app que pregunta (cabecera X-Locus-App)."""
    return TripPlannerService(session, settings, app)


PlannerDep = Annotated[TripPlannerService, Depends(get_planner)]


@router.get("", response_model=list[TripView])
async def list_trips(
    planner: PlannerDep, current_user: CurrentUserDep
) -> list[TripView]:
    return await planner.list_trips(current_user)


@router.post("", response_model=TripView, status_code=201)
async def create_trip(
    planner: PlannerDep, current_user: CurrentUserDep
) -> TripView:
    return planner.view(await planner.create(current_user), current_user.id)


@router.get("/{trip_id}", response_model=TripView)
async def get_trip(
    trip_id: str, planner: PlannerDep, current_user: CurrentUserDep
) -> TripView:
    try:
        return planner.view(await planner.get(current_user, trip_id), current_user.id)
    except TripError as error:
        raise _http(error) from error


@router.post("/{trip_id}/locate", response_model=TripView)
async def locate_trip(
    planner: PlannerDep,
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep,
    locale: str | None = Query(default=None),
    accept_language: str | None = Header(default=None),
) -> TripView:
    """Pone en el mapa la ciudad y las paradas que aún no tienen sitio.

    Los sitios se buscan en el idioma del viaje. La app publicada no manda
    `locale` aquí, así que sin él se toma el primero de `Accept-Language`.
    """
    idioma = locale or (accept_language or "es").split(",", 1)[0].split(";", 1)[0].strip()
    try:
        trip = await planner.locate_missing(
            current_user, trip_id, idioma or "es"
        )
    except TripError as error:
        raise _http(error) from error
    except ChatConfigurationError as error:
        raise HTTPException(503, "El planificador no está disponible") from error
    return planner.view(trip, current_user.id)


@router.get("/{trip_id}/days/{day}/route", response_model=DayRoute | None)
async def get_day_route(
    planner: PlannerDep,
    trip_id: str,
    day: date,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> DayRoute | None:
    """El camino a pie del día, o null si no hay dos paradas con sitio en el mapa."""
    try:
        trip = await planner.get(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    return await day_route(session, settings, trip, day)


@router.delete("/{trip_id}", status_code=204)
async def delete_trip(
    planner: PlannerDep,
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> None:
    try:
        await planner.delete(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error


@router.post("/{trip_id}/messages", response_model=TripView)
async def planner_message(
    planner: PlannerDep,
    trip_id: str,
    payload: PlannerMessage,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    # Como el chat del mapa: sin saldo no se llama al modelo.
    wallet = await session.scalar(select(Wallet).where(Wallet.user_id == current_user.id))
    if wallet is None or wallet.balance_cents <= 0:
        raise HTTPException(402, "Saldo insuficiente")
    try:
        trip = await planner.send_message(
            current_user, trip_id, payload.text, payload.locale
        )
    except TripError as error:
        raise _http(error) from error
    except ChatConfigurationError as error:
        raise HTTPException(503, "El planificador no está disponible") from error
    return planner.view(trip, current_user.id)


@router.post("/{trip_id}/proposals/{proposal_id}/accept", response_model=TripView)
async def accept_proposal(
    planner: PlannerDep,
    trip_id: str,
    proposal_id: str,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await planner.accept_proposal(
            current_user, trip_id, proposal_id
        )
    except TripError as error:
        raise _http(error) from error
    return planner.view(trip, current_user.id)


@router.post("/{trip_id}/proposals/{proposal_id}/undo", response_model=TripView)
async def undo_proposal(
    planner: PlannerDep,
    trip_id: str,
    proposal_id: str,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await planner.undo_proposal(
            current_user, trip_id, proposal_id
        )
    except TripError as error:
        raise _http(error) from error
    return planner.view(trip, current_user.id)


class InviteResponse(BaseModel):
    code: str


@router.post("/{trip_id}/invite", response_model=InviteResponse)
async def invite_to_trip(
    planner: PlannerDep,
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> InviteResponse:
    """El código del enlace para sumarse al viaje (/trips/join/<código>)."""
    try:
        code = await planner.invite(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    return InviteResponse(code=code)


@router.post("/join/{code}", response_model=TripView)
async def join_trip(
    planner: PlannerDep,
    code: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    try:
        trip = await planner.join(current_user, code)
    except TripError as error:
        raise _http(error) from error
    return planner.view(trip, current_user.id)


@router.delete("/{trip_id}/members/me", status_code=204)
async def leave_trip(
    planner: PlannerDep,
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> None:
    try:
        await planner.leave(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error


@router.delete("/{trip_id}/members/{member_id}", response_model=TripView)
async def remove_trip_member(
    planner: PlannerDep,
    trip_id: str,
    member_id: int,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await planner.remove_member(
            current_user, trip_id, member_id
        )
    except TripError as error:
        raise _http(error) from error
    return planner.view(trip, current_user.id)


@router.post("/{trip_id}/share/guide", response_model=TripView)
async def share_with_guide(
    trip_id: str, planner: PlannerDep, current_user: CurrentUserDep
) -> TripView:
    """«Llevar este plan a Locus»: el viaje de Borea pasa a verse también en Locus Guide."""
    try:
        trip = await planner.share_with(current_user, trip_id, ClientApp.GUIDE)
    except TripError as error:
        raise _http(error) from error
    return planner.view(trip, current_user.id)


@router.get("/{trip_id}/calendar.ics")
async def trip_calendar(
    trip_id: str, planner: PlannerDep, current_user: CurrentUserDep
) -> Response:
    """El plan como calendario, para añadirlo al del iPhone desde el menú de compartir."""
    try:
        trip = await planner.get(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    return Response(
        trip_ics(trip, utc_now()),
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="viaje-{trip.public_id[:8]}.ics"'},
    )


@router.get("/{trip_id}/weather", response_model=TripWeather)
async def trip_forecast(
    trip_id: str, planner: PlannerDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripWeather:
    """La previsión de cada día si el viaje está a diez días o menos (WeatherKit)."""
    try:
        trip = await planner.get(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    centro = trip.center_json or next((i.map_json for i in trip.items if i.map_json), None) or {}
    return await trip_weather(
        settings, centro.get("lat"), centro.get("lng"),
        trip.starts_at.date() if trip.starts_at else None,
        trip.ends_at.date() if trip.ends_at else None,
        utc_now().date(),
    )


@router.post("/{trip_id}/voice", response_model=TripView)
async def planner_voice(
    trip_id: str,
    planner: PlannerDep,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    audio: Annotated[UploadFile, File(description="La nota de voz (m4a, mp4, webm, ogg…)")],
    locale: Annotated[str, Form()] = "es",
) -> TripView:
    """Nota de voz, como en BACUL: se transcribe, el audio no se guarda y llega el texto."""
    wallet = await session.scalar(select(Wallet).where(Wallet.user_id == current_user.id))
    if wallet is None or wallet.balance_cents <= 0:
        raise HTTPException(402, "Saldo insuficiente")
    data = await audio.read(MAX_VOICE_BYTES + 1)
    if not data:
        raise HTTPException(400, "La nota de voz está vacía")
    if len(data) > MAX_VOICE_BYTES:
        raise HTTPException(413, "La nota de voz es demasiado larga")
    mime = (audio.content_type or "audio/mp4").split(";")[0]
    if not mime.startswith("audio/") and mime not in {"video/mp4", "video/webm"}:
        raise HTTPException(415, "Eso no es un audio")
    try:
        texto, usage = await transcribe(settings, data, mime.replace("video/", "audio/"))
    except VoiceError as error:
        raise HTTPException(503, str(error)) from error
    await planner.bill_transcription(current_user, usage)
    if not texto:
        raise HTTPException(422, "No se ha entendido la nota de voz")
    try:
        trip = await planner.send_message(current_user, trip_id, texto, locale)
    except TripError as error:
        raise _http(error) from error
    except ChatConfigurationError as error:
        raise HTTPException(503, "El planificador no está disponible") from error
    return planner.view(trip, current_user.id)
