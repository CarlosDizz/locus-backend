"""Mis viajes y el planificador."""

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.billing.models import Wallet
from locus_v2.chat.configuration import ChatConfigurationError
from locus_v2.trips.routes import DayRoute, day_route
from locus_v2.trips.service import TripError, TripPlannerService, TripView, trip_view

router = APIRouter(prefix="/api/trips", tags=["trips"])


class PlannerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    locale: str = "es"


def _http(error: TripError) -> HTTPException:
    return HTTPException(error.status, str(error))


@router.get("", response_model=list[TripView])
async def list_trips(
    session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> list[TripView]:
    return await TripPlannerService(session, settings).list_trips(current_user)


@router.post("", response_model=TripView, status_code=201)
async def create_trip(
    session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    return trip_view(
        await TripPlannerService(session, settings).create(current_user), current_user.id
    )


@router.get("/{trip_id}", response_model=TripView)
async def get_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    try:
        return trip_view(
            await TripPlannerService(session, settings).get(current_user, trip_id),
            current_user.id,
        )
    except TripError as error:
        raise _http(error) from error


@router.post("/{trip_id}/locate", response_model=TripView)
async def locate_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    """Pone en el mapa la ciudad y las paradas que aún no tienen sitio."""
    try:
        trip = await TripPlannerService(session, settings).locate_missing(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    except ChatConfigurationError as error:
        raise HTTPException(503, "El planificador no está disponible") from error
    return trip_view(trip, current_user.id)


@router.get("/{trip_id}/days/{day}/route", response_model=DayRoute | None)
async def get_day_route(
    trip_id: str,
    day: date,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> DayRoute | None:
    """El camino a pie del día, o null si no hay dos paradas con sitio en el mapa."""
    try:
        trip = await TripPlannerService(session, settings).get(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    return await day_route(session, settings, trip, day)


@router.delete("/{trip_id}", status_code=204)
async def delete_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> None:
    try:
        await TripPlannerService(session, settings).delete(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error


@router.post("/{trip_id}/messages", response_model=TripView)
async def planner_message(
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
        trip = await TripPlannerService(session, settings).send_message(
            current_user, trip_id, payload.text, payload.locale
        )
    except TripError as error:
        raise _http(error) from error
    except ChatConfigurationError as error:
        raise HTTPException(503, "El planificador no está disponible") from error
    return trip_view(trip, current_user.id)


@router.post("/{trip_id}/proposals/{proposal_id}/accept", response_model=TripView)
async def accept_proposal(
    trip_id: str,
    proposal_id: str,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await TripPlannerService(session, settings).accept_proposal(
            current_user, trip_id, proposal_id
        )
    except TripError as error:
        raise _http(error) from error
    return trip_view(trip, current_user.id)


@router.post("/{trip_id}/proposals/{proposal_id}/undo", response_model=TripView)
async def undo_proposal(
    trip_id: str,
    proposal_id: str,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await TripPlannerService(session, settings).undo_proposal(
            current_user, trip_id, proposal_id
        )
    except TripError as error:
        raise _http(error) from error
    return trip_view(trip, current_user.id)


class InviteResponse(BaseModel):
    code: str


@router.post("/{trip_id}/invite", response_model=InviteResponse)
async def invite_to_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> InviteResponse:
    """El código del enlace para sumarse al viaje (/trips/join/<código>)."""
    try:
        code = await TripPlannerService(session, settings).invite(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error
    return InviteResponse(code=code)


@router.post("/join/{code}", response_model=TripView)
async def join_trip(
    code: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    try:
        trip = await TripPlannerService(session, settings).join(current_user, code)
    except TripError as error:
        raise _http(error) from error
    return trip_view(trip, current_user.id)


@router.delete("/{trip_id}/members/me", status_code=204)
async def leave_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> None:
    try:
        await TripPlannerService(session, settings).leave(current_user, trip_id)
    except TripError as error:
        raise _http(error) from error


@router.delete("/{trip_id}/members/{member_id}", response_model=TripView)
async def remove_trip_member(
    trip_id: str,
    member_id: int,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TripView:
    try:
        trip = await TripPlannerService(session, settings).remove_member(
            current_user, trip_id, member_id
        )
    except TripError as error:
        raise _http(error) from error
    return trip_view(trip, current_user.id)
