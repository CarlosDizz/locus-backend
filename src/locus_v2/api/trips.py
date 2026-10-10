"""Mis viajes y el planificador."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.billing.models import Wallet
from locus_v2.chat.configuration import ChatConfigurationError
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
    return trip_view(await TripPlannerService(session, settings).create(current_user))


@router.get("/{trip_id}", response_model=TripView)
async def get_trip(
    trip_id: str, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TripView:
    try:
        return trip_view(await TripPlannerService(session, settings).get(current_user, trip_id))
    except TripError as error:
        raise _http(error) from error


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
    return trip_view(trip)


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
    return trip_view(trip)
