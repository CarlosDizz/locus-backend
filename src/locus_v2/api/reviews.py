"""Valoraciones de llamadas: la de cada usuario al colgar, y lo público por sitio."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from locus_v2.api.admin_auth import require_admin
from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.calls.models import CallError
from locus_v2.calls.store import RoomStore
from locus_v2.identity.models import User
from locus_v2.infrastructure.redis import get_redis
from locus_v2.reviews.service import (
    CommentModerator,
    PoiReviewSummary,
    ReviewError,
    ReviewInput,
    ReviewService,
)

router = APIRouter(tags=["reviews"])
AdminDep = Annotated[User, Depends(require_admin)]


@router.post("/api/calls/{call_id}/review", status_code=201)
async def review_call(
    call_id: str,
    payload: ReviewInput,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> dict[str, Any]:
    try:
        # La sala vive en Redis mientras la llamada existe y un rato después:
        # la valoración se manda al colgar, así que ahí sigue.
        room = await RoomStore(get_redis()).get(call_id.upper())
    except CallError as error:
        raise HTTPException(404, "Esta llamada ya no existe") from error
    try:
        review = await ReviewService(session, CommentModerator(settings)).submit(
            current_user, room, payload
        )
    except ReviewError as error:
        raise HTTPException(error.status, str(error)) from error
    return {"id": review.id, "place_comment_status": review.place_comment_status}


@router.get("/api/catalog/pois/{poi_id}/reviews", response_model=PoiReviewSummary)
async def poi_reviews(poi_id: int, session: SessionDep) -> PoiReviewSummary:
    try:
        return await ReviewService(session).poi_summary(poi_id)
    except ReviewError as error:
        raise HTTPException(error.status, str(error)) from error


@router.get("/admin/v2/reviews")
async def admin_reviews(session: SessionDep, _admin: AdminDep) -> list[dict[str, Any]]:
    return await ReviewService(session).admin_list()
