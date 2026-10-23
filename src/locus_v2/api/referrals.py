"""/api/r/<token>: cuenta el clic en un referido y lleva a la web del programa."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from locus_v2.affiliates.models import ReferralClick
from locus_v2.affiliates.tracking import verify
from locus_v2.api.auth import SessionDep, SettingsDep

router = APIRouter(prefix="/api/r", tags=["referrals"])


@router.get("/{token}")
async def follow_referral(
    token: str, session: SessionDep, settings: SettingsDep
) -> RedirectResponse:
    link = verify(settings, token)
    if link is None:
        raise HTTPException(404, "Enlace no válido")
    session.add(ReferralClick(
        user_id=link.user_id, app=link.app, program=link.program[:40],
        trip_public_id=link.trip_id[:36], url=link.url[:2048],
    ))
    await session.commit()
    return RedirectResponse(link.url, status_code=302)
