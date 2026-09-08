from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings, get_settings
from locus_v2.identity.application.auth import AdminAuthService, AuthenticatedAdmin
from locus_v2.identity.infrastructure.google_identity import GoogleIdentityVerifier
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(prefix="/admin/v2/auth", tags=["admin-auth"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class GoogleLoginRequest(BaseModel):
    credential: str


class AuthConfig(BaseModel):
    google_client_id: str | None
    local_login_enabled: bool


class AdminView(BaseModel):
    id: str
    email: str
    display_name: str
    avatar_url: str | None
    roles: list[str]


def _view(admin: AuthenticatedAdmin | User) -> AdminView:
    roles = admin.roles if isinstance(admin, AuthenticatedAdmin) else [r.code for r in admin.roles]
    return AdminView(
        id=admin.id if isinstance(admin, AuthenticatedAdmin) else admin.public_id,
        email=admin.email,
        display_name=admin.display_name,
        avatar_url=admin.avatar_url,
        roles=roles,
    )


def _set_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        settings.admin_session_cookie,
        token,
        max_age=settings.admin_session_days * 86400,
        httponly=True,
        secure=settings.env == "production",
        samesite="lax",
        path="/",
    )


async def require_admin(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    token = request.cookies.get(settings.admin_session_cookie)
    authenticated = await AdminAuthService(session, settings).authenticate(token)
    if authenticated is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session required",
        )
    return authenticated[0]


AdminDep = Annotated[User, Depends(require_admin)]


@router.get("/config", response_model=AuthConfig)
async def auth_config(settings: SettingsDep) -> AuthConfig:
    return AuthConfig(
        google_client_id=settings.google_auth_client_ids[0]
        if settings.google_auth_client_ids
        else None,
        local_login_enabled=settings.env == "local" and settings.allow_insecure_local_admin,
    )


@router.post("/google", response_model=AdminView)
async def google_login(
    payload: GoogleLoginRequest,
    request: Request,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
) -> AdminView:
    try:
        admin, token = await AdminAuthService(session, settings).login_google(
            payload.credential, GoogleIdentityVerifier(), request.headers.get("user-agent")
        )
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    _set_cookie(response, token, settings)
    return _view(admin)


@router.post("/local", response_model=AdminView)
async def local_login(
    request: Request,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
) -> AdminView:
    try:
        admin, token = await AdminAuthService(session, settings).login_local(
            request.headers.get("user-agent")
        )
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    _set_cookie(response, token, settings)
    return _view(admin)


@router.get("/me", response_model=AdminView)
async def me(admin: AdminDep) -> AdminView:
    return _view(admin)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
) -> None:
    token = request.cookies.get(settings.admin_session_cookie)
    await AdminAuthService(session, settings).logout(token)
    response.delete_cookie(settings.admin_session_cookie, path="/")
