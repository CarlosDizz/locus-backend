"""V1-compatible mobile auth facade.

Mounted at /api/auth so the Ionic app only ever has to change its base URL (see
docs/roadmap.md section 5). Request/response shapes mirror app/schemas/auth.py exactly.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings, get_settings
from locus_v2.identity.application.mobile_auth import (
    MobileAuthError,
    MobileAuthService,
    MobileUserView,
)
from locus_v2.identity.infrastructure.google_identity import GoogleIdentityVerifier
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_session
from locus_v2.shared.mobile_ids import mobile_id

router = APIRouter(prefix="/api/auth", tags=["mobile-auth"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(min_length=20)


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    auth_provider: str
    avatar_url: str
    is_active: bool
    profile_context: str = ""
    preferred_name: str = ""


class ProfileUpdateRequest(BaseModel):
    # Both optional so the app can send one without clearing the other, and
    # capped so a prompt variable cannot be used to smuggle in a whole essay:
    # this text is pasted into the guide's system instruction, which is billed
    # on every single turn of a call.
    profile_context: str | None = Field(default=None, max_length=1000)
    preferred_name: str | None = Field(default=None, max_length=160)


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


def _view(view: MobileUserView) -> UserResponse:
    return UserResponse(
        id=view.id,
        email=view.email,
        display_name=view.display_name,
        auth_provider=view.auth_provider,
        avatar_url=view.avatar_url,
        is_active=view.is_active,
        profile_context=view.profile_context,
        preferred_name=view.preferred_name,
    )


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


async def get_current_user_required(
    session: SessionDep,
    settings: SettingsDep,
    authorization: str | None = Header(default=None),
) -> User:
    token = _extract_bearer(authorization)
    user = await MobileAuthService(session, settings).authenticate(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user_required)]


async def get_current_user_optional(
    session: SessionDep,
    settings: SettingsDep,
    authorization: str | None = Header(default=None),
) -> User | None:
    token = _extract_bearer(authorization)
    if not token:
        return None
    return await MobileAuthService(session, settings).authenticate(token)


OptionalUserDep = Annotated[User | None, Depends(get_current_user_optional)]


@router.post("/register", response_model=AuthResponse)
async def register(
    payload: RegisterRequest, session: SessionDep, settings: SettingsDep
) -> AuthResponse:
    try:
        user, token = await MobileAuthService(session, settings).register(
            email=payload.email, password=payload.password, display_name=payload.display_name
        )
    except MobileAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(token=token, user=_view(user))


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, session: SessionDep, settings: SettingsDep) -> AuthResponse:
    try:
        user, token = await MobileAuthService(session, settings).login(
            email=payload.email, password=payload.password
        )
    except MobileAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=token, user=_view(user))


@router.post("/google", response_model=AuthResponse)
async def google_auth(
    payload: GoogleAuthRequest, request: Request, session: SessionDep, settings: SettingsDep
) -> AuthResponse:
    try:
        user, token = await MobileAuthService(session, settings).login_google(
            payload.id_token, GoogleIdentityVerifier(), request.headers.get("user-agent")
        )
    except MobileAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=token, user=_view(user))


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUserDep) -> UserResponse:
    return _view(_me_view(current_user))


@router.put("/me/profile", response_model=UserResponse)
async def update_my_profile(
    payload: ProfileUpdateRequest,
    current_user: CurrentUserDep,
    session: SessionDep,
) -> UserResponse:
    """Save what the traveller wants the guide to know about them.

    Deliberately asks for nothing but the text. The app used to save this
    through the session endpoint, which meant taking a GPS fix first: a denied
    or slow permission killed the save before it left the phone, and since the
    caller swallowed the error the button simply did nothing. Nobody had ever
    managed to save a profile in production.
    """
    if payload.profile_context is not None:
        current_user.profile_context = payload.profile_context.strip()
    if payload.preferred_name is not None:
        current_user.preferred_name = payload.preferred_name.strip()
    session.add(current_user)
    await session.commit()
    await session.refresh(current_user)
    return _view(_me_view(current_user))


def _me_view(user: User) -> MobileUserView:
    return MobileUserView(
        id=mobile_id(user),
        email=user.email,
        display_name=user.display_name,
        auth_provider=user.auth_provider,
        avatar_url=user.avatar_url or "",
        is_active=user.status == "active",
        profile_context=user.profile_context or "",
        preferred_name=user.preferred_name or "",
    )
