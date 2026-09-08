import hashlib

import jwt
from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings
from locus_v2.identity.models import AdminSession, User, UserStatus
from locus_v2.shared.clock import utc_now


class VoiceAuthenticationError(PermissionError):
    pass


async def authenticate_voice_user(
    websocket: WebSocket,
    session: AsyncSession,
    settings: Settings,
) -> User:
    token = _bearer_token(websocket) or websocket.query_params.get("access_token")
    if token:
        return await _user_from_access_token(token, session, settings)

    admin_token = websocket.cookies.get(settings.admin_session_cookie)
    if admin_token:
        admin_session = await session.scalar(
            select(AdminSession).where(
                AdminSession.token_hash == hashlib.sha256(admin_token.encode()).hexdigest(),
                AdminSession.revoked_at.is_(None),
                AdminSession.expires_at > utc_now(),
            )
        )
        if admin_session is not None:
            user = await session.get(User, admin_session.user_id)
            if user is not None and user.status == UserStatus.ACTIVE:
                return user

    raise VoiceAuthenticationError("A valid Locus session is required")


def _bearer_token(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


async def _user_from_access_token(
    token: str,
    session: AsyncSession,
    settings: Settings,
) -> User:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
    except jwt.PyJWTError as error:
        raise VoiceAuthenticationError("Invalid or expired Locus access token") from error

    user = await session.scalar(
        select(User).where(
            User.public_id == str(claims["sub"]),
            User.status == UserStatus.ACTIVE,
        )
    )
    if user is None:
        raise VoiceAuthenticationError("The Locus user is not available")
    return user
