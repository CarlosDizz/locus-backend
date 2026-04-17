from __future__ import annotations

from fastapi import Header, HTTPException

from app.schemas.auth import UserResponse
from app.services.auth_service import auth_service


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def get_current_user_optional(authorization: str | None = Header(default=None)) -> UserResponse | None:
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    user = auth_service.get_user_from_token(token)
    if user is None:
        return None
    return UserResponse(id=user.id, email=user.email, display_name=user.display_name, is_active=user.is_active)


def get_current_user_required(authorization: str | None = Header(default=None)) -> UserResponse:
    user = get_current_user_optional(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")
    return user
