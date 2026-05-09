from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import get_current_user_required
from app.schemas.auth import AuthResponse, GoogleAuthRequest, LoginRequest, RegisterRequest, UserResponse
from app.services.auth_service import AuthError, auth_service


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterRequest) -> AuthResponse:
    try:
        token, user = auth_service.register_user(payload.email, payload.password, payload.display_name)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest) -> AuthResponse:
    try:
        token, user = auth_service.login(payload.email, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@router.post("/google", response_model=AuthResponse)
async def google_auth(payload: GoogleAuthRequest) -> AuthResponse:
    try:
        token, user = auth_service.authenticate_google(payload.id_token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserResponse = Depends(get_current_user_required)) -> UserResponse:
    return current_user
