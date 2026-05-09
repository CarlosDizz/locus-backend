from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Any
from datetime import datetime, timedelta, UTC

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import select

from app.config import settings
from app.db.models import AuthToken, User, Wallet
from app.db.session import session_scope
from app.schemas.auth import UserResponse


class AuthError(RuntimeError):
    pass


class AuthService:
    def __init__(self) -> None:
        self.pbkdf2_iterations = 600_000
        self.token_ttl_days = int(getattr(settings, "auth_token_ttl_days", 30))

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, self.pbkdf2_iterations)
        return f"pbkdf2_sha256${self.pbkdf2_iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"

    def _verify_password(self, password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_b64, hash_b64 = encoded.split("$", 3)
        except ValueError:
            return False
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)

    def _token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _normalize_email(self, email: str) -> str:
        return email.strip().lower()

    def _user_to_schema(self, user: User) -> UserResponse:
        return UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            auth_provider=user.auth_provider,
            avatar_url=user.avatar_url,
            is_active=user.is_active,
        )

    def _build_token(self, user_id: int) -> tuple[str, AuthToken]:
        raw_token = secrets.token_urlsafe(32)
        token = AuthToken(
            user_id=user_id,
            token_hash=self._token_hash(raw_token),
            expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=self.token_ttl_days),
        )
        return raw_token, token

    def _issue_token(self, user: User) -> tuple[str, UserResponse, AuthToken]:
        token_value, token_row = self._build_token(user.id)
        return token_value, self._user_to_schema(user), token_row

    def _create_user(
        self,
        *,
        email: str,
        password_hash: str,
        display_name: str,
        auth_provider: str,
        google_sub: str | None = None,
        avatar_url: str = "",
    ) -> User:
        return User(
            email=email,
            password_hash=password_hash,
            display_name=display_name.strip(),
            auth_provider=auth_provider,
            google_sub=google_sub,
            avatar_url=avatar_url.strip(),
            is_active=True,
        )

    def _grant_signup_bonus(self, db, wallet: Wallet, user_id: int) -> None:
        from app.services.billing_service import billing_service

        billing_service.add_ledger_entry(
            db=db,
            wallet=wallet,
            user_id=user_id,
            entry_type="signup_bonus",
            amount_cents=settings.billing_signup_bonus_cents,
            description="Saldo promocional de bienvenida",
            reference_type="user",
            reference_id=str(user_id),
        )

    def _create_wallet_for_user(self, db, user_id: int) -> Wallet:
        wallet = Wallet(user_id=user_id, currency="EUR", balance_cents=0)
        db.add(wallet)
        db.flush()
        self._grant_signup_bonus(db, wallet, user_id)
        return wallet

    def _verify_google_token(self, raw_id_token: str) -> dict[str, Any]:
        client_ids = {client_id for client_id in settings.google_auth_client_ids if client_id}
        if not client_ids:
            raise AuthError("El acceso con Google no está configurado en el backend")

        try:
            token_info = google_id_token.verify_oauth2_token(raw_id_token, google_requests.Request())
        except ValueError as exc:
            raise AuthError("No he podido verificar la cuenta de Google") from exc

        audience = str(token_info.get("aud", "")).strip()
        if audience not in client_ids:
            raise AuthError("El token de Google no pertenece a esta aplicación")

        email = str(token_info.get("email", "")).strip()
        sub = str(token_info.get("sub", "")).strip()
        email_verified = bool(token_info.get("email_verified"))

        if not email or not sub or not email_verified:
            raise AuthError("La cuenta de Google no tiene un email verificado")

        return token_info

    def _sync_google_profile(
        self,
        *,
        db,
        user: User,
        email: str,
        google_sub: str,
        display_name: str,
        avatar_url: str,
    ) -> None:
        if user.google_sub and user.google_sub != google_sub:
            raise AuthError("Esta cuenta ya está vinculada a otro acceso de Google")

        if user.email != email:
            email_owner = db.scalar(select(User).where(User.email == email).where(User.id != user.id))
            if email_owner is not None:
                raise AuthError("El email de Google ya pertenece a otra cuenta")
            user.email = email

        user.google_sub = google_sub
        if display_name and not user.display_name:
            user.display_name = display_name
        if avatar_url:
            user.avatar_url = avatar_url

    def register_user(self, email: str, password: str, display_name: str = "") -> tuple[str, UserResponse]:
        if not settings.auth_enable_password_auth:
            raise AuthError("El acceso con email y contraseña está desactivado")
        normalized_email = self._normalize_email(email)
        with session_scope() as db:
            existing = db.scalar(select(User).where(User.email == normalized_email))
            if existing is not None:
                raise AuthError("Ya existe un usuario con ese email")

            user = self._create_user(
                email=normalized_email,
                password_hash=self._hash_password(password),
                display_name=display_name,
                auth_provider="local",
            )
            db.add(user)
            db.flush()

            self._create_wallet_for_user(db, user.id)
            token_value, user_schema, token_row = self._issue_token(user)
            db.add(token_row)
            return token_value, user_schema

    def login(self, email: str, password: str) -> tuple[str, UserResponse]:
        if not settings.auth_enable_password_auth:
            raise AuthError("Usa el acceso con Google")
        normalized_email = self._normalize_email(email)
        with session_scope() as db:
            user = db.scalar(select(User).where(User.email == normalized_email))
            if user is None or not user.is_active:
                raise AuthError("Credenciales no válidas")
            if not user.password_hash:
                raise AuthError("Esta cuenta usa acceso con Google")
            if not self._verify_password(password, user.password_hash):
                raise AuthError("Credenciales no válidas")
            token_value, user_schema, token_row = self._issue_token(user)
            db.add(token_row)
            return token_value, user_schema

    def authenticate_google(self, raw_id_token: str) -> tuple[str, UserResponse]:
        token_info = self._verify_google_token(raw_id_token)
        normalized_email = self._normalize_email(str(token_info["email"]))
        google_sub = str(token_info["sub"]).strip()
        display_name = str(token_info.get("name", "")).strip()
        avatar_url = str(token_info.get("picture", "")).strip()

        with session_scope() as db:
            user = db.scalar(select(User).where(User.google_sub == google_sub))
            if user is None:
                user = db.scalar(select(User).where(User.email == normalized_email))

            if user is None:
                user = self._create_user(
                    email=normalized_email,
                    password_hash="",
                    display_name=display_name,
                    auth_provider="google",
                    google_sub=google_sub,
                    avatar_url=avatar_url,
                )
                db.add(user)
                db.flush()
                self._create_wallet_for_user(db, user.id)
            else:
                if not user.is_active:
                    raise AuthError("Esta cuenta está desactivada")
                self._sync_google_profile(
                    db=db,
                    user=user,
                    email=normalized_email,
                    google_sub=google_sub,
                    display_name=display_name,
                    avatar_url=avatar_url,
                )

            token_value, user_schema, token_row = self._issue_token(user)
            db.add(token_row)
            return token_value, user_schema

    def get_user_from_token(self, token: str) -> User | None:
        token = token.strip()
        if not token:
            return None
        with session_scope() as db:
            row = db.scalar(
                select(AuthToken)
                .where(AuthToken.token_hash == self._token_hash(token))
                .where(AuthToken.revoked_at.is_(None))
                .where(AuthToken.expires_at > datetime.utcnow())
            )
            if row is None:
                return None
            return db.get(User, row.user_id)

    def revoke_token(self, token: str) -> None:
        with session_scope() as db:
            row = db.scalar(select(AuthToken).where(AuthToken.token_hash == self._token_hash(token)))
            if row is not None and row.revoked_at is None:
                row.revoked_at = datetime.utcnow()


auth_service = AuthService()
