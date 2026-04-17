from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, UTC

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
            is_active=user.is_active,
        )

    def register_user(self, email: str, password: str, display_name: str = "") -> tuple[str, UserResponse]:
        normalized_email = self._normalize_email(email)
        with session_scope() as db:
            existing = db.scalar(select(User).where(User.email == normalized_email))
            if existing is not None:
                raise AuthError("Ya existe un usuario con ese email")

            user = User(
                email=normalized_email,
                password_hash=self._hash_password(password),
                display_name=display_name.strip(),
                is_active=True,
            )
            db.add(user)
            db.flush()

            wallet = Wallet(user_id=user.id, currency="EUR", balance_cents=0)
            db.add(wallet)
            db.flush()

            token_value, token_row = self._build_token(user.id)
            db.add(token_row)
            from app.services.billing_service import billing_service

            billing_service.add_ledger_entry(
                db=db,
                wallet=wallet,
                user_id=user.id,
                entry_type="signup_bonus",
                amount_cents=settings.billing_signup_bonus_cents,
                description="Saldo promocional de bienvenida",
                reference_type="user",
                reference_id=str(user.id),
            )
            return token_value, self._user_to_schema(user)

    def _build_token(self, user_id: int) -> tuple[str, AuthToken]:
        raw_token = secrets.token_urlsafe(32)
        token = AuthToken(
            user_id=user_id,
            token_hash=self._token_hash(raw_token),
            expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=self.token_ttl_days),
        )
        return raw_token, token

    def login(self, email: str, password: str) -> tuple[str, UserResponse]:
        normalized_email = self._normalize_email(email)
        with session_scope() as db:
            user = db.scalar(select(User).where(User.email == normalized_email))
            if user is None or not user.is_active or not self._verify_password(password, user.password_hash):
                raise AuthError("Credenciales no válidas")
            token_value, token_row = self._build_token(user.id)
            db.add(token_row)
            return token_value, self._user_to_schema(user)

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
