"""Mobile-facing identity: the Ionic app's compatibility surface.

Kept deliberately separate from AdminAuthService: different session table
(``UserSession`` vs ``AdminSession``), no cookie, no admin role requirement, and a
bearer token returned in the response body rather than set on the response.

See docs/roadmap.md section 5 and 11 for the V1 contract this mirrors, and section 0
for why sessions are not carried over from V1 (one forced re-login on cutover).
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.application.onboarding import create_signup_wallet
from locus_v2.config import Settings
from locus_v2.identity.models import Role, User, UserSession, UserStatus
from locus_v2.shared.clock import utc_now
from locus_v2.shared.mobile_ids import mobile_id

MOBILE_SESSION_DAYS = 30


class IdentityVerifier(Protocol):
    async def verify(self, credential: str, audiences: list[str]) -> dict[str, Any]: ...


class MobileAuthError(RuntimeError):
    """Raised with the same wording/semantics V1 already returns to Ionic."""


@dataclass(frozen=True)
class MobileUserView:
    id: int
    email: str
    display_name: str
    auth_provider: str
    avatar_url: str
    is_active: bool


class MobileAuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def register(
        self, *, email: str, password: str, display_name: str
    ) -> tuple[MobileUserView, str]:
        if not self.settings.auth_enable_password_auth:
            raise MobileAuthError("El acceso con email y contraseña está desactivado")
        raise MobileAuthError("El registro por email todavía no está disponible en V2")

    async def login(self, *, email: str, password: str) -> tuple[MobileUserView, str]:
        if not self.settings.auth_enable_password_auth:
            raise MobileAuthError("Usa el acceso con Google")
        raise MobileAuthError("Credenciales no válidas")

    async def login_google(
        self, raw_id_token: str, verifier: IdentityVerifier, user_agent: str | None
    ) -> tuple[MobileUserView, str]:
        client_ids = [client_id for client_id in self.settings.google_auth_client_ids if client_id]
        if not client_ids:
            raise MobileAuthError("El acceso con Google no está configurado en el backend")

        try:
            claims = await verifier.verify(raw_id_token, client_ids)
        except ValueError as exc:
            raise MobileAuthError("No he podido verificar la cuenta de Google") from exc

        email = str(claims.get("email", "")).strip().lower()
        google_sub = str(claims.get("sub", "")).strip()
        email_verified = bool(claims.get("email_verified"))
        if not email or not google_sub or not email_verified:
            raise MobileAuthError("La cuenta de Google no tiene un email verificado")

        display_name = str(claims.get("name") or email.split("@", 1)[0]).strip()
        avatar_url = str(claims.get("picture") or "").strip()

        user = await self._find_or_create(
            email=email,
            google_sub=google_sub,
            display_name=display_name,
            avatar_url=avatar_url,
        )
        if user.status != UserStatus.ACTIVE:
            raise MobileAuthError("Esta cuenta está desactivada")

        raw_token = await self._create_session(user, user_agent)
        return self._view(user), raw_token

    async def authenticate(self, raw_token: str | None) -> User | None:
        if not raw_token:
            return None
        now = utc_now()
        auth_session = await self.session.scalar(
            select(UserSession).where(
                UserSession.token_hash == self._hash(raw_token),
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
        )
        if auth_session is None:
            return None
        user = await self.session.scalar(
            select(User).where(User.id == auth_session.user_id, User.status == UserStatus.ACTIVE)
        )
        if user is None:
            return None
        if now - auth_session.last_seen_at > timedelta(minutes=5):
            auth_session.last_seen_at = now
            await self.session.commit()
        return user

    async def logout(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        auth_session = await self.session.scalar(
            select(UserSession).where(UserSession.token_hash == self._hash(raw_token))
        )
        if auth_session is not None and auth_session.revoked_at is None:
            auth_session.revoked_at = utc_now()
            await self.session.commit()

    async def _find_or_create(
        self, *, email: str, google_sub: str, display_name: str, avatar_url: str
    ) -> User:
        user = await self.session.scalar(select(User).where(User.provider_subject == google_sub))
        if user is None:
            user = await self.session.scalar(select(User).where(User.email == email))

        if user is None:
            role = await self.session.scalar(select(Role).where(Role.code == "user"))
            user = User(
                email=email,
                display_name=display_name,
                avatar_url=avatar_url or None,
                auth_provider="google",
                provider_subject=google_sub,
                status=UserStatus.ACTIVE,
                roles=[role] if role else [],
            )
            self.session.add(user)
            await self.session.flush()
            await create_signup_wallet(
                self.session, user.id, self.settings.billing_signup_bonus_cents
            )
        else:
            if user.provider_subject and user.provider_subject != google_sub:
                raise MobileAuthError("Esta cuenta ya está vinculada a otro acceso de Google")
            user.provider_subject = google_sub
            if display_name and not user.display_name:
                user.display_name = display_name
            if avatar_url:
                user.avatar_url = avatar_url

        await self.session.commit()
        return user

    async def _create_session(self, user: User, user_agent: str | None) -> str:
        raw_token = secrets.token_urlsafe(32)
        now = utc_now()
        self.session.add(
            UserSession(
                user_id=user.id,
                token_hash=self._hash(raw_token),
                expires_at=now + timedelta(days=MOBILE_SESSION_DAYS),
                last_seen_at=now,
                user_agent=(user_agent or "")[:500] or None,
            )
        )
        await self.session.commit()
        return raw_token

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _view(user: User) -> MobileUserView:
        return MobileUserView(
            id=mobile_id(user),
            email=user.email,
            display_name=user.display_name,
            auth_provider=user.auth_provider,
            avatar_url=user.avatar_url or "",
            is_active=user.status == UserStatus.ACTIVE,
        )
