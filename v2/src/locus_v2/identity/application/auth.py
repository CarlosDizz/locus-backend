import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from locus_v2.config import Settings
from locus_v2.identity.models import AdminSession, Role, User, UserStatus
from locus_v2.shared.clock import utc_now


class IdentityVerifier(Protocol):
    async def verify(self, credential: str, audiences: list[str]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AuthenticatedAdmin:
    id: str
    email: str
    display_name: str
    avatar_url: str | None
    roles: list[str]


class AdminAuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def login_google(
        self, credential: str, verifier: IdentityVerifier, user_agent: str | None
    ) -> tuple[AuthenticatedAdmin, str]:
        claims = await verifier.verify(credential, self.settings.google_auth_client_ids)
        email = str(claims.get("email", "")).lower()
        if not claims.get("email_verified") or email != str(self.settings.admin_email).lower():
            raise PermissionError("This Google account is not an authorized administrator")
        user = await self._upsert_admin(
            email=email,
            display_name=str(claims.get("name") or email.split("@", 1)[0]),
            avatar_url=claims.get("picture"),
            provider_subject=str(claims["sub"]),
        )
        return self._view(user), await self._create_session(user, user_agent)

    async def login_local(self, user_agent: str | None) -> tuple[AuthenticatedAdmin, str]:
        if self.settings.env != "local" or not self.settings.allow_insecure_local_admin:
            raise PermissionError("Local login is disabled")
        user = await self._upsert_admin(
            email=str(self.settings.admin_email).lower(),
            display_name="Carlos García",
            avatar_url=None,
            provider_subject=None,
        )
        return self._view(user), await self._create_session(user, user_agent)

    async def authenticate(self, raw_token: str | None) -> tuple[User, AdminSession] | None:
        if not raw_token:
            return None
        now = utc_now()
        auth_session = await self.session.scalar(
            select(AdminSession).where(
                AdminSession.token_hash == self._hash(raw_token),
                AdminSession.revoked_at.is_(None),
                AdminSession.expires_at > now,
            )
        )
        if auth_session is None:
            return None
        user = await self.session.scalar(
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == auth_session.user_id, User.status == UserStatus.ACTIVE)
        )
        if user is None or not user.has_role("admin"):
            return None
        if now - auth_session.last_seen_at > timedelta(minutes=5):
            auth_session.last_seen_at = now
            await self.session.commit()
        return user, auth_session

    async def logout(self, raw_token: str | None) -> None:
        if raw_token:
            auth_session = await self.session.scalar(
                select(AdminSession).where(AdminSession.token_hash == self._hash(raw_token))
            )
            if auth_session is not None and auth_session.revoked_at is None:
                auth_session.revoked_at = utc_now()
                await self.session.commit()

    async def _upsert_admin(
        self,
        *,
        email: str,
        display_name: str,
        avatar_url: str | None,
        provider_subject: str | None,
    ) -> User:
        roles = list(
            (
                await self.session.scalars(
                    select(Role).where(Role.code.in_(["admin", "user"]))
                )
            ).all()
        )
        user = await self.session.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == email)
        )
        if user is None:
            user = User(
                email=email,
                display_name=display_name,
                avatar_url=avatar_url,
                provider_subject=provider_subject,
                auth_provider="google",
                status=UserStatus.ACTIVE,
                roles=roles,
            )
            self.session.add(user)
            await self.session.flush()
        else:
            user.display_name = display_name
            user.avatar_url = avatar_url or user.avatar_url
            user.provider_subject = provider_subject or user.provider_subject

        existing = {role.code for role in user.roles}
        user.roles.extend(role for role in roles if role.code not in existing)
        await self.session.commit()
        await self.session.refresh(user, attribute_names=["roles"])
        return user

    async def _create_session(self, user: User, user_agent: str | None) -> str:
        raw_token = secrets.token_urlsafe(48)
        now = utc_now()
        self.session.add(
            AdminSession(
                user_id=user.id,
                token_hash=self._hash(raw_token),
                expires_at=now + timedelta(days=self.settings.admin_session_days),
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
    def _view(user: User) -> AuthenticatedAdmin:
        return AuthenticatedAdmin(
            id=user.public_id,
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            roles=sorted(role.code for role in user.roles),
        )
