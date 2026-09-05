import json
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from locus_v2.ai.enums import PublicationStatus
from locus_v2.ai.models import AIModel, AIProvider, PromptDefinition, PromptVersion, RoutingProfile
from locus_v2.identity.models import AdminAuditEvent, User
from locus_v2.shared.clock import utc_now


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RoutingChange:
    primary_model_id: int
    fallback_model_id: int | None
    prompt_version_id: int


class AdminConfigurationService:
    def __init__(self, session: AsyncSession, actor: User) -> None:
        self.session = session
        self.actor = actor

    async def snapshot(self) -> dict:
        providers = list(
            (
                await self.session.scalars(
                    select(AIProvider)
                    .options(selectinload(AIProvider.models))
                    .order_by(AIProvider.name)
                )
            ).all()
        )
        definitions = list(
            (
                await self.session.scalars(
                    select(PromptDefinition)
                    .options(selectinload(PromptDefinition.versions))
                    .order_by(PromptDefinition.name)
                )
            ).all()
        )
        profiles = list(
            (
                await self.session.scalars(select(RoutingProfile).order_by(RoutingProfile.name))
            ).all()
        )
        return {
            "providers": [
                {
                    "id": provider.id,
                    "code": provider.code,
                    "name": provider.name,
                    "enabled": provider.enabled,
                    "models": [self._model_view(model) for model in provider.models],
                }
                for provider in providers
            ],
            "prompts": [
                {
                    "id": definition.id,
                    "code": definition.code,
                    "name": definition.name,
                    "description": definition.description,
                    "versions": [
                        {
                            "id": version.id,
                            "version": version.version,
                            "status": version.status,
                            "content": version.content,
                            "variables": version.variables_json,
                            "published_at": version.published_at,
                        }
                        for version in sorted(
                            definition.versions, key=lambda item: item.version, reverse=True
                        )
                    ],
                }
                for definition in definitions
            ],
            "routing_profiles": [self._profile_view(profile) for profile in profiles],
        }

    async def set_model_state(self, model_id: int, enabled: bool, selectable: bool) -> dict:
        model = await self.session.get(AIModel, model_id)
        if model is None:
            raise ConfigurationError("Model not found")
        if selectable and not enabled:
            raise ConfigurationError("A disabled model cannot be selectable")
        if not enabled or not selectable:
            active_route = await self.session.scalar(
                select(RoutingProfile.id).where(
                    RoutingProfile.status == PublicationStatus.PUBLISHED,
                    or_(
                        RoutingProfile.primary_model_id == model_id,
                        RoutingProfile.fallback_model_id == model_id,
                    ),
                )
            )
            if active_route is not None:
                raise ConfigurationError(
                    "This model is used by a published route; replace it before disabling it"
                )
        before = self._model_view(model)
        model.enabled = enabled
        model.selectable = selectable
        await self._audit(
            "model.state.changed",
            "ai_model",
            str(model.id),
            before,
            self._model_view(model),
        )
        await self.session.commit()
        return self._model_view(model)

    async def create_prompt_version(self, definition_id: int, content: str) -> dict:
        definition = await self.session.get(PromptDefinition, definition_id)
        if definition is None:
            raise ConfigurationError("Prompt definition not found")
        latest = await self.session.scalar(
            select(func.max(PromptVersion.version)).where(
                PromptVersion.definition_id == definition_id
            )
        )
        version = PromptVersion(
            definition_id=definition_id,
            version=(latest or 0) + 1,
            status=PublicationStatus.DRAFT,
            content=content.strip(),
            variables_json={"required": ["locale", "poi_name"]},
            created_by_user_id=self.actor.id,
        )
        if not version.content:
            raise ConfigurationError("Prompt content cannot be empty")
        self.session.add(version)
        await self.session.flush()
        await self._audit(
            "prompt.version.created",
            "prompt_version",
            str(version.id),
            None,
            {"definition_id": definition_id, "version": version.version},
        )
        await self.session.commit()
        return {"id": version.id, "version": version.version, "status": version.status}

    async def publish_prompt_version(self, version_id: int) -> dict:
        version = await self.session.get(PromptVersion, version_id)
        if version is None:
            raise ConfigurationError("Prompt version not found")
        if version.status == PublicationStatus.PUBLISHED:
            return {"id": version.id, "version": version.version, "status": version.status}
        previous_status = version.status
        current = list(
            (
                await self.session.scalars(
                    select(PromptVersion).where(
                        PromptVersion.definition_id == version.definition_id,
                        PromptVersion.status == PublicationStatus.PUBLISHED,
                    )
                )
            ).all()
        )
        current_ids = [published.id for published in current]
        affected_profiles = (
            list(
                (
                    await self.session.scalars(
                        select(RoutingProfile).where(
                            RoutingProfile.prompt_version_id.in_(current_ids)
                        )
                    )
                ).all()
            )
            if current_ids
            else []
        )
        published_at = utc_now()
        for profile in affected_profiles:
            before = self._profile_view(profile)
            profile.prompt_version_id = version.id
            profile.published_at = published_at
            await self._audit(
                "routing.prompt.advanced",
                "routing_profile",
                str(profile.id),
                before,
                self._profile_view(profile),
            )
        for published in current:
            published.status = PublicationStatus.RETIRED
        version.status = PublicationStatus.PUBLISHED
        version.published_at = published_at
        await self._audit(
            "prompt.version.published",
            "prompt_version",
            str(version.id),
            {"status": previous_status},
            {
                "status": PublicationStatus.PUBLISHED,
                "advanced_routing_profiles": [profile.id for profile in affected_profiles],
            },
        )
        await self.session.commit()
        return {"id": version.id, "version": version.version, "status": version.status}

    async def change_routing(self, profile_id: int, change: RoutingChange) -> dict:
        profile = await self.session.get(RoutingProfile, profile_id)
        if profile is None:
            raise ConfigurationError("Routing profile not found")
        primary = await self._selectable_model(change.primary_model_id)
        fallback = (
            await self._selectable_model(change.fallback_model_id)
            if change.fallback_model_id is not None
            else None
        )
        if primary.service_kind != "voice" or (fallback and fallback.service_kind != "voice"):
            raise ConfigurationError("Voice routing only accepts voice models")
        if fallback and fallback.id == primary.id:
            raise ConfigurationError("Primary and fallback models must be different")
        prompt = await self.session.get(PromptVersion, change.prompt_version_id)
        if prompt is None or prompt.status != PublicationStatus.PUBLISHED:
            raise ConfigurationError("Routing requires a published prompt")
        before = self._profile_view(profile)
        profile.primary_model_id = primary.id
        profile.fallback_model_id = fallback.id if fallback else None
        profile.prompt_version_id = prompt.id
        profile.published_at = utc_now()
        after = self._profile_view(profile)
        await self._audit("routing.changed", "routing_profile", str(profile.id), before, after)
        await self.session.commit()
        return after

    async def _selectable_model(self, model_id: int) -> AIModel:
        model = await self.session.get(AIModel, model_id)
        if model is None or not model.enabled or not model.selectable:
            raise ConfigurationError("Selected model is not available")
        return model

    async def _audit(
        self, action: str, resource_type: str, resource_id: str, before: dict | None, after: dict
    ) -> None:
        self.session.add(
            AdminAuditEvent(
                actor_user_id=self.actor.id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                before_json=json.dumps(before, default=str) if before is not None else None,
                after_json=json.dumps(after, default=str),
                trace_id=str(uuid4()),
            )
        )

    @staticmethod
    def _model_view(model: AIModel) -> dict:
        return {
            "id": model.id,
            "external_id": model.external_id,
            "display_name": model.display_name,
            "service_kind": model.service_kind,
            "adapter_code": model.adapter_code,
            "lifecycle": model.lifecycle,
            "enabled": model.enabled,
            "selectable": model.selectable,
        }

    @staticmethod
    def _profile_view(profile: RoutingProfile) -> dict:
        return {
            "id": profile.id,
            "code": profile.code,
            "name": profile.name,
            "experience_code": profile.experience_code,
            "environment": profile.environment,
            "status": profile.status,
            "voice_mode": profile.voice_mode,
            "primary_model_id": profile.primary_model_id,
            "fallback_model_id": profile.fallback_model_id,
            "prompt_version_id": profile.prompt_version_id,
        }
