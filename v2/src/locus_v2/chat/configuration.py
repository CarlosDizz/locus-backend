"""Minimal chat configuration resolver.

This is deliberately smaller than `voice.configuration.VoiceConfigurationResolver`:
single provider (no fallback chaining yet), no tool wiring, single-turn only.
It exists to prove the AI-services test harness end to end (see
docs/testing-checklist.md, Capítulo 3) — the full Chat domain (persistent
sessions/messages, tools, fallback, V1-compatible router) is still pending
and should follow the same hexagonal pattern as `voice/` when it is built.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import PublicationStatus, ServiceKind
from locus_v2.ai.models import AIModel, AIProvider, RoutingProfile
from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.shared.prompting import PromptRenderingError, localized_field, render_prompt


class ChatConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ChatRequest:
    routing_profile: str
    locale: str
    context_type: str
    context_id: str | None
    message: str


@dataclass(frozen=True)
class ResolvedChatProvider:
    model_id: int
    provider_id: int
    provider_code: str
    adapter_code: str
    model: str
    prompt: str
    provider_options: dict[str, object]


@dataclass(frozen=True)
class ResolvedChatConfiguration:
    routing_profile_id: int
    routing_profile_code: str
    prompt_version_id: int
    primary: ResolvedChatProvider
    context: dict[str, object]


class ChatConfigurationResolver:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def resolve(self, request: ChatRequest) -> ResolvedChatConfiguration:
        profile = await self.session.scalar(
            select(RoutingProfile)
            .options(
                joinedload(RoutingProfile.primary_model).joinedload(AIModel.provider),
                joinedload(RoutingProfile.prompt_version),
            )
            .where(
                RoutingProfile.code == request.routing_profile,
                RoutingProfile.environment == self.settings.env,
                RoutingProfile.service_kind == ServiceKind.CHAT,
                RoutingProfile.status == PublicationStatus.PUBLISHED,
            )
        )
        if profile is None:
            raise ChatConfigurationError("Published chat routing profile not found")

        context = await self._context(request)
        variables: dict[str, str] = {
            "locale": request.locale,
            "poi_name": str(context.get("name", "este lugar")),
            "poi_description": str(context.get("description", "")),
            "city_name": str(context.get("city_name", "")),
        }
        try:
            prompt = render_prompt(profile.prompt_version.content, variables)
        except PromptRenderingError as error:
            raise ChatConfigurationError(str(error)) from error

        model: AIModel = profile.primary_model
        provider: AIProvider = model.provider
        if not provider.enabled or not model.enabled or not model.selectable:
            raise ChatConfigurationError(f"Chat model is disabled: {model.display_name}")
        if model.service_kind != ServiceKind.CHAT:
            raise ChatConfigurationError(
                f"Configured model is not a chat model: {model.display_name}"
            )

        options = dict(model.runtime_defaults_json or {})
        options.update(profile.prompt_version.runtime_config_json or {})
        primary = ResolvedChatProvider(
            model_id=model.id,
            provider_id=provider.id,
            provider_code=provider.code,
            adapter_code=model.adapter_code,
            model=model.external_id,
            prompt=prompt,
            provider_options=options,
        )
        return ResolvedChatConfiguration(
            routing_profile_id=profile.id,
            routing_profile_code=profile.code,
            prompt_version_id=profile.prompt_version_id,
            primary=primary,
            context=context,
        )

    async def _context(self, request: ChatRequest) -> dict[str, object]:
        if request.context_type != "poi" or request.context_id is None:
            raise ChatConfigurationError("A POI context is required for this chat experience")
        poi = await self.session.scalar(
            select(Poi)
            .options(joinedload(Poi.city))
            .where(Poi.public_id == request.context_id, Poi.is_active.is_(True))
        )
        if poi is None:
            raise ChatConfigurationError("POI not found")
        language = request.locale.split("-", 1)[0].lower()
        name = localized_field(poi.names_json, request.locale, language) or poi.name
        description = (
            localized_field(poi.short_descriptions_json, request.locale, language)
            or poi.short_description
        )
        return {
            "public_id": poi.public_id,
            "name": name,
            "description": description,
            "city_name": poi.city.name if poi.city else "",
        }
