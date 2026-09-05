from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import PublicationStatus, ServiceKind
from locus_v2.ai.models import AIModel, AIProvider, RoutingProfile
from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.shared.prompting import PromptRenderingError, localized_field, render_prompt
from locus_v2.voice.protocol import SessionStart
from locus_v2.voice.providers.base import LiveSessionConfig


class VoiceConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedProvider:
    model_id: int
    provider_code: str
    adapter_code: str
    config: LiveSessionConfig


@dataclass(frozen=True)
class ResolvedVoiceConfiguration:
    routing_profile_id: int
    routing_profile_code: str
    prompt_version_id: int
    primary: ResolvedProvider
    fallback: ResolvedProvider | None
    locale: str
    context_type: str
    context_public_id: str | None
    context: dict
    snapshot: dict


class VoiceConfigurationResolver:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def resolve(self, request: SessionStart) -> ResolvedVoiceConfiguration:
        profile = await self.session.scalar(
            select(RoutingProfile)
            .options(
                joinedload(RoutingProfile.primary_model).joinedload(AIModel.provider),
                joinedload(RoutingProfile.fallback_model).joinedload(AIModel.provider),
                joinedload(RoutingProfile.prompt_version),
            )
            .where(
                RoutingProfile.code == request.routing_profile,
                RoutingProfile.environment == self.settings.env,
                RoutingProfile.service_kind == ServiceKind.VOICE,
                RoutingProfile.status == PublicationStatus.PUBLISHED,
            )
        )
        if profile is None:
            raise VoiceConfigurationError("Published voice routing profile not found")

        context = await self._context(request)
        variables = {
            "locale": request.locale,
            "poi_name": context.get("name", "este lugar"),
            "poi_description": context.get("description", ""),
            "city_name": context.get("city_name", ""),
        }
        try:
            prompt = render_prompt(profile.prompt_version.content, variables)
        except PromptRenderingError as error:
            raise VoiceConfigurationError(str(error)) from error
        tools = [
            {
                "type": "function",
                "name": tool["code"],
                "description": tool["description"],
                "parameters": tool["schema"],
            }
            for tool in profile.prompt_version.tools_json
            if tool.get("enabled", True)
        ]
        runtime_config = _deep_merge(
            profile.config_json, profile.prompt_version.runtime_config_json
        )
        primary = self._provider(
            profile.primary_model,
            prompt,
            request.locale,
            request.audio_format,
            runtime_config,
            tools,
        )
        fallback = (
            self._provider(
                profile.fallback_model,
                prompt,
                request.locale,
                request.audio_format,
                runtime_config,
                tools,
            )
            if profile.fallback_model is not None
            else None
        )
        snapshot = {
            "routing_profile": {"id": profile.id, "code": profile.code},
            "prompt_version_id": profile.prompt_version_id,
            "tools": profile.prompt_version.tools_json,
            "runtime_config": runtime_config,
            "primary": _provider_snapshot(primary),
            "fallback": _provider_snapshot(fallback) if fallback else None,
            "voice_mode": profile.voice_mode,
            "locale": request.locale,
            "context": context,
            "audio_persistence": False,
        }
        return ResolvedVoiceConfiguration(
            routing_profile_id=profile.id,
            routing_profile_code=profile.code,
            prompt_version_id=profile.prompt_version_id,
            primary=primary,
            fallback=fallback,
            locale=request.locale,
            context_type=request.context_type,
            context_public_id=request.context_id,
            context=context,
            snapshot=snapshot,
        )

    async def _context(self, request: SessionStart) -> dict:
        if request.context_type != "poi" or request.context_id is None:
            raise VoiceConfigurationError("A POI context is required for this voice experience")
        poi = await self.session.scalar(
            select(Poi).options(joinedload(Poi.city)).where(
                Poi.public_id == request.context_id,
                Poi.is_active.is_(True),
            )
        )
        if poi is None:
            raise VoiceConfigurationError("POI not found")
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
            "wikidata_id": poi.wikidata_id,
            "wikipedia_title": poi.wikipedia_title,
        }

    @staticmethod
    def _provider(
        model: AIModel,
        prompt: str,
        locale: str,
        audio_format,
        runtime_config: dict,
        tools: list[dict],
    ) -> ResolvedProvider:
        provider: AIProvider = model.provider
        if not provider.enabled or not model.enabled or not model.selectable:
            raise VoiceConfigurationError(f"Voice model is disabled: {model.display_name}")
        if model.service_kind != ServiceKind.VOICE:
            raise VoiceConfigurationError(
                f"Configured model is not a voice model: {model.display_name}"
            )
        options = _deep_merge(model.runtime_defaults_json, runtime_config)
        provider_overrides = options.pop("provider_overrides", {})
        options = _deep_merge(options, provider_overrides.get(provider.code, {}))
        voice = options.pop("voice", None)
        return ResolvedProvider(
            model_id=model.id,
            provider_code=provider.code,
            adapter_code=model.adapter_code,
            config=LiveSessionConfig(
                model=model.external_id,
                prompt=prompt,
                locale=locale,
                voice=voice,
                audio_format=audio_format,
                tools=tools,
                provider_options=options,
            ),
        )


def _provider_snapshot(provider: ResolvedProvider) -> dict:
    return {
        "model_id": provider.model_id,
        "provider_code": provider.provider_code,
        "adapter_code": provider.adapter_code,
        "external_model": provider.config.model,
        "runtime_config": provider.config.provider_options,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
