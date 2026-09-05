import asyncio

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from locus_v2.ai.enums import Lifecycle, PublicationStatus, ServiceKind, VoiceMode
from locus_v2.ai.models import AIModel, AIProvider, PromptDefinition, PromptVersion, RoutingProfile
from locus_v2.config import get_settings
from locus_v2.identity.models import Role, User, UserStatus
from locus_v2.infrastructure.database.session import get_database
from locus_v2.shared.clock import utc_now


PROVIDERS = (
    ("openai", "OpenAI"),
    ("google", "Google"),
    ("locus", "Locus test adapter"),
)

MODELS = (
    ("openai", "gpt-5-mini", "GPT-5 mini", ServiceKind.CHAT, "openai_responses", Lifecycle.STABLE, True),
    ("openai", "gpt-realtime-mini", "GPT Realtime mini", ServiceKind.VOICE, "openai_realtime", Lifecycle.STABLE, True),
    ("google", "gemini-3.1-flash-live-preview", "Gemini 3.1 Flash Live", ServiceKind.VOICE, "gemini_live", Lifecycle.PREVIEW, True),
    ("openai", "gpt-live", "GPT Live", ServiceKind.VOICE, "openai_live", Lifecycle.DISABLED, False),
    ("locus", "mock-live", "Mock Live", ServiceKind.VOICE, "mock_live", Lifecycle.STABLE, True),
)

VOICE_PROMPT = """Eres Locus, un guía local natural y bien documentado. Habla en {locale} y céntrate en {poi_name}. Usa las herramientas siempre que necesites hechos, fechas o contexto; no rellenes con frases genéricas ni expliques tus limitaciones. Responde primero a lo que pide la persona y sugiere escenas o paradas solo cuando aporten valor."""


async def seed() -> None:
    settings = get_settings()
    database = get_database()
    async with database.sessions() as session:
        roles: dict[str, Role] = {}
        for code, name in (("admin", "Administrator"), ("user", "User")):
            role = await session.scalar(select(Role).where(Role.code == code))
            if role is None:
                role = Role(code=code, name=name, is_system=True)
                session.add(role)
                await session.flush()
            roles[code] = role

        admin_email = str(settings.admin_email).lower()
        admin = await session.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == admin_email)
        )
        if admin is None:
            admin = User(
                email=admin_email,
                display_name="Carlos García",
                auth_provider="google",
                status=UserStatus.ACTIVE,
                roles=list(roles.values()),
            )
            session.add(admin)
            await session.flush()
        else:
            admin.roles = list(roles.values())

        providers: dict[str, AIProvider] = {}
        for code, name in PROVIDERS:
            provider = await session.scalar(select(AIProvider).where(AIProvider.code == code))
            if provider is None:
                provider = AIProvider(code=code, name=name, enabled=True, config_json={})
                session.add(provider)
                await session.flush()
            providers[code] = provider

        models: dict[str, AIModel] = {}
        for provider_code, external_id, name, kind, adapter, lifecycle, selectable in MODELS:
            provider = providers[provider_code]
            model = await session.scalar(
                select(AIModel).where(
                    AIModel.provider_id == provider.id,
                    AIModel.external_id == external_id,
                )
            )
            if model is None:
                model = AIModel(
                    provider_id=provider.id,
                    external_id=external_id,
                    display_name=name,
                    service_kind=kind,
                    adapter_code=adapter,
                    lifecycle=lifecycle,
                    enabled=selectable,
                    selectable=selectable,
                    capabilities_json={},
                )
                session.add(model)
                await session.flush()
            models[adapter] = model

        definition = await session.scalar(
            select(PromptDefinition).where(PromptDefinition.code == "voice.poi.guide")
        )
        if definition is None:
            definition = PromptDefinition(
                code="voice.poi.guide",
                name="Guía de voz para POI",
                description="Prompt base versionado para experiencias de escena y paradas.",
            )
            session.add(definition)
            await session.flush()

        prompt = await session.scalar(
            select(PromptVersion).where(
                PromptVersion.definition_id == definition.id,
                PromptVersion.version == 1,
            )
        )
        if prompt is None:
            prompt = PromptVersion(
                definition_id=definition.id,
                version=1,
                status=PublicationStatus.PUBLISHED,
                content=VOICE_PROMPT,
                variables_json={"required": ["locale", "poi_name"]},
                published_at=utc_now(),
            )
            session.add(prompt)
            await session.flush()

        profile = await session.scalar(
            select(RoutingProfile).where(RoutingProfile.code == "voice.poi.local")
        )
        if profile is None:
            profile = RoutingProfile(
                code="voice.poi.local",
                name="POI voice local",
                experience_code="poi_guide",
                environment=settings.env,
                status=PublicationStatus.PUBLISHED,
                voice_mode=VoiceMode.PUSH_TO_TALK,
                primary_model_id=models["gemini_live"].id,
                fallback_model_id=models["openai_realtime"].id,
                prompt_version_id=prompt.id,
                config_json={"audio_persistence": False},
                published_at=utc_now(),
            )
            session.add(profile)

        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
