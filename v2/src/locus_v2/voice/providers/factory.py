from locus_v2.config import Settings
from locus_v2.voice.providers.future_openai_live import FutureOpenAILiveProvider
from locus_v2.voice.providers.gemini_live import GeminiLiveProvider
from locus_v2.voice.providers.mock import MockLiveProvider
from locus_v2.voice.providers.openai_realtime import OpenAIRealtimeProvider
from locus_v2.voice.providers.registry import ProviderRegistry


def build_provider_registry(settings: Settings) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(MockLiveProvider.code, MockLiveProvider)
    registry.register(FutureOpenAILiveProvider.code, FutureOpenAILiveProvider)
    openai_key = (
        settings.openai_api_key.get_secret_value().strip()
        if settings.openai_api_key is not None
        else ""
    )
    if openai_key:
        registry.register(
            OpenAIRealtimeProvider.code,
            lambda: OpenAIRealtimeProvider(openai_key),
        )
    gemini_key = (
        settings.gemini_api_key.get_secret_value().strip()
        if settings.gemini_api_key is not None
        else ""
    )
    if gemini_key:
        registry.register(
            GeminiLiveProvider.code,
            lambda: GeminiLiveProvider(gemini_key),
        )
    return registry
