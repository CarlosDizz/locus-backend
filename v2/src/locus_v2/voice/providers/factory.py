from locus_v2.config import Settings
from locus_v2.voice.providers.future_openai_live import FutureOpenAILiveProvider
from locus_v2.voice.providers.mock import MockLiveProvider
from locus_v2.voice.providers.registry import ProviderRegistry


def build_provider_registry(settings: Settings) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(MockLiveProvider.code, MockLiveProvider)
    registry.register(FutureOpenAILiveProvider.code, FutureOpenAILiveProvider)
    return registry
