from collections.abc import Callable

from locus_v2.voice.providers.base import LiveProvider

ProviderFactory = Callable[[], LiveProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, adapter_code: str, factory: ProviderFactory) -> None:
        if adapter_code in self._factories:
            raise ValueError(f"Provider adapter already registered: {adapter_code}")
        self._factories[adapter_code] = factory

    def create(self, adapter_code: str) -> LiveProvider:
        try:
            return self._factories[adapter_code]()
        except KeyError as exc:
            raise LookupError(f"Unknown provider adapter: {adapter_code}") from exc

    def available(self) -> list[str]:
        return sorted(self._factories)
