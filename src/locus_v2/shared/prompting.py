"""Prompt rendering helpers shared by every service kind (voice, chat, ...).

Kept separate from any single domain so Voice and Chat configuration
resolvers do not duplicate the same localization and template logic.
"""

from string import Formatter


class PromptRenderingError(ValueError):
    pass


def localized_field(values: dict[str, object], locale: str, language: str) -> str | None:
    for key in (locale, locale.lower(), language, "local", "en"):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def render_prompt(template: str, variables: dict[str, str]) -> str:
    required = {name for _, name, _, _ in Formatter().parse(template) if name}
    missing = required - variables.keys()
    if missing:
        raise PromptRenderingError(f"Prompt variables are missing: {', '.join(sorted(missing))}")
    return template.format_map(variables)
