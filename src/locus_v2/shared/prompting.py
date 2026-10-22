"""Prompt rendering helpers shared by every service kind (voice, chat, ...).

Kept separate from any single domain so Voice and Chat configuration
resolvers do not duplicate the same localization and template logic.
"""

from string import Formatter


class PromptRenderingError(ValueError):
    pass


# Va en código y no en las plantillas: los prompts viven versionados en la base
# (el de las llamadas en grupo lo publica el panel) y la regla tiene que valer en
# todos. Un «no menciones modelos» genérico no bastó: a «¿qué modelo eres? ¿quién
# te desarrolló?» el chat del mapa contestó «OpenAI» (2026-10-19).
IDENTITY_RULE = (
    "Identidad: eres Locus, el guía de la app Locus. Si te preguntan qué modelo o "
    "inteligencia artificial eres, quién te ha creado o qué empresa o tecnología hay "
    "detrás, di solo que eres el guía de Locus. No nombres a ningún proveedor ni "
    "modelo de IA, ni confirmes o niegues el que te sugieran."
)


def with_identity(prompt: str) -> str:
    return f"{prompt}\n\n{IDENTITY_RULE}"


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
