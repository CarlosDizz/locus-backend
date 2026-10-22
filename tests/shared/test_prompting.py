from locus_v2.shared.prompting import IDENTITY_RULE, render_prompt, with_identity


def test_every_rendered_prompt_can_carry_the_identity_rule() -> None:
    """A «¿qué modelo eres?» el chat del mapa contestó «OpenAI» (2026-10-19)."""
    prompt = with_identity(render_prompt("Eres Locus. Responde en {locale}.", {"locale": "zh"}))

    assert prompt.startswith("Eres Locus. Responde en zh.")
    assert prompt.endswith(IDENTITY_RULE)
    assert "OpenAI" not in IDENTITY_RULE and "Gemini" not in IDENTITY_RULE
