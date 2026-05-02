import json
from pathlib import Path


class PromptService:
    def __init__(self) -> None:
        self.base_path = Path(__file__).resolve().parent.parent / "prompts"
        self._cache: dict[str, dict] = {}

    def load_prompt(self, name: str) -> dict:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        path = self.base_path / name
        with path.open("r", encoding="utf-8") as handle:
            prompt = json.load(handle)
        self._cache[name] = prompt
        return prompt

    def render(self, prompt_name: str, context: dict) -> str:
        shared = self.load_prompt("shared_rules.json")
        prompt = self.load_prompt(prompt_name)

        main = prompt["instrucciones_principales"]
        tools = prompt["tools"]

        sections = [
            "INSTRUCCIONES PRINCIPALES:\n"
            + "\n".join(
                [
                    f"IDENTIDAD:\n{main['identidad']}",
                    f"CONTEXTO DE LA APLICACION:\n{main['contexto_aplicacion']}",
                    "MISION:\n" + "\n".join(f"- {item}" for item in main["mision"]),
                    "REGLAS:\n" + "\n".join(f"- {item}" for item in shared["hard_rules"] + main["reglas"]),
                    "ESTILO:\n" + "\n".join(f"- {item}" for item in main["estilo"]),
                ]
            ),
            f"CONTEXTO DE USUARIO:\n{prompt['contexto_usuario']['descripcion']}",
            "TOOLS:\n"
            + "\n".join(
                [
                    "POLITICA GENERAL:\n" + "\n".join(f"- {item}" for item in tools["politica_general"]),
                    "LISTA DISPONIBLE:\n"
                    + "\n".join(f"- {name}: {description}" for name, description in tools["lista"].items()),
                ]
            ),
        ]

        dynamic_lines = []
        for key, label in prompt["dynamic_context"].items():
            value = context.get(key)
            dynamic_lines.append(f"{label}:\n{value or '(sin dato)'}")
        sections.append("CONTEXTO DINAMICO:\n" + "\n\n".join(dynamic_lines))
        result = "\n\n".join(sections)
        for key, value in context.items():
            result = result.replace("{{" + key + "}}", value or "")
        return result


prompt_service = PromptService()
